#!/usr/bin/env bash
# Scan this repo for private infrastructure details and credentials before publishing.
#
#   scripts/check-public.sh           precise patterns; exit 1 on any finding
#   scripts/check-public.sh --broad   also list every secret / password / api-key / bearer / authorization mention
#                                     for a manual read (informational, does not fail)
#
# Scans every tracked and untracked (not ignored) file except the vendor/TensorFold submodule and this script.
#
# Your own private names (hostnames, user names, tailnet name, home paths, internal repos) belong in
# .check-public.local (git-ignored, one extended regex per line, # comments allowed) or in CHECK_PUBLIC_EXTRA='a|b'.
# They are matched case-insensitively. Lines of the form  regex<TAB>path-glob  skip that path for that regex
# (for a known false positive such as a benchmark question).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SELF="scripts/check-public.sh"
LOCAL_LIST=".check-public.local"

# 0600's image-URL hardening refuses private / link-local / CGNAT / metadata addresses; its tests, patch and doc list
# example addresses of each range (10.0.0.1, 192.168.1.1, 100.64.0.1, 100.100.100.200, ...), none of them ours.
SSRF_FIXTURES="tests/test_upstream_ports.py patches/0600-glm-upstream-ports.patch docs/UPSTREAM-PORTS.md"

# name|regex (grep -E, case-insensitive)|path globs excluded for this pattern (space-separated, may be empty)
PATTERNS=(
    "private IPv4 192.168/16|\\b192\\.168\\.[0-9]{1,3}\\.[0-9]{1,3}\\b|$SSRF_FIXTURES"
    "private IPv4 10/8 (the launcher tests use a fake 10.0.0.1)|\\b10\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\b|tests/test_serve_ops.py $SSRF_FIXTURES"
    "private IPv4 172.16/12|\\b172\\.(1[6-9]|2[0-9]|3[01])\\.[0-9]{1,3}\\.[0-9]{1,3}\\b|$SSRF_FIXTURES"
    "CGNAT / Tailscale IPv4|\\b100\\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\\.[0-9]{1,3}\\.[0-9]{1,3}\\b|$SSRF_FIXTURES"
    'Tailscale IPv6|fd7a:|'
    'Tailscale names|\.ts\.net\b|tailnet|tailscale|'
    'gmail address|[a-z0-9._%+-]+@gmail\.com|'
    'MAC address|\b([0-9a-f]{2}:){5}[0-9a-f]{2}\b|'
    'serial number|serial[ _-]?(number|no\.?)[ :=]|'
    'SSH public key|ssh-(ed25519|rsa|dss) AAAA|'
    'private key block|BEGIN [A-Z ]*PRIVATE KEY|'
    'Hugging Face token|\bhf_[A-Za-z0-9]{30,}|'
    'OpenAI-style key|\bsk-[A-Za-z0-9_-]{20,}|'
    'GitHub token|\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}|'
    'Slack token|\bxox[abprs]-[A-Za-z0-9-]{10,}|'
    'AWS key id|\bAKIA[0-9A-Z]{16}\b|'
    'literal bearer token|bearer [A-Za-z0-9._~+/-]{6,}|'
    "literal secret assignment|(password|passwd|secret|api[_-]?key|access[_-]?token)[\"']?[[:space:]]*[:=][[:space:]]*[\"'][^\"'<\$ {]{6,}[\"']|"
    'home directories|/home/[a-z0-9_-]+/|'
)
if [[ -f "$LOCAL_LIST" ]]; then
    while IFS= read -r line; do
        [[ -z "${line//[[:space:]]/}" || "$line" == \#* ]] && continue
        re="${line%%$'\t'*}"; skip=""
        [[ "$line" == *$'\t'* ]] && skip="${line#*$'\t'}"
        PATTERNS+=("local list ($LOCAL_LIST)|$re|$skip")
    done < "$LOCAL_LIST"
fi
[[ -n "${CHECK_PUBLIC_EXTRA:-}" ]] && PATTERNS+=("CHECK_PUBLIC_EXTRA|$CHECK_PUBLIC_EXTRA|")

mapfile -t FILES < <(git ls-files -co --exclude-standard 2>/dev/null |
                     grep -vE '^vendor/TensorFold(/|$)' | grep -vxF "$SELF" | grep -vxF "$LOCAL_LIST" |
                     while IFS= read -r f; do [[ -f "$f" ]] && printf '%s\n' "$f"; done)
if [[ ${#FILES[@]} -eq 0 ]]; then echo "no files to scan (not a git checkout?)" >&2; exit 2; fi

# /home/<name>/ is allowed for these generic placeholders
HOME_OK='/home/(user|you|<you>|<user>|ubuntu)/'

fail=0
for entry in "${PATTERNS[@]}"; do
    name="${entry%%|*}"; rest="${entry#*|}"
    skip="${rest##*|}"; re="${rest%|*}"
    hits=$(for f in "${FILES[@]}"; do
               # shellcheck disable=SC2053
               skipped=0
               read -ra globs <<< "$skip"
               for g in "${globs[@]}"; do [[ "$f" == $g ]] && { skipped=1; break; }; done
               [[ $skipped -eq 1 ]] && continue
               grep -HnIiE -- "$re" "$f" 2>/dev/null
           done)
    [[ "$name" == "home directories" && -n "$hits" ]] && hits=$(grep -viE "$HOME_OK" <<< "$hits")
    if [[ -n "$hits" ]]; then
        fail=1
        echo "== $name"
        cut -c1-220 <<< "$hits"
    fi
done

# e-mail addresses other than placeholder / example domains (model outputs in results/ invent a few)
emails=$(for f in "${FILES[@]}"; do grep -HnoIiE '[a-z0-9][a-z0-9._%+-]*@[a-z0-9-]+(\.[a-z0-9-]+)*\.[a-z]{2,}' "$f" 2>/dev/null; done |
         grep -viE '@(([a-z0-9-]+\.)*example(\.(com|org|net))?|company\.com|novatech\.io)$|:noreply@anthropic\.com$' | cut -c1-200)
if [[ -n "$emails" ]]; then fail=1; echo "== e-mail addresses (not on the example-domain allowlist)"; echo "$emails"; fi

# files that should never be published
bad=$(printf '%s\n' "${FILES[@]}" | grep -E '\.(log|out|pem|key)$|(^|/)\.env$|^config/[^/]+\.env$|id_(rsa|ed25519)' || true)
if [[ -n "$bad" ]]; then fail=1; echo "== files that should not be published"; echo "$bad"; fi

if [[ "${1:-}" == --broad ]]; then
    echo "== broad keyword list (manual review; informational)"
    for f in "${FILES[@]}"; do
        grep -HnIiE 'secret|password|api[_-]?key|bearer|authorization' "$f" 2>/dev/null
    done | cut -c1-200
fi

if [[ $fail -ne 0 ]]; then
    echo; echo "check-public: FINDINGS above (${#FILES[@]} files scanned)"; exit 1
fi
echo "check-public: clean (${#FILES[@]} files scanned$([[ -f "$LOCAL_LIST" ]] && echo ", plus $LOCAL_LIST"))"
