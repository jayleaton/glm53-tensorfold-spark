#!/usr/bin/env bash
# W19 capture (a TRACE=1 load; W11's cap.sh + a prefill segment): warm-up, the same requests uncaptured (ctl-NAME.jsonl),
# then ONE nsys window on both ranks (cap-NAME.jsonl), segments 4 s apart: (1) a cold ~24.5k prompt (ab.py's filler,
# unique tag, 16 tokens), (2) prose chat 256, (3) code (LRU) 384, (4) 4 x prose 384 concurrent. Greedy, thinking off.
#   cap.sh NAME
cd $HOME/glm53-tensorfold-spark
R=results/W19; N=$1; C=$R/nsysctl.sh; Q="python3 $R/w19req.py"
pf() { python3 - "$1" <<'PY'
import json, sys, time, urllib.request
F = ("The quick brown fox jumps over the lazy dog while the committee reviews quarterly logistics, "
     "inventory forecasts, and the maintenance schedule for the northern warehouse. ")
p = f"[w19cap-{time.time_ns()}] " + F * (24500 // 32) + "\n\nIn one sentence, what is the committee reviewing?"
b = {"model": "GLM-5.3-Flash-EXL3", "messages": [{"role": "user", "content": p}], "max_tokens": 16, "temperature": 0,
     "top_p": 1, "chat_template_kwargs": {"enable_thinking": False}}
t0 = time.time()
d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions",
    data=json.dumps(b).encode(), headers={"Content-Type": "application/json"}), timeout=900))
tf = d.get("tensorfold", {})
r = {"kind": "prefill", "t0": t0, "t1": time.time(), "prompt": d["usage"]["prompt_tokens"], "prefill_s": tf.get("prefill_s"),
     "sha": tf.get("sha256")}
open(sys.argv[1], "a").write(json.dumps(r) + "\n")
print("prefill", r["prompt"], r["prefill_s"], round(r["prompt"] / r["prefill_s"], 1) if r["prefill_s"] else None)
PY
}
$Q one $R/warm-cap-$N.jsonl chat 64 > /dev/null
$Q conc $R/warm-cap-$N.jsonl 4 64 > /dev/null
for f in ctl cap; do
  J=$R/$f-$N.jsonl
  if [[ $f == cap ]]; then echo "nsys start $(date +%T.%N)" >> $R/cap-$N.times; $C start cap-$N; sleep 2; fi
  pf $J; sleep 4
  $Q one $J chat 256; sleep 4
  $Q one $J code-lru 384; sleep 4
  $Q conc $J 4 384; sleep 2
  if [[ $f == cap ]]; then $C stop; echo "nsys stop $(date +%T.%N)" >> $R/cap-$N.times; fi
  sleep 3
done
for i in $(seq 1 120); do
  a=$(ls /var/tmp/w19/out/cap-$N-r0.nsys-rep 2>/dev/null); b=$(ssh -o BatchMode=yes $WORKER_SSH ls /var/tmp/w19/out/cap-$N-r1.nsys-rep 2>/dev/null)
  if [[ -n $a && -n $b ]] && ! pgrep -f QdstrmImporter >/dev/null && ! ssh -o BatchMode=yes $WORKER_SSH pgrep -f QdstrmImporter >/dev/null; then echo "report cap-$N ready $(date +%T)"; break; fi
  sleep 5
done
ls -la /var/tmp/w19/out | grep cap-$N; ssh -o BatchMode=yes $WORKER_SSH ls -la /var/tmp/w19/out | grep cap-$N
echo "cap $N DONE $(date +%T)"
