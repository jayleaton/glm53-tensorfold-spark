#!/usr/bin/env bash
# W18 (from W17) per-load A/B set: ab.sh TAG = warm-up, exact 10/10, batchexact 4/4, W9 transcripts (== W9 load A), ab.py 24.5k /
# 98k once (reply sha 8794a3463259cc2f, prefill), glmbench tf,tweet,kit,edit x3 (1 stream; reply hashes), concurrent 4
# streams x3 twice (6 reps), lone requests over slots 0-3 (slots.py), /health, engine errors
cd $HOME/glm53-tensorfold-spark
B=http://127.0.0.1:8001; export BASE=$B; M=GLM-5.3-Flash-EXL3; R=results/W19; tag=$1
echo "ab $tag start $(date +%T)"
python3 $R/req-p.py prefill $R/warm-$tag.jsonl 12000 8 '{}' > /dev/null 2>&1
python3 bench/glmbench.py --base $B --model $M --suites exact --out $R/exact-$tag.json > $R/exact-$tag.log 2>&1
python3 bench/multiturn.py --base $B --model $M --modes batchexact --out $R/batchexact-$tag.json > $R/batchexact-$tag.log 2>&1
echo "exact: $(tail -1 $R/exact-$tag.log | grep -o true | wc -l)/10"; grep -E 'batched == alone' $R/batchexact-$tag.log
python3 $R/transcripts-p.py $R/transcripts-$tag.json > $R/transcripts-$tag.log 2>&1
python3 - $R/transcripts-$tag.log results/W9/transcripts-A.log <<'PY'
import sys; a, b = [open(f).read().splitlines() for f in sys.argv[1:3]]
print("transcripts alone == W9 A:", bool(a) and a[0] == b[0], "|", a[-1] if a else "NONE")
PY
python3 $R/ab-p.py $R/ab-$tag.json 24500,98000 '{"prod":{}}' > $R/ab-$tag.log 2>&1; cut -c1-110,200- $R/ab-$tag.log | tail -2
python3 bench/glmbench.py --base $B --model $M --suites tf,tweet,kit,edit --reps 3 --long-tokens 512 --label $tag --out $R/glmbench-$tag.json > $R/glmbench-$tag.log 2>&1
grep -E 'median' $R/glmbench-$tag.log | cut -c1-80
for pass in a b; do
  python3 bench/multiturn.py --base $B --model $M --modes concurrent --streams 4 --reps 3 --long-tokens 512 --out $R/conc-$tag-$pass.json > $R/conc-$tag-$pass.log 2>&1
  grep -E 'streams rep' $R/conc-$tag-$pass.log | cut -c1-60
done
python3 $R/slots.py $R/slots-$tag.json > $R/slots-$tag.log 2>&1; tail -1 $R/slots-$tag.log
curl -s $B/health > $R/health-$tag.json; cat $R/health-$tag.json; echo
docker logs glm53-tf-r0 2>&1 | grep -ciE 'traceback|error' | sed 's/^/r0 error lines: /'
echo "ab $tag done $(date +%T)"
