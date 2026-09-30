#!/usr/bin/env bash
# W19 (W11's): nsys-rep -> sqlite for cap-NAME (both ranks; rank 1's report copied from the worker node), in a throwaway b10
# container at nice 19 on 4 little cores (prod may be serving), then w19att.py per rank.  export.sh NAME
cd $HOME/glm53-tensorfold-spark; R=results/W19; N=$1; O=/var/tmp/w19/out
scp -q $WORKER_SSH:$O/cap-$N-r1.nsys-rep $O/
nice -n 19 docker run --rm --name w19-export-$N --cpuset-cpus 0-3 --entrypoint bash -v $O:/o glm53-tensorfold:b10 -c \
  "for r in r0 r1; do nsys export --type sqlite -f true -o /o/cap-$N-\$r.sqlite /o/cap-$N-\$r.nsys-rep > /dev/null 2>&1; done; ls -la /o | grep $N"
for r in r0 r1; do nice -n 19 python3 $R/w19att.py $O/cap-$N-$r.sqlite $R/cap-$N.jsonl $R/att-$N-$r.json | tee $R/att-$N-$r.txt; done
