#!/usr/bin/env bash
# W18 failsafe: deadman.sh HHMM: at that time (+07), if the test-window lease still exists, restore prod (restore.sh deadman)
cd $HOME/glm53-tensorfold-spark
while [ "$(date +%H%M)" != "$1" ]; do sleep 20; done
[ -f $HOME/.test-window-lease ] && bash results/W18/restore.sh deadman > results/W18/deadman-$1.out 2>&1
