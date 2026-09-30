#!/usr/bin/env bash
# W19 (W11's): container entrypoint wrapper -- the image's entrypoint with /w19/bin first on PATH, so its final
# `exec tensorfold ...` runs under `nsys launch` (W19_NSYS=1) in an idle session "w19"
export PATH=/w19/bin:$PATH
exec /usr/local/bin/glm53-tf-entrypoint "$@"
