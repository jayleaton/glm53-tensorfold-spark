== B10
  exact 10/10, 10/10 | batchexact [['4/4'], ['4/4']] | transcripts together == alone: {'1-s': True, '3-s': True, '0-g': True, '2-g': True}
  reply sha ['8794a3463259cc2f'] OK
  prefill 24500: 1,658, 1,663, 1,657, 1,658  (min 1,657, mean 1,659)
  prefill 98000: 1,653, 1,656, 1,652, 1,654  (min 1,652, mean 1,654)
  glmbench 1 stream geomean vs B10 +0.00%; greedy hashes 11/11, all 13/13
  4 streams mean of 6 88.58 (sd 3.39) [92.5, 84.6, 87.8, 93.5, 85.0, 88.1]
  mmlu200 accuracy 0.880 (176/200);   refusals 0/10
  N1 drafted == serial: 6/6 | N1 batched == alone: 6/6
  needle 314267 tokens: cold found True prefill 221.9578 s (1415.9 tok/s); resend found True cached 314240
  B10: MemAvailable min by phase (head / worker GiB)
    pre       15.60 /  15.10
    ab        14.35 /  13.79
    ab2       13.80 /  13.32
    n1        13.34 /  12.77
    stress    10.72 /  10.77
    mmlu      11.25 /  11.19
    needle    11.38 /  11.19
    ab3       12.19 /  11.73
    ab4       11.31 /  10.84
    end       11.62 /  11.16
    stress+mmlu+needle min 10.72 / 10.77  -> PASS (>= 8 GiB)
    stress min 10.72 / 10.77
    needle min 11.38 / 11.19
    whole load min 10.72 / 10.77
    MemFree steps >= 1 GiB (r0): mmlu 1, needle 1
    MemFree steps >= 1 GiB (r1): mmlu 1, needle 1
  oom / nvrm lines after: 0 0 nvrm-nomem 0 0 (before: 0 0 nvrm-nomem 0 0)
== RECIPE
  exact 10/10, 10/10 | batchexact [['4/4'], ['4/4']] | transcripts together == alone: {'1-s': True, '3-s': True, '0-g': True, '2-g': True}
  reply sha ['8794a3463259cc2f'] OK
  prefill 24500: 1,663, 1,666, 1,656, 1,661  (min 1,656, mean 1,661)
  prefill 98000: 1,651, 1,656, 1,650, 1,651  (min 1,650, mean 1,652)
  glmbench 1 stream geomean vs B10 +0.21%; greedy hashes 11/11, all 13/13
  4 streams mean of 6 87.22 (sd 2.99) [90.9, 83.7, 86.3, 91.5, 84.4, 86.5]
  mmlu200 accuracy 0.880 (176/200);   refusals 0/10
  N1 drafted == serial: 6/6 | N1 batched == alone: 6/6
  needle 314217 tokens: cold found True prefill 222.1839 s (1414.2 tok/s); resend found True cached 314176
  RECIPE: MemAvailable min by phase (head / worker GiB)
    pre       15.70 /  15.16
    ab        14.15 /  13.56
    ab2       13.60 /  13.07
    n1        12.94 /  12.36
    stress    10.57 /  10.68
    mmlu      10.78 /  10.75
    needle    11.06 /  10.70
    ab3       11.68 /  11.31
    ab4       10.86 /  10.43
    stress+mmlu+needle min 10.57 / 10.68  -> PASS (>= 8 GiB)
    stress min 10.57 / 10.68
    needle min 11.06 / 10.70
    whole load min 10.57 / 10.43
    MemFree steps >= 1 GiB (r0): mmlu 1, needle 1
    MemFree steps >= 1 GiB (r1): mmlu 1, needle 1
  oom / nvrm lines after: 0 0 nvrm-nomem 0 0 (before: 0 0 nvrm-nomem 0 0)

| metric | B10 | RECIPE |
| --- | --- | --- |
| idle MemAvailable min r0 / r1 (GiB) | 17.93 / 17.69 | 17.59 / 17.31 |
| idle MemFree mean r0 / r1 (GiB) | 15.51 / 16.49 | 15.43 / 16.27 |
| stress min MemAvailable r0 / r1 | 10.72 / 10.77 | 10.57 / 10.68 |
| needle min MemAvailable r0 / r1 | 10.72 / 10.77 | 10.57 / 10.68 |
| stress+mmlu+needle min r0 / r1 | 10.72 / 10.77 | 10.57 / 10.68 |
| 1s: skew mean / p90 (us) | None / None | None / None |
| 1s: beyond transport r0+r1 (us/exch) | None | None |
| 1s: rank 0 late (%) | None | None |
| 4s: skew mean / p90 (us) | 39.77 / 88.13 | 45.96 / 121.41 |
| 4s: beyond transport r0+r1 (us/exch) | 39.77 | 45.96 |
| 4s: rank 0 late (%) | 50 | 55 |
| transport p50 1s / 4s (us) | None / 4.51 | None / 4.1 |
| 4s decode: busy % X925 r0 / r1 | 20.57 / 20.61 | 20.58 / 20.52 |
| 4s decode: busy % A725 r0 / r1 | 1.83 / 0.83 | 1.03 / 0.99 |

(decode / prefill / 4-stream / MMLU / exactness: summ.py block above; glmbench geomean is vs the first name)
