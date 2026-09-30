rounds: 3 (final)

| suite | cell | mode | tokens | mean | min | max | W20 b10 | vs B10 | yesterday | vLLM kit | vs vLLM | hashes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| tf | code | sampled (T=1) | 64 | **51.1** | 51.0 | 51.1 | 51.8 | -1.4% |  |  |  | rounds agree |
| tf | chat | sampled (T=1) | 64 | **48.6** | 48.1 | 48.9 | 49.5 | -1.7% |  |  |  | rounds agree |
| tf | code | greedy (T=0) | 64 | **89.6** | 89.3 | 90.0 | 88.9 | +0.8% | 77.6 | 41.9 | 2.14x | same |
| tf | chat | greedy (T=0) | 64 | **51.6** | 51.5 | 51.6 | 49.8 | +3.6% | 44.6 | 22.8 | 2.26x | same |
| tweet | sequence | greedy (T=0) | 512 | **105.1** | 105.0 | 105.1 | 105.0 | +0.1% |  |  |  | same |
| tweet | code | greedy (T=0) | 512 | **75.9** | 75.8 | 76.0 | 70.8 | +7.3% |  |  |  | same |
| tweet | json | greedy (T=0) | 512 | **84.0** | 84.0 | 84.0 | 84.3 | -0.4% |  |  |  | same |
| kit | hashmap | greedy (T=0) | 200 | **59.6** | 59.5 | 59.7 | 60.0 | -0.8% |  | 30.0 | 1.99x | same |
| kit | structured | greedy (T=0) | 200 | **112.3** | 111.9 | 113.1 | 113.9 | -1.4% | 100.6 | 72.7 | 1.54x | same |
| kit | essay | greedy (T=0) | 200 | **50.5** | 50.4 | 50.6 | 50.6 | -0.3% |  | 26.1 | 1.93x | same |
| edit | edit-rename | greedy (T=0) | 1024 | **124.1** | 123.1 | 124.7 | 125.1 | -0.8% |  |  |  | same |
| edit | edit-comments | greedy (T=0) | 1024 | **108.0** | 105.9 | 109.2 | 109.9 | -1.7% |  |  |  | same |
| edit | edit-print-to-log | greedy (T=0) | 1024 | **126.5** | 125.7 | 127.1 | 128.7 | -1.7% |  |  |  | same |

geomean vs B10 over 13 cells: +0.09%
