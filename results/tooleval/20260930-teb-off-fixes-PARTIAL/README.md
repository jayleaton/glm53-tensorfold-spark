# Tool-calling eval, W20 production (PARTIAL: one run)

Server: W20 production (image b11, `GLM53_TF_TOOL_FIXES=all`, `config/prod.env.example`), thinking off
(`chat_template_kwargs {"enable_thinking": false}`), temperature 0. Runner: `scripts/tooleval/run.sh both off fixes`.
Client on another machine over an SSH port forward (the `base_url` port in the JSON is that forward).

- **tool-eval-bench** ([SeraphimSerapis/tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench), commit
  `c7b5b95`): complete, 69/69 scenarios. **Final score 90** (124 / 138 points), category C multi-step chains
  **8/8** (4/4 scenarios), deployability 86, responsiveness 78. Safety gate: one warning (TC-51, batched two calls
  that should have waited on each other). `teb-summary.json` = the bench's JSON with each scenario's raw log, per-turn
  timings and the local host fields removed.
- **spark-bench** TrueScore: stopped mid-run on purpose (the rig was needed for other work). Not a score; to be re-run.

This is **one run on our checkpoint** (abliterated EXL3 4-bit GLM-5.3-Flash), not an average, and there is no
same-day run with the fixes off to pair it with; read it as "the fixes do not break anything and chains pass", not as
a measured improvement. docs/TOOL-CALLING.md has the bugs 0620 fixes and the before / after host tests.
