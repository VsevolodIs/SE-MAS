# TutorFlow AI — A03 evaluation

This script produces **technical benchmark summaries** and **conservative objective-check candidates**.

- `evaluation_summary.csv`: completed/truncated counts, TTFT/TPS, plus counts of candidate checks.
- `evaluation_details.csv`: all answers and preliminary automatic flags; fill `human_verdict` with `pass`, `partial`, or `fail`, and explain in `human_comment` after manual checking.
- `candidate_pass` is **not** verified mathematical accuracy: it only checks selected final values, not the validity of each step.
- `candidate_fail` flags a likely wrong final answer; confirm manually when formatting is ambiguous.
- `manual_review` means no safe automated verifier exists for this test yet.
- `not_evaluated` means the generation ended prematurely or failed.
- The tool does **not** compute an overall educational quality percentage; a separate, human-reviewed rubric is needed.
- SymPy is used for deterministic comparisons of numbers, rational expressions, and root values in a narrow whitelist of straightforward tasks. It does not validate all algebraic steps.
- These results do not measure RAM or VRAM.

To add deeper automatic validation, develop a dedicated verifier for each task type and cross-check its decisions against independently hand-reviewed examples.
