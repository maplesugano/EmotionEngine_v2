# CAA Decomposition Steering — Results Summary
Primary layer: 13
Sample size: 20

## Vector norms at layer 13
| emotion | ||c_e|| | c_e·g | ||resid_raw|| |
|---------|--------|-------|------------|
| joy            | 1.0000 | 0.9916 | 0.1293 |
| trust          | 1.0000 | 0.9904 | 0.1382 |
| fear           | 1.0000 | 0.9939 | 0.1100 |
| surprise       | 1.0000 | 0.9893 | 0.1457 |
| sadness        | 1.0000 | 0.9960 | 0.0894 |
| disgust        | 1.0000 | 0.9948 | 0.1016 |
| anger          | 1.0000 | 0.9934 | 0.1144 |
| anticipation   | 1.0000 | 0.9935 | 0.1138 |

## Output files
- `generation_outputs.jsonl` — Exp 1 raw generations
- `generation_eval_requests.jsonl` — LLM eval batch (submit to OpenAI)
- `generation_eval_results.jsonl` — LLM eval results (after batch)
- `summary_by_condition.csv` — Exp 1 metrics aggregated by condition
- `summary_by_emotion.csv` — Exp 1 metrics by emotion × condition
- `multiplier_sweep_outputs.jsonl` — Exp 3 sweep generations
- `multiplier_sweep_summary.csv` — Exp 3 metrics (after eval)
- `sweep_heatmap_{emotion}.png` — Exp 3 2D control surface heatmaps
- `local_pc_generation_outputs.jsonl` — Exp 4 PC± generations
- `local_pc_axis_interpretation_requests.jsonl` — Exp 4 LLM axis labelling batch
- `local_pc_axis_interpretations.json` — Exp 4 axis labels (after batch)
- `local_pc_axis_summary.md` — Exp 4 human-readable axis labels
- `exp5_prompt_vs_steering.jsonl` — Exp 5 prompt vs steering generations
- `metrics_by_condition.png`, `tradeoff_scatter.png`, `confusion_*.png`

## Questions to answer
1. What does `g` do during generation?
2. What does `resid_e` do during generation?
3. Is `g + resid_e` better controlled than original CAA?
4. Which emotions are easiest / hardest to steer?
5. Do local PCs produce consistent, interpretable sub-emotion changes?
6. Which local PCs are promising as EmotionEngine DJ knobs?
7. What steering recipe should be used in the next prototype?
