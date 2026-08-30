# Released Results

This repository includes compact aggregate artifacts used by the paper. It does not redistribute datasets, model weights, signal caches, or full per-video VLM predictions.

## Headline Evaluation

All headline files use `Qwen/Qwen2-VL-7B-Instruct`. EARS uses the single frozen configuration in `configs/frozen_config.json` for every dataset and frame budget.

| Artifact | Videos | Description |
| --- | ---: | --- |
| `results/headline/xdviolence_ears_qwen2vl7b.json` | 504 | EARS, K=1/2/4/6/8/10 |
| `results/headline/xdviolence_uniform_qwen2vl7b.json` | 504 | Matched Uniform control |
| `results/headline/ucfcrime_ears_qwen2vl7b.json` | 440 | EARS, K=1/2/4/6/8/10 |
| `results/headline/ucfcrime_uniform_qwen2vl7b.json` | 440 | Matched Uniform control |

The XD-Violence result shows a large low-budget gain. UCF-Crime shows negative transfer relative to Uniform at every reported K and is included deliberately.

## Ablations

- `results/ablations/heldout_tuning.json`: held-out parameter search.
- `results/ablations/random_xdviolence_5seeds.json`: five-seed Random control.
- `results/ablations/random_ucfcrime_5seeds.json`: five-seed Random control.

## Efficiency and Robustness

- `results/efficiency/runtime_end_to_end_k4_n24.json`: end-to-end K=4 timing.
- `results/efficiency/runtime_bench_cpu.json`: CPU signal timing.
- `results/efficiency/e5_selector_cost.json`: selector-cost comparison.
- `results/efficiency/motion_proxy_cost.json`: motion proxy timing.
- `results/efficiency/flow_scale_selection.json`: selected-frame agreement across optical-flow scales.

Runtime values are hardware- and software-stack dependent. They are released as measurement records, not universal throughput claims.

## Figures

PNG previews and vector PDFs are under `assets/figures/`. The success and failure figures use frames from public research datasets; consult the source dataset terms before redistributing modified derivatives elsewhere.
