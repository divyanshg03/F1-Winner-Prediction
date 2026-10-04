# Llama 3.1 8B (local, GPU via Ollama) as a race-winner predictor

Setup: temperature 0, JSON output, no fine-tuning. The model sees the same pre-race table the trained models use (grid, driver form, team form, recent win rate, qualifying gap). Zero-shot, and few-shot with three worked 2024 examples. Evaluated on the unseen 2025 (24 races) and 2026 (15 races) seasons. Llama 3.1's training data ends ~Dec 2023, so these races are unseen to it too. Ran on the RTX 4060 (5.8 GB of the model in VRAM), about 10 s per race.

| 2025 (24 races) | log-loss [95% CI] | top-1 |
|---|---|---|
| grid-only logit | 1.111 [0.866, 1.365] | 67% |
| trained ensemble (ens_post) | 0.946 [0.739, 1.171] | 62% |
| Llama zero-shot | 1.433 [1.262, 1.612] | 67% |
| Llama few-shot | 1.667 [1.563, 1.791] | 67% |
| 50/50 blend (ensemble + Llama few-shot) | 1.160 [1.021, 1.321] | 58% |

Paired (nats/race, + = first better): Llama few-shot vs trained ensemble **-0.721 [-0.926, -0.508]**; Llama few-shot vs grid-only -0.556 [-0.794, -0.318]; Llama zero-shot vs grid-only -0.322 [-0.496, -0.153]; blend vs ensemble -0.214 [-0.311, -0.110].

2026 (15 races): Llama few-shot 2.021 (top-1 27%), zero-shot 1.343 (top-1 73%) vs ensemble 1.270 and grid-only 1.416. Few-shot vs ensemble -0.751 [-1.036, -0.444].

## Reading
- **Llama is significantly worse than the trained models** on probability quality, and adding it to the ensemble makes things worse.
- **It mostly echoes the grid.** Its zero-shot top pick is usually the pole-sitter (top-1 matches pole-sitter's 67% in 2025) but the probabilities are badly under-confident: mean probability on the pole-sitter was 32% (zero-shot) and 20% (few-shot) in 2025, against an actual pole win rate of 67%. That is what drives the poor log-loss.
- **Few-shot hurt**, rather than helped: the three worked examples made it spread probability more flat (mean max probability 23%), and in 2026 its top pick was right only 27% of the time.
- **2026 zero-shot is not a clean number.** In 5 of 15 races (rounds 3, 4, 5, 8, 10) the zero-shot JSON named only 6-7 of the 22 drivers; the rest were assigned a floor probability and renormalised. The 73% top-1 and 1.343 log-loss there come from a mix of good and degraded outputs, so don't read it as Llama being competitive. With 15 races the intervals are wide in any case.
- An 8B language model has no mechanism for calibrated numeric probabilities; it pattern-matches the table. The trained models learn the grid-to-win relationship from ~190 races.

Reproduce: `ollama pull llama3.1:8b`, then `python scripts/llm_eval.py` (needs `reports/final_scores_*.csv` from `scripts/train_final.py`).
