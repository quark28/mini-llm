## Benchmark results

| Model                               | Params |     PIQA | HellaSwag* |   ARC-e* |   ARC-c* |    WinoG |     SciQ |  LAMBADA |    BLiMP |
|:------------------------------------|-------:|---------:|-----------:|---------:|---------:|---------:|---------:|---------:|---------:|
| ours · SFT                          |  19.8M |     52.6 |       26.0 |     30.3 |     22.4 |     51.9 |     33.7 |      0.0 |     58.9 |
| ours · RL                           |  19.8M |     52.8 |       25.8 |     30.7 |     22.2 |     50.0 |     29.5 |      0.0 |     58.5 |
| EleutherAI/pythia-14m               |    14M |     56.9 |       26.5 |     32.7 |     21.0 |     51.3 |     50.4 |     10.3 |     67.7 |
| EleutherAI/pythia-70m               |    70M |     59.8 |       27.4 |     35.2 |     21.8 |     52.9 |     64.0 |     22.7 |     74.2 |
| openai-community/gpt2               |   124M |     62.9 |       31.1 |     39.5 |     22.7 |     51.6 |     75.2 |     32.6 | **82.1** |
| HuggingFaceTB/SmolLM2-135M          |   135M | **68.4** |   **43.1** | **58.6** | **29.6** | **53.0** | **84.0** | **42.9** |     79.0 |
| HuggingFaceTB/SmolLM2-135M-Instruct |   135M |     66.7 |       42.9 |     48.8 |     27.7 |     51.1 |     81.8 |     39.7 |     74.6 |
| *random*                            |      — |     50.0 |       25.0 |     25.0 |     25.0 |     50.0 |     25.0 |      0.0 |     50.0 |

Zero-shot, lm-evaluation-harness 0.4.13. Accuracy, %; \* — length-normalized accuracy (`acc_norm`). **Bold** — best in column. Ours: context 256, answer slot 64, diffusion likelihood via left-to-right unmasking.

### Truncated answer options, % (slot = 64 tokens)

| Model      | PIQA | HellaSwag | ARC-e | ARC-c | WinoG | SciQ | LAMBADA | BLiMP |
|:-----------|-----:|----------:|------:|------:|------:|-----:|--------:|------:|
| ours · SFT |  8.5 |       1.7 |   0.0 |   0.0 |   0.0 |  0.0 |     0.0 |   0.0 |
| ours · RL  |  8.5 |       1.7 |   0.0 |   0.0 |   0.0 |  0.0 |     0.0 |   0.0 |
