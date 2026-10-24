## Benchmark results

| Model         | Params |     PIQA | HellaSwag* |   ARC-e* |   ARC-c* |    WinoG |     SciQ | LAMBADA |    BLiMP |
|:--------------|-------:|---------:|-----------:|---------:|---------:|---------:|---------:|--------:|---------:|
| classic · SFT |  20.1M |     54.5 |       26.7 | **31.9** |     21.0 |     49.2 | **48.7** |     0.0 |     56.0 |
| classic · RL  |  20.1M | **54.8** |   **26.8** |     30.0 | **21.6** | **49.6** |     45.6 | **0.3** | **56.2** |
| *random*      |      — |     50.0 |       25.0 |     25.0 |     25.0 |     50.0 |     25.0 |     0.0 |     50.0 |

Zero-shot, lm-evaluation-harness 0.4.13. Accuracy, %; \* — length-normalized accuracy (`acc_norm`). **Bold** — best in column. Ours: context 256, answer slot 64, diffusion likelihood via left-to-right unmasking.

### Truncated answer options, % (slot = 64 tokens)

| Model         | PIQA | HellaSwag | ARC-e | ARC-c | WinoG | SciQ | LAMBADA | BLiMP |
|:--------------|-----:|----------:|------:|------:|------:|-----:|--------:|------:|
| classic · SFT |  8.5 |       1.7 |   0.0 |   0.0 |   0.0 |  0.0 |     0.0 |   0.0 |
| classic · RL  |  8.5 |       1.7 |   0.0 |   0.0 |   0.0 |  0.0 |     0.0 |   0.0 |
