| depth | init | logit std at init | ‖dL/dW‖ layer 1 at init | val acc after 3 epochs (%) |
|---|---|---|---|---|
| 10 | xavier | 0.046 | 0.074 | 96.2 ± 0.4 |
| 20 | xavier | 0.0014 | 0.0022 | 95.2 ± 0.6 |
| 30 | xavier | 5.7e-05 | 7e-05 | 71.2 ± 4.8 |
| 10 | he | 1.2 | 3 | 96.3 ± 0.4 |
| 20 | he | 1.2 | 2.4 | 95.7 ± 0.4 |
| 30 | he | 1.5 | 4.1 | 93.5 ± 3.5 |
