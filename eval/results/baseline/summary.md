| SQL model | EX (all) | Easy | Medium | Hard | Declined unanswerable | False refusals | Table recall@6 | Final schema recall | Avg SQL latency | Self-corrected |
|---|---|---|---|---|---|---|---|---|---|---|
| `openai/gpt-oss-120b` | **95.3%** | 100.0% | 100.0% | 85.7% | 4/4 | 2 | 99.2% | 97.7% | 0.89s | 0 |
| `openai/gpt-oss-20b` | **93.0%** | 100.0% | 100.0% | 78.6% | 4/4 | 2 | 99.2% | 97.7% | 3.84s | 0 |

Baseline run (before the two fixes described in the README), helper model `qwen/qwen3.8-27b`, 47 questions. Latency here includes free-tier rate-limit waits.
