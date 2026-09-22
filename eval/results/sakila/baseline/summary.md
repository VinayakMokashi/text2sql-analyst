| SQL model | EX (all) | Easy | Medium | Hard | Declined unanswerable | False refusals | Table recall@6 | Final schema recall | Avg SQL latency | Self-corrected |
|---|---|---|---|---|---|---|---|---|---|---|
| `openai/gpt-oss-120b` | **92.9%** | 100.0% | 100.0% | 75.0% | 2/2 | 0 | 96.8% | 98.6% | 1.09s | 0 |
| `openai/gpt-oss-20b` | **92.9%** | 100.0% | 100.0% | 75.0% | 2/2 | 1 | 96.8% | 98.6% | 0.65s | 0 |
| `qwen/qwen3.8-27b` | **100.0%** | 100.0% | 100.0% | 100.0% | 2/2 | 0 | 96.8% | 98.6% | 0.45s | 0 |

Provider: `groq`, helper model (table selection): `qwen/qwen3.8-27b`, 16 questions, updated 2026-09-22.
