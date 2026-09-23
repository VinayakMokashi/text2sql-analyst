| SQL model | EX (all) | Easy | Medium | Hard | Declined unanswerable | False refusals | Table recall@6 | Final schema recall | Avg SQL latency | Self-corrected |
|---|---|---|---|---|---|---|---|---|---|---|
| `openai/gpt-oss-120b` | **100.0%** | 100.0% | 100.0% | 100.0% | 4/4 | 0 | 99.2% | 100.0% | 1.14s | 0 |
| `openai/gpt-oss-20b` | **97.7%** | 100.0% | 100.0% | 92.9% | 4/4 | 0 | 99.2% | 100.0% | 0.79s | 0 |
| `qwen/qwen3.8-27b` | **100.0%** | 100.0% | 100.0% | 100.0% | 4/4 | 0 | 99.2% | 100.0% | 0.33s | 0 |

Provider: `groq`; helper model (table selection): `qwen/qwen3.8-27b`; 47 questions; updated 2026-09-23.
