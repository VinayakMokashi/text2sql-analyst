| SQL model | EX (all) | Easy | Medium | Hard | Declined unanswerable | False refusals | Table recall@6 | Final schema recall | Avg SQL latency | Self-corrected |
|---|---|---|---|---|---|---|---|---|---|---|
| `openai/gpt-oss-120b` | **100.0%** | 100.0% | 100.0% | 100.0% | 4/4 | 0 | 98.0% | 95.0% | 1.16s | 0 |
| `openai/gpt-oss-20b` | **95.0%** | 100.0% | 87.5% | 100.0% | 4/4 | 0 | 98.0% | 95.0% | 0.76s | 0 |
| `qwen/qwen3.8-27b` | **100.0%** | 100.0% | 100.0% | 100.0% | 4/4 | 0 | 98.0% | 95.0% | 0.34s | 0 |

Provider: `groq`; helper model (table selection): `qwen/qwen3.8-27b`; 24 questions; updated 2026-09-23.
