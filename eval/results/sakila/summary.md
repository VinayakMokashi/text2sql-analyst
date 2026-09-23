| SQL model | EX (all) | Easy | Medium | Hard | Declined unanswerable | False refusals | Table recall@6 | Final schema recall | Avg SQL latency | Self-corrected |
|---|---|---|---|---|---|---|---|---|---|---|
| `openai/gpt-oss-120b` | **100.0%** | 100.0% | 100.0% | 100.0% | 2/2 | 0 | 96.8% | 100.0% | 1.17s | 0 |
| `openai/gpt-oss-20b` | **100.0%** | 100.0% | 100.0% | 100.0% | 2/2 | 0 | 96.8% | 100.0% | 0.86s | 0 |
| `qwen/qwen3.8-27b` | **100.0%** | 100.0% | 100.0% | 100.0% | 2/2 | 0 | 96.8% | 100.0% | 0.35s | 0 |

Provider: `groq`; helper model (table selection): `qwen/qwen3.8-27b`; 16 questions; updated 2026-09-23.
