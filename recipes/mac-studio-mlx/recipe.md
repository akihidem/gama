# Mac Studio MLX — light stack ties a 122B

A sovereign **light** stack matches a single **122B** on a hard 12-task suite, fully local.

## Hardware / runtime
- Mac Studio, running **MLX** (`mlx_lm.server`, OpenAI-compatible on `localhost:8080`),
  reached over **SSH** (no open port; prompt on stdin).
- Set `ssh_host` in `config.json` to your Mac's SSH host (`user@host`).

## The combination (gama routing over light lanes)
| task class | lane |
|---|---|
| `qa` (math) | **tool** — 7B writes Python, we run it (exact arithmetic) |
| `code_implementation` | **32B-Coder** |
| `research` | **heterogeneous ensemble** (7B + 24B + 32B, 32B aggregates) |

vs baseline: a single **Qwen3.5-122B**.

## Result (big suite, 12 tasks)
| | gama-light (7B+24B+32B+tool) | 122B single |
|---|---|---|
| **score** | **0.92** | **0.92** |
| misses | r4 (day-of-week mod arithmetic) | c3 (roman numerals) |

**Tied**, with complementary blind spots — the light stack is fully local & sovereign.
Read as *competitive*, not a clean win: an earlier apparent edge was a measurement
artifact (122B answer truncation), fixed by a fair token budget.

## Reproduce
```bash
# edit config.json: ssh_host -> your Mac
python3 -m experiments.moa_vs_strong recipes/mac-studio-mlx/config.json
```

## Notes
- Identical-copy ensembles add nothing; **diversity** + **tools** are what close gaps.
- The math lane uses a tool because no small model here does big arithmetic mentally.
- Single run; small N. Numbers will wobble — re-run and PR your own.
