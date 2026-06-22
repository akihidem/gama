# Contributing to gama

gama grows with the community. Three easy ways in:

## 1. Add a recipe 🌱 (most wanted)
A *recipe* is a combination that matched (or approached) a bigger model on your hardware.
1. `cp -r recipes/TEMPLATE recipes/<your-name>`
2. Edit `config.json` — your combination, as `build_backend` specs (`gama` / `ensemble` /
   `tool` over `ollama` / `ssh-openai` lanes). **Use placeholder hosts** (`user@host`) —
   never real IPs, hostnames, or keys.
3. Measure it:
   ```bash
   gama bench --backends gama,<your-big-baseline> --config recipes/<name>/config.json --tier large
   ```
   Paste the table into `recipe.md` (models, hardware, scores, suite).
4. Open a PR. CI must stay green.

## 2. Add a backend
Subclass `ModelBackend` in `gama/backends.py`:
```python
class MyBackend(ModelBackend):
    name = "my-runtime"
    available = True
    def complete(self, prompt, tier, **kwargs) -> str:
        ...  # return the model's text
```
Register it in `_BACKENDS`. **Stdlib only** (subprocess / urllib) — no new dependencies.
Add a unit test that mocks the subprocess/HTTP (see the `Fake*` backends in `tests/`).

## 3. Add benchmark cases
The default suite is in `gama/benchmark.py` (`DEFAULT_SUITE`); the harder suites are in
`experiments/moa_vs_strong.py`. A case is `BenchCase(case_id, task_type, prompt, checker)`
with a **deterministic** checker — exact value, executed code, or required-element
presence. Never an LLM judge (that would measure opinion, not correctness).

## Ground rules
- **stdlib only** in the core — `pip install` must stay dependency-free.
- **No secrets / real hosts** in committed files — examples and recipes use placeholders.
- Tests pass: `python3 -m unittest discover -s tests -t .`
- Checkers are deterministic; routing is *measured*, not guessed.
- Be honest in `recipe.md`: note blind spots, single-run variance, and any measurement
  caveats. A tie is a tie; don't claim a win.
