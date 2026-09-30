# First real source improvement

On 2026-09-30, the AWS mission ran two Astra proposals concurrently, obtained an
independent Claude review for each, evaluated both changes, and selected a new
JSON extractor. The completed evaluation reported `improved`; the operator then
advanced `codex/gama-parallel-rsi` to the selected source commit.

The parser now uses Python's `JSONDecoder.raw_decode` for the first object or
array within surrounding text. This handles escaped quotes, backslashes,
delimiters inside strings, and mixed nesting while preserving complete JSON
values and rejecting a malformed first container.

| Check | Original | Selected candidate |
|---|---:|---:|
| Public search split | 16/20 | 20/20 |
| Public confirmation split, each of three repeats | 16/20 | 20/20 |
| Reserved final split, each of three repeats | 16/20 | 20/20 |
| Separately prepared generated cases | 186/288 | 288/288 |
| Existing regression suite | 636 passed | 636 passed |

The measured scope is this parser and these fixtures. All three fixed evaluation
splits are public. The additional 288 cases were prepared before proposal
generation using JSON serialization, varied strings/nesting, and malformed first
containers; their harness is preserved in
[`validation/json_extraction.py`](../validation/json_extraction.py).

- Seed commit: `173fd0b539c17b28ec5d46cee33c77daf596942c`.
- Selected Astra commit: `0f8c2dce44f0bb86749478d62e95c165d9e15cef`.
- Core run: `94dee6bb434740c0af6fbe225f0374ee`, one completed round, two reserved proposals.
- First cycle: 22:45:59–22:54:42 Asia/Tokyo.
- Actual routes: Astra (`bedrock-astra`) and Claude (`global.anthropic.claude-opus-5`).
- Operating implementation: Astra Loop `20260930T133503-0f38416b`, twelve integration
  checks passed and independent Claude integration review passed.

The [AWS timer](rsi_operations.md) is enabled for 09:00 and 21:00 Asia/Tokyo.
Each cycle permits two concurrent proposers and at most four reservations,
including failed or interrupted proposals. This first mission is now
`saturated`; a service invocation after completion preserved the two reservations
and generated no new model artifacts. A new goal uses a separate mission.

Run state, model prompts, actual identities/usage, reviews, and patches are
retained outside the source checkout. The AWS delivery index is
`/home/akhd/outbox/projects/20260930T133503-0f38416b/README.md`.
