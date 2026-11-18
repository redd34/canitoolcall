# Wishlist

Work we would like done and will review quickly. Humans and AI agents are equally welcome to take any item. Last updated 2026-10-14.

**Where this is going:** CanIToolCall should be the thing people and tools consult before trusting tool calling on a local model: a neutral fixture corpus that engines run in their own CI, a matrix that answers "will this model's tool calls parse on this engine?", and a probe that checks the server on your own machine.

## How to take an item

1. Pick an item. If it links an issue, comment there; if it says *no issue yet*, open an [Agent check-in](https://github.com/redd34/canitoolcall/issues/new?template=agent_checkin.yml) or a normal issue naming the item, so two people don't do the same work.
2. Read [AGENTS.md](AGENTS.md) (setup, commands, rules) and [CONTRIBUTING.md](CONTRIBUTING.md).
3. Open one PR per item. Say what you ran and what you could not run. A partial result with an honest boundary is useful; an invented one is not.

Sizes: **S** is an hour or two, **M** is a day, **L** is several days or needs a design first.

## Now: wanted for 0.3.0

0.2.0 shipped on 2026-11-15 (see [CHANGELOG.md](CHANGELOG.md)). These items were not finished for it and carry over.

| Item | Size | What done looks like |
|---|---|---|
| Triage the fixture left in `transformers-other` ([#23](https://github.com/redd34/canitoolcall/issues/23)) | S | Each fixture is in a finding with a one-line summary and a repro command; `triage.py` regenerates `triage.jsonl` byte for byte. |
| Triage the 14 fixtures in `sglang-other` ([#24](https://github.com/redd34/canitoolcall/issues/24)) | M | Same as above. [#13](https://github.com/redd34/canitoolcall/pull/13) is a good model of the result. |
| Docs: pointing `probe` at hosted OpenAI-compatible providers ([#26](https://github.com/redd34/canitoolcall/issues/26)) | S | A short doc page with tested commands and the key-safety notes. |
| Verify vllm#58829 against the llama fixtures ([#16](https://github.com/redd34/canitoolcall/issues/16)) | S | The verifier output, base and head SHAs, OS, and the supported/unsupported split posted on the issue. |
| Spec: how generation ended, and fixtures that assume an ignored stop token ([#9](https://github.com/redd34/canitoolcall/issues/9)) | M | The fields landed in [#38](https://github.com/redd34/canitoolcall/pull/38) (spec 0.2). Left: whatever #9 still lists after that first PR. |
| One shared report validator (*no issue yet*) | S | The report checks added in [#31](https://github.com/redd34/canitoolcall/pull/31) live in one function that the compiled-engine verifier ([#33](https://github.com/redd34/canitoolcall/pull/33)) also uses, once that lands. |
| Native Windows paths (*no issue yet*) | S | `.venvs/<engine>/bin/python` is hard-coded in `adapters/__init__.py` and `verify_upstream_pr.py`; use `Scripts\python.exe` on Windows. Needs a Windows run to confirm. |
| Verify vllm#51577 (*no issue yet*) | S | The llama3 JSON parser rewrite that replaced the closed vllm#58826. It moves the parser to a new module, so the overlay may exit 3 on vLLM 0.30.0; report whichever happens. |

## Next: wider coverage

| Item | Size | What done looks like |
|---|---|---|
| New family: MiniMax M2.x / M3 ([#4](https://github.com/redd34/canitoolcall/issues/4)) | M | Format notes with sources, a generator script, fixtures with provenance, one PR. |
| New family: NVIDIA Nemotron 3 / 3.5 ([#5](https://github.com/redd34/canitoolcall/issues/5)) | M | Same. |
| llama.cpp: Llama 4 fixtures with an authoritative chat template ([#6](https://github.com/redd34/canitoolcall/issues/6)) | M | The Llama 4 fixtures run as supported on llama.cpp. |
| Ollama: the legacy template tools parser ([#8](https://github.com/redd34/canitoolcall/issues/8)) | L | The 208 fixtures that are `unsupported` on Ollama today are replayed through the template parser. |
| SGLang: DeepSeek V4.1 ([#7](https://github.com/redd34/canitoolcall/issues/7)) | S | Blocked until an SGLang release contains sglang#39665; then one `ParserRule` and a replay. |
| Ollama and llama.cpp fix-PR verification ([#22](https://github.com/redd34/canitoolcall/issues/22)) | L | In review as [#33](https://github.com/redd34/canitoolcall/pull/33). Still wanted: the first real llama.cpp replay recorded on the issue. |
| Probe recipes for local runtimes (*no issue yet*) | S each | One short, tested page per runtime showing `probe` against its OpenAI-compatible server: LM Studio, mlx-lm, llama-cpp-python, KoboldCpp, Jan, TabbyAPI. Paste the real output and the runtime version; a runtime you could not run is not a recipe. |
| Demo patch: the pytest plugin vendored into an engine's test suite ([#28](https://github.com/redd34/canitoolcall/issues/28)) | M | A prepared patch, not sent upstream. |

## Later: the building-block layer

These need a design agreed in the issue before code.

| Item | Size | What done looks like |
|---|---|---|
| Main-branch engine results ([#25](https://github.com/redd34/canitoolcall/issues/25), [design](docs/DESIGN-main-branch-columns.md)) | L | First step only: a manual-dispatch workflow that builds one engine's main at a resolved SHA, replays, and uploads the result. No site change until that is measured. |
| Provider observations ([#27](https://github.com/redd34/canitoolcall/issues/27), [design](docs/DESIGN-provider-matrix.md)) | L | Not started; the scope is undecided. Comments on the design are welcome. |

## Help that isn't code

- **Run the probe on your own setup** and [report a wrong or surprising result](https://github.com/redd34/canitoolcall/issues/new/choose). Real endpoints find things fixtures don't.
- **Tell us about a model or engine we don't cover**, with a link to its format documentation.
- **Maintain an engine?** Vendor the fixtures into your test suite (see the README) and tell us what got in the way.
- **Building an agent or a client?** Link the matrix where your users ask why tool calling broke, and tell us which cell you needed that wasn't there.
- **Star or watch the repository** if it saved you time. It is how other people find it.
