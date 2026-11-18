# Release checklist: canitoolcall v0.1.0

Nothing in this repository has been pushed, published or posted. Every step below is for a human maintainer to do by hand, in order. `docs/PUBLISHING.md` has the background for sections 1 to 4; this file is the one to follow.

**State at hand-off (2026-09-26):**
- All work is on the local branch `integrate/v0.1`. `main` still holds only the skeleton commit `0c80b52`.
- Verified on that branch:
  - `uv run pytest`: 830 passed, with all five engine venvs present.
  - `ruff check`, `ruff format --check` and `mypy --strict` are clean.
  - `canitoolcall validate`: 45 files, 0 issues.
  - `uv build` and `twine check --strict` pass for both the sdist and the wheel.
  - `canitoolcall matrix --results results/2026-09-25` renders the site from the committed real runs.
- Repository location decided: **https://github.com/redd34/canitoolcall**, under the personal account `redd34` (no org). The owner placeholders are filled in (step 1.3).
- `gh` on this machine is currently authenticated as `sauravl-yenta`, not `redd34`. Authenticate as `redd34` before creating or pushing the repository (step 2.1).
- Name check (read-only GETs on 2026-09-26): `https://pypi.org/pypi/canitoolcall/json` returned 404; `https://api.github.com/users/canitoolcall` returned 404; a GitHub repository search for `canitoolcall` found 0 repositories.
- `python scripts/check_release.py` reports 1 blocker: the Code of Conduct contact (step 1.3b), which is still undecided.

---

## 1. Before the first push

- [ ] **1.1 Re-check the names**, in case they were taken since 2026-09-26:
  ```sh
  curl -s -o /dev/null -w "%{http_code}\n" https://pypi.org/pypi/canitoolcall/json   # expect 404
  curl -s -o /dev/null -w "%{http_code}\n" https://api.github.com/repos/redd34/canitoolcall  # expect 404 until you create it
  ```
- [x] **1.2 Decide the publishing identity.** RESOLVED 2026-09-26: all commits are authored as `Saurav Lall <saurav.lall1+github@gmail.com>` (history rewritten before any push; repo-local git config set).
- [x] **1.3 Fill in the repository location.** Done: the owner is `redd34`, and the repository is `https://github.com/redd34/canitoolcall`.
  - `README.md`: the matrix URL (`https://redd34.github.io/canitoolcall/`) and the clone URL.
  - `pyproject.toml`: the `fancy-pypi-readme` substitution `replacement`, which turns the README's relative links into `https://github.com/redd34/canitoolcall/blob/main/...` on PyPI, and `[project.urls]` (`Homepage`, `Issues`, `Changelog`, `Matrix`).
  - `SECURITY.md`: links to `https://github.com/redd34/canitoolcall/security/advisories/new` (enable it in step 2.3).
- [ ] **1.3b Fill in the Code of Conduct contact.** `CODE_OF_CONDUCT.md` line 39: replace the `CONTACT METHOD` placeholder with a real address. Not decided yet.
  - Then run `uv run python scripts/check_release.py`. It must report 0 blockers.
- [x] **1.4 Set the version.** In `src/canitoolcall/__init__.py`, set `__version__ = "0.1.0"`, currently `0.1.0.dev0`. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [0.1.0] - YYYY-MM-DD`. Also remove the "Status: pre-release (0.1.0.dev0)" note in `README.md`, and switch the quickstart text from `uv run` to `uvx`.
- [x] **1.5 Run the full local check, then commit:**
  ```sh
  uv run ruff check src tests scripts && uv run ruff format --check src tests scripts && uv run mypy
  uv run pytest
  uv run canitoolcall validate
  rm -rf dist && uv build && uvx twine check --strict dist/*
  uv run python scripts/check_release.py
  git commit -am "Release 0.1.0"
  ```
- [x] **1.6 Audit the fixtures.** Re-run every `scripts/fixtures/*/build.py` (their Hub revisions are pinned), then run `git status`. It must stay clean, because template-rendered fixtures must regenerate byte for byte.
- [ ] **1.7 Put the work on `main`.** `main` is a direct ancestor of the branch, so this is a fast-forward:
  ```sh
  git checkout main && git merge --ff-only integrate/v0.1
  ```

## 2. Create the GitHub repository and push

- [ ] **2.1 Authenticate as `redd34`, then create an empty public repository `redd34/canitoolcall`.** `gh` is currently logged in as `sauravl-yenta`; creating the repository with that login would put it under the wrong account. Do not add a README, licence or .gitignore, since the repository already has them.
  ```sh
  gh auth login                     # log in as redd34 (or: gh auth switch --user redd34)
  gh auth status                    # the active account must be redd34
  ssh -T git@github.com             # must greet redd34; otherwise use the https remote URL
  gh repo create redd34/canitoolcall --public \
    --description "caniuse.com for tool calling: conformance suite and matrix for tool-call and reasoning parsers across inference engines" \
    --homepage "https://redd34.github.io/canitoolcall/"
  git remote add origin git@github.com:redd34/canitoolcall.git
  git push -u origin main
  ```
  Only push `integrate/v0.1` as well if you want to keep the branch name.
- [ ] **2.2 Settings → General:** add topics: `tool-calling`, `function-calling`, `vllm`, `sglang`, `llama-cpp`, `ollama`, `llm`, `conformance-testing`.
- [ ] **2.3 Settings → Code security:** enable **Private vulnerability reporting**. `SECURITY.md` points reporters there.
- [ ] **2.4 Settings → Branches:** protect `main` and require the `ci` workflow's jobs (lint + types, pytest, build + twine check) to pass before merging.
- [ ] **2.5 Check that CI goes green on the first push** (Actions → ci). The workflows passed actionlint locally but have never actually run.

## 3. Enable GitHub Pages and publish the matrix

- [ ] **3.1 Settings → Pages:** under **Build and deployment → Source**, choose **GitHub Actions**. The `github-pages` environment is created on the first deploy.
- [ ] **3.2 Optional: Settings → Secrets → Actions:** add `HF_TOKEN`, a read-only token. The nightly prefetch works without it, but hits the anonymous Hub rate limit sooner.
- [ ] **3.3 Actions → nightly → Run workflow** (on `main`). Then check that:
  - each engine job uploads a results file, or shows a "not run" notice with the reason;
  - the `deploy to GitHub Pages` job succeeds, and `https://redd34.github.io/canitoolcall/` loads;
  - the per-cell drill-down pages load, and the footer shows the "Built with Llama" notice.
- [ ] **3.4 Compare the first nightly matrix with `results/2026-09-25/`.** On Linux x86_64, pass counts should match the committed snapshot. If they don't, investigate before announcing. Note that nightly writes `.json` files while the snapshot uses `.json.gz`, and the matrix reads both.

## 4. Configure PyPI trusted publishing

- [x] **4.1** On pypi.org, go to **Account → Publishing → Add a new pending publisher → GitHub**:

  | Field | Value |
  |---|---|
  | PyPI project name | `canitoolcall` |
  | Owner | `redd34` |
  | Repository name | `canitoolcall` |
  | Workflow name | `release.yml` |
  | Environment name | `pypi` |

- [x] **4.2** On GitHub, go to **Settings → Environments → New environment `pypi`**. Add yourself as a required reviewer, and restrict deployments to tags matching `v*`.
- [ ] **4.3 Optional dry run:** add a pending publisher on test.pypi.org as well, and run the build job only. The release workflow fails fast if placeholders remain or if the tag does not match the built version.

## 5. Tag v0.1.0 and release

- [x] **5.1** Tag the release commit, which is the one from step 1.5, and push the tag:
  ```sh
  git tag -a v0.1.0 -m "canitoolcall 0.1.0"
  git push origin v0.1.0
  ```
- [x] **5.2 Actions → release → Run workflow:** under "Use workflow from", pick **tag `v0.1.0`**. Approve the `pypi` environment when prompted. The workflow refuses to run on anything but a `v*` tag.
- [x] **5.3 Create the GitHub release:**
  ```sh
  gh release create v0.1.0 --title "v0.1.0" --notes-file release-notes.md
  ```
  Here `release-notes.md` is the `## [0.1.0]` section of `CHANGELOG.md`, copied by hand; don't commit it. Attach nothing. PyPI has the artifacts.
- [x] **5.4 Verify from a clean machine:**
  ```sh
  uvx canitoolcall --version            # 0.1.0
  uvx canitoolcall probe --help
  uvx canitoolcall validate             # uses the bundled corpus
  ```
- [x] **5.5** Open the PyPI page and check that the README renders, the relative links resolve to GitHub, and the "Built with Llama" paragraph is present.
- [x] **5.6** Bump `__version__` to `0.2.0.dev0` and re-add `## [Unreleased]` to the CHANGELOG.

## 6. Announcement plan

Order matters. Engine maintainers hear about each finding first, then the public.

**6.1 Day 0, file upstream.** Review each draft in section 7, re-check it against the engine's latest release, and file only the ones that still reproduce. Each draft links the fixture file on GitHub. The URLs work once the repository is public, so file after step 3.
- Where an issue already exists (section 7, "Known upstream"), add a comment with the fixture instead of opening a new issue.

**6.2 Day 0, comment on the existing threads** that motivated the project. Keep each comment short: which fixtures cover the thread, the one-line command to reproduce it, and the matrix link.

| Thread | What to say |
|---|---|
| [vercel/ai#19512](https://github.com/vercel/ai/issues/19512) (shared provider conformance suites) | canitoolcall is a neutral, language-neutral fixture corpus plus a live `probe` for any OpenAI-compatible endpoint. Offer the JSONL fixtures and the spec as a base for provider conformance tests. |
| [vllm-project/vllm#55079](https://github.com/vllm-project/vllm/pull/55079) (split-invariance tests) | Link the cross-engine split-invariance results and the pytest plugin (`canitoolcall.pytest_plugin.assert_conforms`, filtered with `--canitoolcall-family`), and offer the fixtures for vLLM's common suite. |
| [vllm-project/vllm#48294](https://github.com/vllm-project/vllm/issues/48294), [#56840](https://github.com/vllm-project/vllm/issues/56840), [#57699](https://github.com/vllm-project/vllm/issues/57699), [#57826](https://github.com/vllm-project/vllm/issues/57826), [#56263](https://github.com/vllm-project/vllm/issues/56263), [#57353](https://github.com/vllm-project/vllm/issues/57353), [#57688](https://github.com/vllm-project/vllm/issues/57688) | Each still reproduces on vLLM 0.30.0: link its fixture(s) and matrix cell. The comment text is in section 7.1. |
| [sgl-project/sglang#40739](https://github.com/sgl-project/sglang/issues/40739), [#35562](https://github.com/sgl-project/sglang/issues/35562) | Same approach, and note that vLLM has the same bugs. |
| [ggml-org/llama.cpp#20837](https://github.com/ggml-org/llama.cpp/issues/20837) | The matrix shows which Qwen3.x tool-call shapes llama.cpp `a25c9865` parses. |
| [ollama/ollama#18390](https://github.com/ollama/ollama/issues/18390), [#18354](https://github.com/ollama/ollama/issues/18354), [#18421](https://github.com/ollama/ollama/issues/18421), [#18530](https://github.com/ollama/ollama/issues/18530) | Each still reproduces on Ollama `7af39318`: link its fixture. |
| The kenashe.ai post, [local tool-use evals are measuring your server too](https://kenashe.ai/blog/2026-09-23-local-tool-use-evals-are-measuring-your-server-too) | Email or reply to the author. The post asks readers to build their own "stack matrix", and this is a shared one. |

**6.3 Day 1, post publicly,** once most upstream issues are filed, so the post can link them.
- **Show HN.** Title: "Show HN: CanIToolCall – caniuse for tool calling across vLLM, SGLang, llama.cpp and Ollama". Body, in three short paragraphs:
  1. The problem: the same model's tool calls parse differently on each engine.
  2. The method: recorded raw outputs, replayed offline through each engine's own parser, with every stream-split tested.
  3. The headline numbers from section 8, with the matrix link, and `uvx canitoolcall probe` for checking your own stack.
- **Reddit r/LocalLLaMA.** Lead with `canitoolcall probe --base-url http://localhost:11434/v1 --model …` for Ollama and llama-server users, then the matrix. Also post to r/LocalLLM, but not to r/MachineLearning, since this is a tool, not research.
- **Tone:** neutral and engine-agnostic. Every number comes from `results/2026-09-25/` and names the engine version. Do not rank engines. Coverage differs a lot: Ollama and transformers mark most fixtures unsupported, so their pass rates are not comparable.

**6.4 Week 1:** in each engine's issue tracker or discussions, offer the pytest plugin for vendoring the fixtures into its CI. Then triage incoming PRs for new model families, in the "one family = one PR" spirit of CONTRIBUTING.md.

---

## 7. Draft upstream bug reports (NOT posted)

These drafts are pre-filled from the committed snapshot `results/2026-09-25/` and from direct re-runs on 2026-09-26. Every "observed" value below came from running the pinned engine's own parser code. None of it was written by hand.

Before filing any of them:
1. Re-check against the engine's **latest** release or main. The pins are vLLM 0.34.0, SGLang 0.5.20, llama.cpp `a25c9865`, Ollama `7af39318` and transformers 5.17.0.
2. Search the tracker for duplicates.
3. Link fixtures as `https://github.com/redd34/canitoolcall/blob/main/fixtures/<family>/<file>.jsonl`.
4. Where a direct repro script exists, lead with it: it uses only the engine's own API.

The harness repro for any finding is:
```sh
uv run canitoolcall run --engine <engine> --id <fixture-id> --observed all --env HF_HUB_OFFLINE=1
```

The harness feeds the parser the tokenizer's exact token ids for the fixture's text. It replays them non-streaming, and then as streams split in several ways:
- `one`: the whole output in one delta
- `special`: split at special tokens
- `token`: one token per delta
- `rand:1:8`: seeded random groups of 1 to 8 tokens

The shared part of every report is: "Found by canitoolcall (https://github.com/redd34/canitoolcall), which replays recorded or template-rendered model outputs through each engine's own parser, offline. The fixture and its provenance are linked below."

### 7.1 vLLM (0.30.0)

#### V1. deepseek_v3 / deepseek_v31: streaming loses the tool call when one delta carries several special tokens
- **Where:** `vllm/tool_parsers/deepseekv3_tool_parser.py`, `deepseekv31_tool_parser.py`, in `extract_tool_calls_streaming`.
- **Input:** the official DeepSeek-V3-0324 chat template rendering of a `get_weather(city="Paris", unit="c")` call:
  ```
  <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"city": "Paris", "unit": "c"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
  ```
- **Observed:**
  - non-streaming, and one token per delta: `get_weather {"city": "Paris", "unit": "c"}`, which is correct;
  - the whole output in one delta, or deltas split at special-token boundaries: `tool_calls: []` and no content. The call is lost silently.
- **Why it matters:** with speculative decoding, or any server-side delta batching, a delta can carry several tokens, and then the call vanishes.
- **Scope:** 20 fixtures, e.g. `deepseek/v3-single-call`, `deepseek/vllm-v3-single-call` (vLLM's own test string) and `deepseek/vllm-v31-text-before-call`.
- **Fixtures:** `fixtures/deepseek/v3.jsonl`, `v31.jsonl`, `engine-tests.jsonl`.
- **Title:** `[Bug]: deepseek_v3 / deepseek_v31 tool parsers drop the tool call when a streaming delta contains more than one special token`

#### V2. Format-marker text inside a JSON string argument breaks the parse (hermes; several other parsers)
- **Direct repro:** `results/2026-09-25/repro/vllm_hermes_marker_in_arguments.py`, run with vLLM 0.30.0 and `Qwen/Qwen3-0.6B@c1899de2`. The input is Qwen3's official template rendering of `echo(text='Use <tool_call> ... </tool_call> tags, or "<think>" blocks.')`:
  ```
  <tool_call>\n{"name": "echo", "arguments": {"text": "Use <tool_call> ... </tool_call> tags, or \"<think>\" blocks."}}\n</tool_call>
  ```
- **Observed (re-run 2026-09-26):**
  ```
  ERROR [hermes_tool_parser.py:117] json.decoder.JSONDecodeError: Unterminated string starting at: line 2 column 40 (char 40)
  tools_called: False
  tool_calls  : []
  content     : '<tool_call>\n{"name": "echo", ...}\n</tool_call>'
  ```
- **Expected:** one `echo` call. The JSON is valid, and the closing tag only appears inside a JSON string.
- **Why it matters:** agents that write docs or code about tool calling, such as "write this README", lose the call, and the raw markup leaks to the user.
- **Also affected** (14 fixtures in total):
  - glm45 and glm47. For example, GLM-4.7 `<arg_value>Wrap each call in <tool_call> and </tool_call>; keys go in <arg_key>.</arg_value>` gives non-streaming `{"path": "docs/format.md"}`, with the `content` argument dropped; streaming truncates the value and leaks the rest as content.
  - qwen3_coder. Qwen3.8's `<parameter=content>` with `</parameter>` in the text is cut to `"Close tags inline: "`.
  - kimi_k2, kimi_k3, gemma4, llama3_json, mistral and deepseek_v31.
- **Note:** for XML-style formats that do not escape values (GLM, Qwen3-Coder, Gemma), this is partly a format ambiguity. Ask for a best-effort rule, such as a newline-anchored closer or the last closer before the call end. For JSON-bodied formats (hermes, llama3_json, deepseek), JSON string escaping makes the input unambiguous, so it is a plain bug.
- **Title:** `[Bug]: hermes tool parser returns no tool call when a JSON string argument contains "</tool_call>"`. File the other parsers as a follow-up list in the same issue.

#### V3. llama3_json streaming swallows content that starts with `{` but is not a call
- **Input:** the Llama 3.3 official template, assistant plain answer `{"capital": "Paris", "country": "France"}` (no tools called).
- **Observed:** non-streaming returns it as `content`, which is correct. Every streaming split returns `content: null`, `tool_calls: []`: the answer disappears.
- **Also affected:**
  - `{"a": 1} is a dict`, from the bug report [sglang#35562](https://github.com/sgl-project/sglang/issues/35562), which is the same bug in SGLang;
  - an empty `{}` answer;
  - a call with empty arguments.
- **Scope:** 5 fixtures, e.g. `llama/l3-json-answer-not-a-call`.
- **Title:** `[Bug]: llama3_json streaming drops assistant content that starts with '{' but is not a tool call (non-streaming keeps it)`

#### V4. llama3_json: text before `<|python_tag|>` is dropped (non-streaming) or the call is returned as content (streaming)
- **Input:** `Let me check. <|python_tag|>{"name": "get_weather", "arguments": {"city": "Tokyo"}}`. This is SGLang's test string, `test_llama32_detector.py#L84-L88`.
- **Observed:**
  - non-streaming returns the call with `content: null`, so "Let me check. " is lost;
  - every streaming split returns `content: 'Let me check. {"name": "get_weather", ...}'` and no tool call.
- **Caveat:** check how your server detokenizes `<|python_tag|>` in streaming (`skip_special_tokens`) before filing the streaming half.
- **Title:** `[Bug]: llama3_json drops text before <|python_tag|> (non-streaming) and returns the call as content (streaming)`

#### V5. gemma4: nested numbers, booleans and null that the schema leaves untyped come back as strings
- **Direct repro:** `results/2026-09-25/repro/vllm_gemma4_nested_numbers.py`:
  ```
  _parse_gemma4_args('location:<|"|>Tokyo<|"|>,details:{temp:25,unit:<|"|>celsius<|"|>},flags:[true,1]')
  -> {'location': 'Tokyo', 'details': {'temp': '25', 'unit': 'celsius'}, 'flags': ['true', '1']}
  ```
- **Expected:** `temp: 25`, `flags: [true, 1]`. Gemma 4 marks strings explicitly with `<|"|>`, so an undelimited value is not a string. SGLang and Ollama keep the types.
- **Note:** vLLM's tests assert string output for schema-less values, so frame this as a design question with evidence, not as a regression.
- **Scope:** 7 fixtures.
- **Title:** `[Gemma4] Untyped nested values are returned as strings although the format marks strings with <|"|>`

#### V6. gpt-oss (Harmony): header markup leaks into content; a stray header becomes a bogus tool name
- **Input 1:** the openai-harmony render, cut by `max_tokens` inside the call header:
  ```
  <|channel|>analysis<|message|>The user asks ...<|end|><|start|>assistant to=functions.get_weather<|channel|>commentary <|constrain|>json
  ```
  **Observed:** `content: "<|start|>assistant to=functions.get_weather<|channel|>commentary <|constrain|>json"`, the raw header in content. **Expected:** no content and no call.
- **Input 2:** the `llama.cpp` test string `<|channel|>commentary to=assistant<|channel|>analysis<|message|>I'm\nthinking<|end|><|start|>assistant<|channel|>final<|message|>Hello, world!\nWhat's up?`.
  **Observed:** a tool call named `assistant<|channel|>analysis` with arguments `I'm\nthinking`.
- **Title:** `[Bug]: gpt-oss parser leaks a truncated call header into content and turns a stray 'to=assistant' header into a tool call`

#### V7. gemma4: non-streaming drops text after a call
- **Input:** SGLang's test string `Some text before <|tool_call>call:get_weather{location:<|"|>Tokyo<|"|>}<tool_call|> after`.
- **Observed:** non-streaming content is `"Some text before"`; streaming is `"Some text before  after"`.
- **Action:** add to [vllm#56263](https://github.com/vllm-project/vllm/issues/56263), which reports this class for deepseekv3 and hermes. Do not open a new issue.

#### Known upstream: comment with the fixture instead of filing
| Issue | Fixture(s) | Still reproduces on 0.30.0 |
|---|---|---|
| [#48294](https://github.com/vllm-project/vllm/issues/48294) llama3_json drops the call in one delta | `llama/l3-bug-whole-call-single-delta` and 14 more | yes, 15 fixtures |
| [#56840](https://github.com/vllm-project/vllm/issues/56840) llama4_pythonic rejects a leading `_` | `llama/l4-bug-leading-underscore-identifier` | yes |
| [#57699](https://github.com/vllm-project/vllm/issues/57699) / [#57826](https://github.com/vllm-project/vllm/issues/57826) missing close tag drops the last argument | `qwen3-xml/bug-missing-close-parameter-before-function`, `glm/glm47-missing-last-close-arg-value` | yes. The #57826 thread says GLM was fixed by #45701, but 0.30.0 still drops the value (`vllm/parser/glm47_moe.py` L56-66); say so politely, with the fixture. |
| [#56263](https://github.com/vllm-project/vllm/issues/56263) non-streaming drops post-call text | `qwen3-xml/bug-coder-text-after-call` (qwen3_coder), `gemma4/sglang-text-around-call` (gemma4) | yes; these parsers are not named in the issue yet |
| [#57353](https://github.com/vllm-project/vllm/issues/57353) kimi_k3 non-streaming returns truncated reasoning as content | `kimi/k3-truncated-in-reasoning`, `kimi/k3-bug-truncated-reasoning-recorded` | yes: non-streaming `content: "The user wants the weather in Paris"`, streaming `reasoning_content` |
| [#57688](https://github.com/vllm-project/vllm/issues/57688) kimi_k3 streaming classifies a response-only completion as reasoning | `kimi/k3-bug-response-only-completion` | yes: streaming `reasoning_content` holds the whole `<|open|>response<|sep|>Hello world...` markup |

### 7.2 SGLang (0.5.20)

#### S1. Detectors lose arguments, later parallel calls or preceding content when one delta carries a whole call
- **Direct repro:** `results/2026-09-25/repro/sglang_qwen25_one_delta.py`. Re-run on 2026-09-26:
  ```
  text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris", "unit": "c"}}\n</tool_call>'
  non-stream:    [ToolCallItem(tool_index=0, name='get_weather', parameters='{"city": "Paris", "unit": "c"}')]
  one delta :    [ToolCallItem(tool_index=0, name='get_weather', parameters='')] | normal_text: ''
  4-char deltas: [('get_weather', ''), (None, '{"city": "P'), (None, 'aris"'), (None, ', "unit": "c"}')]
  ```
- **Expected:** `parameters='{"city": "Paris", "unit": "c"}'` for the single delta.
- **Affected detectors:** qwen25, deepseekv3, deepseekv31, glm45/glm47, llama3 and mistral.
- **Scope:** 59 fixtures for single-delta or special-token-boundary deltas, plus 32 more that also fail with random 1 to 8 token groups (`sglang-stream-split-loss`). Per-token streaming and non-streaming are correct.
- **Why it matters:** speculative decoding (EAGLE/MTP) and `--stream-interval` > 1 both produce multi-token deltas.
- **Title:** `[Bug] Qwen25Detector (and deepseekv3/v31, glm45/47, llama3, mistral) lose tool-call arguments when a streaming increment contains the whole call`

#### S2. glm45 non-streaming leaks `\n<think>` into reasoning_content
- **Input:** the official GLM-4.5 template render:
  ```
  \n<think>The user wants the current weather in Paris, in Celsius.</think>\n<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>...
  ```
- **Observed:** non-streaming `reasoning_content: "\n<think>The user wants the current weather in Paris, in Celsius."`, with the markup leaked. Streaming gives `"\nThe user wants ..."`, so the two paths disagree.
- **Scope:** 10 fixtures (GLM-4.5, 4.5-Air and 4.6).
- **Title:** `[Bug] glm45 reasoning parser leaves "<think>" in non-streaming reasoning_content`

#### S3. gpt-oss detector only recognises one Harmony header form
- **Input:** the llama.cpp test string (`test-chat.cpp#L6404`) `<|channel|>commentary<|message|>Hello, world!\nWhat's up?`, a commentary-channel preamble with no recipient.
- **Observed:** `content: null, tool_calls: []`. The text is dropped entirely, in every mode.
- **Also affected** (5 fixtures): a call as the first message with no `<|start|>assistant` prefix, and a call with no `<|constrain|>json`, both come back as content with the markup leaked.
- **Separately:** 27 fixtures put the recipient in the role header (`<|start|>assistant to=functions.get_weather<|channel|>commentary ...`). The Harmony spec allows this, and it is how openai-harmony and the HF template render history. Here SGLang returns the whole call as content. Generations usually put the recipient after the channel, which SGLang parses, so file this half as low priority.
- **Title:** `[Bug] gpt-oss detector drops commentary-channel text and misses tool calls without "<|start|>assistant" or "<|constrain|>json"`

#### S4. gemma4 detector: `null` and exponent numbers become strings
- **Input:** the official Gemma 4 template render:
  ```
  <|tool_call>call:calculate{exact:true,limit:null,tolerance:1e-05,values:[0,-0.5,1e+21],x:-3,y:2.5}<tool_call|>
  ```
- **Observed:** `{"exact": true, "limit": "null", "tolerance": "1e-05", "values": [0, -0.5, "1e+21"], "x": -3, "y": 2.5}`. This fails the tool's JSON schema.
- **Also:** `none:null` becomes `"null"` in `gemma4/edge-values-key-with-space`; text after a call is dropped in non-streaming; parallel calls are lost in streaming.
- **Scope:** 5 fixtures.
- **Title:** `[Bug] Gemma4Detector returns null and exponent-notation numbers as strings`

#### S5. DeepSeek V3.2 / V4 DSML: string parameter values lose their trailing newline
- **Input:** the official `encoding_dsv32.py` render of `write_file(content="if a < b and c > d:\n    print(...)  # tab\there\n")`.
- **Observed:** the value ends at `...# tab\there`, without the final `\n`. With one token per delta, `content: "\n\n\n"` also appears. File contents written by agents are silently altered.
- **Scope:** 2 fixtures, `deepseek/v32-unescaped-string-value` and `deepseek/v4-unescaped-string-value`.
- **Title:** `[Bug] DeepSeek V3.2/V4 DSML detector strips trailing newlines from string="true" parameter values`

#### S6. Marker text inside argument strings breaks the parse
- **Input:** the same Qwen3 render as V2 (`</tool_call>` inside a JSON string).
- **Observed:** non-streaming is correct. Streaming returns `arguments: ""` for one delta, and for token deltas it truncates the argument and puts `\" blocks."}}\n</tool_call>` into `reasoning_content`.
- **Scope:** 13 fixtures across qwen25, qwen3_coder, glm47, deepseekv31, kimi_k3, gemma4, llama32 and mistral.
- **Title:** `[Bug] Streaming tool-call parsing breaks when an argument string contains the format's own markers`

#### Known upstream
- [sglang#40739](https://github.com/sgl-project/sglang/issues/40739) (qwen3_coder drops text after a call) and [sglang#35562](https://github.com/sgl-project/sglang/issues/35562) (llama3 deletes a leading JSON object): the fixtures `qwen3-xml/bug-coder-text-after-call` and `llama/l3-bug-leading-json-object-in-content` still fail on 0.5.20. They are currently triaged in the `sglang-other` bucket. Confirm them, then comment.

### 7.3 llama.cpp (`a25c9865`)

#### L1. DeepSeek-V3-0324 / V3.1 with the official HF template: every tool call fails to parse
- **Setup:** a vocab-only GGUF converted from `deepseek-ai/DeepSeek-V3-0324@e9b33add`, embedding the HF `chat_template`. `common_chat_templates_init` is called the way llama-server calls it.
- **Input:** the V1 text above.
- **Observed:** every mode raises `The model produced output that does not match the expected peg-native format`. The autoparser derives multi-token `preserved_tokens` such as `function<｜tool▁sep｜>`, so `<｜tool▁sep｜>` is never rendered as a special token.
- **Scope:** 24 fixtures.
- **Before filing:** llama.cpp's own tests use its rewritten `models/templates/deepseek-ai-DeepSeek-V3.1.jinja`, which works. The bug appears only with the HF template, which some popular GGUFs embed: `bartowski/deepseek-ai_DeepSeek-V3.1-GGUF@a7ccff77` embeds it (sha256 `45690185…`). Re-check that GGUF, and `unsloth/DeepSeek-V3.1-GGUF`, on current master.
- **Title:** `Eval bug: DeepSeek-V3.1 / V3-0324 GGUFs with the official HF chat template: tool calls fail with "does not match the expected peg-native format"`

#### L2. Kimi K3: `<|end_of_msg|>` leaks into content
- **Input:** vLLM's test string `step<|close|>think<|sep|><|open|>response<|sep|>answer`, with generation ending at `<|end_of_msg|>`.
- **Observed:** `content: "answer<|end_of_msg|>"`. The token is both a preserved token and the EOG token, so llama-server renders it.
- **Scope:** 1 fixture, plus related K3 cases in `llamacpp-other`.
- **Title:** `Eval bug: Kimi K3 end-of-generation token <|end_of_msg|> leaks into content`

#### L3. Plain answers raise instead of degrading to content
- **Inputs:**
  - the Llama 3.3 official template render `{"capital": "Paris", "country": "France"}` (a plain answer);
  - `{"a": 1} is a dict`, from sglang#35562;
  - the Kimi K3 official `encoding_k3.py` render with XML-escaped attribute keys (`key="q&amp;a"`, `key="say &quot;hi&quot;"`).
- **Observed:** every mode raises `does not match the expected peg-native format`, so the whole response errors.
- **Expected:** the plain answers come back as content, and the K3 call as `save_answer({"q&a": ..., "say \"hi\"": ...})`.
- **Scope:** 6 fixtures.
- **Note:** llama.cpp closed [#27720](https://github.com/ggml-org/llama.cpp/issues/27720) (garbled gpt-oss channel names) as "not feasible from a parsing perspective". These inputs are different: they are well-formed outputs the official templates produce.
- **Title:** `Eval bug: Llama 3.x plain JSON answers and Kimi K3 escaped attribute keys fail the final peg-native parse`

#### L4. Llama 3: special-token text inside a string argument is deleted
- **Input:** a `write_file` call whose `content` is `"Tool calls may start with <|python_tag|>; headers look like <|start_header_id|>ipython<|end_header_id|>."`.
- **Observed:** `"Tool calls may start with ; headers look like ipython."`. The argument value is silently altered.
- **Title:** `Eval bug: Llama 3 tool-call arguments lose special-token text that appears inside JSON strings`

#### L5. Candidate only, DO NOT FILE yet: Mistral-Small-3.2 / Mistral-7B-v0.3 calls are rejected when BOS is passed to `common_chat_templates_init`
- **What was seen:** an A/B through llama.cpp's code in the harness. With the vocab BOS, which is what llama-server passes (`server-context.cpp:1455`), 17 fixtures fail. With `model=nullptr`, which is what test-chat.cpp passes, they pass.
- **Before filing:** confirm it against a live `llama-server` with real weights, using `mistral/vllm-v3-parallel` as the prompt target.

### 7.4 Ollama (`7af39318`)

#### O1. glm-4.7 strips whitespace inside `<arg_value>`
- **Input:** the official GLM-4.7 template render:
  ```
  Keep the spacing exactly.</think><tool_call>write_file<arg_key>path</arg_key><arg_value>pad.txt</arg_value><arg_key>content</arg_key><arg_value>  two  spaces\n</arg_value></tool_call>
  ```
- **Observed:** `{"path":"pad.txt","content":"  two  spaces"}`, with the trailing newline dropped.
- **Expected:** `"  two  spaces\n"`. The template writes the value verbatim, so the whitespace is data.
- **Title:** `glm-4.7: tool-call string arguments lose trailing whitespace/newlines`

#### O2. Marker text inside arguments raises or leaks into content
- **qwen3 (hermes JSON):** the V2 input raises `failed to parse JSON: unexpected end of JSON input` in every mode.
- **gpt-oss (harmony):** a `write_file` whose content mentions `<|call|>` and a Harmony header raises `error parsing tool call: ... unexpected end of JSON input`.
- **gemma4:** `text:<|"|>Wrap calls as <|tool_call>call:f{}<tool_call|> and close thoughts with <channel|>.<|"|>` gives the argument `"Wrap calls as <|tool_call>call:f{"` and leaks the rest into content.
- **glm-4.7:** a value containing `</tool_call>` is cut at `"Wrap each call in <tool_call> and"`, and the rest leaks into content.
- **Scope:** 7 fixtures.
- **Title:** `Tool-call parsers (qwen3, harmony, gemma4, glm-4.7) break when an argument string contains the model's own tool-call markers`

#### Known upstream: comment with the fixture
- [#18390](https://github.com/ollama/ollama/issues/18390) (gemma4 keys with spaces): `gemma4/bug-ollama-18390-key-with-spaces`, `gemma4/edge-values-key-with-space`.
- [#18354](https://github.com/ollama/ollama/issues/18354) (gemma4 placeholder collision): `gemma4/many-strings-then-string-array`.
- [#18421](https://github.com/ollama/ollama/issues/18421) (qwen3-coder int64 clamp): `qwen3-xml/bug-coder-number-outside-int64`.
- [#18530](https://github.com/ollama/ollama/issues/18530) (qwen3-coder missing `<tool_call>` opener): `qwen3-xml/bug-coder-missing-tool-call-opener`. This one is in `ollama-other`, so confirm it first.

### 7.5 HF transformers (5.17.0) / Gemma 4 `response_template`

#### T1. `tokenizer.parse_response` raises for valid Gemma 4 tool calls
- **Direct repro:** `results/2026-09-25/repro/transformers_gemma4_hyphenated_name.py`, with `google/gemma-4-31B-it@842da379`. Re-run on 2026-09-26:
  ```
  get_weather -> {'role': 'assistant', 'tool_calls': [{'function': {'arguments': {'location': 'London'}, 'name': 'get_weather'}, 'type': 'function'}]}
  get-weather -> ValueError json: could not parse after dialect transforms.
  ```
- **Also raises for:**
  - dotted names: `call:weather.get{...}` becomes `Original: '.get{location:...}'`, because the name regex stops at `.`;
  - keys with spaces: `key with space:1.5e-05` is left unquoted after the transform;
  - marker text inside strings.
- **Scope:** 5 fixtures. The format allows all of these, and vLLM's gemma4 tests use such names.
- **Where to file:** huggingface/transformers, because the `json` dialect transform lives in `parse_response`. Cross-link a discussion on the `google/gemma-4-31B-it` Hub repo, because the regex in its `response_template` stops the name at `-` and `.`.
- **Title:** `parse_response: Gemma 4 response_template fails on tool names containing '-' or '.' and on object keys with spaces`

---

## 8. Numbers for the announcement

All numbers come from `results/2026-09-25/`. It holds 469 fixtures across 9 families, each strategy is replayed offline, and `supported` excludes fixtures an engine cannot run.

| Engine (pin) | supported | pass + soft_pass | fail | unsupported |
|---|---|---|---|---|
| vLLM 0.30.0 | 469 | 354 (75.5%) | 115 | 0 |
| SGLang 0.5.20 | 448 | 251 (56.0%) | 197 | 21 |
| llama.cpp `a25c9865` | 438 | 324 (74.0%) | 114 | 31 |
| Ollama `7af39318` | 261 | 234 (89.7%) | 27 | 208 |
| transformers 5.17.0 | 48 | 40 (83.3%) | 8 | 421 |

Of the 461 failing cases:
- 248 fall into 22 new `engine_bug` findings;
- 29 into 10 findings already reported upstream;
- 17 into 1 unconfirmed candidate;
- 70 are truncation-policy cases;
- 47 are other not-a-bug categories (strict-grammar variants, inputs the model does not generate, unverified formats, contested fixtures, history-render uncertainty, stop-token scoping, and behaviour the engine's own tests assert);
- 50 are still `untriaged_discrepancy`.

Quote findings, not failing cases: one finding can cover many fixtures.
