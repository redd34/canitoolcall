# Third-party notices

CanIToolCall's own code is licensed under Apache-2.0 (see `LICENSE`). The fixture
corpus (`fixtures/`, also shipped inside the wheel) and one harness file contain
material from the projects below. Each fixture records its source, revision and
SPDX license in `provenance`. The full license texts that require a copy with
every distribution are in `LICENSES/`.

## Code

| File | Source | License |
|---|---|---|
| `harnesses/llamacpp/replay.cpp` (the block marked "Copied from llama.cpp") | llama.cpp `tools/server/server-common.cpp` at `a25c9865` | MIT, Copyright (c) 2023-2026 The ggml authors; `LICENSES/llama.cpp-MIT.txt` |

## Fixtures copied from engine test suites

| Source | License | Text |
|---|---|---|
| llama.cpp `tests/test-chat.cpp` | MIT, Copyright (c) 2023-2026 The ggml authors | `LICENSES/llama.cpp-MIT.txt` |
| Ollama `model/parsers/*_test.go` | MIT, Copyright (c) Ollama | `LICENSES/ollama-MIT.txt` |
| vLLM, SGLang, HF transformers, openai/harmony tests | Apache-2.0 (none of them ships a NOTICE file) | `LICENSE` |

## Fixtures rendered through model repositories' templates or encoders

| Model repositories (revisions in each fixture) | License | Text |
|---|---|---|
| `deepseek-ai/DeepSeek-V3-0324`, `-V3.1`, `-V3.2`, `-V4-Flash`, `-V4.1-Flash` | MIT, Copyright (c) 2023 DeepSeek | `LICENSES/deepseek-MIT.txt` |
| `zai-org/GLM-4.5`, `zai-org/GLM-4.7` | MIT (declared in the model card metadata; the repositories ship no LICENSE file and name no copyright holder) | the MIT text as in `LICENSES/deepseek-MIT.txt`, without its copyright line |
| `zai-org/GLM-5.3` | GLM-5.3 License, Copyright (c) 2026 Z.AI | `LICENSES/glm-5.3.txt` |
| `moonshotai/Kimi-K2-Instruct-0905` | Modified MIT License, Copyright (c) 2025 Moonshot AI | `LICENSES/kimi-k2-modified-MIT.txt` |
| `moonshotai/Kimi-K2.6` | Modified MIT License, Copyright (c) 2026 Moonshot AI | `LICENSES/kimi-k2.6-modified-MIT.txt` |
| `moonshotai/Kimi-K3` | Kimi K3 License, Copyright (c) 2026 Moonshot AI | `LICENSES/kimi-k3.txt` |
| `meta-llama/Llama-3.3-70B-Instruct` (template via the `unsloth/Llama-3.3-70B-Instruct` mirror) | Llama 3.3 Community License | `LICENSES/llama3.3-community.txt` |
| `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` (also listed: `-3-Nano-30B-A3B-BF16`, `-3-Super-120B-A12B-BF16`) | NVIDIA Nemotron Open Model License (v. December 15, 2025; the repositories ship no NOTICE file) | `LICENSES/nvidia-nemotron-open-model-license.txt` |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (also listed: `-3-Ultra-550B-A55B-BF16`) | OpenMDW-1.1, Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES | `LICENSES/openmdw-1.1.txt` |
| Qwen, Gemma 4, gpt-oss, Mistral repositories and `openai-harmony` | Apache-2.0 | `LICENSE` |

## Short quotes

- Meta `llama-models` prompt-format docs (`models/llama3_3/prompt_format.md`,
  `models/llama4/prompt_format.md` at `0e0b8c51`): four one-line model-response
  examples, fixture kind `spec_example`.
- Public GitHub issues (fixture kind `bug_report`, license `NOASSERTION`): short
  quotes with the issue URL as attribution.

## Required attribution notices

Built with Llama.

Clause 1.b.i of the Llama 3.3 and Llama 4 Community Licenses requires anyone who distributes Llama Materials to provide a copy of the agreement (`LICENSES/llama3.3-community.txt`, `LICENSES/llama4-community.txt`) and to prominently display "Built with Llama" on related documentation. This project displays it in the README License section and in the footer of every page of the matrix site. Clause 1.b.iii requires the notices below.

Llama 3.3 is licensed under the Llama 3.3 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.

Llama 4 is licensed under the Llama 4 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
