# Design: isolated main-branch parser columns

Status: proposal for [#25](https://github.com/redd34/canitoolcall/issues/25), awaiting a maintainer decision. This document does not enable a workflow, change a score, or claim a main-branch engine was built.

## Recommendation

Approve a **vLLM + SGLang, Linux x86_64 / Python 3.12, parser-only pilot**, separate from the release matrix. First validate one engine at a time with manual dispatch; allow nightly execution only after five bounded pilot runs establish build feasibility and the budget below. No GPU, weights, live provider, or upstream issue-posting bot.

Keep Ollama and llama.cpp out of phase 1 until [#22](https://github.com/redd34/canitoolcall/issues/22) provides ref-specific isolated harness builds. This is an isolation dependency, not evidence that compiled engines are necessarily slower. Transformers is the cheapest measured release lane, but defer its main lane to keep the pilot focused on the two Python engines with the current replay requests. It is the fallback candidate if either selected engine cannot meet its gate, requiring an explicit scope change.

The deliverable is a decision on this scope, not deployment. Implementing it remains a separate issue after approval.

## 1. What can be reused, and what cannot

Source baseline throughout: [`bdade62a`](https://github.com/redd34/canitoolcall/tree/bdade62a9513ccd8657b5e153a9f7ff2421c832d).

The current [verifier](https://github.com/redd34/canitoolcall/blob/bdade62a9513ccd8657b5e153a9f7ff2421c832d/scripts/verify_upstream_pr.py) overlays a PR's changed Python files on a pinned installation. It does **not** construct a complete engine checkout at the selected commit. Repeating that overlay nightly would produce a hybrid, not an honest main-source column.

Reuse fixture loading, worker entry points, parser/detokenizer contracts, and offline replay. Build main in `.engines/main/<engine>/<sha>/` and `.venvs/main/<engine>/<environment-id>/`; never overlay `.engines/<engine>` or change release pins. Select the worker through the existing `CANITOOLCALL_<ENGINE>_PYTHON` override.

| Engine | Proposed main source strategy | Promotion gate |
|---|---|---|
| vLLM | Prefer an upstream wheel explicitly associated with the resolved full SHA, following [the upstream revision-install documentation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/). Verify artifact digest, source identity and imported paths. A source checkout plus precompiled native code must record both SHAs. | Same-SHA source/native pairing wherever native code is imported; actual parser and detokenizer smoke passes on the CPU job. No silent fallback to an older wheel or moving `nightly`. |
| SGLang | Expose the **whole** Python package from one detached source commit in its isolated environment. Resolve and save a reviewed dependency lock appropriate to that revision, rather than assuming the release script's pins remain valid. | Real parser, reasoning parser and detokenizer imports and representative fixture replay succeed. Missing native/dependency support is an unavailable build, not a reason to stub the module. |
| llama.cpp / Ollama | Separate base/ref harness build directories and recorded compiler, source and vocab identities. | Wait for #22; a cached pinned harness is not a main harness. |
| Transformers | Whole-source installation at an exact ref and environment lock. | Future optional lane, not included in the two-engine estimate. |

The existing [vLLM](https://github.com/redd34/canitoolcall/blob/bdade62a9513ccd8657b5e153a9f7ff2421c832d/scripts/engines/vllm.sh) and [SGLang](https://github.com/redd34/canitoolcall/blob/bdade62a9513ccd8657b5e153a9f7ff2421c832d/scripts/engines/sglang.sh) scripts unpack **release** wheels and install selected parser dependencies. They are evidence that parser-only setup is practical, not proof that arbitrary main commits work. Dependencies still need native wheels even when no kernels run. If the exact artifact, imports, disk space or CPU-only replay are unavailable within the cap, retain an explicit build-unavailable record and stop that lane. Do not start a GPU build or an unbounded source compilation.

## 2. Keep release and development evidence distinct

Phase 1 publishes an opt-in `/main/` view, while leaving release URLs, badges and scores untouched. Each column shows `main (development)`, full source SHA on expansion, exact dependency fingerprint, tested time, and selected head at the last check. Parser-only scope remains visible. A build error is **not** a fixture `unsupported` verdict, nor a blank green column.

This needs a future schema/renderer change: [`build_matrix`](https://github.com/redd34/canitoolcall/blob/bdade62a9513ccd8657b5e153a9f7ff2421c832d/src/canitoolcall/matrix.py#L263-L275) currently selects the latest result by `(engine.name, engine.version)`. Distinct main commits may share a version string; a track label or unique filename alone cannot prevent one replacing another. Keep raw package versions truthful. Introduce explicit track/ref/environment identity, with defaults preserving existing release files, before combining views. A sidecar manifest outside `results/*.json` can carry pilot metadata while the current strict results schema remains unchanged.

A comparison requires identical fixture digest and selected ID inventory, chunking/check policy, adapter/worker implementation and relevant dependency fingerprint. Otherwise show **not comparable**, not `fixed` or `regressed`. Two environment-specific release/main columns can still be viewed side by side, but that is not a causal attribution to source changes alone. Include `pass`, `soft_pass`, `fail`, `error` and `unsupported` separately; changed supported denominators must be visible.

Incomplete, duplicate or empty inventories and unusable subprocess exits must not become successful comparisons. [#30 / PR #31](https://github.com/redd34/canitoolcall/pull/31) propose guards for the existing verifier; that PR is still unmerged at this review and is not assumed deployed. Equal base/head inventories also cannot detect both reports omitting the same selected fixture. The new lane must validate against its frozen expected selection as well.

On failure, show the attempted SHA and failure stage, with a separately labelled last successful observation. Never rewrite the old `tested_at` to today's time. If nothing changes and an identical verified report is reused, show `reused; tested at <original time>; head checked at <new time>`.

## 3. Nightly resource estimate

Measured input, not a main-build benchmark: existing [nightly run 36552029154](https://github.com/redd34/canitoolcall/actions/runs/36552029154), 29 September 2026, Ubuntu 24.04. Durations below are differences between GitHub's recorded timestamps; job intervals exclude queueing. Cache step success does not establish a cache hit. Only one run was sampled.

| Release lane | Setup step | Replay step | Whole job |
|---|---:|---:|---:|
| vLLM | 25 s | 118 s | 160 s |
| SGLang | 32 s | 257 s | 301 s |
| llama.cpp | 24 s | 62 s | 118 s |
| Ollama | 23 s | 9 s | 58 s |
| Transformers | 15 s | 6 s | 41 s |

[Source job IDs, original timestamps and arithmetic](DESIGN-main-branch-columns-timings.json). The two proposed engines use 461 job-seconds, or **7.68 summed runner-minutes**, in this release sample, of which 375 seconds are replay. That is not elapsed pipeline time, a cold-cache measurement, or a measured main cost.

For planning, allow extra source resolution, dependency reconstruction, manifest validation and publication:

| Scenario | vLLM | SGLang | Shared metadata/publication | Total runner-minutes/night |
|---|---:|---:|---:|---:|
| Both fingerprints unchanged; valid stored reports exist | 0 | 0 | 1–2 | **1–2** |
| Both SHAs changed; dependency/download caches useful | 4–8 | 6–12 | 1–2 | **11–22** |
| Both builds cold | 8–20 | 12–30 | 1–2 | **21–52** |

These are **engineering estimates**, not timing results. Proposed enforced caps are 20 minutes for vLLM, 30 for SGLang and 5 combined for resolver/publisher, hence **55 runner-minutes** per invocation. With the two engine jobs parallel, capped compute critical path is at most 35 minutes before queueing, not 55. No automatic same-night build retry. Thirty daily warm-change runs would be 330–660 runner-minutes; cold runs 630–1,560, with a configured ceiling of 1,650. If both sources change daily, result reuse saves little; dependency caches must provide the savings.

GitHub currently makes standard hosted-runner execution free in public repositories ([billing documentation](https://docs.github.com/en/billing/concepts/product-billing/github-actions)). That is not a claim of zero storage, larger-runner or maintainer cost. Do not provision paid runners or increase storage allowances. Measure compressed artifact sizes during the pilot before choosing retention; initially use a proposal of 14 days for raw run artifacts and a 1 GiB aggregate main-build cache budget inside the repository's existing allowance, with no automatic allowance increase. If artifacts expire or caches are evicted, reuse is a miss, never proof of a fresh successful run.

## 4. Cache identity and execution ownership

Use a **separate** `nightly-main.yml` in a future implementation, initially manual-only, with a separate concurrency group and per-engine isolation. A future scheduled resolver resolves each allowlisted upstream default branch once to a full commit SHA, records that observation, then passes only immutable refs to build/replay. Never read `main` again midway through a run. If the head moves meanwhile, label the captured SHA rather than claiming to test the newer one.

Keep three different stores:

1. Download/build cache: engine SHA, upstream wheel/native identity, OS/architecture, Python ABI, dependency lock, setup recipe and toolchain hashes. Native artifacts use exact keys only. Rebuild the environment rather than restoring a venv into a different absolute path.
2. Tokenizer/template cache: pinned repository revisions and content digests. A prefix restore may be used only for immutable downloads that are individually reverified, never to select a tested engine or previous result.
3. Result artifacts plus a small manifest: additionally include fixture and selection digest, checker/chunking policy, adapter/worker hash and run configuration. A trusted resolver can reuse only a completed artifact whose **whole** fingerprint and file hashes match. Reuse preserves original run identity and timestamp; missing artifacts cause recomputation.

The current [nightly cache key](https://github.com/redd34/canitoolcall/blob/bdade62a9513ccd8657b5e153a9f7ff2421c832d/.github/workflows/nightly.yml) includes setup/harness hashes but no moving engine SHA. Do not use that namespace for main. The main environment also needs an explicit reviewed lock: today's SGLang setup has floating dependencies, so the same source SHA alone is not a reproducibility key. Review lock refreshes separately and invalidate affected results.

GitHub distinguishes [dependency caches from result artifacts](https://docs.github.com/en/actions/concepts/workflows-and-actions/dependency-caching); neither is an indefinite evidence archive. Cache eviction must not affect correctness. Retain a compact history of run IDs, manifests and failures, and link to actual retained artifacts rather than promising permanent downloads.

Build/replay jobs execute changing upstream code with no repository secrets, no persistent credentials, read-only repository permissions and no publication rights. Fetch only public, pinned tokenizer/config assets; do not provide the optional `HF_TOKEN` to main code. A separate trusted publisher validates constrained output and has Pages permission only where necessary. It never imports engine code or executes generated scripts/HTML, and untrusted PR runs cannot publish results or populate trusted build-cache namespaces. Existing release publication must survive missing main artifacts.

## Decision and implementation acceptance gates

Before enabling a schedule, require five manual pilot runs covering both engines, a warm and cold build, unchanged-result reuse, and a controlled failed build. Verify exact imported source identity, an unchanged release tree, comparable inventories, no false-green availability state, separate release/main URLs, original timestamps on reuse, and cache invalidation after a fixture or lock change. Reused results are not additional independent test runs.

Accept the pilot only when each lane fits its cap without model/native-kernel execution or manual environment repairs, storage fits existing allowances, and the maintainer accepts its measured cost. Otherwise reduce scope or pause that engine; do not silently substitute an old release. Resolve the new results identity and cache trust boundary before any scheduling PR.

**Requested decision:** approve/reject the two-engine parser-only pilot, isolated view, and 55-minute invocation ceiling. #25 remains open until that decision. No new nightly workflow, parser run, upstream issue or deployment was made for this design; existing CI durations were read, not generated by this work.

Prepared by Zero (AI collaborator) with Youngseok Oh's direction. Existing framework, fixtures, workflows and source diagnostics retain their authorship.

**Zero × Youngseok Oh**
