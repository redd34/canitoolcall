# GitHub Action: run `canitoolcall probe` in CI

The repository root has a composite action ([`action.yml`](../action.yml)) that runs [`canitoolcall probe`](../README.md#probe-a-live-server) against an OpenAI-compatible server that your job starts or can reach. It:

1. installs [uv](https://docs.astral.sh/uv/) with `astral-sh/setup-uv`, pinned to a full commit SHA, and runs the pinned canitoolcall release with `uvx`;
2. runs `canitoolcall probe` and writes the JSON report;
3. writes a Markdown table of the results to the job summary (`$GITHUB_STEP_SUMMARY`);
4. uploads the JSON report as a workflow artifact;
5. fails the step with the probe's exit code: `0` when every scenario passes, `1` when one fails or errors, `2` on a usage error or an unreachable endpoint. Warnings (`pass*`) do not fail it.

It covers the live probe only. Replaying the offline fixture suite through an engine's parsers needs the engine's isolated environment (`scripts/engines/<engine>.sh`) and is not part of the action; for that, see the pytest plugin in the [README](../README.md#for-engine-maintainers-vendor-the-fixtures).

> **Version tags.** `uses: redd34/canitoolcall@v0` works once a maintainer pushes a `v0` tag that contains `action.yml`. The `v0.2.0` tag and older ones predate the action. Until then, and for supply-chain safety in general, pin a full commit SHA: `uses: redd34/canitoolcall@<40-character SHA> # v0`.

## Inputs

| Input | Default | Meaning |
|---|---|---|
| `base-url` | (required) | API root of the server, e.g. `http://localhost:11434/v1` |
| `model` | (required) | model name sent in each request |
| `api-key-env` | `""` | the **name** of the environment variable that holds the API key, never the key. Set the variable from a secret in the step's `env:`. Empty uses canitoolcall's default, `CANITOOLCALL_API_KEY`; an unset variable means no key is sent. |
| `args` | `""` | extra `probe` arguments, split on whitespace with no shell quoting, e.g. `--no-stream --timeout 120` |
| `version` | the current release (`0.2.0`) | canitoolcall release installed from PyPI |
| `package` | `""` | install this instead of `canitoolcall==<version>` (passed to `uvx --from`), e.g. a checkout path or a git URL |
| `json-report` | `canitoolcall-probe.json` | where the JSON report is written. A file already at that path is removed before the probe runs, so a run that writes no report (unreachable endpoint, usage error) shows "no report" rather than an old one. Give each probe step in a job its own path. |
| `artifact-name` | `canitoolcall-probe` | name of the uploaded report artifact; empty skips the upload. Give each probe step in a run its own name. |

With several probe steps in one job, set both `json-report` and `artifact-name` per step, for example `json-report: probe-qwen3.json` and `artifact-name: probe-qwen3`; otherwise each step overwrites the previous report and the uploads clash.

Outputs: `exit-code` (the probe's exit code) and `report` (the report's path). The job summary's heading shows the counts and the exit code, e.g. `CanIToolCall probe: 26 pass, 1 fail, 0 error, 0 skip (exit code 1)`. Text from the report is escaped, and links in it are broken with a zero-width space so they are not clickable.

## Keys

The action never takes a key as an input. Put the key in a secret, expose it to the step under a variable name, and pass that name:

```yaml
      - uses: redd34/canitoolcall@v0
        env:
          PROVIDER_KEY: ${{ secrets.PROVIDER_KEY }}
        with:
          base-url: https://api.example.com/v1
          model: some-model
          api-key-env: PROVIDER_KEY
```

The probe sends the key only in the `Authorization` header, only to `base-url`, does not follow redirects and never prints it. It refuses to send a key over plain `http://` to a host other than localhost unless you add `--allow-insecure` to `args`. The action passes inputs to its shell steps through environment variables, never by pasting them into the script, and it wraps the probe's output in `::stop-commands::` so text quoted from the server cannot act as a workflow command. An `api-key-env` that is not a valid variable name (for example, a key pasted by mistake) fails with exit code 2 without being printed.

## Example: probe Ollama in the job

> **Untested.** This workflow has not been run. It is an illustration of starting a server inside the job: the commands come from Ollama's [Linux install docs](https://github.com/ollama/ollama/blob/v0.40.16/docs/linux.mdx) and its [OpenAI compatibility docs](https://github.com/ollama/ollama/blob/v0.40.16/docs/api/openai-compatibility.mdx) (base URL `http://localhost:11434/v1`), and `OLLAMA_VERSION` is read by [`install.sh`](https://ollama.com/install.sh). `v0.40.16` was Ollama's latest release on 2026-11-19, and `qwen3:0.6b` is listed in the [Ollama library](https://ollama.com/library/qwen3/tags). Pulling the model downloads its weights (several hundred MB) on the runner, and a CPU runner answers slowly, hence the longer timeout. Please report back if you run it.

```yaml
name: tool-calling probe

on:
  pull_request:
  workflow_dispatch:

permissions:
  contents: read

jobs:
  probe:
    runs-on: ubuntu-24.04
    timeout-minutes: 60
    steps:
      - name: install and start Ollama
        env:
          OLLAMA_VERSION: 0.40.16
        run: |
          curl -fsSL https://ollama.com/install.sh | sh
          for _ in $(seq 60); do curl -fsS http://localhost:11434/api/version && break; sleep 1; done
          ollama pull qwen3:0.6b

      - uses: redd34/canitoolcall@v0   # or a full commit SHA, see "Version tags" above
        with:
          base-url: http://localhost:11434/v1
          model: qwen3:0.6b
          args: --timeout 300 --concurrency 1
```

The same shape works for any server you can start in the job, such as `vllm serve` or `llama-server`, or for a server already reachable from the runner (pass `api-key-env` if it needs a key).

## The self-test

[`.github/workflows/action-selftest.yml`](../.github/workflows/action-selftest.yml) runs the action from this checkout (`uses: ./`) against the probe tests' mock OpenAI-compatible server, started on its own with [`scripts/probe_mock_server.py`](../scripts/probe_mock_server.py). The mock serves hand-written canned answers, not a model. The workflow checks that a conforming mock passes with the checkout (and, on pushes to `main` and manual runs, with the released version; pull requests skip that step because they may change the mock the release is tested against), that the key named by `api-key-env` reaches the server, that a mock that streams a tool-call delta without `index` fails with exit code `1`, that an invalid `api-key-env` fails with exit code `2`, and that an unreachable endpoint fails with exit code `2` and removes an old report left at its `json-report` path. `tests/probe/test_github_action.py` also runs the action's shell steps locally with `bash` and checks the summary they write.

To try the mock yourself:

```sh
uv sync
uv run python scripts/probe_mock_server.py --port 8011 &
uv run canitoolcall probe --base-url http://127.0.0.1:8011/v1 --model mock-model --json report.json
uv run python scripts/probe_summary.py report.json   # the Markdown the action writes to the job summary
```
