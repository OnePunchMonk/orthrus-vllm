# orthrus-vllm

A vLLM **out-of-tree plugin** that adds native `chiennv/Orthrus-*` checkpoint support to vLLM, with `trust_remote_code=False` — no vLLM fork, no monkeypatching, installed the same way any other vLLM model plugin is.

This exists because the equivalent **in-tree** vLLM PR was closed by a maintainer for scope reasons, not correctness. See [Why this is a plugin, not a vLLM PR](#why-this-is-a-plugin-not-a-vllm-pr) below for the exact maintainer feedback and why an OOT plugin is vLLM's own documented answer to that situation.

## Credit

This package integrates **Orthrus**, the original architecture and checkpoints, into vLLM. All of the model design, training, and released weights are the original authors' work:

> **Orthrus: Memory-Efficient Parallel Token Generation via Dual-View Diffusion**
> [chiennv2000/orthrus](https://github.com/chiennv2000/orthrus) · [arXiv:2605.12825](https://arxiv.org/abs/2605.12825) · Chien Nguyen
> Checkpoints: [`chiennv/Orthrus-Qwen3-1.7B`](https://huggingface.co/chiennv/Orthrus-Qwen3-1.7B), [`chiennv/Orthrus-Qwen3-4B`](https://huggingface.co/chiennv/Orthrus-Qwen3-4B), [`chiennv/Orthrus-Qwen3-8B`](https://huggingface.co/chiennv/Orthrus-Qwen3-8B)

This package contains no original model architecture or training contribution — it is an integration layer only.

## What this does

- Registers `OrthrusConfig` so `model_type="orthrus"` checkpoints load without remote code (`AutoConfig.register`).
- Registers `OrthrusForCausalLM` / `OrthrusLM` as a vLLM text-generation architecture (`ModelRegistry.register_model`).
- Serves Orthrus checkpoints through vLLM's **standard autoregressive path** — this is the fully working, supported part of this package.

## What this does *not* do (yet)

Diffusion-mode self-speculative decoding (Orthrus's actual headline feature — up to 2.39-7.8x speedup, strictly lossless, per the original repo) is **not** activated by this plugin. The proposer code is included as tested reference code, not a working feature. Read [`docs/DIFFUSION_MODE.md`](docs/DIFFUSION_MODE.md) for exactly what was validated, what the real, non-toy blocker is (vLLM's spec-decode drafter dispatch has no OOT extension point, unlike model registration), and what upstream change would actually fix it.

## Verification status

The autoregressive-path code in this repository is a direct port of what was validated end-to-end on real GPUs (A10G and A100-80GB) in the [#44792 PR thread](https://github.com/vllm-project/vllm/pull/44792) — see that thread for full setup/reproduction logs. It has **not** been re-run against a live vLLM install in the process of packaging it as this plugin (that requires a CUDA GPU and a full vLLM install, not available in the packaging environment) — before relying on this for anything beyond experimentation, run the smoke test below yourself and open an issue if something doesn't match the original PR's behavior.

## Install

```bash
pip install vllm>=0.11.0
pip install -e .
```

The `register_orthrus` entry point loads automatically whenever vLLM starts, no explicit import needed. To load only this plugin among others, set `VLLM_PLUGINS=register_orthrus`.

## Usage

```python
from vllm import LLM, SamplingParams

llm = LLM(model="chiennv/Orthrus-Qwen3-1.7B", trust_remote_code=False)
outputs = llm.generate(["The capital of France is"], SamplingParams(temperature=0.0, max_tokens=32))
print(outputs[0].outputs[0].text)
```

## Why this is a plugin, not a vLLM PR

This started as two PRs directly against `vllm-project/vllm`:

- **[#44792](https://github.com/vllm-project/vllm/pull/44792)** — `[Model] Add Orthrus model support (autoregressive path)`. Validated end-to-end on an A10G and an A100-80GB (including the sibling `google/diffusiongemma-26B-A4B-it` checkpoint, tested at a maintainer's suggestion), with full setup/reproduction steps posted in the PR thread.
- **[#53753](https://github.com/vllm-project/vllm/pull/53753)** — `[Model][Spec Decode] Orthrus diffusion-mode decoding (WIP)`, an explicit follow-up building on #44792.

**#44792 was closed by maintainer `Isotr0py`, with `DarkLight1337` deferring first:**

> `DarkLight1337`: *"Not sure whether this is popular enough to support tbh. I'll defer to @WoosukKwon @ywang96 and @Isotr0py"*
>
> `Isotr0py`: *"This model only has ~1200 downloads which is likely a toy model for personal research. You can support it through OOT plugin instead of native in-tree support."*

This is a **scope/adoption-policy decision, not a code-quality or correctness rejection** — neither comment raises an issue with the implementation itself. vLLM's in-tree model support has a real, structural bar: maintaining every registered architecture forever has a cost, so new architectures are filtered by external usage before they're filtered by code quality. Orthrus's ~1,200 HF downloads didn't clear that bar. `Isotr0py`'s own suggestion — the OOT plugin path — is exactly what this repository is.

**#53753 was then self-closed** ~3.5 hours after #44792, since it was an explicit follow-up ("should not be merged before it") with no reason to stay open once the base PR was declined. That PR's own comment history includes real, substantive debugging work (see [`docs/DIFFUSION_MODE.md`](docs/DIFFUSION_MODE.md)) that isn't erased by the base PR's scope-based closure — it's reused here as reference code precisely because the engineering was sound even though the in-tree path wasn't viable.

## License

Apache-2.0 for the integration code in this repository (see [`LICENSE`](LICENSE)). The Orthrus architecture and checkpoints themselves are MIT-licensed by their original author — see [`NOTICE`](NOTICE) and the [original repository](https://github.com/chiennv2000/orthrus) for full attribution.
