# OpenVLA

OpenVLA (`openvla/openvla-7b`) runs its vision backbone as one engine and its Llama-2 on Edge-LLM's regular LLM
runtime, so it uses the runtime's KV cache, sampler and
low-bit paths:

- **Vision** (`export_openvla_vision.py`, timm 0.9.10): the fused DINOv2 + SigLIP towers (second-to-last block's
  patches) and the fused-GELU projector, 256 x 4096 embeddings for the 224x224 image.
- **LLM** (`extract_openvla_llm.py`, then `tensorrt-edgellm-export` and `llm_build`): OpenVLA's Llama-2 as a plain
  checkpoint. Its config is `LlamaConfig(**text_config)`, i.e. RMSNorm eps 1e-6 (not Llama-2's published 1e-5), and the
  pad id 32000 is the image placeholder. `LLMGenerationRequest::precomputedImageEmbeddings` hands the projected patches
  to the runtime, which writes them at the placeholder positions right after BOS.
- **Policy** (`OpenvlaPolicy`): PIL's bicubic resize and the processor's two normalizations (DINOv2's constants are
  the bf16-rounded ImageNet values the processor stores), the `In: What action should the robot take to ...?\nOut:`
  prompt with the empty token appended, greedy decoding of one token per action dimension, the bin mapping and the
  dataset's q01 / q99 unnormalization. Llama-2's `tokenizer.json` prepends `\u2581` through a `Prepend` normalizer,
  which Edge-LLM's tokenizer now honours; without it the first word tokenized differently.

AGX Orin, two raw SO101 camera frames, `bridge_orig` statistics, against the official `predict_action` in FP32 on the
CPU: identical prompt ids and preprocessed pixels, the same 7 action tokens on both frames (including one decided by
a 0.04 logit margin), identical actions; 705.8 ms per action in FP16 (vision 22.5 ms, prefill and 7 tokens 683.3 ms).
Consecutive calls in one process match fresh-process results. Building the 7B FP16 engine peaks near a 64 GB Orin's
unified memory: TensorRT 10.3 keeps the parser's 13 GB of weights, about 20 GB of plan (its two optimization profiles
duplicate part of the weights) and about 20 GB of GPU allocations alive at once, 51 GB of host memory at the peak,
and it cannot stream the plan to disk. With other processes and the page cache holding memory the OOM killer stops
it during serialization; with about 47 GB available it completes (`llm_build` exit 0, same action tokens on both
frames), though allocation failures during tactic selection then cost about 10% of LLM time. Build with nothing else
resident; low-bit weights remove the problem.

```bash
python experimental_models/openvla/scripts/openvla_reference.py --checkpoint openvla-7b --image frame.png \
    --instruction "pick up the banana" --unnorm-key bridge_orig --out ref.npz        # transformers 4.40.1
python experimental_models/openvla/scripts/export_openvla_vision.py --checkpoint openvla-7b --check ref.npz \
    --out vision_onnx
trtexec --onnx=vision_onnx/vision.onnx --saveEngine=vision/vision.engine --stronglyTyped && cp vision_onnx/config.json vision/
python experimental_models/openvla/scripts/extract_openvla_llm.py --checkpoint openvla-7b --out openvla_llm
tensorrt-edgellm-export openvla_llm llm_onnx
python experimental_models/openvla/scripts/extract_openvla_llm.py --checkpoint openvla-7b --patch-onnx-config llm_onnx/llm
llm_build --onnxDir llm_onnx/llm --engineDir llm_engine --maxBatchSize 1 --maxInputLen 512 --maxKVCacheCapacity 640
openvla_policy_inference --visionDir vision --llmEngineDir llm_engine --image frame.png --instruction "..." \
    --unnormKey bridge_orig
```
