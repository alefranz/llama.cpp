# DeepSeek V4 Flash Port Plan

## Goal

Port the DeepSeek V4 Flash runtime support from the fork into the upstream-based `llama.cpp` tree, keeping it usable on a `32 GB` VRAM + `96 GB` RAM system with partial CUDA offload.

## Stages

- [x] Compare upstream repo against fork and isolate DeepSeek V4-specific runtime changes.
- [x] Confirm first pass includes CUDA support for the new DeepSeek V4 GGML ops.
- [x] Port architecture, tensor, and hparam scaffolding.
- [x] Port model loader, graph selection, and memory selection.
- [x] Port hybrid memory compressed-KV support.
- [x] Port GGML public API and CPU/meta implementations for DSV4 ops.
- [x] Add DeepSeek V4 graph builder.
- [x] Enable Windows CUDA build with partial offload by letting unsupported DSV4 ops fall back to CPU.
- [x] Build on Windows CUDA configuration and fix compile/runtime issues.
- [ ] Add native CUDA kernels for the DSV4 custom ops if profiling shows CPU fallbacks are a bottleneck.

## Notes

- Immediate target is the fork-generated DeepSeek V4 Flash GGUF runtime path.
- Conversion support can be staged separately if runtime bring-up succeeds first.
- Keep the forward-port focused on DeepSeek V4 changes and avoid unrelated fork drift.
- Current CUDA strategy is partial offload: CUDA handles the existing supported heavy ops, while the new DSV4 custom ops remain on CPU through backend capability checks.
