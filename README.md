# ternary-shard

Sharded ternary data distribution for multi-GPU inference. Partitions ternary weight matrices across GPU nodes, each shard processes independently, results merge via ternary reduce.

## Overview

# ternary-shard

Sharded ternary data distribution for multi-GPU inference.

## Stats

- **Tests**: 7
- **LOC**: 160
- **License**: Apache-2.0

## Part of the Oxide Stack

This crate is part of the [Flux→PTX](https://github.com/SuperInstance/cuda-oxide/blob/main/FLUX_TO_PTX.md) experimental suite, testing synergies between the five layers of the distributed GPU runtime:

1. **open-parallel** — async runtime (tokio fork)
2. **pincher** — "Vector DB as runtime, LLM as compiler"
3. **flux-core** — bytecode VM + A2A agent protocol
4. **cuda-oxide** — Flux→MIR→Pliron→NVVM→PTX compiler
5. **cudaclaw** — persistent GPU kernels, warp-level consensus, SmartCRDT

## Usage

```rust
use ternary_shard::*;
// See tests in src/lib.rs for examples
```

## License

Apache-2.0
