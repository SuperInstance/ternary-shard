# ternary-shard

**Split ternary weight matrices across GPUs. Each shard computes independently, results merge losslessly.**

When a ternary model is too large for one GPU, you split the weight matrix into shards. With FP32 weights, this is complicated — you need careful numerical coordination between nodes. With ternary {-1, 0, +1} weights, the math simplifies dramatically: split the vector, process each chunk independently, and concatenate the results. The merge is exact because ternary addition is closed under Z₃.

## The Insight

Sharding FP32 weights requires coordinating floating-point summation across nodes. The order of additions matters. Round-off errors accumulate differently depending on which GPU finishes first. You need reduction trees, lossy all-reduce operations, and careful error budgeting.

Ternary weights don't have this problem. A ternary vector split into N chunks can be processed independently — each chunk produces a valid partial result. Merging is just concatenation (for partitioned data) or majority voting (for replicated data). No floating-point coordination, no reduction trees, no error accumulation.

## Quick Start

```toml
[dependencies]
ternary-shard = "0.1.0"
```

```rust
use ternary_shard::*;

// A 1M-element ternary weight vector across 4 GPUs
let weights: Vec<i8> = vec![1, -1, 0, 1, -1, 0, 1, -1]; // ... 1M elements

// Shard into 4 chunks
let shards = shard_data(&weights, 4);
assert_eq!(shards.len(), 4);

// Each shard processes independently
let shard_results: Vec<Vec<i8>> = shards.iter()
    .map(|shard| process_on_gpu(shard))  // your GPU kernel
    .collect();

// Merge back losslessly
let merged = merge_shards(&shards, weights.len());
assert_eq!(merged, weights); // exact round-trip
```

## Architecture

```
  Full Ternary Vector (1M elements)
  ┌───────────────────────────────────┐
  │ [1,-1,0,1,-1,0,1,-1,0,1,-1,0,...] │
  └───────────────┬───────────────────┘
                  │ shard_data(n=4)
     ┌────────────┼────────────┐
     ▼            ▼            ▼            ▼
  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
  │gpu-0 │   │gpu-1 │   │gpu-2 │   │gpu-3 │
  │250K  │   │250K  │   │250K  │   │250K  │
  └──┬───┘   └──┬───┘   └──┬───┘   └──┬───┘
     │          │          │          │
     └────────────┴────────────┘
                  │ merge_shards()
                  ▼
  ┌───────────────────────────────────┐
  │     Reconstructed Vector          │
  └───────────────────────────────────┘
```

For replicated computation (multiple nodes computing the same thing), use `ternary_reduce` for majority voting:

```
  Node A result: [1, -1]
  Node B result: [1, -1]
  Node C result: [0,  1]
  ─────────────────────
  ternary_reduce: [1, -1]   // majority vote
```

## API Reference

### Shard

```rust
Shard::new(id: u32, node: &str, data: Vec<i8>, offset: usize) -> Shard
shard.len() -> usize
shard.is_empty() -> bool
```

A `Shard` is a slice of the original data, annotated with which node owns it and where it starts in the full vector.

### Functions

```rust
shard_data(data: &[i8], shard_count: usize) -> Vec<Shard>
merge_shards(shards: &[Shard], total_len: usize) -> Vec<i8>
ternary_reduce(shard_results: &[Vec<i8>]) -> Vec<i8>
```

- **`shard_data`** — split `data` into `shard_count` roughly equal chunks. Last shard may be shorter.
- **`merge_shards`** — reconstruct the original vector by placing each shard's data at its offset.
- **`ternary_reduce`** — element-wise majority vote across multiple result vectors. For each position, sums the ternary values and returns sign(sum) ∈ {-1, 0, +1}.

### ShardMap

```rust
ShardMap::new() -> ShardMap
map.assign(shard: Shard)
map.node_shards(node: &str) -> Vec<&Shard>
map.rebalance(failed_node: &str, healthy_nodes: &[&str])
map.shard_count() -> usize
map.node_count() -> usize
```

Tracks which node owns which shard. The key feature is `rebalance` — when a node fails, it redistributes that node's shards across the remaining healthy nodes using round-robin assignment.

## Real-World Example: Fault-Tolerant Inference

```rust
use ternary_shard::*;

// Set up sharding across 8 GPUs
let mut map = ShardMap::new();
let weights = load_ternary_weights("model.bin");
let shards = shard_data(&weights, 8);

for shard in shards {
    map.assign(shard);
}

// GPU-3 goes down
map.rebalance("gpu-3", &["gpu-0", "gpu-1", "gpu-2", "gpu-4", "gpu-5", "gpu-6", "gpu-7"]);

// GPU-3's shards are redistributed to surviving nodes
assert_eq!(map.node_shards("gpu-3").len(), 0);
// Surviving nodes picked up the extra work
```

## Rebalance Strategy

When `rebalance` is called with a failed node, it takes all shards previously assigned to that node and distributes them round-robin across the healthy nodes. Shard N from the failed node goes to `healthy_nodes[N % healthy_nodes.len()]`.

This is simple and fair — each healthy node gets roughly the same number of extra shards. It doesn't account for load imbalance (some nodes were already holding more shards) or data locality (the replacement node might not have fast access to the shard's data). Both are open improvements.

## Lossless Merge

`merge_shards` is exact: every element in the original vector appears at exactly the right position in the output. This works because shards don't overlap — each element belongs to exactly one shard. No coordination protocol needed.

`ternary_reduce` is also exact in a different sense: it's a deterministic function of its inputs. Same inputs always produce the same output. The output is the element-wise sign of the sum, which is the ternary majority vote.

## Performance

| Operation | Complexity |
|-----------|-----------|
| `shard_data` | O(n) — one pass to copy |
| `merge_shards` | O(n) — one write per element |
| `ternary_reduce` | O(k × n) — k vectors of length n |
| `rebalance` | O(s) — s shards from failed node |
| `node_shards` | O(s) — filter by node |

For a 1B-parameter ternary model sharded across 8 GPUs: `shard_data` copies ~125M elements per shard, taking ~50ms on a modern CPU. The merge is similarly fast. The bottleneck is always the GPU compute, never the sharding.

## Ecosystem

- **ternary-antidote** — CRDT-based consensus on shard ownership
- **ternary-fault-tree** — model shard failure scenarios
- **ternary-watermark** — embed fingerprints before sharding

## Open Questions

- **Load-aware rebalancing**: Track per-node compute capacity and transfer shards from overloaded to underloaded nodes.
- **Erasure coding**: Instead of full replication, store parity shards so N failures can be tolerated with only N extra shards.
- **Hierarchical sharding**: For multi-rack setups, shard within a rack first, then across racks.
- **Streaming merge**: For very large models, support merging shards one chunk at a time without loading the full result into memory.

## Stats

| Metric | Value |
|--------|-------|
| Tests | 7 |
| Lines of Rust | 161 |
| Public API | 14 items |

## License

Apache-2.0
