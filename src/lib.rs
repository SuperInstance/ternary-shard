//! # ternary-shard
//!
//! Sharded ternary data distribution for multi-GPU inference.
//! Partitions weight matrices across nodes, merges results.

use std::collections::HashMap;

/// A shard of ternary data assigned to one GPU node.
#[derive(Debug, Clone)]
pub struct Shard {
    pub id: u32,
    pub node: String,
    pub data: Vec<i8>,
    pub offset: usize, // start position in full matrix
}

impl Shard {
    pub fn new(id: u32, node: &str, data: Vec<i8>, offset: usize) -> Self {
        Self { id, node: node.into(), data, offset }
    }

    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
}

/// Shard a ternary vector across N nodes.
pub fn shard_data(data: &[i8], shard_count: usize) -> Vec<Shard> {
    let shard_size = (data.len() + shard_count - 1) / shard_count;
    (0..shard_count).map(|i| {
        let start = i * shard_size;
        let end = (start + shard_size).min(data.len());
        Shard::new(i as u32, &format!("gpu-{}", i), data[start..end].to_vec(), start)
    }).collect()
}

/// Merge sharded results back into a single vector.
pub fn merge_shards(shards: &[Shard], total_len: usize) -> Vec<i8> {
    let mut result = vec![0i8; total_len];
    for shard in shards {
        for (i, &v) in shard.data.iter().enumerate() {
            let pos = shard.offset + i;
            if pos < total_len { result[pos] = v; }
        }
    }
    result
}

/// Ternary reduce: combine sharded results via Z₃ addition.
pub fn ternary_reduce(shard_results: &[Vec<i8>]) -> Vec<i8> {
    if shard_results.is_empty() { return vec![]; }
    let len = shard_results[0].len();
    (0..len).map(|i| {
        let sum: i32 = shard_results.iter().map(|s| s.get(i).copied().unwrap_or(0) as i32).sum();
        if sum > 0 { 1 } else if sum < 0 { -1 } else { 0 }
    }).collect()
}

/// Shard assignment: track which node owns which shard.
pub struct ShardMap {
    assignments: HashMap<String, Vec<u32>>,
    shards: HashMap<u32, Shard>,
}

impl ShardMap {
    pub fn new() -> Self { Self { assignments: HashMap::new(), shards: HashMap::new() } }

    pub fn assign(&mut self, shard: Shard) {
        let node = shard.node.clone();
        let id = shard.id;
        self.assignments.entry(node).or_default().push(id);
        self.shards.insert(id, shard);
    }

    pub fn node_shards(&self, node: &str) -> Vec<&Shard> {
        self.assignments.get(node)
            .map(|ids| ids.iter().filter_map(|id| self.shards.get(id)).collect())
            .unwrap_or_default()
    }

    pub fn rebalance(&mut self, failed_node: &str, healthy_nodes: &[&str]) {
        let shard_ids = self.assignments.remove(failed_node).unwrap_or_default();
        for (i, &sid) in shard_ids.iter().enumerate() {
            let target = healthy_nodes[i % healthy_nodes.len()];
            if let Some(shard) = self.shards.get_mut(&sid) {
                shard.node = target.to_string();
            }
            self.assignments.entry(target.to_string()).or_default().push(sid);
        }
    }

    pub fn shard_count(&self) -> usize { self.shards.len() }
    pub fn node_count(&self) -> usize { self.assignments.len() }
}

impl Default for ShardMap { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shard_even() {
        let data = vec![1, -1, 0, 1, -1, 0, 1, -1];
        let shards = shard_data(&data, 4);
        assert_eq!(shards.len(), 4);
        assert_eq!(shards.iter().map(|s| s.data.len()).sum::<usize>(), 8);
    }

    #[test]
    fn test_shard_uneven() {
        let data = vec![1, -1, 0, 1, -1];
        let shards = shard_data(&data, 3);
        assert_eq!(shards.len(), 3);
    }

    #[test]
    fn test_merge_roundtrip() {
        let data = vec![1, -1, 0, 1, -1, 0, 1, -1];
        let shards = shard_data(&data, 3);
        let merged = merge_shards(&shards, data.len());
        assert_eq!(merged, data);
    }

    #[test]
    fn test_ternary_reduce() {
        let results = vec![
            vec![1, -1],
            vec![1, -1],
            vec![0, 1],
        ];
        let reduced = ternary_reduce(&results);
        assert_eq!(reduced, vec![1, -1]); // majority
    }

    #[test]
    fn test_shard_map_assign() {
        let mut map = ShardMap::new();
        map.assign(Shard::new(0, "gpu-0", vec![1, -1], 0));
        map.assign(Shard::new(1, "gpu-1", vec![0, 1], 2));
        assert_eq!(map.shard_count(), 2);
        assert_eq!(map.node_count(), 2);
    }

    #[test]
    fn test_rebalance_on_failure() {
        let mut map = ShardMap::new();
        map.assign(Shard::new(0, "gpu-0", vec![1], 0));
        map.assign(Shard::new(1, "gpu-0", vec![-1], 1));
        map.assign(Shard::new(2, "gpu-1", vec![0], 2));
        map.rebalance("gpu-0", &["gpu-1", "gpu-2"]);
        assert_eq!(map.node_shards("gpu-0").len(), 0);
        assert!(map.node_shards("gpu-1").len() > 0);
    }

    #[test]
    fn test_empty_shard() {
        let shards = shard_data(&[], 3);
        assert!(shards.iter().all(|s| s.data.is_empty()));
    }
}
