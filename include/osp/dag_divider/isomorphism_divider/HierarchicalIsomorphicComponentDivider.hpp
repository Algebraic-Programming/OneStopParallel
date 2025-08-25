// /*
// Copyright 2024 Huawei Technologies Co., Ltd.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,.
// See the License for the specific language governing permissions and
// limitations under the License.

// @author Toni Boehnlein, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner
// */

// #pragma once

// #include "IsomorphicComponentDivider.hpp"

// namespace osp {

// /**
//  * @class HierarchicalIsomorphicComponentDivider
//  * @brief Divides a DAG by proactively analyzing the long-term fate of component families.
//  */
// template<typename Graph_t>
// class HierarchicalIsomorphicComponentDivider : public IDagDivider<Graph_t> {
// public:
//     explicit HierarchicalIsomorphicComponentDivider(
//         size_t family_id_level = 5,      // How many wavefronts to wait before identifying families
//         size_t min_family_size = 2,      // Min # of iso-components to be considered a "family"
//         double sub_major_merge_threshold = 0.5 // Threshold for the internal divider
//     ) : family_id_level_(family_id_level), 
//         min_family_size_(min_family_size),
//         sub_divider_(sub_major_merge_threshold) {}

//     std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> divide(const Graph_t &dag) override {
//         dag_ptr_ = &dag;
//         level_sets_ = compute_wavefronts(dag);
//         if (level_sets_.empty()) return {};

//         return partition_range(0, level_sets_.size());
//     }

// private:
//     using VertexType = vertex_idx_t<Graph_t>;

//     const Graph_t* dag_ptr_ = nullptr;
//     std::vector<std::vector<VertexType>> level_sets_;
//     size_t family_id_level_;
//     size_t min_family_size_;
//     IsomorphicComponentDivider<Graph_t> sub_divider_;

//     // Main recursive partitioning function
//     std::vector<std::vector<std::vector<VertexType>>> partition_range(size_t start_level, size_t end_level) {
//         if (start_level >= end_level) return {};

//         // 1. Identify dominant families
//         auto families = find_families(start_level, end_level);

//         // 2. Analyze their fates
//         if (families.size() >= 2) {
//             auto merge_level = find_merge_level(start_level + family_id_level_, end_level, families);

//             if (merge_level.has_value()) {
//                  // Found a merge point. Create a partition there and recurse.
//                  auto first_partition = get_components_for_range(start_level, *merge_level);
//                  auto remaining_partitions = partition_range(*merge_level, end_level);
                 
//                  std::vector<std::vector<std::vector<VertexType>>> result;
//                  result.push_back(first_partition);
//                  result.insert(result.end(), remaining_partitions.begin(), remaining_partitions.end());
//                  return result;
//             }
//         }
        
//         // If no merge point found or not enough families, use the simple divider for the whole range.
//         auto final_partition = get_components_for_range(start_level, end_level);
//         if (final_partition.size() > 1) {
//              return { final_partition };
//         }
//         return {};
//     }

//     // Identifies large groups of isomorphic components
//     std::map<MerkleHash, std::vector<std::vector<VertexType>>> find_families(size_t start_level, size_t end_level) {
//         size_t target_level = std::min(start_level + family_id_level_, end_level);
//         auto components = get_components_for_range(start_level, target_level);
        
//         MerkleHashComputer<Graph_t> hash_computer(*dag_ptr_);
//         std::map<MerkleHash, std::vector<std::vector<VertexType>>> census;
//         for (const auto& comp : components) {
//             census[hash_computer.compute_hash(comp)].push_back(comp);
//         }

//         std::map<MerkleHash, std::vector<std::vector<VertexType>>> families;
//         for (const auto& [hash, comps] : census) {
//             if (comps.size() >= min_family_size_) {
//                 families[hash] = comps;
//             }
//         }
//         return families;
//     }

//     // Look-ahead to find the first level where two distinct families merge
//     std::optional<size_t> find_merge_level(size_t start_level, size_t end_level, 
//         const std::map<MerkleHash, std::vector<std::vector<VertexType>>>& families) {

//         std::map<VertexType, int> vertex_to_family_id;
//         std::queue<std::pair<VertexType, int>> q;
//         int family_id_counter = 0;

//         for (const auto& [hash, comps] : families) {
//             for (const auto& comp : comps) {
//                 for (const auto& v : comp) {
//                     // Find leaves of the component to start the search
//                     bool is_leaf = true;
//                     for (const auto& child : dag_ptr_->children(v)) {
//                          // A simple check if child is "ahead" of this component
//                         if (std::find(comp.begin(), comp.end(), child) == comp.end()) {
//                             is_leaf = false;
//                             break;
//                         }
//                     }
//                     if (is_leaf) {
//                         for (const auto& child : dag_ptr_->children(v)) {
//                              q.push({child, family_id_counter});
//                         }
//                     }
//                 }
//             }
//             family_id_counter++;
//         }

//         std::map<VertexType, int> visited; // Maps vertex to the first family ID that reached it
        
//         while (!q.empty()) {
//             auto [current_v, family_id] = q.front();
//             q.pop();

//             if (visited.count(current_v)) {
//                 if (visited.at(current_v) != family_id) {
//                     // Merge detected! Find which level this vertex belongs to.
//                     for (size_t i = start_level; i < end_level; ++i) {
//                         for (const auto& v_in_level : level_sets_[i]) {
//                             if (v_in_level == current_v) return i;
//                         }
//                     }
//                 }
//                 continue;
//             }
//             visited[current_v] = family_id;

//             for (const auto& child : dag_ptr_->children(current_v)) {
//                 q.push({child, family_id});
//             }
//         }
//         return std::nullopt; // No merge found
//     }

//     // Helper functions (can be shared or inherited)
//     std::vector<std::vector<VertexType>> get_components_for_range(size_t start, size_t end) { /* ... same as in Isomorphic ... */ }
//     std::vector<std::vector<VertexType>> compute_wavefronts(const Graph_t& dag) const { /* ... same as in Isomorphic ... */ }
// };


// template<typename Graph_t>
// std::vector<std::vector<vertex_idx_t<Graph_t>>> HierarchicalIsomorphicComponentDivider<Graph_t>::get_components_for_range(
//     size_t start_level, size_t end_level, const std::vector<std::vector<VertexType>>& levels) {
//     if (start_level >= end_level) return {};
//     union_find_universe_t<Graph_t> uf;
//     for (size_t i = start_level; i < end_level; ++i) {
//         for(const auto& v : levels[i]) {
//             uf.add_object(v, 0, 0);
//             for(const auto& p : dag_ptr_->parents(v)) {
//                 if(uf.is_in_universe(p)) uf.join_by_name(p,v);
//             }
//         }
//     }
//     return uf.get_connected_components();
// }

// template<typename Graph_t>
// std::vector<std::vector<vertex_idx_t<Graph_t>>> HierarchicalIsomorphicComponentDivider<Graph_t>::compute_wavefronts(
//     const Graph_t& dag) const {
//     // Identical implementation to IsomorphicComponentDivider
//     std::vector<std::vector<VertexType>> level_sets;
//     std::vector<int> in_degree(dag.num_vertices(), 0);
//     std::queue<VertexType> q;
//     for (VertexType u = 0; u < dag.num_vertices(); ++u) {
//         in_degree[u] = dag.parents(u).size();
//         if (in_degree[u] == 0) q.push(u);
//     }
//     while (!q.empty()) {
//         size_t level_size = q.size();
//         std::vector<VertexType> current_level;
//         for (size_t i = 0; i < level_size; ++i) {
//             VertexType u = q.front(); q.pop();
//             current_level.push_back(u);
//             for (const auto& v : dag.children(u)) {
//                 if (--in_degree[v] == 0) q.push(v);
//             }
//         }
//         level_sets.push_back(current_level);
//     }
//     return level_sets;
// }


// } // namespace osp