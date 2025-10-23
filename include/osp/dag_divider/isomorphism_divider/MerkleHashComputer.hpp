/*
Copyright 2024 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner
*/

#pragma once

#include <vector>
#include <unordered_map>
#include <set>
#include <stdexcept> 
#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"
#include "osp/auxiliary/hash_util.hpp"

namespace osp {

template<typename Graph_t, typename node_hash_func_t = uniform_node_hash_func<vertex_idx_t<Graph_t>>, bool forward = true>
class MerkleHashComputer {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    static_assert(std::is_invocable_r<std::size_t, node_hash_func_t, vertex_idx_t<Graph_t>>::value, "node_hash_func_t must be invocable with one vertex_idx_t<Graph_t> argument and return std::size_t.");

    using VertexType = vertex_idx_t<Graph_t>;

    std::vector<std::size_t> vertex_hashes;
    std::unordered_map<std::size_t, std::vector<VertexType>> orbits;

    node_hash_func_t node_hash_func;

    inline void compute_hashes_helper(const VertexType &v, std::vector<std::size_t> & parent_child_hashes) {

            std::sort(parent_child_hashes.begin(),parent_child_hashes.end());

            std::size_t hash = node_hash_func(v);
            for (const VertexType& pc_hash : parent_child_hashes) {
                hash_combine(hash, pc_hash); 
            }
   
            vertex_hashes[v] = hash;

            if (orbits.find(hash) == orbits.end()) {
                orbits[hash] = {v};
            } else {
                orbits[hash].push_back(v);
            }
    }

    template<typename RetT = void> 
    std::enable_if_t<forward, RetT> compute_hashes(const Graph_t & graph) {

        vertex_hashes.resize(graph.num_vertices());
        
        for (const VertexType &v : top_sort_view(graph)) {
            std::vector<std::size_t> parent_hashes;
            for (const VertexType& parent : graph.parents(v)) {
                parent_hashes.push_back(vertex_hashes[parent]);
            }
            compute_hashes_helper(v, parent_hashes);
        }
    }

    template<typename RetT = void> 
    std::enable_if_t<not forward, RetT> compute_hashes(const Graph_t & graph) {

        vertex_hashes.resize(graph.num_vertices());
        
        const auto top_sort = GetTopOrderReverse(graph);
        for (auto it = top_sort.cbegin(); it != top_sort.cend(); ++it) {
            const VertexType &v = *it;
            std::vector<std::size_t> child_hashes;
            for (const VertexType& child : graph.children(v)) {
                child_hashes.push_back(vertex_hashes[child]);
            }
            compute_hashes_helper(v, child_hashes);  
        }      
    }

  public:   

    template<typename... Args>
    MerkleHashComputer(const Graph_t &graph, Args &&...args) : node_hash_func(std::forward<Args>(args)...) {
        compute_hashes(graph);        
    }

    virtual ~MerkleHashComputer() = default;

    inline std::size_t get_vertex_hash(const VertexType &v) const { return vertex_hashes[v]; }
    inline const std::vector<std::size_t> &get_vertex_hashes() const { return vertex_hashes; }
    inline std::size_t num_orbits() const { return orbits.size(); }
    
    inline const std::vector<VertexType> &get_orbit(const VertexType &v) const { return get_orbit_from_hash(get_vertex_hash(v)); }
    inline const std::unordered_map<std::size_t, std::vector<VertexType>> &get_orbits() const { return orbits; }

    inline const std::vector<VertexType>& get_orbit_from_hash(const std::size_t& hash) const {
        return orbits.at(hash);
    }
};

/**
 * @brief Tests if two graphs are isomorphic based on their Merkle hashes.
 *
 * This function provides a strong heuristic for isomorphism. It is fast but not a
 * definitive proof. The direction of the hash (forward or backward) and the 
 * node-level hash function can be customized via template parameters.
 *
 * @tparam Graph_t The graph type, which must be a directed graph.
 * @tparam node_hash_func_t The function object type to use for hashing individual nodes.
 * @tparam Forward If true, computes a forward (top-down) hash; if false, a backward (bottom-up) hash.
 * @param g1 The first graph.
 * @param g2 The second graph.
 * @return True if the graphs are likely isomorphic based on Merkle hashes, false otherwise.
 */
template<typename Graph_t, typename node_hash_func_t = uniform_node_hash_func<vertex_idx_t<Graph_t>>, bool Forward = true>
bool are_isomorphic_by_merkle_hash(const Graph_t& g1, const Graph_t& g2) {
    // Basic check: Different numbers of vertices or edges mean they can't be isomorphic.
    if (g1.num_vertices() != g2.num_vertices() || g1.num_edges() != g2.num_edges()) {
        return false;
    }

    // --- Compute Hashes in the Specified Direction ---
    MerkleHashComputer<Graph_t, node_hash_func_t, Forward> hash1(g1);
    MerkleHashComputer<Graph_t, node_hash_func_t, Forward> hash2(g2);
    
    const auto& orbits1 = hash1.get_orbits();
    const auto& orbits2 = hash2.get_orbits();

    if (orbits1.size() != orbits2.size()) {
        return false;
    }

    for (const auto& pair : orbits1) {
        const std::size_t hash = pair.first;
        const auto& orbit_vec = pair.second;

        auto it = orbits2.find(hash);
        if (it == orbits2.end() || it->second.size() != orbit_vec.size()) {
            return false;
        }
    }
    
    return true;
}


template<typename Graph_t>
struct bwd_merkle_node_hash_func {

    MerkleHashComputer<Graph_t, uniform_node_hash_func<vertex_idx_t<Graph_t>>, false> bw_merkle_hash;

    bwd_merkle_node_hash_func(const Graph_t & graph) : bw_merkle_hash(graph) {}

    std::size_t operator()(const vertex_idx_t<Graph_t> & v) const {
        return bw_merkle_hash.get_vertex_hash(v);
    }
};

} // namespace osp