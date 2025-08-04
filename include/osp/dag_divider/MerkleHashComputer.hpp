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
#include "osp/auxiliary/misc.hpp"

namespace osp {

template<typename VertexType, std::size_t ret = 11>
struct default_node_hash_func {

    std::size_t operator()(const VertexType& ) {
        return ret;
    }

};

template<typename Graph_t, typename node_hash_func_t = default_node_hash_func<vertex_idx_t<Graph_t>>, bool forward = true>
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

    MerkleHashComputer(const Graph_t & g) {
        compute_hashes(g);        
    }

    virtual ~MerkleHashComputer() = default;

    std::size_t get_vertex_hash(const VertexType &v) { return vertex_hashes[v]; }
    const std::vector<std::size_t> &get_vertex_hashes() { return vertex_hashes; }
    std::size_t num_orbits() { return orbits.size(); }
    
    const std::vector<VertexType> &get_orbit(const VertexType &v) { return get_orbit_from_hash(get_vertex_hash(v)); }
    const std::unordered_map<std::size_t, std::vector<VertexType>> &get_orbits() { return orbits; }

    const std::vector<VertexType>& get_orbit_from_hash(const std::size_t& hash) const {
        try {
            return orbits.at(hash);
        } catch (const std::out_of_range& oor) {
            throw std::out_of_range("No orbit found for the given hash.");
        }
    }
};


} // namespace osp