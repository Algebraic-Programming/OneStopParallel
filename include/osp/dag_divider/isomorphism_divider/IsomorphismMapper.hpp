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
*/
#pragma once

#include <functional>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "MerkleHashComputer.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"

namespace osp {

/**
 * @brief Finds a correct isomorphic mapping between a known "representative"
 * subgraph and a new "current" subgraph, assuming they are isomorphic.
 *
 * This class uses a backtracking algorithm pruned by Merkle hashes to
 * efficiently find the vertex-to-vertex mapping.
 *
 * @tparam Graph_t The original graph type (for global vertex IDs).
 * @tparam Constr_Graph_t The subgraph/contracted graph type.
 */
template <typename Graph_t, typename Constr_Graph_t>
class IsomorphismMapper {
    using VertexC = vertex_idx_t<Constr_Graph_t>;    // Local vertex ID
    using VertexG = vertex_idx_t<Graph_t>;           // Global vertex ID

    const Constr_Graph_t &rep_graph;
    const MerkleHashComputer<Constr_Graph_t> rep_hasher;

  public:
    /**
     * @brief Constructs an IsomorphismMapper.
     * @param representative_graph The subgraph to use as the "pattern".
     */
    IsomorphismMapper(const Constr_Graph_t &representative_graph)
        : rep_graph(representative_graph), rep_hasher(representative_graph), num_vertices(representative_graph.num_vertices()) {}

    virtual ~IsomorphismMapper() = default;

    /**
     * @brief Finds the isomorphism between the representative graph and a new graph.
     *
     * This method assumes the two graphs are isomorphic and finds one such mapping.
     *
     * @param current_graph The new isomorphic subgraph.
     * @return A map from `current_local_vertex_id` -> `representative_local_vertex_id`.
     */
    std::unordered_map<VertexC, VertexC> find_mapping(const Constr_Graph_t &current_graph) const {
        if (current_graph.num_vertices() != num_vertices) {
            throw std::runtime_error("IsomorphismMapper: Graph sizes do not match.");
        }
        if (num_vertices == 0) { return {}; }

        // 1. Compute hashes and orbits for the current graph.
        MerkleHashComputer<Constr_Graph_t> current_hasher(current_graph);
        const auto &rep_orbits = rep_hasher.get_orbits();
        const auto &current_orbits = current_hasher.get_orbits();

        // 2. Verify that the orbit structures are identical.
        if (rep_orbits.size() != current_orbits.size()) {
            throw std::runtime_error("IsomorphismMapper: Graphs have a different number of orbits.");
        }
        for (const auto &[hash, rep_orbit_nodes] : rep_orbits) {
            auto it = current_orbits.find(hash);
            if (it == current_orbits.end() || it->second.size() != rep_orbit_nodes.size()) {
                throw std::runtime_error("IsomorphismMapper: Mismatched orbit structure between graphs.");
            }
        }

        // 3. Iteratively map all components of the graph.
        std::vector<VertexC> map_current_to_rep(num_vertices, std::numeric_limits<VertexC>::max());
        std::vector<bool> rep_is_mapped(num_vertices, false);
        std::vector<bool> current_is_mapped(num_vertices, false);
        size_t mapped_count = 0;

        while (mapped_count < num_vertices) {
            std::queue<std::pair<VertexC, VertexC>> q;

            // Find an unmapped vertex in the representative graph to seed the next component traversal.
            VertexC rep_seed = std::numeric_limits<VertexC>::max();
            for (VertexC i = 0; i < num_vertices; ++i) {
                if (!rep_is_mapped[i]) {
                    rep_seed = i;
                    break;
                }
            }

            if (rep_seed == std::numeric_limits<VertexC>::max()) {
                break;    // Should be unreachable if mapped_count < num_vertices
            }

            // Find a corresponding unmapped vertex in the current graph's orbit.
            const auto &candidates = current_orbits.at(rep_hasher.get_vertex_hash(rep_seed));
            VertexC current_seed = std::numeric_limits<VertexC>::max();    // Should always be found
            for (const auto &candidate : candidates) {
                if (!current_is_mapped[candidate]) {
                    current_seed = candidate;
                    break;
                }
            }
            if (current_seed == std::numeric_limits<VertexC>::max()) {
                throw std::runtime_error("IsomorphismMapper: Could not find an unmapped candidate to seed component mapping.");
            }

            // Seed the queue and start the traversal for this component.
            q.push({rep_seed, current_seed});
            map_current_to_rep[rep_seed] = current_seed;
            rep_is_mapped[rep_seed] = true;
            current_is_mapped[current_seed] = true;
            mapped_count++;

            while (!q.empty()) {
                auto [u_rep, u_curr] = q.front();
                q.pop();

                // Match neighbors (both parents and children)
                match_neighbors(current_graph,
                                current_hasher,
                                u_rep,
                                u_curr,
                                map_current_to_rep,
                                rep_is_mapped,
                                current_is_mapped,
                                mapped_count,
                                q,
                                true);
                match_neighbors(current_graph,
                                current_hasher,
                                u_rep,
                                u_curr,
                                map_current_to_rep,
                                rep_is_mapped,
                                current_is_mapped,
                                mapped_count,
                                q,
                                false);
            }
        }

        if (mapped_count != num_vertices) { throw std::runtime_error("IsomorphismMapper: Failed to map all vertices."); }

        // 4. Return the inverted map.
        std::unordered_map<VertexC, VertexC> current_local_to_rep_local;
        current_local_to_rep_local.reserve(num_vertices);
        for (VertexC i = 0; i < num_vertices; ++i) { current_local_to_rep_local[map_current_to_rep[i]] = i; }
        return current_local_to_rep_local;
    }

  private:
    const size_t num_vertices;

    void match_neighbors(const Constr_Graph_t &current_graph,
                         const MerkleHashComputer<Constr_Graph_t> &current_hasher,
                         VertexC u_rep,
                         VertexC u_curr,
                         std::vector<VertexC> &map_current_to_rep,
                         std::vector<bool> &rep_is_mapped,
                         std::vector<bool> &current_is_mapped,
                         size_t &mapped_count,
                         std::queue<std::pair<VertexC, VertexC>> &q,
                         bool match_children) const {
        const auto &rep_neighbors_range = match_children ? rep_graph.children(u_rep) : rep_graph.parents(u_rep);
        const auto &curr_neighbors_range = match_children ? current_graph.children(u_curr) : current_graph.parents(u_curr);

        for (const auto &v_rep : rep_neighbors_range) {
            if (rep_is_mapped[v_rep]) { continue; }

            for (const auto &v_curr : curr_neighbors_range) {
                if (current_is_mapped[v_curr]) { continue; }

                if (rep_hasher.get_vertex_hash(v_rep) == current_hasher.get_vertex_hash(v_curr)) {
                    map_current_to_rep[v_rep] = v_curr;
                    rep_is_mapped[v_rep] = true;
                    current_is_mapped[v_curr] = true;
                    mapped_count++;
                    q.push({v_rep, v_curr});
                    break;    // Found a match for v_rep, move to the next rep neighbor.
                }
            }
        }
    }
};

}    // namespace osp
