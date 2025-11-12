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

#include "osp/coarser/coarser_util.hpp"
#include "osp/dag_divider/isomorphism_divider/HashComputer.hpp"
#include "osp/dag_divider/isomorphism_divider/MerkleHashComputer.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"
#include "osp/graph_algorithms/transitive_reduction.hpp"
#include <numeric>
#include <unordered_set>

namespace osp {

/**
 * @class OrbitGraphProcessor
 * @brief A simple processor that groups nodes of a DAG based on their Merkle hash.
 *
 * This class uses a MerkleHashComputer to assign a structural hash to each node.
 * It then partitions the DAG by grouping all nodes with the same hash into an "orbit".
 * A coarse graph is constructed where each node represents one such orbit.
 */
template<typename Graph_t, typename Constr_Graph_t>
class OrbitGraphProcessor {
  public:
    static_assert(is_computational_dag_v<Graph_t>, "Graph must be a computational DAG");
    static_assert(is_computational_dag_v<Constr_Graph_t>, "Constr_Graph_t must be a computational DAG");
    static_assert(is_constructable_cdag_v<Constr_Graph_t>,
                  "Constr_Graph_t must satisfy the constructable_cdag_vertex concept");
    static_assert(std::is_same_v<vertex_idx_t<Graph_t>, vertex_idx_t<Constr_Graph_t>>,
                  "Graph_t and Constr_Graph_t must have the same vertex_idx types");

    using VertexType = vertex_idx_t<Graph_t>;

    static constexpr bool verbose = false;

    // Represents a group of isomorphic subgraphs, corresponding to a single node in a coarse graph.
    struct Group {
        // Each vector of vertices represents one of the isomorphic subgraphs in this group.
        std::vector<std::vector<VertexType>> subgraphs;
        
        inline size_t size() const { return subgraphs.size(); }
        // v_workw_t<Graph_t> work_weight_per_subgraph = 0;
    };

  private:
    // Results from the first (orbit) coarsening step
    Constr_Graph_t coarse_graph_;
    std::vector<VertexType> contraction_map_;

    // Results from the second (custom) coarsening step
    Constr_Graph_t final_coarse_graph_;
    std::vector<VertexType> final_contraction_map_;
    std::vector<Group> final_groups_;
    size_t current_symmetry;    

    size_t symmetry_threshold_ = 8; // max symmetry threshold
    size_t min_symmetry_ = 2; // min symmetry threshold    
    v_workw_t<Constr_Graph_t> work_threshold_ = 0;
    v_workw_t<Constr_Graph_t> critical_path_threshold_ = 0;
    bool merge_different_node_types_ = true;
    double lock_orbit_ratio = 0.2;

    struct PairHasher {
        template<class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2> &p) const {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second);
            // A common way to combine two hashes.
            return h1 ^ (h2 << 1);
        }
    };

    std::unordered_set<std::pair<VertexType, VertexType>, PairHasher> non_viable_edges_cache_;
    std::unordered_set<std::pair<VertexType, VertexType>, PairHasher> non_viable_crit_path_edges_cache_;

    std::unordered_set<VertexType> locked_orbits;

    /**
     * @brief Simulates the merge of node v into u and returns the resulting temporary graph.
     *
     * This function does not modify the current state. It creates a temporary contraction map
     * and uses it to build a potential new coarse graph for inspection.
     *
     * @param u The target node for the merge.
     * @param v The node to be merged into u.
     * @param current_coarse_graph The current coarse graph.
     * @return A pair containing the simulated coarse graph and the contraction map used to create it.
     */
    std::pair<Constr_Graph_t, std::vector<VertexType>>
    simulate_merge(VertexType u, VertexType v, const Constr_Graph_t &current_coarse_graph) const {
        std::vector<VertexType> temp_contraction_map(current_coarse_graph.num_vertices());
        VertexType new_idx = 0;
        for (VertexType i = 0; i < static_cast<VertexType>(temp_contraction_map.size()); ++i) {
            if (i != v) {
                temp_contraction_map[i] = new_idx++;
            }
        }
        // Assign 'v' the same new index as 'u'.
        temp_contraction_map[v] = temp_contraction_map[u];

        Constr_Graph_t temp_coarse_graph;
        coarser_util::construct_coarse_dag(current_coarse_graph, temp_coarse_graph, temp_contraction_map);

        return {std::move(temp_coarse_graph), std::move(temp_contraction_map)};
    }

    /**
     * @brief Commits a merge operation by updating the graph state.
     *
     * This function takes the results of a successful merge simulation and applies them,
     * updating the coarse graph, groups, and main contraction map.
     */
    void commit_merge(VertexType u, VertexType v, Constr_Graph_t &&next_coarse_graph,
                      const std::vector<VertexType> &group_remap,
                      std::vector<std::vector<VertexType>> &&new_subgraphs, Constr_Graph_t &current_coarse_graph,
                      std::vector<Group> &current_groups, std::vector<VertexType> &current_contraction_map) {

        current_coarse_graph = std::move(next_coarse_graph);

        // When we commit the merge, the vertex indices change. We must update our cache.
        std::unordered_set<std::pair<VertexType, VertexType>, PairHasher> next_non_viable_edges;
        for (const auto &non_viable_edge : non_viable_edges_cache_) {
            const VertexType old_u = non_viable_edge.first;
            const VertexType old_v = non_viable_edge.second;
            const VertexType new_u = group_remap[old_u];
            const VertexType new_v = group_remap[old_v];

            if (old_u != v && old_v != v && new_u != new_v) {
                next_non_viable_edges.insert({new_u, new_v});
            }         
        }
        non_viable_edges_cache_ = std::move(next_non_viable_edges);

        std::unordered_set<std::pair<VertexType, VertexType>, PairHasher> next_non_viable_crit_path_edges;
        for (const auto &non_viable_edge : non_viable_crit_path_edges_cache_) {
            const VertexType old_u = non_viable_edge.first;
            const VertexType old_v = non_viable_edge.second;
            const VertexType new_u = group_remap[old_u];
            const VertexType new_v = group_remap[old_v];

            if (old_u != v && old_v != v && new_u != new_v) {
                next_non_viable_crit_path_edges.insert({new_u, new_v});
            }
            
        }
        non_viable_crit_path_edges_cache_ = std::move(next_non_viable_crit_path_edges);


        std::unordered_set<VertexType> next_locked_orbits;
        for (const auto &locked_orbit : locked_orbits) {
            next_locked_orbits.insert(group_remap[locked_orbit]);
        }

        locked_orbits = std::move(next_locked_orbits);

        std::vector<Group> next_groups(current_coarse_graph.num_vertices());
        for (VertexType i = 0; i < static_cast<VertexType>(current_groups.size()); ++i) {
            if (i != u && i != v) {
                next_groups[group_remap[i]] = std::move(current_groups[i]);
            }
        }
        next_groups[group_remap[u]].subgraphs = std::move(new_subgraphs);
        current_groups = std::move(next_groups);

        for (VertexType &node_map : current_contraction_map) {
            node_map = group_remap[node_map];
        }
    }

    void merge_small_orbits(const Graph_t &original_dag, 
        Constr_Graph_t& current_coarse_graph, 
        std::vector<Group>& current_groups, 
        std::vector<VertexType>& current_contraction_map, 
        const v_workw_t<Constr_Graph_t> work_threshold, 
        const v_workw_t<Constr_Graph_t> path_threshold = 0) {

        bool changed = true;
        while (changed) {
            const std::vector<vertex_idx_t<Constr_Graph_t>> vertexPoset =
                get_top_node_distance<Constr_Graph_t, vertex_idx_t<Constr_Graph_t>>(current_coarse_graph);
            const std::vector<vertex_idx_t<Constr_Graph_t>> vertexBotPoset =
                get_bottom_node_distance<Constr_Graph_t, vertex_idx_t<Constr_Graph_t>>(current_coarse_graph);

            changed = false;
            for (const auto u : current_coarse_graph.vertices()) {
                for (const auto v : current_coarse_graph.children(u)) {                   


                    if constexpr (has_typed_vertices_v<Constr_Graph_t>) {
                        if (not merge_different_node_types_) {
                            if (current_coarse_graph.vertex_type(u) != current_coarse_graph.vertex_type(v)) {
                                if constexpr (verbose) {
                                    std::cout << "  - Merge of " << u << " and " << v << " not viable (different node types)\n";
                                }                        
                                continue;
                            }
                        }
                    }

                    if (locked_orbits.count(u) || locked_orbits.count(v)) {
                        if constexpr (verbose) {
                            std::cout << "  - Merge of " << u << " and " << v << " locked. Skipping.\n";
                        }
                        continue;
                    }

                    // Check memoization cache first
                    if (non_viable_edges_cache_.count({u, v}) || non_viable_crit_path_edges_cache_.count({u, v})) {
                        if constexpr (verbose) {
                            std::cout << "  - Merge of " << u << " and " << v << " already checked. Skipping.\n";
                        }
                        continue;
                    }


                    const v_workw_t<Constr_Graph_t> u_work_weight = current_coarse_graph.vertex_work_weight(u);
                    const v_workw_t<Constr_Graph_t> v_work_weight = current_coarse_graph.vertex_work_weight(v);
                    const v_workw_t<Constr_Graph_t> v_threshold = work_threshold * static_cast<v_workw_t<Constr_Graph_t>>(current_groups[v].size());
                    const v_workw_t<Constr_Graph_t> u_threshold = work_threshold * static_cast<v_workw_t<Constr_Graph_t>>(current_groups[u].size());

                    if (u_work_weight > u_threshold && v_work_weight > v_threshold) {
                        if constexpr (verbose) {
                            std::cout << "  - Merge of " << u << " and " << v << " not viable (work threshold)\n";
                        }
                        continue;
                    }


                    if ((vertexPoset[u] + 1 != vertexPoset[v]) && (vertexBotPoset[u] != 1 + vertexBotPoset[v])) {
                        if constexpr (verbose) {
                            std::cout << "  - Merge of " << u << " and " << v
                                      << " not viable poset. poste v: " << vertexBotPoset[v]
                                      << " poste u: " << vertexBotPoset[u] << "\n";
                        }
                        continue;
                    }

                    std::vector<std::vector<VertexType>> new_subgraphs;

                    const VertexType small_weight_vertex = u_work_weight < v_work_weight ? u : v;
                    const VertexType large_weight_vertex = u_work_weight < v_work_weight ? v : u;

                    // --- Check Constraints ---
                    // Symmetry Threshold
                    bool error = false;
                    const bool merge_viable =  is_merge_viable(original_dag, current_groups[u], current_groups[v], new_subgraphs, error);
                    const bool both_below_symmetry_threshold = (current_groups[u].size() < current_symmetry) && (current_groups[v].size() < current_symmetry);
                    const bool merge_small_weight_orbit = (current_groups[small_weight_vertex].size() >= current_symmetry) && (current_groups[large_weight_vertex].size() < current_symmetry);

                    if (error) {
                        if constexpr (verbose) {
                            std::cout << "  - Merge of " << u << " and " << v << " and " << v
                                      << " not viable (error in is_merge_viable)\n";
                        }
                        non_viable_edges_cache_.insert({u, v});
                        continue;
                    }

                    if (!merge_viable && !both_below_symmetry_threshold && !merge_small_weight_orbit) {
                        if constexpr (verbose) {
                            std::cout << "  - Merge of " << u << " and " << v << " not viable (symmetry threshold)\n";
                        }
                        non_viable_edges_cache_.insert({u, v});
                        continue;
                    }

                    // Simulate the merge to get the potential new graph.
                    auto [temp_coarse_graph, temp_contraction_map] = simulate_merge(u, v, current_coarse_graph);

                    if (critical_path_weight(temp_coarse_graph) > (path_threshold * static_cast<v_workw_t<Constr_Graph_t>>(new_subgraphs.size()) + critical_path_weight(current_coarse_graph))) {
                    //if (critical_path_weight(temp_coarse_graph) > critical_path_weight(current_coarse_graph)) {
                        if constexpr (verbose) {
                            std::cout << "  - Merge of " << u << " and " << v << " increases critical path. Old cirtical path: " << critical_path_weight(current_coarse_graph)
                                    << " new critical path: " << critical_path_weight(temp_coarse_graph) << " + " << path_threshold * static_cast<v_workw_t<Constr_Graph_t>>(new_subgraphs.size()) << "\n";
                        }
                        non_viable_crit_path_edges_cache_.insert({u, v});
                        continue;
                    }

                    // If all checks pass, commit the merge.
                    if constexpr (verbose) {
                        std::cout << "  - Merging " << v << " into " << u << ". New coarse graph has "
                                << temp_coarse_graph.num_vertices() << " nodes.\n";
                    }

                    commit_merge(u, v, std::move(temp_coarse_graph), temp_contraction_map, std::move(new_subgraphs),
                                 current_coarse_graph, current_groups, current_contraction_map);

                    changed = true;
                    break; // Restart scan on the new, smaller graph
                }
                if (changed) {
                    break;
                }
            }
        }
    }

    void contract_edges(const Graph_t &original_dag, Constr_Graph_t& current_coarse_graph, std::vector<Group>& current_groups, std::vector<VertexType>& current_contraction_map, const bool merge_symmetry_narrowing, const bool merge_different_node_types, const v_workw_t<Constr_Graph_t> path_threshold = 0) {

        bool changed = true;
        while (changed) {
            const std::vector<vertex_idx_t<Constr_Graph_t>> vertexPoset =
                get_top_node_distance<Constr_Graph_t, vertex_idx_t<Constr_Graph_t>>(current_coarse_graph);
            const std::vector<vertex_idx_t<Constr_Graph_t>> vertexBotPoset =
                get_bottom_node_distance<Constr_Graph_t, vertex_idx_t<Constr_Graph_t>>(current_coarse_graph);

            changed = false;
            for (const auto &edge : edges(current_coarse_graph)) {
                VertexType u = source(edge, current_coarse_graph);
                VertexType v = target(edge, current_coarse_graph);

                // Check memoization cache first
                if (non_viable_edges_cache_.count({u, v}) || non_viable_crit_path_edges_cache_.count({u, v})) {
                    if constexpr (verbose) {
                        std::cout << "  - Merge of " << u << " and " << v << " already checked. Skipping.\n";
                    }
                    continue;
                }

                if constexpr (has_typed_vertices_v<Constr_Graph_t>) {
                    if (not merge_different_node_types) {
                        if (current_coarse_graph.vertex_type(u) != current_coarse_graph.vertex_type(v)) {
                            if constexpr (verbose) {
                                std::cout << "  - Merge of " << u << " and " << v << " not viable (different node types)\n";
                            }                        
                            continue;
                        }
                    }
                }

                if ((vertexPoset[u] + 1 != vertexPoset[v]) && (vertexBotPoset[u] != 1 + vertexBotPoset[v])) {
                    if constexpr (verbose) {
                        std::cout << "  - Merge of " << u << " and " << v
                                  << " not viable poset. poste v: " << vertexBotPoset[v]
                                  << " poste u: " << vertexBotPoset[u] << "\n";
                    }
                    continue;
                }

                std::vector<std::vector<VertexType>> new_subgraphs;

                // --- Check Constraints ---
                // Symmetry Threshold

                const std::size_t u_size = current_groups[u].size();
                const std::size_t v_size = current_groups[v].size();
                bool error = false;
                const bool merge_viable = is_merge_viable(original_dag, current_groups[u], current_groups[v], new_subgraphs, error);
                const bool both_below_symmetry_threshold =
                    (u_size < current_symmetry) &&
                    (v_size < current_symmetry);// && 
                    //  (not ((u_size == 1 && v_size > 1) || (u_size > 1 && v_size == 1)));

                if (error) {
                    if constexpr (verbose) {
                        std::cout << "  - Merge of " << u << " and " << v << " and " << v
                                  << " not viable (error in is_merge_viable)\n";
                    }
                    non_viable_edges_cache_.insert({u, v});
                    continue;
                }

                if (!merge_viable && !both_below_symmetry_threshold) {
                    if constexpr (verbose) {
                        std::cout << "  - Merge of " << u << " and " << v << " not viable (symmetry threshold)\n";
                    }
                    non_viable_edges_cache_.insert({u, v});
                    continue;
                }

                if (not merge_symmetry_narrowing) {
                    const std::size_t min_size = std::min(u_size, v_size);
                    const std::size_t new_size = new_subgraphs.size();

                    if (new_size < min_size) {
                        if constexpr (verbose) {                        std::cout << "  - Merge of " << u << " and " << v
                                  << " not viable (symmetry narrowing: " << u_size << "x" << v_size << " -> "
                                  << new_size << " subgraphs)\n";
                        }
                        continue;
                    }
                }

                // Simulate the merge to get the potential new graph.
                auto [temp_coarse_graph, temp_contraction_map] = simulate_merge(u, v, current_coarse_graph);

                if (critical_path_weight(temp_coarse_graph) > (path_threshold * static_cast<v_workw_t<Constr_Graph_t>>(new_subgraphs.size()) + critical_path_weight(current_coarse_graph))) {
                //if (critical_path_weight(temp_coarse_graph) > critical_path_weight(current_coarse_graph)) {
                    if constexpr (verbose) {
                        std::cout << "  - Merge of " << u << " and " << v << " increases critical path. Old cirtical path: " << critical_path_weight(current_coarse_graph)
                                  << " new critical path: " << critical_path_weight(temp_coarse_graph) << " + " << path_threshold * static_cast<v_workw_t<Constr_Graph_t>>(new_subgraphs.size()) << "\n";
                    }
                    non_viable_crit_path_edges_cache_.insert({u, v});
                    continue;
                }

                // If all checks pass, commit the merge.
                if constexpr (verbose) {
                    std::cout << "  - Merging " << v << " into " << u << ". New coarse graph has "
                              << temp_coarse_graph.num_vertices() << " nodes.\n";
                }

                commit_merge(u, v, std::move(temp_coarse_graph), temp_contraction_map, std::move(new_subgraphs), 
                             current_coarse_graph, current_groups, current_contraction_map);

                changed = true;
                break; // Restart scan on the new, smaller graph
            }
        }
    }


    void contract_edges_adpative_sym(const Graph_t &original_dag, 
        Constr_Graph_t& current_coarse_graph, 
        std::vector<Group>& current_groups, 
        std::vector<VertexType>& current_contraction_map, 
        /* const bool merge_symmetry_narrowing, */ 
        const bool merge_different_node_types, 
        const bool check_below_threshold, 
        const v_workw_t<Constr_Graph_t> path_threshold = 0) {

        bool changed = true;
        while (changed) {
            const std::vector<vertex_idx_t<Constr_Graph_t>> vertexPoset =
                get_top_node_distance<Constr_Graph_t, vertex_idx_t<Constr_Graph_t>>(current_coarse_graph);
            const std::vector<vertex_idx_t<Constr_Graph_t>> vertexBotPoset =
                get_bottom_node_distance<Constr_Graph_t, vertex_idx_t<Constr_Graph_t>>(current_coarse_graph);

            changed = false;
            for (const auto &edge : edges(current_coarse_graph)) {
                VertexType u = source(edge, current_coarse_graph);
                VertexType v = target(edge, current_coarse_graph);

                if (locked_orbits.count(u) || locked_orbits.count(v)) {
                    if constexpr (verbose) {
                        std::cout << "  - Merge of " << u << " and " << v << " locked. Skipping.\n";
                    }
                    continue;
                }

                // Check memoization cache first
                if (non_viable_edges_cache_.count({u, v}) || non_viable_crit_path_edges_cache_.count({u, v})) {
                    if constexpr (verbose) {
                        std::cout << "  - Merge of " << u << " and " << v << " already checked. Skipping.\n";
                    }
                    continue;
                }

                if constexpr (has_typed_vertices_v<Constr_Graph_t>) {
                    if (not merge_different_node_types) {
                        if (current_coarse_graph.vertex_type(u) != current_coarse_graph.vertex_type(v)) {
                            if constexpr (verbose) {
                                std::cout << "  - Merge of " << u << " and " << v << " not viable (different node types)\n";
                            }                        
                            continue;
                        }
                    }
                }

                if ((vertexPoset[u] + 1 != vertexPoset[v]) && (vertexBotPoset[u] != 1 + vertexBotPoset[v])) {
                    if constexpr (verbose) {
                        std::cout << "  - Merge of " << u << " and " << v
                                  << " not viable poset. poste v: " << vertexBotPoset[v]
                                  << " poste u: " << vertexBotPoset[u] << "\n";
                    }
                    continue;
                }

                std::vector<std::vector<VertexType>> new_subgraphs;

                // --- Check Constraints ---
                // Symmetry Threshold

                const std::size_t u_size = current_groups[u].size();
                const std::size_t v_size = current_groups[v].size();
                bool error = false;
                const bool merge_viable = is_merge_viable(original_dag, current_groups[u], current_groups[v], new_subgraphs, error);
                const bool both_below_symmetry_threshold = check_below_threshold &&
                    (u_size < current_symmetry) &&
                    (v_size < current_symmetry);// && 
                    //  (not ((u_size == 1 && v_size > 1) || (u_size > 1 && v_size == 1)));

                if (error) {
                    if constexpr (verbose) {
                        std::cout << "  - Merge of " << u << " and " << v << " and " << v
                                  << " not viable (error in is_merge_viable)\n";
                    }
                    non_viable_edges_cache_.insert({u, v});
                    continue;
                }

                if (!merge_viable && !both_below_symmetry_threshold) {
                    if constexpr (verbose) {
                        std::cout << "  - Merge of " << u << " and " << v << " not viable (symmetry threshold)\n";
                    }
                    non_viable_edges_cache_.insert({u, v});
                    continue;
                }


                

                // Simulate the merge to get the potential new graph.
                auto [temp_coarse_graph, temp_contraction_map] = simulate_merge(u, v, current_coarse_graph);

                if (critical_path_weight(temp_coarse_graph) > (path_threshold * static_cast<v_workw_t<Constr_Graph_t>>(new_subgraphs.size()) + critical_path_weight(current_coarse_graph))) {
                //if (critical_path_weight(temp_coarse_graph) > critical_path_weight(current_coarse_graph)) {
                    if constexpr (verbose) {
                        std::cout << "  - Merge of " << u << " and " << v << " increases critical path. Old cirtical path: " << critical_path_weight(current_coarse_graph)
                                  << " new critical path: " << critical_path_weight(temp_coarse_graph) << " + " << path_threshold * static_cast<v_workw_t<Constr_Graph_t>>(new_subgraphs.size()) << "\n";
                    }
                    non_viable_crit_path_edges_cache_.insert({u, v});
                    continue;
                }

                // If all checks pass, commit the merge.
                if constexpr (verbose) {
                    std::cout << "  - Merging " << v << " into " << u << ". New coarse graph has "
                              << temp_coarse_graph.num_vertices() << " nodes.\n";
                }

                commit_merge(u, v, std::move(temp_coarse_graph), temp_contraction_map, std::move(new_subgraphs), 
                             current_coarse_graph, current_groups, current_contraction_map);

                changed = true;
                break; // Restart scan on the new, smaller graph
            }
        }
    }


  public:
    explicit OrbitGraphProcessor(size_t symmetry_threshold = 2) : symmetry_threshold_(symmetry_threshold) {}

    /**
     * @brief Sets the minimum number of isomorphic subgraphs a merged group must have.
     * @param threshold The symmetry threshold.
     */
    void set_symmetry_threshold(size_t threshold) { symmetry_threshold_ = threshold; }
    void setMergeDifferentNodeTypes(bool flag) { merge_different_node_types_ = flag; }
    void set_work_threshold(v_workw_t<Constr_Graph_t> work_threshold) { work_threshold_ = work_threshold; }
    void setCriticalPathThreshold(v_workw_t<Constr_Graph_t> critical_path_threshold) { critical_path_threshold_ = critical_path_threshold; }
    void setLockRatio(double lock_ratio) { lock_orbit_ratio = lock_ratio; }
    void setMinSymmetry(size_t min_symmetry) { min_symmetry_ = min_symmetry; }

    /**
     * @brief Discovers isomorphic groups (orbits) and constructs a coarse graph.
     * @param dag The input computational DAG.
     */
    void discover_isomorphic_groups(const Graph_t &dag, const HashComputer<VertexType> &hasher) {
        coarse_graph_ = Constr_Graph_t();
        contraction_map_.clear();
        final_coarse_graph_ = Constr_Graph_t();
        final_contraction_map_.clear();
        final_groups_.clear();
        non_viable_edges_cache_.clear();
        non_viable_crit_path_edges_cache_.clear();
        current_symmetry = symmetry_threshold_;

        if (dag.num_vertices() == 0) {
            return;
        }

        const auto &orbits = hasher.get_orbits();

        contraction_map_.assign(dag.num_vertices(), 0);
        VertexType coarse_node_idx = 0;

        for (const auto &hash_vertices_pair : orbits) {
            const auto &vertices = hash_vertices_pair.second;
            for (const auto v : vertices) {
                contraction_map_[v] = coarse_node_idx;
            }
            coarse_node_idx++;
        }

        coarser_util::construct_coarse_dag(dag, coarse_graph_, contraction_map_);
        perform_coarsening_adaptive_symmetry(dag, coarse_graph_);
    }

  private:
    /**
     * @brief Greedily merges nodes in the orbit graph based on structural and symmetry constraints.
     */
    void perform_coarsening(const Graph_t &original_dag, const Constr_Graph_t &initial_coarse_graph) {
        final_coarse_graph_ = Constr_Graph_t();
        final_contraction_map_.clear();

        if (initial_coarse_graph.num_vertices() == 0) {
            return;
        }

        Constr_Graph_t current_coarse_graph = initial_coarse_graph;
        std::vector<Group> current_groups(initial_coarse_graph.num_vertices());
        std::vector<VertexType> current_contraction_map = contraction_map_;

        // Initialize groups: each group corresponds to an orbit.
        for (VertexType i = 0; i < original_dag.num_vertices(); ++i) {
            const VertexType coarse_node = contraction_map_[i];
            current_groups[coarse_node].subgraphs.push_back({i});
        }


        if constexpr (has_typed_vertices_v<Constr_Graph_t>) {
            if constexpr (verbose) {
                std::cout << "Attempting to merge same node types.\n";
            }
            contract_edges(original_dag, current_coarse_graph, current_groups, current_contraction_map, false, false);
            contract_edges(original_dag, current_coarse_graph, current_groups, current_contraction_map, true, false);
        }


        if constexpr (verbose) {
            std::cout << "Attempting to merge different node types.\n";
        }
        contract_edges(original_dag, current_coarse_graph, current_groups, current_contraction_map, false, merge_different_node_types_);
        contract_edges(original_dag, current_coarse_graph, current_groups, current_contraction_map, true, merge_different_node_types_);
    

        if constexpr (verbose) {
            std::cout << "Attempting to merge small orbits.\n";
        }
        merge_small_orbits(original_dag, current_coarse_graph, current_groups, current_contraction_map, work_threshold_);

        non_viable_crit_path_edges_cache_.clear();
        non_viable_edges_cache_.clear();

        contract_edges(original_dag, current_coarse_graph, current_groups, current_contraction_map, true, merge_different_node_types_, work_threshold_);
        
        // --- Finalize ---
        final_coarse_graph_ = std::move(current_coarse_graph);
        final_contraction_map_ = std::move(current_contraction_map);
        final_groups_ = std::move(current_groups);

        if constexpr (verbose) {
            print_final_groups_summary();
        }
    }

    void perform_coarsening_adaptive_symmetry(const Graph_t &original_dag, const Constr_Graph_t &initial_coarse_graph) {
        final_coarse_graph_ = Constr_Graph_t();
        final_contraction_map_.clear();

        if (initial_coarse_graph.num_vertices() == 0) {
            return;
        }

        Constr_Graph_t current_coarse_graph = initial_coarse_graph;
        std::vector<Group> current_groups(initial_coarse_graph.num_vertices());
        std::vector<VertexType> current_contraction_map = contraction_map_;

        // Initialize groups: each group corresponds to an orbit.
        for (VertexType i = 0; i < original_dag.num_vertices(); ++i) {
            const VertexType coarse_node = contraction_map_[i];
            current_groups[coarse_node].subgraphs.push_back({i});
        }

        v_workw_t<Constr_Graph_t> total_work_weight = sumOfVerticesWorkWeights(initial_coarse_graph);
        v_workw_t<Constr_Graph_t> lock_threshold = static_cast<v_workw_t<Constr_Graph_t>>(lock_orbit_ratio * total_work_weight);        

        if constexpr (verbose) {
            std::cout << " Starting adaptive symmetry coarsening with lock threshold: " << lock_threshold << ", critical_path_threshold: " << critical_path_threshold_ << "\n";
        }

        while (current_symmetry >= min_symmetry_) {

            if constexpr (verbose) {
                std::cout << "  Current symmetry threshold: " << current_symmetry << "\n";
            }

            non_viable_edges_cache_.clear();

            const bool is_last_loop = (current_symmetry / 2) < min_symmetry_;   
            contract_edges_adpative_sym(original_dag, current_coarse_graph, current_groups, current_contraction_map, false, is_last_loop);                 
            
            if (merge_different_node_types_)
                contract_edges_adpative_sym(original_dag, current_coarse_graph, current_groups, current_contraction_map, merge_different_node_types_, is_last_loop);
            
            non_viable_crit_path_edges_cache_.clear();
            contract_edges_adpative_sym(original_dag, current_coarse_graph, current_groups, current_contraction_map, merge_different_node_types_, is_last_loop, critical_path_threshold_);

            for (const auto& v : current_coarse_graph.vertices()) {
                if (current_coarse_graph.vertex_work_weight(v) > lock_threshold) {
                    if constexpr (verbose) {
                        std::cout << "  Locking orbit " << v << "\n";
                    }
                    locked_orbits.insert(v);
                }
            }  
            current_symmetry = current_symmetry / 2;
        }
  
        if constexpr (verbose) {
            std::cout << " Merging small orbits with work threshold: " << work_threshold_ << "\n";
        }
        non_viable_edges_cache_.clear();
        merge_small_orbits(original_dag, current_coarse_graph, current_groups, current_contraction_map, work_threshold_);
        
        // --- Finalize ---
        final_coarse_graph_ = std::move(current_coarse_graph);
        final_contraction_map_ = std::move(current_contraction_map);
        final_groups_ = std::move(current_groups);

        if constexpr (verbose) {
            print_final_groups_summary();
        }
    }

    void print_final_groups_summary() const {
        std::cout << "\n--- 📦 Final Groups Summary ---\n";
        std::cout << "Total final groups: " << final_groups_.size() << "\n";
        for (size_t i = 0; i < final_groups_.size(); ++i) {
            const auto &group = final_groups_[i];
            std::cout << "  - Group " << i << " (Size: " << group.subgraphs.size() << ")\n";
            if (!group.subgraphs.empty() && !group.subgraphs[0].empty()) {
                std::cout << "    - Rep. Subgraph size: " << group.subgraphs[0].size() << " nodes\n";
            }
        }
        std::cout << "--------------------------------\n";
    }

    /**
     * @brief Checks if merging two groups is viable based on the resulting number of isomorphic subgraphs.
     * This is analogous to WavefrontOrbitProcessor::is_viable_continuation.
     * If viable, it populates the `out_new_subgraphs` with the structure of the merged group.
     */
    bool is_merge_viable(const Graph_t &original_dag, const Group &group_u, const Group &group_v,
                         std::vector<std::vector<VertexType>> &out_new_subgraphs, bool &error) const {

        std::vector<VertexType> all_nodes;
        all_nodes.reserve(group_u.subgraphs.size() + group_v.subgraphs.size());
        for (const auto &sg : group_u.subgraphs) {
            all_nodes.insert(all_nodes.end(), sg.begin(), sg.end());
        }
        for (const auto &sg : group_v.subgraphs) {
            all_nodes.insert(all_nodes.end(), sg.begin(), sg.end());
        }

        assert([&]() {
            std::vector<VertexType> temp_nodes_for_check = all_nodes;
            std::sort(temp_nodes_for_check.begin(), temp_nodes_for_check.end());
            return std::unique(temp_nodes_for_check.begin(), temp_nodes_for_check.end()) == temp_nodes_for_check.end();
        }() && "Assumption failed: Vertices in groups being merged are not disjoint.");

        std::sort(all_nodes.begin(), all_nodes.end());

        Constr_Graph_t induced_subgraph;

        auto map = create_induced_subgraph_map(original_dag, induced_subgraph, all_nodes);
        std::vector<VertexType> components; // local -> component_id
        size_t num_components = compute_weakly_connected_components(induced_subgraph, components);
        out_new_subgraphs.assign(num_components, std::vector<VertexType>());
        for (const auto &node : all_nodes) {
            out_new_subgraphs[components[map[node]]].push_back(node);
        }

        if (num_components > 1) {
            const size_t first_sg_size = out_new_subgraphs[0].size();
            Constr_Graph_t rep_sg;
            create_induced_subgraph(original_dag, rep_sg, out_new_subgraphs[0]);

            for (size_t i = 1; i < num_components; ++i) {
                if (out_new_subgraphs[i].size() != first_sg_size) {
                    error = true;
                    return false;
                }

                Constr_Graph_t current_sg;
                create_induced_subgraph(original_dag, current_sg, out_new_subgraphs[i]);
                if (!are_isomorphic_by_merkle_hash(rep_sg, current_sg)) {
                    error = true;
                    return false;
                }
            }
        }

        return num_components >= current_symmetry;
    }

  public:
    const Graph_t &get_coarse_graph() const { return coarse_graph_; }
    const std::vector<VertexType> &get_contraction_map() const { return contraction_map_; }
    const Graph_t &get_final_coarse_graph() const { return final_coarse_graph_; }
    const std::vector<VertexType> &get_final_contraction_map() const { return final_contraction_map_; }
    const std::vector<Group> &get_final_groups() const { return final_groups_; }
};

} // namespace osp