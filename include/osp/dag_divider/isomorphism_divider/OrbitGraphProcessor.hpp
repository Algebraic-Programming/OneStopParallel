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
#include <algorithm>
#include <map>

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

    /**
     * @brief Heuristics for selecting which symmetry levels to test during coarsening.
     */
    enum class SymmetryLevelHeuristic {
        /**
         * @brief Original logic: Select levels where cumulative work passes an increasing threshold.
         */
        CURRENT_DEFAULT,
        /**
         * @brief Select levels that correspond to fixed work-load percentiles.
         */
        PERCENTILE_BASED,
        /**
         * @brief Select levels based on the orbit size or count distribution.
         */
        NATURAL_BREAKS
    };

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
    double lock_orbit_ratio = 0.5;

    SymmetryLevelHeuristic symmetry_level_heuristic_ = SymmetryLevelHeuristic::NATURAL_BREAKS;
    std::vector<double> work_percentiles_ = {0.50, 0.75};
    double natural_breaks_count_percentage_ = 0.2; 

    struct PairHasher {
        template<class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2> &p) const {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second);
            return h1 ^ (h2 << 1);
        }
    };

    std::unordered_set<std::pair<VertexType, VertexType>, PairHasher> non_viable_edges_cache_;
    std::unordered_set<std::pair<VertexType, VertexType>, PairHasher> non_viable_crit_path_edges_cache_;

    /**
     * @brief Simulates the merge of node v into u and returns the resulting temporary graph.
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
        temp_contraction_map[v] = temp_contraction_map[u];

        Constr_Graph_t temp_coarse_graph;
        coarser_util::construct_coarse_dag(current_coarse_graph, temp_coarse_graph, temp_contraction_map);

        return {std::move(temp_coarse_graph), std::move(temp_contraction_map)};
    }

    /**
     * @brief Commits a merge operation by updating the graph state.
     */
    void commit_merge(VertexType u, VertexType v, Constr_Graph_t &&next_coarse_graph,
                      const std::vector<VertexType> &group_remap,
                      std::vector<std::vector<VertexType>> &&new_subgraphs, Constr_Graph_t &current_coarse_graph,
                      std::vector<Group> &current_groups, std::vector<VertexType> &current_contraction_map) {

        current_coarse_graph = std::move(next_coarse_graph);

        // Update caches for new vertex indices
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

        // Update groups
        std::vector<Group> next_groups(current_coarse_graph.num_vertices());
        for (VertexType i = 0; i < static_cast<VertexType>(current_groups.size()); ++i) {
            if (i != u && i != v) {
                next_groups[group_remap[i]] = std::move(current_groups[i]);
            }
        }
        next_groups[group_remap[u]].subgraphs = std::move(new_subgraphs);
        current_groups = std::move(next_groups);

        // Update main contraction map
        for (VertexType &node_map : current_contraction_map) {
            node_map = group_remap[node_map];
        }
    }

    /**
     * @brief Merges small orbits based on work threshold (final cleanup pass).
     */
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
                    const bool merge_is_valid = is_merge_viable(original_dag, current_groups[u], current_groups[v], new_subgraphs);
                    
                    if (!merge_is_valid) {
                        if constexpr (verbose) {
                            std::cout << "  - Merge of " << u << " and " << v << " and " << v
                                      << " not viable (error in is_merge_viable)\n";
                        }
                        non_viable_edges_cache_.insert({u, v});
                        continue;
                    }

                    auto [temp_coarse_graph, temp_contraction_map] = simulate_merge(u, v, current_coarse_graph);

                    if (critical_path_weight(temp_coarse_graph) > (path_threshold * static_cast<v_workw_t<Constr_Graph_t>>(new_subgraphs.size()) + critical_path_weight(current_coarse_graph))) {
                        if constexpr (verbose) {
                            std::cout << "  - Merge of " << u << " and " << v << " increases critical path. Old cirtical path: " << critical_path_weight(current_coarse_graph)
                                      << " new critical path: " << critical_path_weight(temp_coarse_graph) << " + " << path_threshold * static_cast<v_workw_t<Constr_Graph_t>>(new_subgraphs.size()) << "\n";
                        }
                        non_viable_crit_path_edges_cache_.insert({u, v});
                        continue;
                    }

                    if constexpr (verbose) {
                        std::cout << "  - Merging " << v << " into " << u << ". New coarse graph has "
                                  << temp_coarse_graph.num_vertices() << " nodes.\n";
                    }

                    commit_merge(u, v, std::move(temp_coarse_graph), temp_contraction_map, std::move(new_subgraphs),
                                 current_coarse_graph, current_groups, current_contraction_map);

                    changed = true;
                    break;
                }
                if (changed) {
                    break;
                }
            }
        }
    }

    /**
     * @brief Deprecated non-adaptive merge function.
     */
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

                if (non_viable_edges_cache_.count({u, v}) || non_viable_crit_path_edges_cache_.count({u, v})) {
                    continue;
                }
                if constexpr (has_typed_vertices_v<Constr_Graph_t>) {
                    if (not merge_different_node_types) {
                        if (current_coarse_graph.vertex_type(u) != current_coarse_graph.vertex_type(v)) {
                            continue;
                        }
                    }
                }
                if ((vertexPoset[u] + 1 != vertexPoset[v]) && (vertexBotPoset[u] != 1 + vertexBotPoset[v])) {
                    continue;
                }

                std::vector<std::vector<VertexType>> new_subgraphs;
                const std::size_t u_size = current_groups[u].size();
                const std::size_t v_size = current_groups[v].size();
                const bool merge_is_valid = is_merge_viable(original_dag, current_groups[u], current_groups[v], new_subgraphs);
                const std::size_t new_size = new_subgraphs.size();
                
                const bool merge_viable = (new_size >= current_symmetry); 
                const bool both_below_symmetry_threshold = (u_size < current_symmetry) && (v_size < current_symmetry);

                if (!merge_is_valid) {
                    non_viable_edges_cache_.insert({u, v});
                    continue;
                }
                if (!merge_viable && !both_below_symmetry_threshold) {
                    non_viable_edges_cache_.insert({u, v});
                    continue;
                }
                if (not merge_symmetry_narrowing) {
                    if (new_size < std::min(u_size, v_size)) {
                        continue;
                    }
                }

                auto [temp_coarse_graph, temp_contraction_map] = simulate_merge(u, v, current_coarse_graph);

                if (critical_path_weight(temp_coarse_graph) > (path_threshold * static_cast<v_workw_t<Constr_Graph_t>>(new_subgraphs.size()) + critical_path_weight(current_coarse_graph))) {
                    non_viable_crit_path_edges_cache_.insert({u, v});
                    continue;
                }

                commit_merge(u, v, std::move(temp_coarse_graph), temp_contraction_map, std::move(new_subgraphs), 
                             current_coarse_graph, current_groups, current_contraction_map);
                changed = true;
                break;
            }
        }
    }


    /**
     * @brief Core adaptive merging function.
     */
    void contract_edges_adpative_sym(const Graph_t &original_dag, 
        Constr_Graph_t& current_coarse_graph, 
        std::vector<Group>& current_groups, 
        std::vector<VertexType>& current_contraction_map, 
        const bool merge_different_node_types, 
        const bool merge_below_threshold,
        const std::vector<v_workw_t<Graph_t>>& lock_threshold_per_type,
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
                const std::size_t u_size = current_groups[u].size();
                const std::size_t v_size = current_groups[v].size();
                
                const bool merge_is_valid = is_merge_viable(original_dag, current_groups[u], current_groups[v], new_subgraphs);
                const std::size_t new_size = new_subgraphs.size();
                
                if (!merge_is_valid) {
                    if constexpr (verbose) {
                        std::cout << "  - Merge of " << u << " and " << v << " and " << v
                                  << " not viable (error in is_merge_viable)\n";
                    }
                    non_viable_edges_cache_.insert({u, v});
                    continue;
                }

                const bool merge_viable = (new_size >= current_symmetry);
                const bool both_below_minimal_threshold = merge_below_threshold && (u_size < min_symmetry_) && (v_size < min_symmetry_);
                
                if (!merge_viable && !both_below_minimal_threshold) {
                    if constexpr (verbose) {
                        std::cout << "  - Merge of " << u << " and " << v << " not viable (Symmetry Threshold)\n";
                        std::cout << "    - u_sym: " << u_size << ", v_sym: " << v_size << " -> new_sym: " << new_size
                                  << " (current_threshold: " << current_symmetry 
                                  << ", global_min_threshold: " << min_symmetry_ << ")\n";
                    }
                    non_viable_edges_cache_.insert({u, v});
                    continue;
                }

                v_type_t<Graph_t> u_type = 0;
                v_type_t<Graph_t> v_type = 0;
                if (not merge_different_node_types && has_typed_vertices_v<Graph_t> ) {
                    u_type = current_coarse_graph.vertex_type(u);
                    v_type = current_coarse_graph.vertex_type(v);
                }

                const bool u_is_significant = (u_size >= min_symmetry_) && 
                    (current_coarse_graph.vertex_work_weight(u) > lock_threshold_per_type[u_type]);
                const bool v_is_significant = (v_size >= min_symmetry_) && 
                    (current_coarse_graph.vertex_work_weight(v) > lock_threshold_per_type[v_type]);

                if (u_is_significant && v_is_significant)
                {
                    // Both are significant ---
                    if (new_size < std::min(u_size, v_size)) {
                        if constexpr (verbose) {
                            std::cout << "  - Merge of " << u << " and " << v << " not viable (Symmetry Narrowing below min of two significant nodes)\n";
                            std::cout << "    - u_sym: " << u_size << ", v_sym: " << v_size << " -> new_sym: " << new_size << "\n";
                        }
                        non_viable_edges_cache_.insert({u, v});
                        continue;
                    }
                }
                else if (u_is_significant || v_is_significant)
                {
                    // Exactly one is significant ---
                    const std::size_t significant_node_size = u_is_significant ? u_size : v_size;
                    
                    if (new_size < significant_node_size) {
                        if constexpr (verbose) {
                            std::cout << "  - Merge of " << u << " and " << v << " not viable (Symmetry Narrowing of a single significant node)\n";
                            std::cout << "    - u_sym: " << u_size << " (sig: " << u_is_significant << ")"
                                      << ", v_sym: " << v_size << " (sig: " << v_is_significant << ")"
                                      << " -> new_sym: " << new_size << "\n";
                        }
                        non_viable_edges_cache_.insert({u, v});
                        continue;
                    }
                }
                
                // Critical Path Check
                auto [temp_coarse_graph, temp_contraction_map] = simulate_merge(u, v, current_coarse_graph);

                if (critical_path_weight(temp_coarse_graph) > (path_threshold * static_cast<v_workw_t<Constr_Graph_t>>(new_subgraphs.size()) + critical_path_weight(current_coarse_graph))) {
                    if constexpr (verbose) {
                        std::cout << "  - Merge of " << u << " and " << v << " increases critical path. Old cirtical path: " << critical_path_weight(current_coarse_graph)
                                  << " new critical path: " << critical_path_weight(temp_coarse_graph) << " + " << path_threshold * static_cast<v_workw_t<Constr_Graph_t>>(new_subgraphs.size()) << "\n";
                    }
                    non_viable_crit_path_edges_cache_.insert({u, v});
                    continue;
                }

                // Commit Merge
                if constexpr (verbose) {
                    std::cout << "  - Merging " << v << " into " << u << ". New coarse graph has "
                              << temp_coarse_graph.num_vertices() << " nodes.\n";
                }

                commit_merge(u, v, std::move(temp_coarse_graph), temp_contraction_map, std::move(new_subgraphs), 
                             current_coarse_graph, current_groups, current_contraction_map);

                changed = true;
                break;
            }
        }
    }


  public:

    explicit OrbitGraphProcessor(size_t symmetry_threshold = 2) : symmetry_threshold_(symmetry_threshold) {}

    void set_symmetry_threshold(size_t threshold) { symmetry_threshold_ = threshold; }
    void setMergeDifferentNodeTypes(bool flag) { merge_different_node_types_ = flag; }
    void set_work_threshold(v_workw_t<Constr_Graph_t> work_threshold) { work_threshold_ = work_threshold; }
    void setCriticalPathThreshold(v_workw_t<Constr_Graph_t> critical_path_threshold) { critical_path_threshold_ = critical_path_threshold; }
    void setLockRatio(double lock_ratio) { lock_orbit_ratio = lock_ratio; }
    void setMinSymmetry(size_t min_symmetry) { min_symmetry_ = min_symmetry; }
    void setSymmetryLevelHeuristic(SymmetryLevelHeuristic heuristic) { symmetry_level_heuristic_ = heuristic; }
    void setWorkPercentiles(const std::vector<double>& percentiles) {
        work_percentiles_ = percentiles;
        std::sort(work_percentiles_.begin(), work_percentiles_.end());
    }

    void setNaturalBreaksCountPercentage(double percentage) { natural_breaks_count_percentage_ = percentage; }


    /**
     * @brief Discovers isomorphic groups (orbits) and constructs a coarse graph.
     */
    void discover_isomorphic_groups(const Graph_t &dag, const HashComputer<VertexType> &hasher) {
        coarse_graph_ = Constr_Graph_t();
        contraction_map_.clear();
        final_coarse_graph_ = Constr_Graph_t();
        final_contraction_map_.clear();
        final_groups_.clear();
        non_viable_edges_cache_.clear();
        non_viable_crit_path_edges_cache_.clear();

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
    
        std::vector<v_workw_t<Graph_t>> work_per_vertex_type;
        work_per_vertex_type.resize(merge_different_node_types_ ? 1U : dag.num_vertex_types(), 0);
        
        std::map<size_t, size_t> orbit_size_counts;
        std::map<size_t, v_workw_t<Graph_t>> work_per_orbit_size;
        v_workw_t<Graph_t> total_work = 0;
        for (const auto &[hash, vertices] : orbits) {
            const size_t orbit_size = vertices.size();
            orbit_size_counts[orbit_size]++;

            v_workw_t<Graph_t> orbit_work = 0;
            for (const auto v : vertices) {
                orbit_work += dag.vertex_work_weight(v);
            }

            if (not merge_different_node_types_ && has_typed_vertices_v<Graph_t>) {
                work_per_vertex_type[dag.vertex_type(vertices[0])] += orbit_work;
            } else {
                work_per_vertex_type[0] += orbit_work;
            }

            work_per_orbit_size[orbit_size] += orbit_work;
            total_work += orbit_work;
        }

        std::vector<v_workw_t<Graph_t>> lock_threshold_per_type(work_per_vertex_type.size());
        for (size_t i = 0; i < work_per_vertex_type.size(); ++i) {
            lock_threshold_per_type[i] = static_cast<v_workw_t<Graph_t>>(lock_orbit_ratio * work_per_vertex_type[i]);
        }
        
        std::vector<double> rel_acc_work_per_orbit_size;
        std::vector<size_t> symmetry_levels_to_test = compute_symmetry_levels(rel_acc_work_per_orbit_size, work_per_orbit_size, total_work, orbit_size_counts);

        if constexpr (verbose) {
            std::cout << "\n--- Orbit Analysis ---\n";
            for (auto const& [size, count] : orbit_size_counts) {
                if (total_work > 0)
                    std::cout << "  - Orbits of size " << size << ": " << count << " groups, weight: " << 100.0 * static_cast<double>(work_per_orbit_size[size]) / static_cast<double>(total_work) << "%\n";            
                else
                    std::cout << "  - Orbits of size " << size << ": " << count << " groups, weight: 0.0%\n";
            }
            std::cout << "  Cumulative work distribution by orbit size (largest to smallest):\n";
            size_t i = 0;
            for (auto it = orbit_size_counts.rbegin(); it != orbit_size_counts.rend() && i < rel_acc_work_per_orbit_size.size(); ++it, ++i) {
                std::cout << "    - Orbits with size >= " << it->first << ": "
                          << std::fixed << std::setprecision(2) << rel_acc_work_per_orbit_size[i] * 100 << "%\n";
            }
            std::cout << "  Work distribution by vertex type:\n";
            for (size_t j = 0; j < work_per_vertex_type.size(); ++j) {
                if (total_work > 0)
                    std::cout << "    - Vertex type " << j << ": " << 100.0 * static_cast<double>(work_per_vertex_type[j]) / static_cast<double>(total_work) << "%\n";
                else
                     std::cout << "    - Vertex type " << j << ": 0.0%\n";
            }
            
            std::cout << "--------------------------------\n";
            std::cout << " Symmetry levels to test: " << "\n";
            for (const auto level : symmetry_levels_to_test) {
                std::cout << "  - " << level << "\n";
            }
            std::cout << "--------------------------------\n";             
        }       

        coarser_util::construct_coarse_dag(dag, coarse_graph_, contraction_map_);
        perform_coarsening_adaptive_symmetry(dag, coarse_graph_, lock_threshold_per_type, symmetry_levels_to_test);
    }

  private:

    std::vector<size_t> compute_symmetry_levels(std::vector<double> & rel_acc_work_per_orbit_size, const std::map<size_t, v_workw_t<Graph_t>> work_per_orbit_size, const v_workw_t<Graph_t> total_work, const std::map<size_t, size_t> orbit_size_counts) {

        std::vector<size_t> symmetry_levels_to_test;
        min_symmetry_ = 2;

        switch (symmetry_level_heuristic_) {
            case SymmetryLevelHeuristic::PERCENTILE_BASED:
            {
                if constexpr (verbose) { std::cout << "Using PERCENTILE_BASED heuristic for symmetry levels.\n"; }
                size_t percentile_idx = 0;
                v_workw_t<Graph_t> cumulative_work = 0;
                for (auto it = work_per_orbit_size.rbegin(); 
                     it != work_per_orbit_size.rend(); 
                     ++it) 
                {
                    cumulative_work += it->second;
                    if (total_work == 0) continue; // Avoid division by zero
                    double current_work_ratio = static_cast<double>(cumulative_work) / static_cast<double>(total_work);
                    rel_acc_work_per_orbit_size.push_back(current_work_ratio); // For printing

                    if (percentile_idx < work_percentiles_.size() && current_work_ratio >= work_percentiles_[percentile_idx]) {
                        if (it->first > min_symmetry_) {
                            symmetry_levels_to_test.push_back(it->first);
                        }
                        while (percentile_idx < work_percentiles_.size() &&
                               current_work_ratio >= work_percentiles_[percentile_idx]) {
                            percentile_idx++;
                        }
                    }
                }
                break;
            }

            case SymmetryLevelHeuristic::NATURAL_BREAKS:
            {
                if constexpr (verbose) { std::cout << "Using NATURAL_BREAKS heuristic for symmetry levels.\n"; }

                size_t total_orbit_groups = 0;
                for (const auto& [size, count] : orbit_size_counts) {
                    total_orbit_groups += count;
                }
                size_t count_threshold = static_cast<size_t>(static_cast<double>(total_orbit_groups) * natural_breaks_count_percentage_);
                if (count_threshold == 0 && total_orbit_groups > 0) {
                    count_threshold = 1; // Ensure threshold is at least 1 if possible
                }
                if constexpr (verbose) { std::cout << "  - Total orbit groups: " << total_orbit_groups << ", count threshold: " << count_threshold << "\n"; }

                std::vector<size_t> sorted_sizes;
                sorted_sizes.reserve(orbit_size_counts.size());
                for (const auto& [size, count] : orbit_size_counts) {
                    sorted_sizes.push_back(size);
                }
                std::sort(sorted_sizes.rbegin(), sorted_sizes.rend()); // Sort descending

                if (!sorted_sizes.empty()) {
                    for (size_t i = 0; i < sorted_sizes.size(); ++i) {
                        const size_t current_size = sorted_sizes[i];
                        if (current_size < min_symmetry_) continue;

                        // Add if this size's count is significant
                        const size_t current_count = orbit_size_counts.at(current_size);
                        bool count_significant = (current_count >= count_threshold);
                        
                        if (count_significant) {
                            symmetry_levels_to_test.push_back(current_size);
                            continue;
                        }
                    }
                }

                if (symmetry_levels_to_test.empty()) {
                    size_t max_count = 0;
                    size_t size_with_max_count = 0;
                    for (const auto& [size, count] : orbit_size_counts) {
                        if (count > max_count) {
                            max_count = count;
                            size_with_max_count = size;
                        }
                    }
                    if (size_with_max_count > 0) {
                        symmetry_levels_to_test.push_back(size_with_max_count);
                    }
                }

                // Verbose print data
                v_workw_t<Graph_t> cumulative_work = 0;
                for (auto it = work_per_orbit_size.rbegin(); it != work_per_orbit_size.rend(); ++it) {
                    cumulative_work += it->second;
                    if (total_work > 0)
                        rel_acc_work_per_orbit_size.push_back(static_cast<double>(cumulative_work) / static_cast<double>(total_work));
                }
                break;
            }

            case SymmetryLevelHeuristic::CURRENT_DEFAULT:
            default:
            {
                if constexpr (verbose) { std::cout << "Using CURRENT_DEFAULT heuristic for symmetry levels.\n"; }
                double threshold = lock_orbit_ratio;
                v_workw_t<Graph_t> cumulative_work = 0;
                for (auto it = work_per_orbit_size.rbegin(); it != work_per_orbit_size.rend(); ++it) {
                    cumulative_work += it->second;
                    const double rel_work = (total_work == 0) ? 0 : static_cast<double>(cumulative_work) / static_cast<double>(total_work);         
                    rel_acc_work_per_orbit_size.push_back(rel_work); // For printing
                    
                    if (rel_work >= threshold && it->first > min_symmetry_) {
                        symmetry_levels_to_test.push_back(it->first);
                        threshold += lock_orbit_ratio * 0.5;
                    }
                }
                break;
            }
        }
        
        if (symmetry_levels_to_test.empty()) 
            symmetry_levels_to_test.push_back(2);

        min_symmetry_ = symmetry_levels_to_test.back();
        
        // De-duplicate and sort descending
        std::sort(symmetry_levels_to_test.rbegin(), symmetry_levels_to_test.rend());
        auto last = std::unique(symmetry_levels_to_test.begin(), symmetry_levels_to_test.end());
        symmetry_levels_to_test.erase(last, symmetry_levels_to_test.end());

        return symmetry_levels_to_test;
    }


    /**
     * @brief Non-adaptive coarsening (deprecated).
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
        
        final_coarse_graph_ = std::move(current_coarse_graph);
        final_contraction_map_ = std::move(current_contraction_map);
        final_groups_ = std::move(current_groups);

        if constexpr (verbose) {
            print_final_groups_summary();
        }
    }

    void perform_coarsening_adaptive_symmetry(const Graph_t &original_dag, const Constr_Graph_t &initial_coarse_graph, const std::vector<v_workw_t<Graph_t>>& lock_threshold_per_type, const std::vector<size_t>& symmetry_levels_to_test) {
        final_coarse_graph_ = Constr_Graph_t();
        final_contraction_map_.clear();

        if (initial_coarse_graph.num_vertices() == 0) {
            return;
        }

        Constr_Graph_t current_coarse_graph = initial_coarse_graph;
        std::vector<Group> current_groups(initial_coarse_graph.num_vertices());
        std::vector<VertexType> current_contraction_map = contraction_map_;

        for (VertexType i = 0; i < original_dag.num_vertices(); ++i) {
            const VertexType coarse_node = contraction_map_[i];
            current_groups[coarse_node].subgraphs.push_back({i});
        }
    
        if constexpr (verbose) {
            std::cout << " Starting adaptive symmetry coarsening with critical_path_threshold: " << critical_path_threshold_ << "\n";
        }

        for (const auto sym : symmetry_levels_to_test) {
            current_symmetry = sym;
            const bool is_last_loop = (sym == symmetry_levels_to_test.back());
            if constexpr (verbose) {
                std::cout << "  Current symmetry threshold: " << current_symmetry << "\n";
            }

            non_viable_edges_cache_.clear();

            contract_edges_adpative_sym(original_dag, current_coarse_graph, current_groups, current_contraction_map, false, is_last_loop, lock_threshold_per_type);
            
            if (merge_different_node_types_)
                contract_edges_adpative_sym(original_dag, current_coarse_graph, current_groups, current_contraction_map, merge_different_node_types_, is_last_loop, lock_threshold_per_type);
            
            non_viable_crit_path_edges_cache_.clear();
            contract_edges_adpative_sym(original_dag, current_coarse_graph, current_groups, current_contraction_map, merge_different_node_types_, is_last_loop, lock_threshold_per_type, critical_path_threshold_);

        }
    
        if constexpr (verbose) {
            std::cout << " Merging small orbits with work threshold: " << work_threshold_ << "\n";
        }
        non_viable_edges_cache_.clear();
        merge_small_orbits(original_dag, current_coarse_graph, current_groups, current_contraction_map, work_threshold_);
        
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
     * @brief Checks if merging two groups is structurally viable.
     */
    bool is_merge_viable(const Graph_t &original_dag, const Group &group_u, const Group &group_v,
                         std::vector<std::vector<VertexType>> &out_new_subgraphs) const {

        std::vector<VertexType> all_nodes;
        all_nodes.reserve(group_u.subgraphs.size() * (group_u.subgraphs.empty() ? 0 : group_u.subgraphs[0].size()) + 
                          group_v.subgraphs.size() * (group_v.subgraphs.empty() ? 0 : group_v.subgraphs[0].size()));
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
        
        if (all_nodes.empty()) { // Handle empty graph case
             return true;
        }

        for (const auto &node : all_nodes) {
            out_new_subgraphs[components[map[node]]].push_back(node);
        }

        if (num_components > 1) {
            const size_t first_sg_size = out_new_subgraphs[0].size();
            Constr_Graph_t rep_sg;
            create_induced_subgraph(original_dag, rep_sg, out_new_subgraphs[0]);

            for (size_t i = 1; i < num_components; ++i) {
                if (out_new_subgraphs[i].size() != first_sg_size) {
                    return false;
                }

                Constr_Graph_t current_sg;
                create_induced_subgraph(original_dag, current_sg, out_new_subgraphs[i]);
                if (!are_isomorphic_by_merkle_hash(rep_sg, current_sg)) {
                    return false;
                }
            }
        }
        return true;
    }

  public:
    const Graph_t &get_coarse_graph() const { return coarse_graph_; }
    const std::vector<VertexType> &get_contraction_map() const { return contraction_map_; }
    const Graph_t &get_final_coarse_graph() const { return final_coarse_graph_; }
    const std::vector<VertexType> &get_final_contraction_map() const { return final_contraction_map_; }
    const std::vector<Group> &get_final_groups() const { return final_groups_; }
};

} // namespace osp