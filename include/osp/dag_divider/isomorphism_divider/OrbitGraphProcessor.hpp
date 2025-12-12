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

#include <algorithm>
#include <map>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "osp/coarser/coarser_util.hpp"
#include "osp/dag_divider/isomorphism_divider/HashComputer.hpp"
#include "osp/dag_divider/isomorphism_divider/MerkleHashComputer.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"
#include "osp/graph_algorithms/transitive_reduction.hpp"

namespace osp {

/**
 * @class OrbitGraphProcessor
 * @brief A simple processor that groups nodes of a DAG based on their Merkle hash.
 *
 * This class uses a MerkleHashComputer to assign a structural hash to each node.
 * It then partitions the DAG by grouping all nodes with the same hash into an "orbit".
 * A coarse graph is constructed where each node represents one such orbit.
 */
template <typename GraphT, typename ConstrGraphT>
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

    static_assert(IsComputationalDagV<Graph_t>, "Graph must be a computational DAG");
    static_assert(IsComputationalDagV<Constr_Graph_t>, "Constr_Graph_t must be a computational DAG");
    static_assert(IsConstructableCdagV<Constr_Graph_t>, "Constr_Graph_t must satisfy the constructable_cdag_vertex concept");
    static_assert(std::is_same_v<vertex_idx_t<Graph_t>, vertex_idx_t<Constr_Graph_t>>,
                  "Graph_t and Constr_Graph_t must have the same vertex_idx types");

    using VertexType = vertex_idx_t<Graph_t>;

    static constexpr bool verbose_ = false;

    // Represents a group of isomorphic subgraphs, corresponding to a single node in a coarse graph.
    struct Group {
        // Each vector of vertices represents one of the isomorphic subgraphs in this group.
        std::vector<std::vector<VertexType>> subgraphs_;

        inline size_t size() const { return subgraphs.size(); }
    };

  private:
    // Results from the first (orbit) coarsening step
    ConstrGraphT coarseGraph_;
    std::vector<VertexType> contractionMap_;

    // Results from the second (custom) coarsening step
    ConstrGraphT finalCoarseGraph_;
    std::vector<VertexType> finalContractionMap_;
    std::vector<Group> finalGroups_;
    size_t currentSymmetry_;

    size_t minSymmetry_ = 2;    // min symmetry threshold
    v_workw_t<Constr_Graph_t> workThreshold_ = 0;
    v_workw_t<Constr_Graph_t> criticalPathThreshold_ = 0;
    bool mergeDifferentNodeTypes_ = true;
    double lockOrbitRatio_ = 0.5;

    SymmetryLevelHeuristic symmetryLevelHeuristic_ = SymmetryLevelHeuristic::NATURAL_BREAKS;
    std::vector<double> workPercentiles_ = {0.50, 0.75};
    double naturalBreaksCountPercentage_ = 0.2;

    bool useAdaptiveSymmetryThreshold_ = true;

    struct PairHasher {
        template <class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2> &p) const {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second);
            return h1 ^ (h2 << 1);
        }
    };

    std::unordered_set<std::pair<VertexType, VertexType>, PairHasher> nonViableEdgesCache_;
    std::unordered_set<std::pair<VertexType, VertexType>, PairHasher> nonViableCritPathEdgesCache_;

    /**
     * @brief Simulates the merge of node v into u and returns the resulting temporary graph.
     */
    std::pair<Constr_Graph_t, std::vector<VertexType>> SimulateMerge(VertexType u,
                                                                     VertexType v,
                                                                     const ConstrGraphT &currentCoarseGraph) const {
        std::vector<VertexType> tempContractionMap(currentCoarseGraph.NumVertices());
        VertexType newIdx = 0;
        for (VertexType i = 0; i < static_cast<VertexType>(temp_contraction_map.size()); ++i) {
            if (i != v) {
                tempContractionMap[i] = new_idx++;
            }
        }
        tempContractionMap[v] = temp_contraction_map[u];

        ConstrGraphT tempCoarseGraph;
        coarser_util::construct_coarse_dag(current_coarse_graph, temp_coarse_graph, temp_contraction_map);

        return {std::move(tempCoarseGraph), std::move(temp_contraction_map)};
    }

    /**
     * @brief Commits a merge operation by updating the graph state.
     */
    void CommitMerge(VertexType u,
                     VertexType v,
                     ConstrGraphT &&nextCoarseGraph,
                     const std::vector<VertexType> &groupRemap,
                     std::vector<std::vector<VertexType>> &&newSubgraphs,
                     ConstrGraphT &currentCoarseGraph,
                     std::vector<Group> &currentGroups,
                     std::vector<VertexType> &currentContractionMap) {
        currentCoarseGraph = std::move(nextCoarseGraph);

        // Update caches for new vertex indices
        std::unordered_set<std::pair<VertexType, VertexType>, PairHasher> nextNonViableEdges;
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

        std::unordered_set<std::pair<VertexType, VertexType>, PairHasher> nextNonViableCritPathEdges;
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
        std::vector<Group> nextGroups(currentCoarseGraph.NumVertices());
        for (VertexType i = 0; i < static_cast<VertexType>(currentGroups.size()); ++i) {
            if (i != u && i != v) {
                nextGroups[group_remap[i]] = std::move(currentGroups[i]);
            }
        }
        nextGroups[group_remap[u]].subgraphs = std::move(new_subgraphs);
        currentGroups = std::move(nextGroups);

        // Update main contraction map
        for (VertexType &node_map : current_contraction_map) {
            node_map = group_remap[node_map];
        }
    }

    /**
     * @brief Merges small orbits based on work threshold (final cleanup pass).
     */
    void MergeSmallOrbits(const GraphT &originalDag,
                          ConstrGraphT &currentCoarseGraph,
                          std::vector<Group> &currentGroups,
                          std::vector<VertexType> &currentContractionMap,
                          const v_workw_t<Constr_Graph_t> workThreshold,
                          const v_workw_t<Constr_Graph_t> pathThreshold = 0) {
        bool changed = true;
        while (changed) {
            const std::vector<vertex_idx_t<Constr_Graph_t>> vertexPoset
                = get_top_node_distance<Constr_Graph_t, vertex_idx_t<Constr_Graph_t>>(current_coarse_graph);
            const std::vector<vertex_idx_t<Constr_Graph_t>> vertexBotPoset
                = get_bottom_node_distance<Constr_Graph_t, vertex_idx_t<Constr_Graph_t>>(current_coarse_graph);

            changed = false;
            for (const auto u : currentCoarseGraph.vertices()) {
                for (const auto v : currentCoarseGraph.children(u)) {
                    if constexpr (HasTypedVerticesV<Constr_Graph_t>) {
                        if (not mergeDifferentNodeTypes_) {
                            if (currentCoarseGraph.vertex_type(u) != currentCoarseGraph.vertex_type(v)) {
                                if constexpr (verbose_) {
                                    std::cout << "  - Merge of " << u << " and " << v << " not viable (different node types)\n";
                                }
                                continue;
                            }
                        }
                    }

                    if (non_viable_edges_cache_.count({u, v}) || non_viable_crit_path_edges_cache_.count({u, v})) {
                        if constexpr (verbose_) {
                            std::cout << "  - Merge of " << u << " and " << v << " already checked. Skipping.\n";
                        }
                        continue;
                    }

                    const v_workw_t<Constr_Graph_t> uWorkWeight = currentCoarseGraph.vertex_work_weight(u);
                    const v_workw_t<Constr_Graph_t> vWorkWeight = currentCoarseGraph.vertex_work_weight(v);
                    const v_workw_t<Constr_Graph_t> vThreshold
                        = work_threshold * static_cast<v_workw_t<Constr_Graph_t>>(currentGroups[v].size());
                    const v_workw_t<Constr_Graph_t> uThreshold
                        = work_threshold * static_cast<v_workw_t<Constr_Graph_t>>(currentGroups[u].size());

                    if (uWorkWeight > u_threshold && v_work_weight > v_threshold) {
                        if constexpr (verbose_) {
                            std::cout << "  - Merge of " << u << " and " << v << " not viable (work threshold)\n";
                        }
                        continue;
                    }

                    if ((vertexPoset[u] + 1 != vertexPoset[v]) && (vertexBotPoset[u] != 1 + vertexBotPoset[v])) {
                        if constexpr (verbose_) {
                            std::cout << "  - Merge of " << u << " and " << v
                                      << " not viable poset. poste v: " << vertexBotPoset[v] << " poste u: " << vertexBotPoset[u]
                                      << "\n";
                        }
                        continue;
                    }

                    std::vector<std::vector<VertexType>> newSubgraphs;
                    const bool mergeIsValid = is_merge_viable(original_dag, current_groups[u], current_groups[v], new_subgraphs);

                    if (!mergeIsValid) {
                        if constexpr (verbose_) {
                            std::cout << "  - Merge of " << u << " and " << v << " and " << v
                                      << " not viable (error in is_merge_viable)\n";
                        }
                        non_viable_edges_cache_.insert({u, v});
                        continue;
                    }

                    auto [temp_coarse_graph, temp_contraction_map] = simulate_merge(u, v, current_coarse_graph);

                    if (critical_path_weight(temp_coarse_graph)
                        > (pathThreshold * static_cast<v_workw_t<Constr_Graph_t>>(new_subgraphs.size())
                           + critical_path_weight(currentCoarseGraph))) {
                        if constexpr (verbose_) {
                            std::cout << "  - Merge of " << u << " and " << v << " increases critical path. Old cirtical path: "
                                      << critical_path_weight(currentCoarseGraph)
                                      << " new critical path: " << critical_path_weight(temp_coarse_graph) << " + "
                                      << path_threshold * static_cast<v_workw_t<Constr_Graph_t>>(new_subgraphs.size()) << "\n";
                        }
                        non_viable_crit_path_edges_cache_.insert({u, v});
                        continue;
                    }

                    if constexpr (verbose_) {
                        std::cout << "  - Merging " << v << " into " << u << ". New coarse graph has "
                                  << temp_coarse_graph.NumVertices() << " nodes.\n";
                    }

                    commit_merge(u,
                                 v,
                                 std::move(temp_coarse_graph),
                                 temp_contraction_map,
                                 std::move(new_subgraphs),
                                 current_coarse_graph,
                                 current_groups,
                                 current_contraction_map);

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
    void ContractEdges(const GraphT &originalDag,
                       ConstrGraphT &currentCoarseGraph,
                       std::vector<Group> &currentGroups,
                       std::vector<VertexType> &currentContractionMap,
                       const bool mergeSymmetryNarrowing,
                       const bool mergeDifferentNodeTypes,
                       const v_workw_t<Constr_Graph_t> pathThreshold = 0) {
        bool changed = true;
        while (changed) {
            const std::vector<vertex_idx_t<Constr_Graph_t>> vertexPoset
                = get_top_node_distance<Constr_Graph_t, vertex_idx_t<Constr_Graph_t>>(current_coarse_graph);
            const std::vector<vertex_idx_t<Constr_Graph_t>> vertexBotPoset
                = get_bottom_node_distance<Constr_Graph_t, vertex_idx_t<Constr_Graph_t>>(current_coarse_graph);

            changed = false;
            for (const auto &edge : edges(currentCoarseGraph)) {
                VertexType u = source(edge, currentCoarseGraph);
                VertexType v = target(edge, currentCoarseGraph);

                if (non_viable_edges_cache_.count({u, v}) || non_viable_crit_path_edges_cache_.count({u, v})) {
                    continue;
                }
                if constexpr (HasTypedVerticesV<Constr_Graph_t>) {
                    if (not mergeDifferentNodeTypes) {
                        if (currentCoarseGraph.vertex_type(u) != currentCoarseGraph.vertex_type(v)) {
                            continue;
                        }
                    }
                }
                if ((vertexPoset[u] + 1 != vertexPoset[v]) && (vertexBotPoset[u] != 1 + vertexBotPoset[v])) {
                    continue;
                }

                std::vector<std::vector<VertexType>> newSubgraphs;
                const std::size_t uSize = currentGroups[u].size();
                const std::size_t vSize = currentGroups[v].size();
                const bool mergeIsValid = is_merge_viable(original_dag, current_groups[u], current_groups[v], new_subgraphs);
                const std::size_t newSize = new_subgraphs.size();

                const bool mergeViable = (newSize >= currentSymmetry_);
                const bool bothBelowSymmetryThreshold = (uSize < currentSymmetry_) && (vSize < currentSymmetry_);

                if (!mergeIsValid) {
                    non_viable_edges_cache_.insert({u, v});
                    continue;
                }
                if (!mergeViable && !bothBelowSymmetryThreshold) {
                    non_viable_edges_cache_.insert({u, v});
                    continue;
                }
                if (not mergeSymmetryNarrowing) {
                    if (newSize < std::min(uSize, vSize)) {
                        continue;
                    }
                }

                auto [temp_coarse_graph, temp_contraction_map] = simulate_merge(u, v, current_coarse_graph);

                if (critical_path_weight(temp_coarse_graph)
                    > (pathThreshold * static_cast<v_workw_t<Constr_Graph_t>>(new_subgraphs.size())
                       + critical_path_weight(currentCoarseGraph))) {
                    non_viable_crit_path_edges_cache_.insert({u, v});
                    continue;
                }

                commit_merge(u,
                             v,
                             std::move(temp_coarse_graph),
                             temp_contraction_map,
                             std::move(new_subgraphs),
                             current_coarse_graph,
                             current_groups,
                             current_contraction_map);
                changed = true;
                break;
            }
        }
    }

    /**
     * @brief Core adaptive merging function.
     */
    void ContractEdgesAdpativeSym(const GraphT &originalDag,
                                  ConstrGraphT &currentCoarseGraph,
                                  std::vector<Group> &currentGroups,
                                  std::vector<VertexType> &currentContractionMap,
                                  const bool mergeDifferentNodeTypes,
                                  const bool mergeBelowThreshold,
                                  const std::vector<v_workw_t<Graph_t>> &lockThresholdPerType,
                                  const v_workw_t<Constr_Graph_t> pathThreshold = 0) {
        bool changed = true;
        while (changed) {
            const std::vector<vertex_idx_t<Constr_Graph_t>> vertexPoset
                = get_top_node_distance<Constr_Graph_t, vertex_idx_t<Constr_Graph_t>>(current_coarse_graph);
            const std::vector<vertex_idx_t<Constr_Graph_t>> vertexBotPoset
                = get_bottom_node_distance<Constr_Graph_t, vertex_idx_t<Constr_Graph_t>>(current_coarse_graph);

            changed = false;
            for (const auto &edge : edges(currentCoarseGraph)) {
                VertexType u = source(edge, currentCoarseGraph);
                VertexType v = target(edge, currentCoarseGraph);

                if (non_viable_edges_cache_.count({u, v}) || non_viable_crit_path_edges_cache_.count({u, v})) {
                    if constexpr (verbose_) {
                        std::cout << "  - Merge of " << u << " and " << v << " already checked. Skipping.\n";
                    }
                    continue;
                }

                if constexpr (HasTypedVerticesV<Constr_Graph_t>) {
                    if (not mergeDifferentNodeTypes) {
                        if (currentCoarseGraph.vertex_type(u) != currentCoarseGraph.vertex_type(v)) {
                            if constexpr (verbose_) {
                                std::cout << "  - Merge of " << u << " and " << v << " not viable (different node types)\n";
                            }
                            continue;
                        }
                    }
                }

                if ((vertexPoset[u] + 1 != vertexPoset[v]) && (vertexBotPoset[u] != 1 + vertexBotPoset[v])) {
                    if constexpr (verbose_) {
                        std::cout << "  - Merge of " << u << " and " << v << " not viable poset. poste v: " << vertexBotPoset[v]
                                  << " poste u: " << vertexBotPoset[u] << "\n";
                    }
                    continue;
                }

                std::vector<std::vector<VertexType>> newSubgraphs;
                const std::size_t uSize = currentGroups[u].size();
                const std::size_t vSize = currentGroups[v].size();

                const bool mergeIsValid = is_merge_viable(original_dag, current_groups[u], current_groups[v], new_subgraphs);
                const std::size_t newSize = new_subgraphs.size();

                if (!mergeIsValid) {
                    if constexpr (verbose_) {
                        std::cout << "  - Merge of " << u << " and " << v << " and " << v
                                  << " not viable (error in is_merge_viable)\n";
                    }
                    non_viable_edges_cache_.insert({u, v});
                    continue;
                }

                const bool mergeViable = (newSize >= currentSymmetry_);
                const bool bothBelowMinimalThreshold = mergeBelowThreshold && (uSize < minSymmetry_) && (vSize < minSymmetry_);

                if (!mergeViable && !bothBelowMinimalThreshold) {
                    if constexpr (verbose_) {
                        std::cout << "  - Merge of " << u << " and " << v << " not viable (Symmetry Threshold)\n";
                        std::cout << "    - u_sym: " << uSize << ", v_sym: " << vSize << " -> new_sym: " << newSize
                                  << " (current_threshold: " << currentSymmetry_ << ", global_min_threshold: " << minSymmetry_
                                  << ")\n";
                    }
                    non_viable_edges_cache_.insert({u, v});
                    continue;
                }

                v_type_t<Graph_t> uType = 0;
                v_type_t<Graph_t> vType = 0;
                if (not merge_different_node_types && HasTypedVerticesV<Graph_t>) {
                    uType = currentCoarseGraph.vertex_type(u);
                    vType = currentCoarseGraph.vertex_type(v);
                }

                const bool uIsSignificant = (uSize >= minSymmetry_)
                                            && (currentCoarseGraph.vertex_work_weight(u) > lock_threshold_per_type[u_type]);
                const bool vIsSignificant = (vSize >= minSymmetry_)
                                            && (currentCoarseGraph.vertex_work_weight(v) > lock_threshold_per_type[v_type]);

                if (uIsSignificant && vIsSignificant) {
                    // Both are significant ---
                    if (newSize < std::min(uSize, vSize)) {
                        if constexpr (verbose_) {
                            std::cout << "  - Merge of " << u << " and " << v
                                      << " not viable (Symmetry Narrowing below min of two significant nodes)\n";
                            std::cout << "    - u_sym: " << uSize << ", v_sym: " << vSize << " -> new_sym: " << newSize << "\n";
                        }
                        non_viable_edges_cache_.insert({u, v});
                        continue;
                    }
                } else if (uIsSignificant || vIsSignificant) {
                    // Exactly one is significant ---
                    const std::size_t significantNodeSize = uIsSignificant ? uSize : vSize;

                    if (newSize < significantNodeSize) {
                        if constexpr (verbose_) {
                            std::cout << "  - Merge of " << u << " and " << v
                                      << " not viable (Symmetry Narrowing of a single significant node)\n";
                            std::cout << "    - u_sym: " << uSize << " (sig: " << uIsSignificant << ")"
                                      << ", v_sym: " << vSize << " (sig: " << vIsSignificant << ")"
                                      << " -> new_sym: " << newSize << "\n";
                        }
                        non_viable_edges_cache_.insert({u, v});
                        continue;
                    }
                }

                // Critical Path Check
                auto [temp_coarse_graph, temp_contraction_map] = simulate_merge(u, v, current_coarse_graph);

                if (critical_path_weight(temp_coarse_graph)
                    > (pathThreshold * static_cast<v_workw_t<Constr_Graph_t>>(new_subgraphs.size())
                       + critical_path_weight(currentCoarseGraph))) {
                    if constexpr (verbose_) {
                        std::cout << "  - Merge of " << u << " and " << v
                                  << " increases critical path. Old cirtical path: " << critical_path_weight(currentCoarseGraph)
                                  << " new critical path: " << critical_path_weight(temp_coarse_graph) << " + "
                                  << path_threshold * static_cast<v_workw_t<Constr_Graph_t>>(new_subgraphs.size()) << "\n";
                    }
                    non_viable_crit_path_edges_cache_.insert({u, v});
                    continue;
                }

                // Commit Merge
                if constexpr (verbose_) {
                    std::cout << "  - Merging " << v << " into " << u << ". New coarse graph has "
                              << temp_coarse_graph.NumVertices() << " nodes.\n";
                }

                commit_merge(u,
                             v,
                             std::move(temp_coarse_graph),
                             temp_contraction_map,
                             std::move(new_subgraphs),
                             current_coarse_graph,
                             current_groups,
                             current_contraction_map);

                changed = true;
                break;
            }
        }
    }

  public:
    explicit OrbitGraphProcessor() {}

    void SetMergeDifferentNodeTypes(bool flag) { mergeDifferentNodeTypes_ = flag; }

    void SetWorkThreshold(v_workw_t<Constr_Graph_t> workThreshold) { work_threshold_ = work_threshold; }

    void SetCriticalPathThreshold(v_workw_t<Constr_Graph_t> criticalPathThreshold) {
        critical_path_threshold_ = critical_path_threshold;
    }

    void SetLockRatio(double lockRatio) { lockOrbitRatio_ = lockRatio; }

    void SetSymmetryLevelHeuristic(SymmetryLevelHeuristic heuristic) { symmetryLevelHeuristic_ = heuristic; }

    void SetWorkPercentiles(const std::vector<double> &percentiles) {
        workPercentiles_ = percentiles;
        std::sort(workPercentiles_.begin(), workPercentiles_.end());
    }

    void SetUseStaticSymmetryLevel(size_t staticSymmetryLevel) {
        symmetryLevelHeuristic_ = SymmetryLevelHeuristic::NATURAL_BREAKS;
        useAdaptiveSymmetryThreshold_ = false;
        currentSymmetry_ = staticSymmetryLevel;
    }

    void SetNaturalBreaksCountPercentage(double percentage) { naturalBreaksCountPercentage_ = percentage; }

    /**
     * @brief Discovers isomorphic groups (orbits) and constructs a coarse graph.
     */
    void DiscoverIsomorphicGroups(const GraphT &dag, const HashComputer<VertexType> &hasher) {
        coarseGraph_ = ConstrGraphT();
        contraction_map_.clear();
        finalCoarseGraph_ = ConstrGraphT();
        final_contraction_map_.clear();
        finalGroups_.clear();
        non_viable_edges_cache_.clear();
        non_viable_crit_path_edges_cache_.clear();

        if (dag.NumVertices() == 0) {
            return;
        }

        const auto &orbits = hasher.get_orbits();

        contraction_map_.assign(dag.NumVertices(), 0);
        VertexType coarseNodeIdx = 0;

        for (const auto &hash_vertices_pair : orbits) {
            const auto &vertices = hash_vertices_pair.second;
            for (const auto v : vertices) {
                contraction_map_[v] = coarse_node_idx;
            }
            coarse_node_idx++;
        }

        std::vector<v_workw_t<Graph_t>> workPerVertexType;
        workPerVertexType.resize(mergeDifferentNodeTypes_ ? 1U : dag.num_vertex_types(), 0);

        std::map<size_t, size_t> orbitSizeCounts;
        std::map<size_t, v_workw_t<Graph_t>> workPerOrbitSize;
        v_workw_t<Graph_t> totalWork = 0;
        for (const auto &[hash, vertices] : orbits) {
            const size_t orbit_size = vertices.size();

            if (orbit_size == 1U) {
                continue;    // exclude single node orbits from total work
            }

            orbit_size_counts[orbit_size]++;

            v_workw_t<Graph_t> orbit_work = 0;
            for (const auto v : vertices) {
                orbit_work += dag.vertex_work_weight(v);
            }

            if (not merge_different_node_types_ && HasTypedVerticesV<Graph_t>) {
                work_per_vertex_type[dag.vertex_type(vertices[0])] += orbit_work;
            } else {
                work_per_vertex_type[0] += orbit_work;
            }

            work_per_orbit_size[orbit_size] += orbit_work;
            total_work += orbit_work;
        }

        std::vector<v_workw_t<Graph_t>> lockThresholdPerType(workPerVertexType.size());
        for (size_t i = 0; i < workPerVertexType.size(); ++i) {
            lockThresholdPerType[i] = static_cast<v_workw_t<Graph_t>>(lockOrbitRatio_ * work_per_vertex_type[i]);
        }

        std::vector<double> relAccWorkPerOrbitSize;
        std::vector<size_t> symmetryLevelsToTest
            = compute_symmetry_levels(rel_acc_work_per_orbit_size, work_per_orbit_size, total_work, orbit_size_counts);

        if constexpr (verbose_) {
            std::cout << "\n--- Orbit Analysis ---\n";
            for (auto const &[size, count] : orbitSizeCounts) {
                if (totalWork > 0) {
                    std::cout << "  - Orbits of size " << size << ": " << count << " groups, weight: "
                              << 100.0 * static_cast<double>(work_per_orbit_size[size]) / static_cast<double>(total_work) << "%\n";
                } else {
                    std::cout << "  - Orbits of size " << size << ": " << count << " groups, weight: 0.0%\n";
                }
            }
            std::cout << "  Cumulative work distribution by orbit size (largest to smallest):\n";
            size_t i = 0;
            for (auto it = orbitSizeCounts.rbegin(); it != orbitSizeCounts.rend() && i < relAccWorkPerOrbitSize.size(); ++it, ++i) {
                std::cout << "    - Orbits with size >= " << it->first << ": " << std::fixed << std::setprecision(2)
                          << relAccWorkPerOrbitSize[i] * 100 << "%\n";
            }
            std::cout << "  Work distribution by vertex type:\n";
            for (size_t j = 0; j < workPerVertexType.size(); ++j) {
                if (totalWork > 0) {
                    std::cout << "    - Vertex type " << j << ": "
                              << 100.0 * static_cast<double>(work_per_vertex_type[j]) / static_cast<double>(total_work) << "%\n";
                } else {
                    std::cout << "    - Vertex type " << j << ": 0.0%\n";
                }
            }

            std::cout << "--------------------------------\n";
            std::cout << " Symmetry levels to test: " << "\n";
            for (const auto level : symmetryLevelsToTest) {
                std::cout << "  - " << level << "\n";
            }
            std::cout << "--------------------------------\n";
        }

        coarser_util::construct_coarse_dag(dag, coarse_graph_, contraction_map_);

        if (useAdaptiveSymmetryThreshold_) {
            perform_coarsening_adaptive_symmetry(dag, coarse_graph_, lock_threshold_per_type, symmetry_levels_to_test);
        } else {
            size_t totalSizeCount = 0U;
            for (const auto &[size, count] : orbitSizeCounts) {
                totalSizeCount += count;
            }

            for (const auto &[size, count] : orbitSizeCounts) {
                if (size == 1U || size > currentSymmetry_) {
                    continue;
                }

                if (count > totalSizeCount / 2) {
                    if constexpr (verbose_) {
                        std::cout << "Setting current_symmetry to " << size << " because " << count << " orbits of size " << size
                                  << " are more than half of the total number of orbits.\n";
                    }
                    currentSymmetry_ = size;
                }
            }

            PerformCoarsening(dag, coarseGraph_);
        }
    }

  private:
    std::vector<size_t> ComputeSymmetryLevels(std::vector<double> &relAccWorkPerOrbitSize,
                                              const std::map<size_t, v_workw_t<Graph_t>> workPerOrbitSize,
                                              const v_workw_t<Graph_t> totalWork,
                                              const std::map<size_t, size_t> orbitSizeCounts) {
        std::vector<size_t> symmetryLevelsToTest;
        minSymmetry_ = 2;

        switch (symmetryLevelHeuristic_) {
            case SymmetryLevelHeuristic::PERCENTILE_BASED: {
                if constexpr (verbose_) {
                    std::cout << "Using PERCENTILE_BASED heuristic for symmetry levels.\n";
                }
                size_t percentileIdx = 0;
                v_workw_t<Graph_t> cumulativeWork = 0;
                for (auto it = work_per_orbit_size.rbegin(); it != work_per_orbit_size.rend(); ++it) {
                    cumulativeWork += it->second;
                    if (totalWork == 0) {
                        continue;    // Avoid division by zero
                    }
                    double currentWorkRatio = static_cast<double>(cumulative_work) / static_cast<double>(total_work);
                    relAccWorkPerOrbitSize.push_back(currentWorkRatio);    // For printing

                    if (percentileIdx < workPercentiles_.size() && currentWorkRatio >= workPercentiles_[percentileIdx]) {
                        if (it->first > minSymmetry_) {
                            symmetryLevelsToTest.push_back(it->first);
                        }
                        while (percentileIdx < workPercentiles_.size() && currentWorkRatio >= workPercentiles_[percentileIdx]) {
                            percentileIdx++;
                        }
                    }
                }
                break;
            }

            case SymmetryLevelHeuristic::NATURAL_BREAKS: {
                if constexpr (verbose_) {
                    std::cout << "Using NATURAL_BREAKS heuristic for symmetry levels.\n";
                }

                size_t totalOrbitGroups = 0;
                for (const auto &[size, count] : orbitSizeCounts) {
                    totalOrbitGroups += count;
                }
                size_t countThreshold = static_cast<size_t>(static_cast<double>(totalOrbitGroups) * naturalBreaksCountPercentage_);
                if (countThreshold == 0 && totalOrbitGroups > 0) {
                    countThreshold = 1;    // Ensure threshold is at least 1 if possible
                }
                if constexpr (verbose_) {
                    std::cout << "  - Total orbit groups: " << totalOrbitGroups << ", count threshold: " << countThreshold << "\n";
                }

                std::vector<size_t> sortedSizes;
                sortedSizes.reserve(orbitSizeCounts.size());
                for (const auto &[size, count] : orbitSizeCounts) {
                    sortedSizes.push_back(size);
                }
                std::sort(sortedSizes.rbegin(), sortedSizes.rend());    // Sort descending

                if (!sortedSizes.empty()) {
                    for (size_t i = 0; i < sortedSizes.size(); ++i) {
                        const size_t currentSize = sortedSizes[i];
                        if (currentSize < minSymmetry_) {
                            continue;
                        }

                        // Add if this size's count is significant
                        const size_t currentCount = orbitSizeCounts.at(currentSize);
                        bool countSignificant = (currentCount >= countThreshold);

                        if (countSignificant) {
                            symmetryLevelsToTest.push_back(currentSize);
                            continue;
                        }
                    }
                }

                if (symmetryLevelsToTest.empty()) {
                    size_t maxCount = 0;
                    size_t sizeWithMaxCount = 0;
                    for (const auto &[size, count] : orbitSizeCounts) {
                        if (count > maxCount) {
                            maxCount = count;
                            sizeWithMaxCount = size;
                        }
                    }
                    if (sizeWithMaxCount > 0) {
                        symmetryLevelsToTest.push_back(sizeWithMaxCount);
                    }
                }

                // Verbose print data
                v_workw_t<Graph_t> cumulativeWork = 0;
                for (auto it = work_per_orbit_size.rbegin(); it != work_per_orbit_size.rend(); ++it) {
                    cumulativeWork += it->second;
                    if (totalWork > 0) {
                        relAccWorkPerOrbitSize.push_back(static_cast<double>(cumulative_work) / static_cast<double>(total_work));
                    }
                }
                break;
            }

            case SymmetryLevelHeuristic::CURRENT_DEFAULT:
            default: {
                if constexpr (verbose_) {
                    std::cout << "Using CURRENT_DEFAULT heuristic for symmetry levels.\n";
                }
                double threshold = lockOrbitRatio_;
                v_workw_t<Graph_t> cumulativeWork = 0;
                for (auto it = work_per_orbit_size.rbegin(); it != work_per_orbit_size.rend(); ++it) {
                    cumulativeWork += it->second;
                    const double relWork
                        = (totalWork == 0) ? 0 : static_cast<double>(cumulative_work) / static_cast<double>(total_work);
                    relAccWorkPerOrbitSize.push_back(relWork);    // For printing

                    if (relWork >= threshold && it->first > minSymmetry_) {
                        symmetryLevelsToTest.push_back(it->first);
                        threshold += lockOrbitRatio_ * 0.5;
                    }
                }
                break;
            }
        }

        if (symmetryLevelsToTest.empty()) {
            symmetryLevelsToTest.push_back(2);
        }

        minSymmetry_ = symmetryLevelsToTest.back();

        // De-duplicate and sort descending
        std::sort(symmetryLevelsToTest.rbegin(), symmetryLevelsToTest.rend());
        auto last = std::unique(symmetryLevelsToTest.begin(), symmetryLevelsToTest.end());
        symmetryLevelsToTest.erase(last, symmetryLevelsToTest.end());

        return symmetryLevelsToTest;
    }

    /**
     * @brief Non-adaptive coarsening (deprecated).
     */
    void PerformCoarsening(const GraphT &originalDag, const ConstrGraphT &initialCoarseGraph) {
        finalCoarseGraph_ = ConstrGraphT();
        final_contraction_map_.clear();

        if (initialCoarseGraph.NumVertices() == 0) {
            return;
        }

        ConstrGraphT currentCoarseGraph = initialCoarseGraph;
        std::vector<Group> currentGroups(initialCoarseGraph.NumVertices());
        std::vector<VertexType> currentContractionMap = contraction_map_;

        // Initialize groups: each group corresponds to an orbit.
        for (VertexType i = 0; i < originalDag.NumVertices(); ++i) {
            const VertexType coarseNode = contraction_map_[i];
            currentGroups[coarse_node].subgraphs.push_back({i});
        }

        if constexpr (HasTypedVerticesV<Constr_Graph_t>) {
            if constexpr (verbose_) {
                std::cout << "Attempting to merge same node types.\n";
            }
            contract_edges(original_dag, current_coarse_graph, current_groups, current_contraction_map, false, false);
            contract_edges(original_dag, current_coarse_graph, current_groups, current_contraction_map, true, false);
        }

        if constexpr (verbose_) {
            std::cout << "Attempting to merge different node types.\n";
        }
        contract_edges(
            original_dag, current_coarse_graph, current_groups, current_contraction_map, false, merge_different_node_types_);
        contract_edges(
            original_dag, current_coarse_graph, current_groups, current_contraction_map, true, merge_different_node_types_);

        if constexpr (verbose_) {
            std::cout << "Attempting to merge small orbits.\n";
        }
        merge_small_orbits(original_dag, current_coarse_graph, current_groups, current_contraction_map, work_threshold_);

        non_viable_crit_path_edges_cache_.clear();
        non_viable_edges_cache_.clear();

        contract_edges(original_dag,
                       current_coarse_graph,
                       current_groups,
                       current_contraction_map,
                       true,
                       merge_different_node_types_,
                       work_threshold_);

        finalCoarseGraph_ = std::move(currentCoarseGraph);
        final_contraction_map_ = std::move(current_contraction_map);
        finalGroups_ = std::move(currentGroups);

        if constexpr (verbose_) {
            PrintFinalGroupsSummary();
        }
    }

    void PerformCoarseningAdaptiveSymmetry(const GraphT &originalDag,
                                           const ConstrGraphT &initialCoarseGraph,
                                           const std::vector<v_workw_t<Graph_t>> &lockThresholdPerType,
                                           const std::vector<size_t> &symmetryLevelsToTest) {
        finalCoarseGraph_ = ConstrGraphT();
        final_contraction_map_.clear();

        if (initialCoarseGraph.NumVertices() == 0) {
            return;
        }

        ConstrGraphT currentCoarseGraph = initialCoarseGraph;
        std::vector<Group> currentGroups(initialCoarseGraph.NumVertices());
        std::vector<VertexType> currentContractionMap = contraction_map_;

        for (VertexType i = 0; i < originalDag.NumVertices(); ++i) {
            const VertexType coarseNode = contraction_map_[i];
            currentGroups[coarse_node].subgraphs.push_back({i});
        }

        if constexpr (verbose_) {
            std::cout << " Starting adaptive symmetry coarsening with critical_path_threshold: " << critical_path_threshold_
                      << "\n";
        }

        for (const auto sym : symmetryLevelsToTest) {
            currentSymmetry_ = sym;
            const bool isLastLoop = (sym == symmetryLevelsToTest.back());
            if constexpr (verbose_) {
                std::cout << "  Current symmetry threshold: " << currentSymmetry_ << "\n";
            }

            non_viable_edges_cache_.clear();

            contract_edges_adpative_sym(original_dag,
                                        current_coarse_graph,
                                        current_groups,
                                        current_contraction_map,
                                        false,
                                        is_last_loop,
                                        lock_threshold_per_type);

            if (mergeDifferentNodeTypes_) {
                contract_edges_adpative_sym(original_dag,
                                            current_coarse_graph,
                                            current_groups,
                                            current_contraction_map,
                                            merge_different_node_types_,
                                            is_last_loop,
                                            lock_threshold_per_type);
            }

            non_viable_crit_path_edges_cache_.clear();
            contract_edges_adpative_sym(original_dag,
                                        current_coarse_graph,
                                        current_groups,
                                        current_contraction_map,
                                        merge_different_node_types_,
                                        is_last_loop,
                                        lock_threshold_per_type,
                                        critical_path_threshold_);
        }

        if constexpr (verbose_) {
            std::cout << " Merging small orbits with work threshold: " << work_threshold_ << "\n";
        }
        non_viable_edges_cache_.clear();
        merge_small_orbits(original_dag, current_coarse_graph, current_groups, current_contraction_map, work_threshold_);

        finalCoarseGraph_ = std::move(currentCoarseGraph);
        final_contraction_map_ = std::move(current_contraction_map);
        finalGroups_ = std::move(currentGroups);

        if constexpr (verbose_) {
            PrintFinalGroupsSummary();
        }
    }

    void PrintFinalGroupsSummary() const {
        std::cout << "\n--- 📦 Final Groups Summary ---\n";
        std::cout << "Total final groups: " << finalGroups_.size() << "\n";
        for (size_t i = 0; i < finalGroups_.size(); ++i) {
            const auto &group = finalGroups_[i];
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
    bool IsMergeViable(const GraphT &originalDag,
                       const Group &groupU,
                       const Group &groupV,
                       std::vector<std::vector<VertexType>> &outNewSubgraphs) const {
        std::vector<VertexType> allNodes;
        allNodes.reserve(groupU.subgraphs_.size() * (groupU.subgraphs_.empty() ? 0 : groupU.subgraphs_[0].size())
                         + groupV.subgraphs_.size() * (groupV.subgraphs_.empty() ? 0 : groupV.subgraphs_[0].size()));
        for (const auto &sg : groupU.subgraphs_) {
            allNodes.insert(all_nodes.end(), sg.begin(), sg.end());
        }
        for (const auto &sg : groupV.subgraphs_) {
            allNodes.insert(all_nodes.end(), sg.begin(), sg.end());
        }

        assert([&]() {
            std::vector<VertexType> tempNodesForCheck = all_nodes;
            std::sort(temp_nodes_for_check.begin(), temp_nodes_for_check.end());
            return std::unique(temp_nodes_for_check.begin(), temp_nodes_for_check.end()) == temp_nodes_for_check.end();
        }() && "Assumption failed: Vertices in groups being merged are not disjoint.");

        std::sort(all_nodes.begin(), all_nodes.end());

        ConstrGraphT inducedSubgraph;

        auto map = create_induced_subgraph_map(originalDag, inducedSubgraph, all_nodes);
        std::vector<VertexType> components;    // local -> component_id
        size_t numComponents = compute_weakly_connected_components(inducedSubgraph, components);
        out_new_subgraphs.assign(num_components, std::vector<VertexType>());

        if (allNodes.empty()) {    // Handle empty graph case
            return true;
        }

        for (const auto &node : all_nodes) {
            out_new_subgraphs[components[map[node]]].push_back(node);
        }

        if (numComponents > 1) {
            const size_t firstSgSize = out_new_subgraphs[0].size();
            ConstrGraphT repSg;
            create_induced_subgraph(originalDag, repSg, out_new_subgraphs[0]);

            for (size_t i = 1; i < numComponents; ++i) {
                if (outNewSubgraphs[i].size() != firstSgSize) {
                    return false;
                }

                ConstrGraphT currentSg;
                create_induced_subgraph(originalDag, currentSg, out_new_subgraphs[i]);
                if (!are_isomorphic_by_merkle_hash(repSg, currentSg)) {
                    return false;
                }
            }
        }
        return true;
    }

  public:
    const GraphT &GetCoarseGraph() const { return coarseGraph_; }

    const std::vector<VertexType> &GetContractionMap() const { return contraction_map_; }

    const GraphT &GetFinalCoarseGraph() const { return finalCoarseGraph_; }

    const std::vector<VertexType> &GetFinalContractionMap() const { return final_contraction_map_; }

    const std::vector<Group> &GetFinalGroups() const { return finalGroups_; }
};

}    // namespace osp
