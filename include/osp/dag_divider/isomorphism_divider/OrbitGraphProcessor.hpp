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

    static_assert(IsComputationalDagV<GraphT>, "Graph must be a computational DAG");
    static_assert(IsComputationalDagV<ConstrGraphT>, "Constr_Graph_t must be a computational DAG");
    static_assert(IsConstructableCdagV<ConstrGraphT>, "Constr_Graph_t must satisfy the constructable_cdag_vertex concept");
    static_assert(std::is_same_v<VertexIdxT<GraphT>, VertexIdxT<ConstrGraphT>>,
                  "Graph_t and Constr_Graph_t must have the same VertexIdx types");

    using VertexType = VertexIdxT<GraphT>;

    static constexpr bool verbose_ = false;

    // Represents a group of isomorphic subgraphs, corresponding to a single node in a coarse graph.
    struct Group {
        // Each vector of vertices represents one of the isomorphic subgraphs in this group.
        std::vector<std::vector<VertexType>> subgraphs_;

        inline size_t size() const { return subgraphs_.size(); }
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
    VWorkwT<ConstrGraphT> workThreshold_ = 0;
    VWorkwT<ConstrGraphT> criticalPathThreshold_ = 0;
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
            HashCombine(h1, h2);
            return h1;
        }
    };

    std::unordered_set<std::pair<VertexType, VertexType>, PairHasher> nonViableEdgesCache_;
    std::unordered_set<std::pair<VertexType, VertexType>, PairHasher> nonViableCritPathEdgesCache_;

    /**
     * @brief Simulates the merge of node v into u and returns the resulting temporary graph.
     */
    std::pair<ConstrGraphT, std::vector<VertexType>> SimulateMerge(VertexType u,
                                                                   VertexType v,
                                                                   const ConstrGraphT &currentCoarseGraph) const {
        std::vector<VertexType> tempContractionMap(currentCoarseGraph.NumVertices());
        VertexType newIdx = 0;
        for (VertexType i = 0; i < static_cast<VertexType>(tempContractionMap.size()); ++i) {
            if (i != v) {
                tempContractionMap[i] = newIdx++;
            }
        }
        tempContractionMap[v] = tempContractionMap[u];

        ConstrGraphT tempCoarseGraph;
        coarser_util::ConstructCoarseDag(currentCoarseGraph, tempCoarseGraph, tempContractionMap);

        return {std::move(tempCoarseGraph), std::move(tempContractionMap)};
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
        for (const auto &nonViableEdge : nonViableEdgesCache_) {
            const VertexType oldU = nonViableEdge.first;
            const VertexType oldV = nonViableEdge.second;
            const VertexType newU = groupRemap[oldU];
            const VertexType newV = groupRemap[oldV];

            if (oldU != v && oldV != v && newU != newV) {
                nextNonViableEdges.insert({newU, newV});
            }
        }
        nonViableEdgesCache_ = std::move(nextNonViableEdges);

        std::unordered_set<std::pair<VertexType, VertexType>, PairHasher> nextNonViableCritPathEdges;
        for (const auto &nonViableEdge : nonViableCritPathEdgesCache_) {
            const VertexType oldU = nonViableEdge.first;
            const VertexType oldV = nonViableEdge.second;
            const VertexType newU = groupRemap[oldU];
            const VertexType newV = groupRemap[oldV];

            if (oldU != v && oldV != v && newU != newV) {
                nextNonViableCritPathEdges.insert({newU, newV});
            }
        }
        nonViableCritPathEdgesCache_ = std::move(nextNonViableCritPathEdges);

        // Update groups
        std::vector<Group> nextGroups(currentCoarseGraph.NumVertices());
        for (VertexType i = 0; i < static_cast<VertexType>(currentGroups.size()); ++i) {
            if (i != u && i != v) {
                nextGroups[groupRemap[i]] = std::move(currentGroups[i]);
            }
        }
        nextGroups[groupRemap[u]].subgraphs_ = std::move(newSubgraphs);
        currentGroups = std::move(nextGroups);

        // Update main contraction map
        for (VertexType &nodeMap : currentContractionMap) {
            nodeMap = groupRemap[nodeMap];
        }
    }

    /**
     * @brief Merges small orbits based on work threshold (final cleanup pass).
     */
    void MergeSmallOrbits(const GraphT &originalDag,
                          ConstrGraphT &currentCoarseGraph,
                          std::vector<Group> &currentGroups,
                          std::vector<VertexType> &currentContractionMap,
                          const VWorkwT<ConstrGraphT> workThreshold,
                          const VWorkwT<ConstrGraphT> pathThreshold = 0) {
        bool changed = true;
        while (changed) {
            const std::vector<VertexIdxT<ConstrGraphT>> vertexPoset
                = GetTopNodeDistance<ConstrGraphT, VertexIdxT<ConstrGraphT>>(currentCoarseGraph);
            const std::vector<VertexIdxT<ConstrGraphT>> vertexBotPoset
                = GetBottomNodeDistance<ConstrGraphT, VertexIdxT<ConstrGraphT>>(currentCoarseGraph);

            changed = false;
            for (const auto u : currentCoarseGraph.Vertices()) {
                for (const auto v : currentCoarseGraph.Children(u)) {
                    if constexpr (HasTypedVerticesV<ConstrGraphT>) {
                        if (not mergeDifferentNodeTypes_) {
                            if (currentCoarseGraph.VertexType(u) != currentCoarseGraph.VertexType(v)) {
                                if constexpr (verbose_) {
                                    std::cout << "  - Merge of " << u << " and " << v << " not viable (different node types)\n";
                                }
                                continue;
                            }
                        }
                    }

                    if (nonViableEdgesCache_.count({u, v}) || nonViableCritPathEdgesCache_.count({u, v})) {
                        if constexpr (verbose_) {
                            std::cout << "  - Merge of " << u << " and " << v << " already checked. Skipping.\n";
                        }
                        continue;
                    }

                    const VWorkwT<ConstrGraphT> uWorkWeight = currentCoarseGraph.VertexWorkWeight(u);
                    const VWorkwT<ConstrGraphT> vWorkWeight = currentCoarseGraph.VertexWorkWeight(v);
                    const VWorkwT<ConstrGraphT> vThreshold
                        = workThreshold * static_cast<VWorkwT<ConstrGraphT>>(currentGroups[v].size());
                    const VWorkwT<ConstrGraphT> uThreshold
                        = workThreshold * static_cast<VWorkwT<ConstrGraphT>>(currentGroups[u].size());

                    if (uWorkWeight > uThreshold && vWorkWeight > vThreshold) {
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
                    const bool mergeIsValid = IsMergeViable(originalDag, currentGroups[u], currentGroups[v], newSubgraphs);

                    if (!mergeIsValid) {
                        if constexpr (verbose_) {
                            std::cout << "  - Merge of " << u << " and " << v << " and " << v
                                      << " not viable (error in is_merge_viable)\n";
                        }
                        nonViableEdgesCache_.insert({u, v});
                        continue;
                    }

                    auto [tempCoarseGraph, tempContractionMap] = SimulateMerge(u, v, currentCoarseGraph);

                    if (CriticalPathWeight(tempCoarseGraph)
                        > (pathThreshold * static_cast<VWorkwT<ConstrGraphT>>(newSubgraphs.size())
                           + CriticalPathWeight(currentCoarseGraph))) {
                        if constexpr (verbose_) {
                            std::cout << "  - Merge of " << u << " and " << v
                                      << " increases critical path. Old cirtical path: " << CriticalPathWeight(currentCoarseGraph)
                                      << " new critical path: " << CriticalPathWeight(tempCoarseGraph) << " + "
                                      << pathThreshold * static_cast<VWorkwT<ConstrGraphT>>(newSubgraphs.size()) << "\n";
                        }
                        nonViableCritPathEdgesCache_.insert({u, v});
                        continue;
                    }

                    if constexpr (verbose_) {
                        std::cout << "  - Merging " << v << " into " << u << ". New coarse graph has "
                                  << tempCoarseGraph.NumVertices() << " nodes.\n";
                    }

                    CommitMerge(u,
                                v,
                                std::move(tempCoarseGraph),
                                tempContractionMap,
                                std::move(newSubgraphs),
                                currentCoarseGraph,
                                currentGroups,
                                currentContractionMap);

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
                       const VWorkwT<ConstrGraphT> pathThreshold = 0) {
        bool changed = true;
        while (changed) {
            const std::vector<VertexIdxT<ConstrGraphT>> vertexPoset
                = GetTopNodeDistance<ConstrGraphT, VertexIdxT<ConstrGraphT>>(currentCoarseGraph);
            const std::vector<VertexIdxT<ConstrGraphT>> vertexBotPoset
                = GetBottomNodeDistance<ConstrGraphT, VertexIdxT<ConstrGraphT>>(currentCoarseGraph);

            changed = false;
            for (const auto &edge : Edges(currentCoarseGraph)) {
                VertexType u = Source(edge, currentCoarseGraph);
                VertexType v = Target(edge, currentCoarseGraph);

                if (nonViableEdgesCache_.count({u, v}) || nonViableCritPathEdgesCache_.count({u, v})) {
                    continue;
                }
                if constexpr (HasTypedVerticesV<ConstrGraphT>) {
                    if (not mergeDifferentNodeTypes) {
                        if (currentCoarseGraph.VertexType(u) != currentCoarseGraph.VertexType(v)) {
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
                const bool mergeIsValid = IsMergeViable(originalDag, currentGroups[u], currentGroups[v], newSubgraphs);
                const std::size_t newSize = newSubgraphs.size();

                const bool mergeViable = (newSize >= currentSymmetry_);
                const bool bothBelowSymmetryThreshold = (uSize < currentSymmetry_) && (vSize < currentSymmetry_);

                if (!mergeIsValid) {
                    nonViableEdgesCache_.insert({u, v});
                    continue;
                }
                if (!mergeViable && !bothBelowSymmetryThreshold) {
                    nonViableEdgesCache_.insert({u, v});
                    continue;
                }
                if (not mergeSymmetryNarrowing) {
                    if (newSize < std::min(uSize, vSize)) {
                        continue;
                    }
                }

                auto [tempCoarseGraph, tempContractionMap] = SimulateMerge(u, v, currentCoarseGraph);

                if (CriticalPathWeight(tempCoarseGraph) > (pathThreshold * static_cast<VWorkwT<ConstrGraphT>>(newSubgraphs.size())
                                                           + CriticalPathWeight(currentCoarseGraph))) {
                    nonViableCritPathEdgesCache_.insert({u, v});
                    continue;
                }

                CommitMerge(u,
                            v,
                            std::move(tempCoarseGraph),
                            tempContractionMap,
                            std::move(newSubgraphs),
                            currentCoarseGraph,
                            currentGroups,
                            currentContractionMap);
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
                                  const std::vector<VWorkwT<GraphT>> &lockThresholdPerType,
                                  const VWorkwT<ConstrGraphT> pathThreshold = 0) {
        bool changed = true;
        while (changed) {
            const std::vector<VertexIdxT<ConstrGraphT>> vertexPoset
                = GetTopNodeDistance<ConstrGraphT, VertexIdxT<ConstrGraphT>>(currentCoarseGraph);
            const std::vector<VertexIdxT<ConstrGraphT>> vertexBotPoset
                = GetBottomNodeDistance<ConstrGraphT, VertexIdxT<ConstrGraphT>>(currentCoarseGraph);

            changed = false;
            for (const auto &edge : Edges(currentCoarseGraph)) {
                VertexType u = Source(edge, currentCoarseGraph);
                VertexType v = Target(edge, currentCoarseGraph);

                if (nonViableEdgesCache_.count({u, v}) || nonViableCritPathEdgesCache_.count({u, v})) {
                    if constexpr (verbose_) {
                        std::cout << "  - Merge of " << u << " and " << v << " already checked. Skipping.\n";
                    }
                    continue;
                }

                if constexpr (HasTypedVerticesV<ConstrGraphT>) {
                    if (not mergeDifferentNodeTypes) {
                        if (currentCoarseGraph.VertexType(u) != currentCoarseGraph.VertexType(v)) {
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

                const bool mergeIsValid = IsMergeViable(originalDag, currentGroups[u], currentGroups[v], newSubgraphs);
                const std::size_t newSize = newSubgraphs.size();

                if (!mergeIsValid) {
                    if constexpr (verbose_) {
                        std::cout << "  - Merge of " << u << " and " << v << " and " << v
                                  << " not viable (error in is_merge_viable)\n";
                    }
                    nonViableEdgesCache_.insert({u, v});
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
                    nonViableEdgesCache_.insert({u, v});
                    continue;
                }

                VTypeT<GraphT> uType = 0;
                VTypeT<GraphT> vType = 0;
                if (not mergeDifferentNodeTypes && HasTypedVerticesV<GraphT>) {
                    uType = currentCoarseGraph.VertexType(u);
                    vType = currentCoarseGraph.VertexType(v);
                }

                const bool uIsSignificant = (uSize >= minSymmetry_)
                                            && (currentCoarseGraph.VertexWorkWeight(u) > lockThresholdPerType[uType]);
                const bool vIsSignificant = (vSize >= minSymmetry_)
                                            && (currentCoarseGraph.VertexWorkWeight(v) > lockThresholdPerType[vType]);

                if (uIsSignificant && vIsSignificant) {
                    // Both are significant ---
                    if (newSize < std::min(uSize, vSize)) {
                        if constexpr (verbose_) {
                            std::cout << "  - Merge of " << u << " and " << v
                                      << " not viable (Symmetry Narrowing below min of two significant nodes)\n";
                            std::cout << "    - u_sym: " << uSize << ", v_sym: " << vSize << " -> new_sym: " << newSize << "\n";
                        }
                        nonViableEdgesCache_.insert({u, v});
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
                        nonViableEdgesCache_.insert({u, v});
                        continue;
                    }
                }

                // Critical Path Check
                auto [tempCoarseGraph, tempContractionMap] = SimulateMerge(u, v, currentCoarseGraph);

                if (CriticalPathWeight(tempCoarseGraph) > (pathThreshold * static_cast<VWorkwT<ConstrGraphT>>(newSubgraphs.size())
                                                           + CriticalPathWeight(currentCoarseGraph))) {
                    if constexpr (verbose_) {
                        std::cout << "  - Merge of " << u << " and " << v
                                  << " increases critical path. Old cirtical path: " << CriticalPathWeight(currentCoarseGraph)
                                  << " new critical path: " << CriticalPathWeight(tempCoarseGraph) << " + "
                                  << pathThreshold * static_cast<VWorkwT<ConstrGraphT>>(newSubgraphs.size()) << "\n";
                    }
                    nonViableCritPathEdgesCache_.insert({u, v});
                    continue;
                }

                // Commit Merge
                if constexpr (verbose_) {
                    std::cout << "  - Merging " << v << " into " << u << ". New coarse graph has "
                              << tempCoarseGraph.NumVertices() << " nodes.\n";
                }

                CommitMerge(u,
                            v,
                            std::move(tempCoarseGraph),
                            tempContractionMap,
                            std::move(newSubgraphs),
                            currentCoarseGraph,
                            currentGroups,
                            currentContractionMap);

                changed = true;
                break;
            }
        }
    }

  public:
    explicit OrbitGraphProcessor() {}

    void SetMergeDifferentNodeTypes(bool flag) { mergeDifferentNodeTypes_ = flag; }

    void SetWorkThreshold(VWorkwT<ConstrGraphT> workThreshold) { workThreshold_ = workThreshold; }

    void SetCriticalPathThreshold(VWorkwT<ConstrGraphT> criticalPathThreshold) { criticalPathThreshold_ = criticalPathThreshold; }

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
        contractionMap_.clear();
        finalCoarseGraph_ = ConstrGraphT();
        finalContractionMap_.clear();
        finalGroups_.clear();
        nonViableEdgesCache_.clear();
        nonViableCritPathEdgesCache_.clear();

        if (dag.NumVertices() == 0) {
            return;
        }

        const auto &orbits = hasher.GetOrbits();

        contractionMap_.assign(dag.NumVertices(), 0);
        VertexType coarseNodeIdx = 0;

        for (const auto &hashVerticesPair : orbits) {
            const auto &vertices = hashVerticesPair.second;
            for (const auto v : vertices) {
                contractionMap_[v] = coarseNodeIdx;
            }
            coarseNodeIdx++;
        }

        std::vector<VWorkwT<GraphT>> workPerVertexType;
        workPerVertexType.resize(mergeDifferentNodeTypes_ ? 1U : dag.NumVertexTypes(), 0);

        std::map<size_t, size_t> orbitSizeCounts;
        std::map<size_t, VWorkwT<GraphT>> workPerOrbitSize;
        VWorkwT<GraphT> totalWork = 0;
        for (const auto &[hash, vertices] : orbits) {
            const size_t orbitSize = vertices.size();

            if (orbitSize == 1U) {
                continue;    // exclude single node orbits from total work
            }

            orbitSizeCounts[orbitSize]++;

            VWorkwT<GraphT> orbitWork = 0;
            for (const auto v : vertices) {
                orbitWork += dag.VertexWorkWeight(v);
            }

            if (not mergeDifferentNodeTypes_ && HasTypedVerticesV<GraphT>) {
                workPerVertexType[dag.VertexType(vertices[0])] += orbitWork;
            } else {
                workPerVertexType[0] += orbitWork;
            }

            workPerOrbitSize[orbitSize] += orbitWork;
            totalWork += orbitWork;
        }

        std::vector<VWorkwT<GraphT>> lockThresholdPerType(workPerVertexType.size());
        for (size_t i = 0; i < workPerVertexType.size(); ++i) {
            lockThresholdPerType[i] = static_cast<VWorkwT<GraphT>>(lockOrbitRatio_ * workPerVertexType[i]);
        }

        std::vector<double> relAccWorkPerOrbitSize;
        std::vector<size_t> symmetryLevelsToTest
            = ComputeSymmetryLevels(relAccWorkPerOrbitSize, workPerOrbitSize, totalWork, orbitSizeCounts);

        if constexpr (verbose_) {
            std::cout << "\n--- Orbit Analysis ---\n";
            for (auto const &[size, count] : orbitSizeCounts) {
                if (totalWork > 0) {
                    std::cout << "  - Orbits of size " << size << ": " << count << " groups, weight: "
                              << 100.0 * static_cast<double>(workPerOrbitSize[size]) / static_cast<double>(totalWork) << "%\n";
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
                              << 100.0 * static_cast<double>(workPerVertexType[j]) / static_cast<double>(totalWork) << "%\n";
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

        coarser_util::ConstructCoarseDag(dag, coarseGraph_, contractionMap_);

        if (useAdaptiveSymmetryThreshold_) {
            PerformCoarseningAdaptiveSymmetry(dag, coarseGraph_, lockThresholdPerType, symmetryLevelsToTest);
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
                                              const std::map<size_t, VWorkwT<GraphT>> workPerOrbitSize,
                                              const VWorkwT<GraphT> totalWork,
                                              const std::map<size_t, size_t> orbitSizeCounts) {
        std::vector<size_t> symmetryLevelsToTest;
        minSymmetry_ = 2;

        switch (symmetryLevelHeuristic_) {
            case SymmetryLevelHeuristic::PERCENTILE_BASED: {
                if constexpr (verbose_) {
                    std::cout << "Using PERCENTILE_BASED heuristic for symmetry levels.\n";
                }
                size_t percentileIdx = 0;
                VWorkwT<GraphT> cumulativeWork = 0;
                for (auto it = workPerOrbitSize.rbegin(); it != workPerOrbitSize.rend(); ++it) {
                    cumulativeWork += it->second;
                    if (totalWork == 0) {
                        continue;    // Avoid division by zero
                    }
                    double currentWorkRatio = static_cast<double>(cumulativeWork) / static_cast<double>(totalWork);
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
                VWorkwT<GraphT> cumulativeWork = 0;
                for (auto it = workPerOrbitSize.rbegin(); it != workPerOrbitSize.rend(); ++it) {
                    cumulativeWork += it->second;
                    if (totalWork > 0) {
                        relAccWorkPerOrbitSize.push_back(static_cast<double>(cumulativeWork) / static_cast<double>(totalWork));
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
                VWorkwT<GraphT> cumulativeWork = 0;
                for (auto it = workPerOrbitSize.rbegin(); it != workPerOrbitSize.rend(); ++it) {
                    cumulativeWork += it->second;
                    const double relWork
                        = (totalWork == 0) ? 0 : static_cast<double>(cumulativeWork) / static_cast<double>(totalWork);
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
        finalContractionMap_.clear();

        if (initialCoarseGraph.NumVertices() == 0) {
            return;
        }

        ConstrGraphT currentCoarseGraph = initialCoarseGraph;
        std::vector<Group> currentGroups(initialCoarseGraph.NumVertices());
        std::vector<VertexType> currentContractionMap = contractionMap_;

        // Initialize groups: each group corresponds to an orbit.
        for (VertexType i = 0; i < originalDag.NumVertices(); ++i) {
            const VertexType coarseNode = contractionMap_[i];
            currentGroups[coarseNode].subgraphs_.push_back({i});
        }

        if constexpr (HasTypedVerticesV<ConstrGraphT>) {
            if constexpr (verbose_) {
                std::cout << "Attempting to merge same node types.\n";
            }
            ContractEdges(originalDag, currentCoarseGraph, currentGroups, currentContractionMap, false, false);
            ContractEdges(originalDag, currentCoarseGraph, currentGroups, currentContractionMap, true, false);
        }

        if constexpr (verbose_) {
            std::cout << "Attempting to merge different node types.\n";
        }
        ContractEdges(originalDag, currentCoarseGraph, currentGroups, currentContractionMap, false, mergeDifferentNodeTypes_);
        ContractEdges(originalDag, currentCoarseGraph, currentGroups, currentContractionMap, true, mergeDifferentNodeTypes_);

        if constexpr (verbose_) {
            std::cout << "Attempting to merge small orbits.\n";
        }
        MergeSmallOrbits(originalDag, currentCoarseGraph, currentGroups, currentContractionMap, workThreshold_);

        nonViableCritPathEdgesCache_.clear();
        nonViableEdgesCache_.clear();

        ContractEdges(
            originalDag, currentCoarseGraph, currentGroups, currentContractionMap, true, mergeDifferentNodeTypes_, workThreshold_);

        finalCoarseGraph_ = std::move(currentCoarseGraph);
        finalContractionMap_ = std::move(currentContractionMap);
        finalGroups_ = std::move(currentGroups);

        if constexpr (verbose_) {
            PrintFinalGroupsSummary();
        }
    }

    void PerformCoarseningAdaptiveSymmetry(const GraphT &originalDag,
                                           const ConstrGraphT &initialCoarseGraph,
                                           const std::vector<VWorkwT<GraphT>> &lockThresholdPerType,
                                           const std::vector<size_t> &symmetryLevelsToTest) {
        finalCoarseGraph_ = ConstrGraphT();
        finalContractionMap_.clear();

        if (initialCoarseGraph.NumVertices() == 0) {
            return;
        }

        ConstrGraphT currentCoarseGraph = initialCoarseGraph;
        std::vector<Group> currentGroups(initialCoarseGraph.NumVertices());
        std::vector<VertexType> currentContractionMap = contractionMap_;

        for (VertexType i = 0; i < originalDag.NumVertices(); ++i) {
            const VertexType coarseNode = contractionMap_[i];
            currentGroups[coarseNode].subgraphs_.push_back({i});
        }

        if constexpr (verbose_) {
            std::cout << " Starting adaptive symmetry coarsening with critical_path_threshold: " << criticalPathThreshold_ << "\n";
        }

        for (const auto sym : symmetryLevelsToTest) {
            currentSymmetry_ = sym;
            const bool isLastLoop = (sym == symmetryLevelsToTest.back());
            if constexpr (verbose_) {
                std::cout << "  Current symmetry threshold: " << currentSymmetry_ << "\n";
            }

            nonViableEdgesCache_.clear();

            ContractEdgesAdpativeSym(
                originalDag, currentCoarseGraph, currentGroups, currentContractionMap, false, isLastLoop, lockThresholdPerType);

            if (mergeDifferentNodeTypes_) {
                ContractEdgesAdpativeSym(originalDag,
                                            currentCoarseGraph,
                                            currentGroups,
                                            currentContractionMap,
                                            mergeDifferentNodeTypes_,
                                            isLastLoop,
                                            lockThresholdPerType);
            }

            nonViableCritPathEdgesCache_.clear();
            ContractEdgesAdpativeSym(originalDag,
                                        currentCoarseGraph,
                                        currentGroups,
                                        currentContractionMap,
                                        mergeDifferentNodeTypes_,
                                        isLastLoop,
                                        lockThresholdPerType,
                                        criticalPathThreshold_);
        }

        if constexpr (verbose_) {
            std::cout << " Merging small orbits with work threshold: " << workThreshold_ << "\n";
        }
        nonViableEdgesCache_.clear();
        MergeSmallOrbits(originalDag, currentCoarseGraph, currentGroups, currentContractionMap, workThreshold_);

        finalCoarseGraph_ = std::move(currentCoarseGraph);
        finalContractionMap_ = std::move(currentContractionMap);
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
            std::cout << "  - Group " << i << " (Size: " << group.subgraphs_.size() << ")\n";
            if (!group.subgraphs_.empty() && !group.subgraphs_[0].empty()) {
                std::cout << "    - Rep. Subgraph size: " << group.subgraphs_[0].size() << " nodes\n";
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
            allNodes.insert(allNodes.end(), sg.begin(), sg.end());
        }
        for (const auto &sg : groupV.subgraphs_) {
            allNodes.insert(allNodes.end(), sg.begin(), sg.end());
        }

        assert([&]() {
            std::vector<VertexType> tempNodesForCheck = allNodes;
            std::sort(tempNodesForCheck.begin(), tempNodesForCheck.end());
            return std::unique(tempNodesForCheck.begin(), tempNodesForCheck.end()) == tempNodesForCheck.end();
        }() && "Assumption failed: Vertices in groups being merged are not disjoint.");

        std::sort(allNodes.begin(), allNodes.end());

        ConstrGraphT inducedSubgraph;

        auto map = CreateInducedSubgraphMap(originalDag, inducedSubgraph, allNodes);
        std::vector<VertexType> components;    // local -> component_id
        size_t numComponents = ComputeWeaklyConnectedComponents(inducedSubgraph, components);
        outNewSubgraphs.assign(numComponents, std::vector<VertexType>());

        if (allNodes.empty()) {    // Handle empty graph case
            return true;
        }

        for (const auto &node : allNodes) {
            outNewSubgraphs[components[map[node]]].push_back(node);
        }

        if (numComponents > 1) {
            const size_t firstSgSize = outNewSubgraphs[0].size();
            ConstrGraphT repSg;
            CreateInducedSubgraph(originalDag, repSg, outNewSubgraphs[0]);

            for (size_t i = 1; i < numComponents; ++i) {
                if (outNewSubgraphs[i].size() != firstSgSize) {
                    return false;
                }

                ConstrGraphT currentSg;
                CreateInducedSubgraph(originalDag, currentSg, outNewSubgraphs[i]);
                if (!AreIsomorphicByMerkleHash(repSg, currentSg)) {
                    return false;
                }
            }
        }
        return true;
    }

  public:
    const GraphT &GetCoarseGraph() const { return coarseGraph_; }

    const std::vector<VertexType> &GetContractionMap() const { return contractionMap_; }

    const GraphT &GetFinalCoarseGraph() const { return finalCoarseGraph_; }

    const std::vector<VertexType> &GetFinalContractionMap() const { return finalContractionMap_; }

    const std::vector<Group> &GetFinalGroups() const { return finalGroups_; }
};

}    // namespace osp
