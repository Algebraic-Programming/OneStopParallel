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
 * @tparam ConstrGraphT The subgraph/contracted graph type.
 */
template <typename GraphT, typename ConstrGraphT>
class IsomorphismMapper {
    using VertexC = VertexIdxT<ConstrGraphT>;    // Local vertex ID
    using VertexG = VertexIdxT<GraphT>;            // Global vertex ID

    const ConstrGraphT &repGraph_;
    const MerkleHashComputer<ConstrGraphT> repHasher_;

  public:
    /**
     * @brief Constructs an IsomorphismMapper.
     * @param representative_graph The subgraph to use as the "pattern".
     */
    IsomorphismMapper(const ConstrGraphT &representativeGraph)
        : repGraph_(representativeGraph), repHasher_(representativeGraph), numVertices_(representativeGraph.NumVertices()) {}

    virtual ~IsomorphismMapper() = default;

    /**
     * @brief Finds the isomorphism between the representative graph and a new graph.
     *
     * This method assumes the two graphs are isomorphic and finds one such mapping.
     *
     * @param current_graph The new isomorphic subgraph.
     * @return A map from `current_local_vertex_id` -> `representative_local_vertex_id`.
     */
    std::unordered_map<VertexC, VertexC> FindMapping(const ConstrGraphT &currentGraph) const {
        if (currentGraph.NumVertices() != numVertices_) {
            throw std::runtime_error("IsomorphismMapper: Graph sizes do not match.");
        }
        if (numVertices_ == 0) {
            return {};
        }

        // 1. Compute hashes and orbits for the current graph.
        MerkleHashComputer<ConstrGraphT> currentHasher(currentGraph);
        const auto &repOrbits = repHasher_.GetOrbits();
        const auto &currentOrbits = currentHasher.GetOrbits();

        // 2. Verify that the orbit structures are identical.
        if (repOrbits.size() != currentOrbits.size()) {
            throw std::runtime_error("IsomorphismMapper: Graphs have a different number of orbits.");
        }
        for (const auto &[hash, repOrbitNodes] : repOrbits) {
            auto it = currentOrbits.find(hash);
            if (it == currentOrbits.end() || it->second.size() != repOrbitNodes.size()) {
                throw std::runtime_error("IsomorphismMapper: Mismatched orbit structure between graphs.");
            }
        }

        // 3. Iteratively map all components of the graph.
        std::vector<VertexC> mapCurrentToRep(numVertices_, std::numeric_limits<VertexC>::max());
        std::vector<bool> repIsMapped(numVertices_, false);
        std::vector<bool> currentIsMapped(numVertices_, false);
        size_t mappedCount = 0;

        while (mappedCount < numVertices_) {
            std::queue<std::pair<VertexC, VertexC>> q;

            // Find an unmapped vertex in the representative graph to seed the next component traversal.
            VertexC repSeed = std::numeric_limits<VertexC>::max();
            for (VertexC i = 0; i < numVertices_; ++i) {
                if (!repIsMapped[i]) {
                    repSeed = i;
                    break;
                }
            }

            if (repSeed == std::numeric_limits<VertexC>::max()) {
                break;    // Should be unreachable if mapped_count < NumVertices
            }

            // Find a corresponding unmapped vertex in the current graph's orbit.
            const auto &candidates = currentOrbits.at(repHasher_.GetVertexHash(repSeed));
            VertexC currentSeed = std::numeric_limits<VertexC>::max();    // Should always be found
            for (const auto &candidate : candidates) {
                if (!currentIsMapped[candidate]) {
                    currentSeed = candidate;
                    break;
                }
            }
            if (currentSeed == std::numeric_limits<VertexC>::max()) {
                throw std::runtime_error("IsomorphismMapper: Could not find an unmapped candidate to seed component mapping.");
            }

            // Seed the queue and start the traversal for this component.
            q.push({repSeed, currentSeed});
            mapCurrentToRep[repSeed] = currentSeed;
            repIsMapped[repSeed] = true;
            currentIsMapped[currentSeed] = true;
            mappedCount++;

            while (!q.empty()) {
                auto [u_rep, u_curr] = q.front();
                q.pop();

                // Match neighbors (both parents and children)
                MatchNeighbors(
                    currentGraph, currentHasher, u_rep, u_curr, mapCurrentToRep, repIsMapped, currentIsMapped, mappedCount, q, true);
                MatchNeighbors(
                    currentGraph, currentHasher, u_rep, u_curr, mapCurrentToRep, repIsMapped, currentIsMapped, mappedCount, q, false);
            }
        }

        if (mappedCount != numVertices_) {
            throw std::runtime_error("IsomorphismMapper: Failed to map all vertices.");
        }

        // 4. Return the inverted map.
        std::unordered_map<VertexC, VertexC> currentLocalToRepLocal;
        currentLocalToRepLocal.reserve(numVertices_);
        for (VertexC i = 0; i < numVertices_; ++i) {
            currentLocalToRepLocal[mapCurrentToRep[i]] = i;
        }
        return currentLocalToRepLocal;
    }

  private:
    const size_t numVertices_;

    void MatchNeighbors(const ConstrGraphT &currentGraph,
                        const MerkleHashComputer<ConstrGraphT> &currentHasher,
                        VertexC uRep,
                        VertexC uCurr,
                        std::vector<VertexC> &mapCurrentToRep,
                        std::vector<bool> &repIsMapped,
                        std::vector<bool> &currentIsMapped,
                        size_t &mappedCount,
                        std::queue<std::pair<VertexC, VertexC>> &q,
                        bool matchChildren) const {
        const auto &repNeighborsRange = matchChildren ? repGraph_.Children(uRep) : repGraph_.Parents(uRep);
        const auto &currNeighborsRange = matchChildren ? currentGraph.Children(uCurr) : currentGraph.Parents(uCurr);

        for (const auto &vRep : repNeighborsRange) {
            if (repIsMapped[vRep]) {
                continue;
            }

            for (const auto &vCurr : currNeighborsRange) {
                if (currentIsMapped[vCurr]) {
                    continue;
                }

                if (repHasher_.GetVertexHash(vRep) == currentHasher.GetVertexHash(vCurr)) {
                    mapCurrentToRep[vRep] = vCurr;
                    repIsMapped[vRep] = true;
                    currentIsMapped[vCurr] = true;
                    mappedCount++;
                    q.push({vRep, vCurr});
                    break;    // Found a match for vRep, move to the next rep neighbor.
                }
            }
        }
    }
};

}    // namespace osp
