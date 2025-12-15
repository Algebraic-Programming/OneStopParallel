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

#include "osp/bsp/model/BspInstance.hpp"
#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/dag_divider/ConnectedComponentDivider.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"
#include "osp/pebbling/pebblers/pebblingILP/partialILP/AcyclicPartitioningILP.hpp"

namespace osp {

template <typename GraphT>
class AcyclicDagDivider {
    static_assert(IsComputationalDagV<GraphT>, "PebblingSchedule can only be used with computational DAGs.");

  protected:
    using VertexIdx = VertexIdxT<GraphT>;

    unsigned minPartitionSize_ = 40, maxPartitionSize_ = 80;
    bool ignoreSourcesInSize_ = true;

    std::vector<unsigned> GetTopologicalSplit(const GraphT &g,
                                              std::pair<unsigned, unsigned> minAndMax,
                                              const std::vector<bool> &isOriginalSource) const;

    VCommwT<GraphT> static GetSplitCost(const GraphT &g, const std::vector<unsigned> &nodeToPart);

  public:
    AcyclicDagDivider() {}

    virtual ~AcyclicDagDivider() = default;

    std::vector<unsigned> ComputePartitioning(const BspInstance<GraphT> &instance);

    // getters and setters for problem parameters
    inline std::pair<unsigned, unsigned> GetMinAndMaxSize() const { return std::make_pair(minPartitionSize_, maxPartitionSize_); }

    inline void SetMinAndMaxSize(const std::pair<unsigned, unsigned> minAndMax) {
        minPartitionSize_ = minAndMax.first;
        maxPartitionSize_ = minAndMax.second;
    }

    inline void SetIgnoreSources(const bool ignore) { ignoreSourcesInSize_ = ignore; }
};

template <typename GraphT>
std::vector<unsigned> AcyclicDagDivider<GraphT>::ComputePartitioning(const BspInstance<GraphT> &instance) {
    const unsigned n = static_cast<unsigned>(instance.NumberOfVertices());

    // split to connected components first
    ConnectedComponentDivider<GraphT, GraphT> connectedComp;
    connectedComp.Divide(instance.GetComputationalDag());

    std::vector<GraphT> subDags = connectedComp.GetSubDags();
    std::vector<std::pair<unsigned, VertexIdx>> nodeToSubdagAndIndex(n);
    std::vector<std::vector<VertexIdx>> originalId(subDags.size());
    for (VertexIdx node = 0; node < n; ++node) {
        nodeToSubdagAndIndex[node] = {connectedComp.GetComponent()[node], connectedComp.GetVertexMap()[node]};
        originalId[connectedComp.GetComponent()[node]].push_back(node);
    }

    // TODO extend with splits at directed articulation points in future?

    // split components further with ILPs or heuristics
    while (true) {
        bool existsTooLarge = false;
        std::vector<bool> dagIsTooLarge(subDags.size(), false);
        std::vector<unsigned> dagRealSize(subDags.size(), 0);

        for (unsigned idx = 0; idx < subDags.size(); ++idx) {
            const GraphT &dag = subDags[idx];
            if (!ignoreSourcesInSize_) {
                dagRealSize[idx] = static_cast<unsigned>(dag.NumVertices());
                if (dag.NumVertices() > maxPartitionSize_) {
                    dagIsTooLarge[idx] = true;
                    existsTooLarge = true;
                }
            } else {
                for (VertexIdx localId = 0; localId < dag.NumVertices(); ++localId) {
                    if (instance.GetComputationalDag().InDegree(originalId[idx][localId]) > 0) {
                        ++dagRealSize[idx];
                    }
                }
            }
            if (dagRealSize[idx] > maxPartitionSize_) {
                dagIsTooLarge[idx] = true;
                existsTooLarge = true;
            }
        }

        if (!existsTooLarge) {
            break;
        }

        std::vector<GraphT> newDagList;
        std::vector<std::vector<VertexIdx>> originalIdUpdated;

        for (unsigned idx = 0; idx < subDags.size(); ++idx) {
            const GraphT &dag = subDags[idx];
            if (!dagIsTooLarge[idx]) {
                for (VertexIdx localId = 0; localId < dag.NumVertices(); ++localId) {
                    nodeToSubdagAndIndex[originalId[idx][localId]].first = static_cast<unsigned>(newDagList.size());
                }

                originalIdUpdated.push_back(originalId[idx]);
                newDagList.push_back(dag);
            } else {
                std::vector<unsigned> ilpAssignment;
                // unsigned newMin = dag_real_size[idx]/3, minPartitionSize); minimum condition removed - it can cause very strict bisections
                unsigned newMin = dagRealSize[idx] / 3;
                unsigned newMax = dagRealSize[idx] - newMin;

                // mark the source nodes of the original DAG
                std::vector<bool> isOriginalSource(dag.NumVertices());
                for (VertexIdx localId = 0; localId < dag.NumVertices(); ++localId) {
                    isOriginalSource[localId] = (instance.GetComputationalDag().InDegree(originalId[idx][localId]) == 0);
                }

                // heuristic splitting
                std::vector<unsigned> heuristicAssignment = GetTopologicalSplit(dag, {newMin, newMax}, isOriginalSource);
                unsigned heuristicCost = GetSplitCost(dag, heuristicAssignment);
                unsigned ilpCost = UINT_MAX;

                // ILP-based splitting
                AcyclicPartitioningILP<GraphT> partitioner;
                partitioner.SetTimeLimitSeconds(120);
                partitioner.SetMinAndMaxSize({newMin, newMax});
                partitioner.SetIsOriginalSource(isOriginalSource);
                partitioner.SetNumberOfParts(2);    // note - if set to more than 2, ILP is MUCH more inefficient
                BspInstance partialInstance(dag, instance.GetArchitecture(), instance.GetNodeProcessorCompatibilityMatrix());
                ReturnStatus status = partitioner.ComputePartitioning(partialInstance, ilpAssignment);
                if (status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND) {
                    ilpCost = GetSplitCost(dag, ilpAssignment);
                }

                std::vector<unsigned> assignment = ilpCost < heuristicCost ? ilpAssignment : heuristicAssignment;

                // split DAG according to labels
                std::vector<GraphT> splitDags = create_induced_subgraphs<GraphT, GraphT>(dag, assignment);
                /*std::cout<<"SPLIT DONE: "<<dag.NumberOfVertices()<<" nodes to ";
                for(auto sdag : splitDags)
                    std::cout<<sdag.NumberOfVertices()<<" + ";
                std::cout<<std::endl;*/

                // update labels
                std::vector<VertexIdx> nodeIdxInNewSubDag(dag.NumVertices());
                std::vector<unsigned> nrNodesInNewSubDag(splitDags.size(), 0);
                for (VertexIdx localId = 0; localId < dag.NumVertices(); ++localId) {
                    nodeIdxInNewSubDag[localId] = nrNodesInNewSubDag[assignment[localId]];
                    ++nrNodesInNewSubDag[assignment[localId]];
                }

                for (auto nextDag : splitDags) {
                    originalIdUpdated.emplace_back(nextDag.NumVertices());
                }

                for (VertexIdx localId = 0; localId < dag.NumVertices(); ++localId) {
                    nodeToSubdagAndIndex[originalId[idx][localId]]
                        = {newDagList.size() + assignment[localId], nodeIdxInNewSubDag[localId]};
                    originalIdUpdated[newDagList.size() + assignment[localId]][nodeIdxInNewSubDag[localId]]
                        = originalId[idx][localId];
                }
                for (auto nextDag : splitDags) {
                    newDagList.push_back(nextDag);
                }
            }
        }

        subDags = newDagList;
        originalId = original_id_updated;
    }

    // output final cost
    std::vector<unsigned> finalAssignment(n);
    for (VertexIdx node = 0; node < n; ++node) {
        finalAssignment[node] = nodeToSubdagAndIndex[node].first;
    }
    std::cout << "Final cut cost of acyclic DAG divider is " << GetSplitCost(instance.GetComputationalDag(), finalAssignment)
              << std::endl;

    return finalAssignment;
}

template <typename GraphT>
std::vector<unsigned> AcyclicDagDivider<GraphT>::GetTopologicalSplit(const GraphT &g,
                                                                     std::pair<unsigned, unsigned> minAndMax,
                                                                     const std::vector<bool> &isOriginalSource) const {
    std::vector<unsigned> nodeToPart(g.NumVertices());

    std::vector<VertexIdx> topOrder = GetTopOrder(g);
    std::vector<unsigned> topOrderIdx(g.NumVertices());
    for (unsigned idx = 0; idx < g.NumVertices(); ++idx) {
        topOrderIdx[topOrder[idx]] = idx;
    }

    std::vector<unsigned> lastNodeIdxInHyperedge(g.NumVertices());
    for (unsigned node = 0; node < g.NumVertices(); ++node) {
        lastNodeIdxInHyperedge[node] = topOrderIdx[node];
        for (const auto &succ : g.Children(node)) {
            lastNodeIdxInHyperedge[node] = std::max(lastNodeIdxInHyperedge[node], topOrderIdx[succ]);
        }
    }

    unsigned index = 0;
    unsigned currentPartId = 0;

    unsigned nodesRemaining = static_cast<unsigned>(g.NumVertices());
    if (ignoreSourcesInSize_) {
        nodesRemaining = 0;
        for (unsigned node = 0; node < g.NumVertices(); ++node) {
            if (!isOriginalSource[node]) {
                ++nodesRemaining;
            }
        }
    }

    while (nodesRemaining > min_and_max.second) {
        unsigned bestCost = UINT_MAX;
        unsigned bestEnd = index;

        unsigned end;
        unsigned newlyAddedNodes = 0;
        for (end = index + 1; index < g.NumVertices() && newlyAddedNodes < minAndMax.first; ++end) {
            if (!ignoreSourcesInSize_ || !isOriginalSource[end]) {
                ++newlyAddedNodes;
            }
        }

        while (end < g.NumVertices() && newlyAddedNodes < minAndMax.second) {
            unsigned extraCost = 0;

            // check the extra cut cost of the potential endpoint
            for (unsigned topOrderPos = index; topOrderPos <= end; ++topOrderPos) {
                VertexIdx node = topOrder[topOrderPos];
                if (lastNodeIdxInHyperedge[node] > end) {
                    extraCost += g.VertexCommWeight(node);
                }

                for (const auto &pred : g.Parents(node)) {
                    if (last_node_idx_in_hyperedge[pred] > end) {
                        extra_cost += G.VertexCommWeight(pred);
                    }
                }
            }

            if (extraCost < bestCost) {
                bestCost = extraCost;
                bestEnd = end;
            }

            ++end;
            if (!ignoreSourcesInSize_ || !is_original_source[end]) {
                ++newlyAddedNodes;
            }
        }

        for (VertexIdx idx = index; idx <= bestEnd; ++idx) {
            nodeToPart[topOrder[idx]] = currentPartId;
            if (!ignoreSourcesInSize_ || !isOriginalSource[idx]) {
                --nodesRemaining;
            }
        }
        index = bestEnd + 1;
        ++currentPartId;
    }

    // remaining nodes go into last part
    for (VertexIdx idx = index; idx < g.NumVertices(); ++idx) {
        nodeToPart[topOrder[idx]] = currentPartId;
    }

    return nodeToPart;
}

template <typename GraphT>
VCommwT<GraphT> AcyclicDagDivider<GraphT>::GetSplitCost(const GraphT &g, const std::vector<unsigned> &nodeToPart) {
    VCommwT<GraphT> cost = 0;

    for (VertexIdx node = 0; node < g.NumVertices(); ++node) {
        std::set<unsigned> partsIncluded;
        partsIncluded.insert(nodeToPart[node]);
        for (const auto &succ : g.Children(node)) {
            partsIncluded.insert(nodeToPart[succ]);
        }

        cost += static_cast<VCommwT<GraphT>>(partsIncluded.size() - 1) * g.VertexCommWeight(node);
    }

    return cost;
}

}    // namespace osp
