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
    static_assert(IsComputationalDagV<Graph_t>, "PebblingSchedule can only be used with computational DAGs.");

  protected:
    using vertex_idx = vertex_idx_t<Graph_t>;

    unsigned minPartitionSize_ = 40, maxPartitionSize_ = 80;
    bool ignoreSourcesInSize_ = true;

    std::vector<unsigned> GetTopologicalSplit(const GraphT &g,
                                              std::pair<unsigned, unsigned> minAndMax,
                                              const std::vector<bool> &isOriginalSource) const;

    v_commw_t<Graph_t> static GetSplitCost(const GraphT &g, const std::vector<unsigned> &nodeToPart);

  public:
    AcyclicDagDivider() {}

    virtual ~AcyclicDagDivider() = default;

    std::vector<unsigned> ComputePartitioning(const BspInstance<GraphT> &instance);

    // getters and setters for problem parameters
    inline std::pair<unsigned, unsigned> GetMinAndMaxSize() const { return std::make_pair(minPartitionSize_, maxPartitionSize_); }

    inline void SetMinAndMaxSize(const std::pair<unsigned, unsigned> minAndMax) {
        minPartitionSize_ = min_and_max.first;
        maxPartitionSize_ = min_and_max.second;
    }

    inline void SetIgnoreSources(const bool ignore) { ignoreSourcesInSize_ = ignore; }
};

template <typename GraphT>
std::vector<unsigned> AcyclicDagDivider<GraphT>::ComputePartitioning(const BspInstance<GraphT> &instance) {
    const unsigned n = static_cast<unsigned>(instance.numberOfVertices());

    // split to connected components first
    ConnectedComponentDivider<GraphT, GraphT> connectedComp;
    connectedComp.divide(instance.getComputationalDag());

    std::vector<Graph_t> subDags = connectedComp.get_sub_dags();
    std::vector<std::pair<unsigned, vertex_idx>> nodeToSubdagAndIndex(n);
    std::vector<std::vector<vertex_idx>> originalId(subDags.size());
    for (vertex_idx node = 0; node < n; ++node) {
        nodeToSubdagAndIndex[node] = {connectedComp.get_component()[node], connectedComp.get_vertex_map()[node]};
        originalId[connectedComp.get_component()[node]].push_back(node);
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
                for (vertex_idx localId = 0; local_ID < dag.NumVertices(); ++local_ID) {
                    if (instance.getComputationalDag().in_degree(original_id[idx][local_ID]) > 0) {
                        ++dag_real_size[idx];
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

        std::vector<Graph_t> newDagList;
        std::vector<std::vector<vertex_idx>> originalIdUpdated;

        for (unsigned idx = 0; idx < subDags.size(); ++idx) {
            const GraphT &dag = subDags[idx];
            if (!dag_is_too_large[idx]) {
                for (vertex_idx localId = 0; local_ID < dag.NumVertices(); ++local_ID) {
                    nodeToSubdagAndIndex[original_id[idx][local_ID]].first = static_cast<unsigned>(newDagList.size());
                }

                originalIdUpdated.push_back(original_id[idx]);
                newDagList.push_back(dag);
            } else {
                std::vector<unsigned> ilpAssignment;
                // unsigned newMin = dag_real_size[idx]/3, minPartitionSize); minimum condition removed - it can cause very strict bisections
                unsigned newMin = dag_real_size[idx] / 3;
                unsigned newMax = dag_real_size[idx] - newMin;

                // mark the source nodes of the original DAG
                std::vector<bool> isOriginalSource(dag.NumVertices());
                for (vertex_idx localId = 0; local_ID < dag.NumVertices(); ++local_ID) {
                    isOriginalSource[local_ID] = (instance.getComputationalDag().in_degree(original_id[idx][local_ID]) == 0);
                }

                // heuristic splitting
                std::vector<unsigned> heuristicAssignment = getTopologicalSplit(dag, {newMin, newMax}, is_original_source);
                unsigned heuristicCost = getSplitCost(dag, heuristic_assignment);
                unsigned ilpCost = UINT_MAX;

                // ILP-based splitting
                AcyclicPartitioningILP<GraphT> partitioner;
                partitioner.setTimeLimitSeconds(120);
                partitioner.setMinAndMaxSize({newMin, newMax});
                partitioner.setIsOriginalSource(is_original_source);
                partitioner.setNumberOfParts(2);    // note - if set to more than 2, ILP is MUCH more inefficient
                BspInstance partialInstance(dag, instance.getArchitecture(), instance.getNodeProcessorCompatibilityMatrix());
                RETURN_STATUS status = partitioner.computePartitioning(partial_instance, ILP_assignment);
                if (status == RETURN_STATUS::OSP_SUCCESS || status == RETURN_STATUS::BEST_FOUND) {
                    ilpCost = getSplitCost(dag, ILP_assignment);
                }

                std::vector<unsigned> assignment = ilpCost < heuristicCost ? ILP_assignment : heuristic_assignment;

                // split DAG according to labels
                std::vector<Graph_t> splitDags = create_induced_subgraphs<GraphT, GraphT>(dag, assignment);
                /*std::cout<<"SPLIT DONE: "<<dag.numberOfVertices()<<" nodes to ";
                for(auto sdag : splitDags)
                    std::cout<<sdag.numberOfVertices()<<" + ";
                std::cout<<std::endl;*/

                // update labels
                std::vector<vertex_idx> nodeIdxInNewSubDag(dag.NumVertices());
                std::vector<unsigned> nrNodesInNewSubDag(splitDags.size(), 0);
                for (vertex_idx localId = 0; local_ID < dag.NumVertices(); ++local_ID) {
                    nodeIdxInNewSubDag[local_ID] = nr_nodes_in_new_subDag[assignment[local_ID]];
                    ++nr_nodes_in_new_subDag[assignment[local_ID]];
                }

                for (auto next_dag : splitDags) {
                    original_id_updated.emplace_back(next_dag.NumVertices());
                }

                for (vertex_idx localId = 0; local_ID < dag.NumVertices(); ++local_ID) {
                    nodeToSubdagAndIndex[original_id[idx][local_ID]]
                        = {newDagList.size() + assignment[local_ID], node_idx_in_new_subDag[local_ID]};
                    originalIdUpdated[newDagList.size() + assignment[local_ID]][node_idx_in_new_subDag[local_ID]]
                        = original_id[idx][local_ID];
                }
                for (auto next_dag : splitDags) {
                    newDagList.push_back(next_dag);
                }
            }
        }

        subDags = newDagList;
        originalId = original_id_updated;
    }

    // output final cost
    std::vector<unsigned> finalAssignment(n);
    for (vertex_idx node = 0; node < n; ++node) {
        finalAssignment[node] = node_to_subdag_and_index[node].first;
    }
    std::cout << "Final cut cost of acyclic DAG divider is " << getSplitCost(instance.getComputationalDag(), final_assignment)
              << std::endl;

    return final_assignment;
}

template <typename GraphT>
std::vector<unsigned> AcyclicDagDivider<GraphT>::GetTopologicalSplit(const GraphT &g,
                                                                     std::pair<unsigned, unsigned> minAndMax,
                                                                     const std::vector<bool> &isOriginalSource) const {
    std::vector<unsigned> nodeToPart(g.NumVertices());

    std::vector<vertex_idx> topOrder = GetTopOrder(g);
    std::vector<unsigned> topOrderIdx(g.NumVertices());
    for (unsigned idx = 0; idx < g.NumVertices(); ++idx) {
        topOrderIdx[top_order[idx]] = idx;
    }

    std::vector<unsigned> lastNodeIdxInHyperedge(g.NumVertices());
    for (unsigned node = 0; node < g.NumVertices(); ++node) {
        lastNodeIdxInHyperedge[node] = top_order_idx[node];
        for (const auto &succ : g.children(node)) {
            lastNodeIdxInHyperedge[node] = std::max(last_node_idx_in_hyperedge[node], top_order_idx[succ]);
        }
    }

    unsigned index = 0;
    unsigned currentPartId = 0;

    unsigned nodesRemaining = static_cast<unsigned>(g.NumVertices());
    if (ignoreSourcesInSize_) {
        nodesRemaining = 0;
        for (unsigned node = 0; node < g.NumVertices(); ++node) {
            if (!is_original_source[node]) {
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
            if (!ignoreSourcesInSize_ || !is_original_source[end]) {
                ++newlyAddedNodes;
            }
        }

        while (end < g.NumVertices() && newlyAddedNodes < minAndMax.second) {
            unsigned extraCost = 0;

            // check the extra cut cost of the potential endpoint
            for (unsigned topOrderPos = index; topOrderPos <= end; ++topOrderPos) {
                vertex_idx node = top_order[topOrderPos];
                if (lastNodeIdxInHyperedge[node] > end) {
                    extraCost += g.VertexCommWeight(node);
                }

                for (const auto &pred : G.parents(node)) {
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

        for (vertex_idx idx = index; idx <= bestEnd; ++idx) {
            nodeToPart[top_order[idx]] = currentPartId;
            if (!ignoreSourcesInSize_ || !is_original_source[idx]) {
                --nodesRemaining;
            }
        }
        index = bestEnd + 1;
        ++currentPartId;
    }

    // remaining nodes go into last part
    for (vertex_idx idx = index; idx < g.NumVertices(); ++idx) {
        nodeToPart[top_order[idx]] = currentPartId;
    }

    return node_to_part;
}

template <typename GraphT>
v_commw_t<Graph_t> AcyclicDagDivider<GraphT>::GetSplitCost(const GraphT &g, const std::vector<unsigned> &nodeToPart) {
    v_commw_t<Graph_t> cost = 0;

    for (vertex_idx node = 0; node < g.NumVertices(); ++node) {
        std::set<unsigned> partsIncluded;
        partsIncluded.insert(node_to_part[node]);
        for (const auto &succ : G.children(node)) {
            parts_included.insert(node_to_part[succ]);
        }

        cost += static_cast<v_commw_t<Graph_t>>(parts_included.size() - 1) * g.VertexCommWeight(node);
    }

    return cost;
}

}    // namespace osp
