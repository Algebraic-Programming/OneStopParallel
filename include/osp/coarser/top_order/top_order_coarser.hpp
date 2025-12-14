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

#include "osp/coarser/Coarser.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"

namespace osp {

template <typename GraphTIn, typename GraphTOut, std::vector<VertexIdxT<GraphTIn>> (*topSortFunc)(const GraphTIn &)>
class TopOrderCoarser : public Coarser<GraphTIn, GraphTOut> {
  private:
    using VertexType = VertexIdxT<GraphTIn>;

    // parameters
    VWorkwT<GraphTIn> workThreshold_ = std::numeric_limits<VWorkwT<GraphTIn>>::max();
    VMemwT<GraphTIn> memoryThreshold_ = std::numeric_limits<VMemwT<GraphTIn>>::max();
    VCommwT<GraphTIn> communicationThreshold_ = std::numeric_limits<VCommwT<GraphTIn>>::max();
    unsigned degreeThreshold_ = std::numeric_limits<unsigned>::max();
    unsigned nodeDistThreshold_ = std::numeric_limits<unsigned>::max();
    VertexType superNodeSizeThreshold_ = std::numeric_limits<VertexType>::max();

    // internal data strauctures
    VMemwT<GraphTIn> currentMemory_ = 0;
    VWorkwT<GraphTIn> currentWork_ = 0;
    VCommwT<GraphTIn> currentCommunication_ = 0;
    VertexType currentSuperNodeIdx_ = 0;

    void FinishSuperNodeAddEdges(const GraphTIn &dagIn,
                                 GraphTOut &dagOut,
                                 const std::vector<VertexType> &nodes,
                                 std::vector<VertexIdxT<GraphTOut>> &reverseVertexMap) {
        dagOut.SetVertexMemWeight(currentSuperNodeIdx_, currentMemory_);
        dagOut.SetVertexWorkWeight(currentSuperNodeIdx_, currentWork_);
        dagOut.SetVertexCommWeight(currentSuperNodeIdx_, currentCommunication_);

        for (const auto &node : nodes) {
            if constexpr (HasEdgeWeightsV<GraphTIn> && HasEdgeWeightsV<GraphTOut>) {
                for (const auto &inEdge : InEdges(node, dagIn)) {
                    const VertexType parentRev = reverseVertexMap[Source(inEdge, dagIn)];
                    if (parentRev != currentSuperNodeIdx_ && parentRev != std::numeric_limits<VertexType>::max()) {
                        auto pair = EdgeDesc(parentRev, currentSuperNodeIdx_, dagOut);
                        if (pair.second) {
                            dagOut.SetEdgeCommWeight(pair.first, dagOut.EdgeCommWeight(pair.first) + dagIn.EdgeCommWeight(inEdge));
                        } else {
                            dagOut.AddEdge(parentRev, currentSuperNodeIdx_, dagIn.EdgeCommWeight(inEdge));
                        }
                    }
                }
            } else {
                for (const auto &parent : dagIn.Parents(node)) {
                    const VertexType parentRev = reverseVertexMap[parent];
                    if (parentRev != currentSuperNodeIdx_ && parentRev != std::numeric_limits<VertexType>::max()) {
                        if (not Edge(parentRev, currentSuperNodeIdx_, dagOut)) {
                            dagOut.AddEdge(parentRev, currentSuperNodeIdx_);
                        }
                    }
                }
            }
        }
    }

    void AddNewSuperNode(const GraphTIn &dagIn, GraphTOut &dagOut, VertexType node) {
        // int node_mem = dag_in.nodeMemoryWeight(node);

        // if (memory_constraint_type == LOCAL_INC_EDGES_2) {

        //     if (not dag_in.isSource(node)) {
        //         node_mem = 0;
        //     }
        // }

        currentMemory_ = dagIn.VertexMemWeight(node);
        currentWork_ = dagIn.VertexWorkWeight(node);
        currentCommunication_ = dagIn.VertexCommWeight(node);

        if constexpr (IsComputationalDagTypedVerticesV<GraphTIn> && IsComputationalDagTypedVerticesV<GraphTOut>) {
            currentSuperNodeIdx_ = dagOut.AddVertex(currentWork_, currentCommunication_, currentMemory_, dagIn.VertexType(node));
        } else {
            currentSuperNodeIdx_ = dagOut.AddVertex(currentWork_, currentCommunication_, currentMemory_);
        }
    }

  public:
    TopOrderCoarser() {};
    virtual ~TopOrderCoarser() = default;

    inline void SetDegreeThreshold(unsigned degreeThreshold) { degreeThreshold_ = degreeThreshold; }

    inline void SetWorkThreshold(VWorkwT<GraphTIn> workThreshold) { workThreshold_ = workThreshold; }

    inline void SetMemoryThreshold(VMemwT<GraphTIn> memoryThreshold) { memoryThreshold_ = memoryThreshold; }

    inline void SetCommunicationThreshold(VCommwT<GraphTIn> communicationThreshold) {
        communicationThreshold_ = communicationThreshold;
    }

    inline void SetSuperNodeSizeThreshold(VertexType superNodeSizeThreshold) { superNodeSizeThreshold_ = superNodeSizeThreshold; }

    inline void SetNodeDistThreshold(unsigned nodeDistThreshold) { nodeDistThreshold_ = nodeDistThreshold; }

    // inline void set_memory_constraint_type(MemoryConstraintType memory_constraint_type_) { memory_constraint_type =
    // memory_constraint_type_; }

    virtual std::string getCoarserName() const override { return "top_order_coarser"; };

    virtual bool CoarsenDag(const GraphTIn &dagIn, GraphTOut &dagOut, std::vector<VertexIdxT<GraphTOut>> &reverseVertexMap) override {
        assert(dagOut.NumVertices() == 0);
        if (dagIn.NumVertices() == 0) {
            reverseVertexMap = std::vector<VertexIdxT<GraphTOut>>();
            return true;
        }

        std::vector<VertexType> topOrdering = topSortFunc(dagIn);

        std::vector<unsigned> sourceNodeDist = GetTopNodeDistance(dagIn);

        reverseVertexMap.resize(dagIn.NumVertices(), std::numeric_limits<VertexType>::max());

        std::vector<std::vector<VertexType>> vertexMap;
        vertexMap.push_back(std::vector<VertexType>({topOrdering[0]}));

        AddNewSuperNode(dagIn, dagOut, topOrdering[0]);
        reverseVertexMap[topOrdering[0]] = currentSuperNodeIdx_;

        for (size_t i = 1; i < topOrdering.size(); i++) {
            const auto v = topOrdering[i];

            // int node_mem = dag_in.VertexMemWeight(v);

            // if (memory_constraint_type == LOCAL_INC_EDGES_2) {

            //     if (not dag_in.isSource(v)) {
            //         node_mem = 0;
            //     }
            // }

            const unsigned dist = sourceNodeDist[v] - sourceNodeDist[topOrdering[i - 1]];

            // start new super node if thresholds are exceeded
            if (((currentMemory_ + dagIn.VertexMemWeight(v) > memoryThreshold_)
                 || (currentWork_ + dagIn.VertexWorkWeight(v) > workThreshold_)
                 || (vertexMap.back().size() >= superNodeSizeThreshold_)
                 || (currentCommunication_ + dagIn.VertexCommWeight(v) > communicationThreshold_))
                || (dist > nodeDistThreshold_) ||
                // or prev node high out degree
                (dagIn.OutDegree(topOrdering[i - 1]) > degreeThreshold_)) {
                FinishSuperNodeAddEdges(dagIn, dagOut, vertexMap.back(), reverseVertexMap);
                vertexMap.push_back(std::vector<VertexType>({v}));
                AddNewSuperNode(dagIn, dagOut, v);

            } else {    // grow current super node

                if constexpr (IsComputationalDagTypedVerticesV<GraphTIn> && IsComputationalDagTypedVerticesV<GraphTOut>) {
                    if (dagOut.VertexType(currentSuperNodeIdx_) != dagIn.VertexType(v)) {
                        FinishSuperNodeAddEdges(dagIn, dagOut, vertexMap.back(), reverseVertexMap);
                        vertexMap.push_back(std::vector<VertexType>({v}));
                        AddNewSuperNode(dagIn, dagOut, v);

                    } else {
                        currentMemory_ += dagIn.VertexMemWeight(v);
                        currentWork_ += dagIn.VertexWorkWeight(v);
                        currentCommunication_ += dagIn.VertexCommWeight(v);

                        vertexMap.back().push_back(v);
                    }

                } else {
                    currentMemory_ += dagIn.VertexMemWeight(v);
                    currentWork_ += dagIn.VertexWorkWeight(v);
                    currentCommunication_ += dagIn.VertexCommWeight(v);

                    vertexMap.back().push_back(v);
                }
            }

            reverseVertexMap[v] = currentSuperNodeIdx_;
        }

        if (!vertexMap.back().empty()) {
            FinishSuperNodeAddEdges(dagIn, dagOut, vertexMap.back(), reverseVertexMap);
        }

        return true;
    }
};

}    // namespace osp
