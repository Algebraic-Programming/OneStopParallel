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

#include <limits>

#include "osp/coarser/Coarser.hpp"
#include "osp/graph_algorithms/directed_graph_edge_desc_util.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"

namespace osp {

template <typename GraphTIn, typename GraphTOut>
class HdaggCoarser : public CoarserGenContractionMap<GraphTIn, GraphTOut> {
    static_assert(IsDirectedGraphEdgeDescV<GraphTIn>, "GraphTIn must satisfy the directed_graph edge desc concept");
    static_assert(HasHashableEdgeDescV<GraphTIn>, "GraphTIn must satisfy the has_hashable_edge_desc concept");
    static_assert(HasTypedVerticesV<GraphTIn>, "GraphTIn must have typed vertices");

  private:
    using VertexTypeIn = VertexIdxT<GraphTIn>;
    using VertexTypeOut = VertexIdxT<GraphTOut>;

  protected:
    VWorkwT<GraphTIn> workThreshold_ = std::numeric_limits<VWorkwT<GraphTIn>>::max();
    VMemwT<GraphTIn> memoryThreshold_ = std::numeric_limits<VMemwT<GraphTIn>>::max();
    VCommwT<GraphTIn> communicationThreshold_ = std::numeric_limits<VCommwT<GraphTIn>>::max();

    std::size_t superNodeSizeThreshold_ = std::numeric_limits<std::size_t>::max();

    // MemoryConstraintType memory_constraint_type = NONE;

    // internal data strauctures
    VMemwT<GraphTIn> currentMemory_ = 0;
    VWorkwT<GraphTIn> currentWork_ = 0;
    VCommwT<GraphTIn> currentCommunication_ = 0;
    VertexTypeOut currentSuperNodeIdx_ = 0;
    VTypeT<GraphTIn> currentVType_ = 0;

    void AddNewSuperNode(const GraphTIn &dagIn, VertexTypeIn node) {
        VMemwT<GraphTIn> nodeMem = dagIn.VertexMemWeight(node);

        currentMemory_ = nodeMem;
        currentWork_ = dagIn.VertexWorkWeight(node);
        currentCommunication_ = dagIn.VertexCommWeight(node);
        currentVType_ = dagIn.VertexType(node);
    }

  public:
    HdaggCoarser() {};

    virtual ~HdaggCoarser() = default;

    virtual std::string GetCoarserName() const override { return "hdagg_coarser"; };

    virtual std::vector<VertexIdxT<GraphTOut>> GenerateVertexContractionMap(const GraphTIn &dagIn) override {
        std::vector<bool> visited(dagIn.NumVertices(), false);
        std::vector<VertexTypeOut> reverseVertexMap(dagIn.NumVertices());

        std::vector<std::vector<VertexTypeIn>> vertexMap;

        auto edgeMask = LongEdgesInTriangles(dagIn);
        const auto edgeMastEnd = edgeMask.cend();

        for (const auto &sink : SinkVerticesView(dagIn)) {
            vertexMap.push_back(std::vector<VertexTypeIn>({sink}));
        }

        std::size_t partInd = 0;
        std::size_t partitionSize = vertexMap.size();
        while (partInd < partitionSize) {
            std::size_t vertInd = 0;
            std::size_t partSize = vertexMap[partInd].size();

            AddNewSuperNode(dagIn, vertexMap[partInd][vertInd]);

            while (vertInd < partSize) {
                const VertexTypeIn vert = vertexMap[partInd][vertInd];
                reverseVertexMap[vert] = currentSuperNodeIdx_;
                bool indegreeOne = true;

                for (const auto &inEdge : InEdges(vert, dagIn)) {
                    if (edgeMask.find(inEdge) != edgeMastEnd) {
                        continue;
                    }

                    unsigned count = 0;
                    for (const auto &outEdge : OutEdges(Source(inEdge, dagIn), dagIn)) {
                        if (edgeMask.find(outEdge) != edgeMastEnd) {
                            continue;
                        }

                        count++;
                        if (count > 1) {
                            indegreeOne = false;
                            break;
                        }
                    }

                    if (not indegreeOne) {
                        break;
                    }
                }

                if (indegreeOne) {
                    for (const auto &inEdge : InEdges(vert, dagIn)) {
                        if (edgeMask.find(inEdge) != edgeMastEnd) {
                            continue;
                        }

                        const auto &edgeSource = Source(inEdge, dagIn);

                        VMemwT<GraphTIn> nodeMem = dagIn.VertexMemWeight(edgeSource);

                        if (((currentMemory_ + nodeMem > memoryThreshold_)
                             || (currentWork_ + dagIn.VertexWorkWeight(edgeSource) > workThreshold_)
                             || (vertexMap[partInd].size() >= superNodeSizeThreshold_)
                             || (currentCommunication_ + dagIn.VertexCommWeight(edgeSource) > communicationThreshold_))
                            ||
                            // or node type changes
                            (currentVType_ != dagIn.VertexType(edgeSource))) {
                            if (!visited[edgeSource]) {
                                vertexMap.push_back(std::vector<VertexTypeIn>({edgeSource}));
                                partitionSize++;
                                visited[edgeSource] = true;
                            }

                        } else {
                            currentMemory_ += nodeMem;
                            currentWork_ += dagIn.VertexWorkWeight(edgeSource);
                            currentCommunication_ += dagIn.VertexCommWeight(edgeSource);

                            vertexMap[partInd].push_back(edgeSource);
                            partSize++;
                        }
                    }
                } else {
                    for (const auto &inEdge : InEdges(vert, dagIn)) {
                        if (edgeMask.find(inEdge) != edgeMastEnd) {
                            continue;
                        }

                        const auto &edgeSource = Source(inEdge, dagIn);

                        if (!visited[edgeSource]) {
                            vertexMap.push_back(std::vector<VertexTypeIn>({edgeSource}));
                            partitionSize++;
                            visited[edgeSource] = true;
                        }
                    }
                }
                vertInd++;
            }

            partInd++;
        }

        return reverseVertexMap;
    }

    inline void SetWorkThreshold(VWorkwT<GraphTIn> workThreshold) { workThreshold_ = workThreshold; }

    inline void SetMemoryThreshold(VMemwT<GraphTIn> memoryThreshold) { memoryThreshold_ = memoryThreshold; }

    inline void SetCommunicationThreshold(VCommwT<GraphTIn> communicationThreshold) {
        communicationThreshold_ = communicationThreshold;
    }

    inline void SetSuperNodeSizeThreshold(std::size_t superNodeSizeThreshold) {
        superNodeSizeThreshold_ = superNodeSizeThreshold;
    }
};

}    // namespace osp
