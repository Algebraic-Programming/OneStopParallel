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
    static_assert(IsDirectedGraphEdgeDescV<Graph_t_in>, "Graph_t_in must satisfy the directed_graph edge desc concept");
    static_assert(has_hashable_edge_desc_v<Graph_t_in>, "Graph_t_in must satisfy the has_hashable_edge_desc concept");
    static_assert(HasTypedVerticesV<Graph_t_in>, "Graph_t_in must have typed vertices");

  private:
    using VertexType_in = vertex_idx_t<Graph_t_in>;
    using VertexType_out = vertex_idx_t<Graph_t_out>;

  protected:
    VWorkwT<Graph_t_in> workThreshold_ = std::numeric_limits<VWorkwT<Graph_t_in>>::max();
    VMemwT<Graph_t_in> memoryThreshold_ = std::numeric_limits<VMemwT<Graph_t_in>>::max();
    VCommwT<Graph_t_in> communicationThreshold_ = std::numeric_limits<VCommwT<Graph_t_in>>::max();

    std::size_t superNodeSizeThreshold_ = std::numeric_limits<std::size_t>::max();

    // MemoryConstraintType memory_constraint_type = NONE;

    // internal data strauctures
    VMemwT<Graph_t_in> currentMemory_ = 0;
    VWorkwT<Graph_t_in> currentWork_ = 0;
    VCommwT<Graph_t_in> currentCommunication_ = 0;
    VertexType_out currentSuperNodeIdx_ = 0;
    v_type_t<Graph_t_in> currentVType_ = 0;

    void AddNewSuperNode(const GraphTIn &dagIn, VertexType_in node) {
        VMemwT<Graph_t_in> nodeMem = dagIn.VertexMemWeight(node);

        current_memory = node_mem;
        current_work = dag_in.VertexWorkWeight(node);
        current_communication = dag_in.VertexCommWeight(node);
        current_v_type = dag_in.VertexType(node);
    }

  public:
    HdaggCoarser() {};

    virtual ~HdaggCoarser() = default;

    virtual std::string getCoarserName() const override { return "hdagg_coarser"; };

    virtual std::vector<vertex_idx_t<Graph_t_out>> generate_vertex_contraction_map(const GraphTIn &dagIn) override {
        std::vector<bool> visited(dagIn.NumVertices(), false);
        std::vector<VertexType_out> reverseVertexMap(dagIn.NumVertices());

        std::vector<std::vector<VertexType_in>> vertexMap;

        auto edgeMask = long_edges_in_triangles(dagIn);
        const auto edgeMastEnd = edgeMask.cend();

        for (const auto &sink : sink_vertices_view(dagIn)) {
            vertex_map.push_back(std::vector<VertexType_in>({sink}));
        }

        std::size_t partInd = 0;
        std::size_t partitionSize = vertex_map.size();
        while (partInd < partitionSize) {
            std::size_t vertInd = 0;
            std::size_t partSize = vertex_map[partInd].size();

            add_new_super_node(dag_in, vertex_map[part_ind][vert_ind]);

            while (vertInd < partSize) {
                const VertexType_in vert = vertex_map[partInd][vertInd];
                reverse_vertex_map[vert] = current_super_node_idx;
                bool indegreeOne = true;

                for (const auto &in_edge : InEdges(vert, dag_in)) {
                    if (edge_mask.find(in_edge) != edge_mast_end) {
                        continue;
                    }

                    unsigned count = 0;
                    for (const auto &out_edge : OutEdges(Source(in_edge, dag_in), dag_in)) {
                        if (edge_mask.find(out_edge) != edge_mast_end) {
                            continue;
                        }

                        count++;
                        if (count > 1) {
                            indegree_one = false;
                            break;
                        }
                    }

                    if (not indegree_one) {
                        break;
                    }
                }

                if (indegreeOne) {
                    for (const auto &in_edge : InEdges(vert, dag_in)) {
                        if (edge_mask.find(in_edge) != edge_mast_end) {
                            continue;
                        }

                        const auto &edge_source = Source(in_edge, dag_in);

                        VMemwT<Graph_t_in> node_mem = dag_in.VertexMemWeight(edge_source);

                        if (((current_memory + node_mem > memory_threshold)
                             || (current_work + dag_in.VertexWorkWeight(edge_source) > work_threshold)
                             || (vertex_map[part_ind].size() >= super_node_size_threshold)
                             || (current_communication + dag_in.VertexCommWeight(edge_source) > communication_threshold))
                            ||
                            // or node type changes
                            (current_v_type != dag_in.VertexType(edge_source))) {
                            if (!visited[edge_source]) {
                                vertex_map.push_back(std::vector<VertexType_in>({edge_source}));
                                partition_size++;
                                visited[edge_source] = true;
                            }

                        } else {
                            current_memory += node_mem;
                            current_work += dag_in.VertexWorkWeight(edge_source);
                            current_communication += dag_in.VertexCommWeight(edge_source);

                            vertex_map[part_ind].push_back(edge_source);
                            part_size++;
                        }
                    }
                } else {
                    for (const auto &in_edge : InEdges(vert, dag_in)) {
                        if (edge_mask.find(in_edge) != edge_mast_end) {
                            continue;
                        }

                        const auto &edge_source = Source(in_edge, dag_in);

                        if (!visited[edge_source]) {
                            vertex_map.push_back(std::vector<VertexType_in>({edge_source}));
                            partition_size++;
                            visited[edge_source] = true;
                        }
                    }
                }
                vertInd++;
            }

            partInd++;
        }

        return reverse_vertex_map;
    }

    inline void SetWorkThreshold(VWorkwT<Graph_t_in> workThreshold) { work_threshold = work_threshold_; }

    inline void SetMemoryThreshold(VMemwT<Graph_t_in> memoryThreshold) { memory_threshold = memory_threshold_; }

    inline void SetCommunicationThreshold(VCommwT<Graph_t_in> communicationThreshold) {
        communication_threshold = communication_threshold_;
    }

    inline void SetSuperNodeSizeThreshold(std::size_t superNodeSizeThreshold) {
        superNodeSizeThreshold_ = superNodeSizeThreshold;
    }
};

}    // namespace osp
