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

template <typename GraphTIn, typename GraphTOut, std::vector<vertex_idx_t<Graph_t_in>> (*topSortFunc)(const GraphTIn &)>
class TopOrderCoarser : public Coarser<GraphTIn, GraphTOut> {
  private:
    using VertexType = vertex_idx_t<Graph_t_in>;

    // parameters
    v_workw_t<Graph_t_in> workThreshold_ = std::numeric_limits<v_workw_t<Graph_t_in>>::max();
    v_memw_t<Graph_t_in> memoryThreshold_ = std::numeric_limits<v_memw_t<Graph_t_in>>::max();
    v_commw_t<Graph_t_in> communicationThreshold_ = std::numeric_limits<v_commw_t<Graph_t_in>>::max();
    unsigned degreeThreshold_ = std::numeric_limits<unsigned>::max();
    unsigned nodeDistThreshold_ = std::numeric_limits<unsigned>::max();
    VertexType superNodeSizeThreshold_ = std::numeric_limits<VertexType>::max();

    // internal data strauctures
    v_memw_t<Graph_t_in> currentMemory_ = 0;
    v_workw_t<Graph_t_in> currentWork_ = 0;
    v_commw_t<Graph_t_in> currentCommunication_ = 0;
    VertexType currentSuperNodeIdx_ = 0;

    void FinishSuperNodeAddEdges(const GraphTIn &dagIn,
                                 GraphTOut &dagOut,
                                 const std::vector<VertexType> &nodes,
                                 std::vector<vertex_idx_t<Graph_t_out>> &reverseVertexMap) {
        dag_out.SetVertexMemWeight(current_super_node_idx, current_memory);
        dag_out.SetVertexWorkWeight(current_super_node_idx, current_work);
        dag_out.SetVertexCommWeight(current_super_node_idx, current_communication);

        for (const auto &node : nodes) {
            if constexpr (HasEdgeWeightsV<Graph_t_in> && HasEdgeWeightsV<Graph_t_out>) {
                for (const auto &in_edge : in_edges(node, dag_in)) {
                    const VertexType parent_rev = reverse_vertex_map[source(in_edge, dag_in)];
                    if (parent_rev != current_super_node_idx && parent_rev != std::numeric_limits<VertexType>::max()) {
                        auto pair = edge_desc(parent_rev, current_super_node_idx, dag_out);
                        if (pair.second) {
                            dag_out.SetEdgeCommWeight(pair.first,
                                                      dag_out.EdgeCommWeight(pair.first) + dag_in.EdgeCommWeight(in_edge));
                        } else {
                            dag_out.add_edge(parent_rev, current_super_node_idx, dag_in.EdgeCommWeight(in_edge));
                        }
                    }
                }
            } else {
                for (const auto &parent : dag_in.parents(node)) {
                    const VertexType parent_rev = reverse_vertex_map[parent];
                    if (parent_rev != current_super_node_idx && parent_rev != std::numeric_limits<VertexType>::max()) {
                        if (not edge(parent_rev, current_super_node_idx, dag_out)) {
                            dag_out.add_edge(parent_rev, current_super_node_idx);
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

        current_memory = dag_in.VertexMemWeight(node);
        current_work = dag_in.VertexWorkWeight(node);
        current_communication = dag_in.VertexCommWeight(node);

        if constexpr (IsComputationalDagTypedVerticesV<Graph_t_in> && IsComputationalDagTypedVerticesV<Graph_t_out>) {
            current_super_node_idx
                = dag_out.add_vertex(current_work, current_communication, current_memory, dag_in.VertexType(node));
        } else {
            current_super_node_idx = dag_out.add_vertex(current_work, current_communication, current_memory);
        }
    }

  public:
    TopOrderCoarser() {};
    virtual ~TopOrderCoarser() = default;

    inline void SetDegreeThreshold(unsigned degreeThreshold) { degreeThreshold_ = degreeThreshold; }

    inline void SetWorkThreshold(v_workw_t<Graph_t_in> workThreshold) { work_threshold = work_threshold_; }

    inline void SetMemoryThreshold(v_memw_t<Graph_t_in> memoryThreshold) { memory_threshold = memory_threshold_; }

    inline void SetCommunicationThreshold(v_commw_t<Graph_t_in> communicationThreshold) {
        communication_threshold = communication_threshold_;
    }

    inline void SetSuperNodeSizeThreshold(VertexType superNodeSizeThreshold) {
        super_node_size_threshold = super_node_size_threshold_;
    }

    inline void SetNodeDistThreshold(unsigned nodeDistThreshold) { nodeDistThreshold_ = nodeDistThreshold; }

    // inline void set_memory_constraint_type(MEMORY_CONSTRAINT_TYPE memory_constraint_type_) { memory_constraint_type =
    // memory_constraint_type_; }

    virtual std::string getCoarserName() const override { return "top_order_coarser"; };

    virtual bool coarsenDag(const GraphTIn &dagIn,
                            GraphTOut &dagOut,
                            std::vector<vertex_idx_t<Graph_t_out>> &reverseVertexMap) override {
        assert(dagOut.NumVertices() == 0);
        if (dagIn.NumVertices() == 0) {
            reverse_vertex_map = std::vector<vertex_idx_t<Graph_t_out>>();
            return true;
        }

        std::vector<VertexType> topOrdering = topSortFunc(dagIn);

        std::vector<unsigned> sourceNodeDist = get_top_node_distance(dagIn);

        reverse_vertex_map.resize(dag_in.NumVertices(), std::numeric_limits<VertexType>::max());

        std::vector<std::vector<VertexType>> vertexMap;
        vertex_map.push_back(std::vector<VertexType>({top_ordering[0]}));

        add_new_super_node(dag_in, dag_out, top_ordering[0]);
        reverse_vertex_map[top_ordering[0]] = current_super_node_idx;

        for (size_t i = 1; i < topOrdering.size(); i++) {
            const auto v = top_ordering[i];

            // int node_mem = dag_in.VertexMemWeight(v);

            // if (memory_constraint_type == LOCAL_INC_EDGES_2) {

            //     if (not dag_in.isSource(v)) {
            //         node_mem = 0;
            //     }
            // }

            const unsigned dist = sourceNodeDist[v] - sourceNodeDist[top_ordering[i - 1]];

            // start new super node if thresholds are exceeded
            if (((current_memory + dag_in.VertexMemWeight(v) > memory_threshold)
                 || (current_work + dag_in.VertexWorkWeight(v) > work_threshold)
                 || (vertex_map.back().size() >= super_node_size_threshold)
                 || (current_communication + dag_in.VertexCommWeight(v) > communication_threshold))
                || (dist > node_dist_threshold) ||
                // or prev node high out degree
                (dag_in.out_degree(top_ordering[i - 1]) > degree_threshold)) {
                finish_super_node_add_edges(dag_in, dag_out, vertex_map.back(), reverse_vertex_map);
                vertex_map.push_back(std::vector<VertexType>({v}));
                add_new_super_node(dag_in, dag_out, v);

            } else {    // grow current super node

                if constexpr (IsComputationalDagTypedVerticesV<Graph_t_in> && IsComputationalDagTypedVerticesV<Graph_t_out>) {
                    if (dag_out.VertexType(current_super_node_idx) != dag_in.VertexType(v)) {
                        finish_super_node_add_edges(dag_in, dag_out, vertex_map.back(), reverse_vertex_map);
                        vertex_map.push_back(std::vector<VertexType>({v}));
                        add_new_super_node(dag_in, dag_out, v);

                    } else {
                        current_memory += dag_in.VertexMemWeight(v);
                        current_work += dag_in.VertexWorkWeight(v);
                        current_communication += dag_in.VertexCommWeight(v);

                        vertexMap.back().push_back(v);
                    }

                } else {
                    current_memory += dag_in.VertexMemWeight(v);
                    current_work += dag_in.VertexWorkWeight(v);
                    current_communication += dag_in.VertexCommWeight(v);

                    vertexMap.back().push_back(v);
                }
            }

            reverse_vertex_map[v] = current_super_node_idx;
        }

        if (!vertex_map.back().empty()) {
            finish_super_node_add_edges(dag_in, dag_out, vertex_map.back(), reverse_vertex_map);
        }

        return true;
    }
};

}    // namespace osp
