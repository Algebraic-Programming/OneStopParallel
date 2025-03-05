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

#include "coarser/top_order/top_order.hpp"

RETURN_STATUS top_order::coarseDag(const ComputationalDag &dag_in, ComputationalDag &dag_out,
                                   std::vector<std::vector<VertexType>> &vertex_map) {

    assert(dag_in.numberOfVertices() == top_ordering.size());
    assert(dag_out.numberOfVertices() == 0);

    reverse_vertex_map.resize(dag_in.numberOfVertices(), std::numeric_limits<VertexType>::max());

    vertex_map.clear();
    vertex_map.push_back(std::vector<VertexType>({top_ordering[0]}));

    add_new_super_node(dag_in, dag_out, top_ordering[0]);
    reverse_vertex_map[top_ordering[0]] = current_super_node_idx;

    for (size_t i = 1; i < top_ordering.size(); i++) {

        const auto v = top_ordering[i];

        int node_mem = dag_in.nodeMemoryWeight(v);

        if (memory_constraint_type == LOCAL_INC_EDGES_2) {
    
            if (not dag_in.isSource(v)) {
                node_mem = 0;
            }
        }

        // start new super node if thresholds are exceeded
        if (((current_memory + node_mem > memory_threshold) ||
             (current_work + dag_in.nodeWorkWeight(v) > work_threshold) ||
             (vertex_map.back().size() >= super_node_size_threshold) ||
             (current_communication + dag_in.nodeCommunicationWeight(v) > communication_threshold)) ||
            // or prev node high out degree
            (dag_in.numberOfChildren(top_ordering[i - 1]) > degree_threshold) ||
            // or node type changes
            (dag_out.nodeType(current_super_node_idx) != dag_in.nodeType(v))) {

            finish_super_node_add_edges(dag_in, dag_out, vertex_map.back());
            vertex_map.push_back(std::vector<VertexType>({v}));
            add_new_super_node(dag_in, dag_out, v);

        } else { // grow current super node

            current_memory += node_mem;
            current_work += dag_in.nodeWorkWeight(v);
            current_communication += dag_in.nodeCommunicationWeight(v);

            vertex_map.back().push_back(v);
 
        }

        reverse_vertex_map[v] = current_super_node_idx;
    }

    if (!vertex_map.back().empty()) {
        finish_super_node_add_edges(dag_in, dag_out, vertex_map.back());
    }

    return SUCCESS;
}

void top_order::finish_super_node_add_edges(const ComputationalDag &dag_in, ComputationalDag &dag_out,
                                            const std::vector<VertexType> &nodes) {

    dag_out.setNodeMemoryWeight(current_super_node_idx, current_memory);
    dag_out.setNodeWorkWeight(current_super_node_idx, current_work);
    dag_out.setNodeCommunicationWeight(current_super_node_idx, current_communication);

    for (const auto &node : nodes) {

        for (const auto &in_edge : dag_in.in_edges(node)) {

            const VertexType parent_rev = reverse_vertex_map[in_edge.m_source];
            if (parent_rev != current_super_node_idx && parent_rev != std::numeric_limits<VertexType>::max()) {

                auto pair = boost::edge(parent_rev, current_super_node_idx, dag_out.getGraph());
                if (pair.second) {
                    dag_out.setEdgeCommunicationWeight(pair.first, dag_out.edgeCommunicationWeight(pair.first) +
                                                                       dag_in.edgeCommunicationWeight(in_edge));
                } else {
                    dag_out.addEdge(parent_rev, current_super_node_idx, dag_in.edgeCommunicationWeight(in_edge));

                }
            } 
        }
    }
}

void top_order::add_new_super_node(const ComputationalDag &dag_in, ComputationalDag &dag_out, VertexType node) {

    int node_mem = dag_in.nodeMemoryWeight(node);

    if (memory_constraint_type == LOCAL_INC_EDGES_2) {

        if (not dag_in.isSource(node)) {
            node_mem = 0;
        }
    }

    current_memory = node_mem;
    current_work = dag_in.nodeWorkWeight(node);
    current_communication = dag_in.nodeCommunicationWeight(node);
    current_super_node_idx = dag_out.addVertex(current_work, current_communication, current_memory, dag_in.nodeType(node));

}