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

#include "scheduler/Coarsers/HDaggCoarser.hpp"

RETURN_STATUS HDaggCoarser::run_contractions() {
    std::cout   << "Coarsen Step: " << dag_history.size()
                << ", Number of nodes: " << dag_history.back()->numberOfVertices()
                << ", Number of edges: " << dag_history.back()->getComputationalDag().numberOfEdges()
                << ", Log ratio: " << std::log(dag_history.back()->getComputationalDag().numberOfEdges()) / std::log(dag_history.back()->numberOfVertices()) << std::endl;

    const ComputationalDag& graph = getOriginalInstance()->getComputationalDag();
    std::vector<std::vector<VertexType>> partition;
    std::vector<bool> visited(getOriginalInstance()->numberOfVertices(), false);
    
    std::unordered_set<EdgeType, EdgeType_hash> edge_mask = getOriginalInstance()->getComputationalDag().long_edges_in_triangles_parallel();
    
    for (const auto& sink : graph.sinkVertices()) {
        partition.push_back(std::vector<VertexType>({sink}));
    }

    size_t part_ind = 0;
    size_t partition_size = partition.size();
    while (part_ind < partition_size) {
        size_t vert_ind = 0;
        size_t part_size = partition[part_ind].size();
        while( vert_ind < part_size ) {
            VertexType vert = partition[part_ind][vert_ind];
            bool indegree_one = true;
            for (const auto& in_edge : graph.in_edges(vert)) {
                if ( edge_mask.find(in_edge) != edge_mask.cend() ) continue;
                unsigned count = 0;
                for (const auto& out_edge : graph.out_edges(in_edge.m_source)) {
                    if ( edge_mask.find(out_edge) != edge_mask.cend() ) continue; 
                    count++;
                                }
                if (count != 1) {
                    indegree_one = false;
                }
            }

            if (indegree_one) {
                for (const auto& in_edge : graph.in_edges(vert)) {
                    if ( edge_mask.find(in_edge) != edge_mask.cend() ) continue;
                    partition[part_ind].push_back(in_edge.m_source);
                    part_size++;
                }
            } else {
                for (const auto& in_edge : graph.in_edges(vert)) {
                    if ( edge_mask.find(in_edge) != edge_mask.cend() ) continue;
                    if (!visited[in_edge.m_source]) {
                        partition.push_back(std::vector<VertexType>({in_edge.m_source}));
                        partition_size++;
                        visited[in_edge.m_source] = true;
                    }
                }
            }
            vert_ind++;
        }
        part_ind++;
    }

    
    std::vector<std::unordered_set<VertexType>> partition_other_format(partition.size());
    for (size_t i = 0; i < partition.size(); i++) {
        for (auto vert : partition[i]) {
            partition_other_format[i].emplace(vert);
        }
    }


    add_contraction(partition_other_format);

    std::cout   << "Coarsen Step: " << dag_history.size()
                << ", Number of nodes: " << dag_history.back()->numberOfVertices()
                << ", Number of edges: " << dag_history.back()->getComputationalDag().numberOfEdges()
                << ", Log ratio: " << std::log(dag_history.back()->getComputationalDag().numberOfEdges()) / std::log(dag_history.back()->numberOfVertices()) << std::endl;

    return SUCCESS;
}
