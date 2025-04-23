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
#include <algorithm>

#include "model/ComputationalDag.hpp"

namespace dag_algorithms {

    ComputationalDag construct_dag_without_long_edges_in_triangles(const ComputationalDag &dag) {

        ComputationalDag new_dag(dag);

        std::unordered_set<EdgeType, EdgeType_hash> deleted_edges;

        for (const auto &vertex : dag.getGraph().vertex_set()) {
    
            std::unordered_set<VertexType> children_set;
    
            for (const auto &v : dag.children(vertex)) {
                children_set.emplace(v);
            }
    
            for (const auto &edge : boost::extensions::make_source_iterator_range(boost::out_edges(vertex, dag.getGraph()))) {
    
                const auto &child = boost::target(edge, dag.getGraph());
    
                for (const auto &parent :
                     boost::extensions::make_source_iterator_range(boost::inv_adjacent_vertices(child, dag.getGraph()))) {
    
                    // const auto &pair = boost::edge(vertex, parent, graph);
    
                    if (children_set.find(parent) != children_set.cend()) {
                        deleted_edges.emplace(edge);
                        break;
                    }
                }
            }
        }
    
        for (const auto &edge : deleted_edges) {
            new_dag.getGraph().remove_edge(edge);
        }

        return new_dag;
       
    }



} // namespace dag_algorithms