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

#include "model/BspInstance_csr.hpp"
#include "model/ComputationalDag.hpp"
#include <unordered_set>

#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/filtered_graph.hpp>

// struct approx_transitive_edge_reduction_csr {

//     std::unordered_set<csr_edge> deleted_edges;

//     approx_transitive_edge_reduction_csr() {}
//     approx_transitive_edge_reduction_csr(const csr_graph &graph) {

//         for (const auto &vertex : boost::make_iterator_range(boost::vertices(graph))) {

//             std::unordered_set<VertexType> children_set;

//             for (const auto &out_edge : boost::make_iterator_range(boost::out_edges(vertex, graph))) {
//                 // const VertexType v = ;

//                 // for (const auto &v :
//                 //      boost::extensions::make_source_iterator_range(boost::adjacent_vertices(vertex, graph))) {
//                 children_set.emplace(boost::target(out_edge, graph));
//             }

//             for (const auto &edge : boost::make_iterator_range(boost::out_edges(vertex, graph))) {
//                 // for (const auto &edge : boost::extensions::make_source_iterator_range(boost::out_edges(vertex,
//                 // graph))) {

//                 const auto &child = boost::target(edge, graph);

//                 for (const auto &in_edge : boost::make_iterator_range(boost::in_edges(child, graph))) {
//                     const VertexType parent = boost::source(edge, graph);

//                     // for (const auto &parent :
//                     //      boost::extensions::make_source_iterator_range(boost::inv_adjacent_vertices(child, graph))) {

//                     const auto &pair = boost::edge(vertex, parent, graph);

//                     if (children_set.find(parent) != children_set.cend()) {
//                         deleted_edges.emplace(edge);
//                     }
//                 }
//             }
//         }
//     }

//     template<typename Edge>
//     bool operator()(const Edge &e) const {
//         return deleted_edges.find(e) == deleted_edges.end();
//     }
// };
struct approx_transitive_edge_reduction {

    std::unordered_set<EdgeType, EdgeType_hash> deleted_edges;

    approx_transitive_edge_reduction() {}
    approx_transitive_edge_reduction(const GraphType &graph) {

        for (const auto &vertex : graph.vertex_set()) {

            std::unordered_set<VertexType> children_set;

            for (const auto &v :
                 boost::extensions::make_source_iterator_range(boost::adjacent_vertices(vertex, graph))) {
                children_set.emplace(v);
            }

            for (const auto &edge : boost::extensions::make_source_iterator_range(boost::out_edges(vertex, graph))) {

                const auto &child = boost::target(edge, graph);

                for (const auto &parent :
                     boost::extensions::make_source_iterator_range(boost::inv_adjacent_vertices(child, graph))) {

                    const auto &pair = boost::edge(vertex, parent, graph);

                    if (children_set.find(parent) != children_set.cend()) {
                        deleted_edges.emplace(edge);
                    }
                }
            }
        }
    }

    template<typename Edge>
    bool operator()(const Edge &e) const {
        return deleted_edges.find(e) == deleted_edges.end();
    }
};
