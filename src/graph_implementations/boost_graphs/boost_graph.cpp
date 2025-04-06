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

#include "graph_implementations/boost_graphs/boost_graph.hpp"

using namespace osp;

void boost_graph::updateNumberOfVertexTypes() {
    number_of_vertex_types = 0;
    for (const auto &v : vertices()) {
        if (vertex_type(v) >= number_of_vertex_types) {
            number_of_vertex_types = vertex_type(v) + 1;
        }
    }
}

std::pair<boost::detail::edge_desc_impl<boost::bidirectional_tag, std::size_t>, bool> boost_graph::add_edge(const boost_graph::vertex_idx &src, const boost_graph::vertex_idx &tar, int comm_weight) {

    const auto pair = boost::add_edge(src, tar, {comm_weight}, graph);

    number_of_vertex_types = std::max(number_of_vertex_types, 1u); // in case adding edges adds vertices
    return pair;
}