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

#include "concepts/computational_dag_concept.hpp"

namespace osp {

template<typename Graph_t>
int max_memory_weight(const Graph_t &graph) {
    int max_memory_weight = 0;

    for (const auto &v : graph.vertices()) {
        max_memory_weight = std::max(max_memory_weight, graph.vertex_memory_weight(v));
    }
    return max_memory_weight;
}

template<typename Graph_t>
int max_memory_weight(unsigned nodeType_, const Graph_t &graph) {
    int max_memory_weight = 0;

    for (const auto &node : graph.vertices()) {
        if (graph.node_type(node) == nodeType_) {
            max_memory_weight = std::max(max_memory_weight, graph.vertex_memory_weight(node));
        }
    }
    return max_memory_weight;
}

} // namespace osp