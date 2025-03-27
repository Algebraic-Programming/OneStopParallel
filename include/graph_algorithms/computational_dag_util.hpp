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