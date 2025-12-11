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

#include <cmath>
#include <numeric>

#include "osp/concepts/constructable_computational_dag_concept.hpp"

namespace osp {

std::vector<std::string> tiny_spaa_graphs() {
    return {"data/spaa/tiny/instance_bicgstab.hdag",
            "data/spaa/tiny/instance_CG_N2_K2_nzP0d75.hdag",
            "data/spaa/tiny/instance_CG_N3_K1_nzP0d5.hdag",
            "data/spaa/tiny/instance_CG_N4_K1_nzP0d35.hdag",
            "data/spaa/tiny/instance_exp_N4_K2_nzP0d5.hdag",
            "data/spaa/tiny/instance_exp_N5_K3_nzP0d4.hdag",
            "data/spaa/tiny/instance_exp_N6_K4_nzP0d25.hdag",
            "data/spaa/tiny/instance_k-means.hdag",
            "data/spaa/tiny/instance_k-NN_3_gyro_m.hdag",
            "data/spaa/tiny/instance_kNN_N4_K3_nzP0d5.hdag",
            "data/spaa/tiny/instance_kNN_N5_K3_nzP0d3.hdag",
            "data/spaa/tiny/instance_kNN_N6_K4_nzP0d2.hdag",
            "data/spaa/tiny/instance_pregel.hdag",
            "data/spaa/tiny/instance_spmv_N6_nzP0d4.hdag",
            "data/spaa/tiny/instance_spmv_N7_nzP0d35.hdag",
            "data/spaa/tiny/instance_spmv_N10_nzP0d25.hdag"};
}

std::vector<std::string> large_spaa_graphs() {
    return {"data/spaa/large/instance_exp_N50_K12_nzP0d15.hdag",
            "data/spaa/large/instance_CG_N24_K22_nzP0d2.hdag",
            "data/spaa/large/instance_kNN_N45_K15_nzP0d16.hdag",
            "data/spaa/large/instance_spmv_N120_nzP0d18.hdag"};
}

std::vector<std::string> test_graphs() {
    return {"data/spaa/tiny/instance_k-means.hdag",
            "data/spaa/tiny/instance_bicgstab.hdag",
            "data/spaa/tiny/instance_CG_N3_K1_nzP0d5.hdag"};
}

/**
 * @brief Constructs a DAG with multiple identical, parallel pipelines.
 *
 * Example: num_pipelines=2, pipeline_len=3 creates:
 * 0 -> 1 -> 2
 * 3 -> 4 -> 5
 *
 * Nodes at the same stage in different pipelines are identical (same work weight).
 *
 * @tparam Graph_t The graph type to construct, must be a constructable computational DAG.
 * @param num_pipelines The number of parallel pipelines.
 * @param pipeline_len The length of each pipeline.
 * @return A Graph_t object representing the DAG.
 */
template <typename Graph_t>
inline Graph_t construct_multi_pipeline_dag(unsigned num_pipelines, unsigned pipeline_len) {
    static_assert(is_constructable_cdag_v<Graph_t>, "Graph_t must be a constructable computational DAG");
    Graph_t dag;
    if (num_pipelines == 0 || pipeline_len == 0) { return dag; }

    for (unsigned i = 0; i < num_pipelines; ++i) {
        for (unsigned j = 0; j < pipeline_len; ++j) {
            // Nodes at the same stage 'j' have the same work weight
            dag.add_vertex(10 * (j + 1), 1, 1);
        }
    }

    for (unsigned i = 0; i < num_pipelines; ++i) {
        for (unsigned j = 0; j < pipeline_len - 1; ++j) { dag.add_edge(i * pipeline_len + j, i * pipeline_len + j + 1); }
    }
    return dag;
}

/**
 * @brief Constructs a "ladder" graph with a specified number of rungs.
 *
 * Each rung is a complete bipartite graph K(2,2) connecting to the next rung.
 * All "left" side nodes are identical, and all "right" side nodes are identical.
 *
 * @tparam Graph_t The graph type to construct.
 * @param num_rungs The number of rungs in the ladder.
 * @return A Graph_t object representing the DAG.
 */
template <typename Graph_t>
inline Graph_t construct_ladder_dag(unsigned num_rungs) {
    static_assert(is_constructable_cdag_v<Graph_t>, "Graph_t must be a constructable computational DAG");
    Graph_t dag;
    if (num_rungs == 0) { return dag; }

    for (unsigned i = 0; i < num_rungs + 1; ++i) {
        dag.add_vertex(10, 1, 1);    // Left side node
        dag.add_vertex(20, 1, 1);    // Right side node
    }

    for (unsigned i = 0; i < num_rungs; ++i) {
        auto u1 = 2 * i;
        auto v1 = 2 * i + 1;
        auto u2 = 2 * (i + 1);
        auto v2 = 2 * (i + 1) + 1;
        dag.add_edge(u1, u2);
        dag.add_edge(u1, v2);
        dag.add_edge(v1, u2);
        dag.add_edge(v1, v2);
    }
    return dag;
}

/**
 * @brief Constructs a graph with no structural symmetries.
 *
 * Creates a simple chain where each node has a unique work weight,
 * ensuring no two nodes will be in the same initial orbit.
 *
 * @tparam Graph_t The graph type to construct.
 * @param num_nodes The number of nodes in the chain.
 * @return A Graph_t object representing the DAG.
 */
template <typename Graph_t>
inline Graph_t construct_asymmetric_dag(unsigned num_nodes) {
    static_assert(is_constructable_cdag_v<Graph_t>, "Graph_t must be a constructable computational DAG");
    Graph_t dag;
    for (unsigned i = 0; i < num_nodes; ++i) {
        dag.add_vertex(10 * (i + 1), 1, 1);
        if (i > 0) { dag.add_edge(i - 1, i); }
    }
    return dag;
}

/**
 * @brief Constructs a complete binary tree that fans out from a single source.
 * @tparam Graph_t The graph type to construct.
 * @param height The height of the tree. A height of 0 is a single node. Total nodes: 2^(height+1) - 1.
 * @return A Graph_t object representing the out-tree.
 */
template <typename Graph_t>
inline Graph_t construct_binary_out_tree(unsigned height) {
    static_assert(is_constructable_cdag_v<Graph_t>, "Graph_t must be a constructable computational DAG");
    Graph_t dag;
    unsigned num_nodes = (1U << (height + 1)) - 1;
    if (num_nodes == 0) { return dag; }

    for (unsigned i = 0; i < num_nodes; ++i) { dag.add_vertex(10, 1, 1); }

    for (unsigned i = 0; i < num_nodes / 2; ++i) {
        dag.add_edge(i, 2 * i + 1);
        dag.add_edge(i, 2 * i + 2);
    }
    return dag;
}

/**
 * @brief Constructs a complete binary tree that fans into a single sink (root).
 * @tparam Graph_t The graph type to construct.
 * @param height The height of the tree. A height of 0 is a single node. Total nodes: 2^(height+1) - 1.
 * @return A Graph_t object representing the in-tree.
 */
template <typename Graph_t>
inline Graph_t construct_binary_in_tree(unsigned height) {
    static_assert(is_constructable_cdag_v<Graph_t>, "Graph_t must be a constructable computational DAG");
    Graph_t dag;
    unsigned num_nodes = (1U << (height + 1)) - 1;
    if (num_nodes == 0) { return dag; }

    for (unsigned i = 0; i < num_nodes; ++i) { dag.add_vertex(10, 1, 1); }

    for (unsigned i = 0; i < num_nodes / 2; ++i) {
        dag.add_edge(2 * i + 1, i);
        dag.add_edge(2 * i + 2, i);
    }
    return dag;
}

/**
 * @brief Constructs a 2D grid graph.
 * @tparam Graph_t The graph type to construct.
 * @param rows The number of rows in the grid.
 * @param cols The number of columns in the grid.
 * @return A Graph_t object representing the grid.
 */
template <typename Graph_t>
inline Graph_t construct_grid_dag(unsigned rows, unsigned cols) {
    static_assert(is_constructable_cdag_v<Graph_t>, "Graph_t must be a constructable computational DAG");
    Graph_t dag;
    if (rows == 0 || cols == 0) { return dag; }

    for (unsigned i = 0; i < rows * cols; ++i) { dag.add_vertex(10, 1, 1); }

    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned c = 0; c < cols; ++c) {
            if (r + 1 < rows) { dag.add_edge(r * cols + c, (r + 1) * cols + c); }
            if (c + 1 < cols) { dag.add_edge(r * cols + c, r * cols + (c + 1)); }
        }
    }
    return dag;
}

/**
 * @brief Constructs a butterfly graph, similar to FFT communication patterns.
 * @tparam Graph_t The graph type to construct.
 * @param stages The number of stages (log2 of the number of inputs). Total nodes: (stages+1) * 2^stages.
 * @return A Graph_t object representing the butterfly graph.
 */
template <typename Graph_t>
inline Graph_t construct_butterfly_dag(unsigned stages) {
    static_assert(is_constructable_cdag_v<Graph_t>, "Graph_t must be a constructable computational DAG");
    Graph_t dag;
    if (stages == 0) { return dag; }

    unsigned N = 1U << stages;
    for (unsigned i = 0; i < (stages + 1) * N; ++i) { dag.add_vertex(10, 1, 1); }

    for (unsigned s = 0; s < stages; ++s) {
        for (unsigned i = 0; i < N; ++i) {
            unsigned current_node = s * N + i;
            unsigned next_node_straight = (s + 1) * N + i;
            unsigned next_node_cross = (s + 1) * N + (i ^ (1U << (stages - 1 - s)));
            dag.add_edge(current_node, next_node_straight);
            dag.add_edge(current_node, next_node_cross);
        }
    }
    return dag;
}

}    // namespace osp
