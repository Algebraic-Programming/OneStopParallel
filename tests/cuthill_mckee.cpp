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

#define BOOST_TEST_MODULE cuthill_mckee
#include <boost/test/unit_test.hpp>
#include <filesystem>

#include "osp/graph_algorithms/cuthill_mckee.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"

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

using namespace osp;

using ComputationalDag = boost_graph_int_t;
using VertexType = vertex_idx_t<ComputationalDag>;

BOOST_AUTO_TEST_CASE(cuthill_mckee_1) {

    ComputationalDag dag;

    dag.add_vertex(2, 9);
    dag.add_vertex(3, 8);
    dag.add_vertex(4, 7);
    dag.add_vertex(5, 6);
    dag.add_vertex(6, 5);
    dag.add_vertex(7, 4);
    dag.add_vertex(8, 3);
    dag.add_vertex(9, 2);

    dag.add_edge(0, 1, 2);
    dag.add_edge(0, 2, 3);
    dag.add_edge(0, 3, 4);
    dag.add_edge(1, 4, 5);
    dag.add_edge(2, 4, 6);
    dag.add_edge(2, 5, 7);
    dag.add_edge(1, 6, 8);
    dag.add_edge(4, 7, 9);
    dag.add_edge(3, 7, 9);

    std::vector<VertexType> cm_wavefront = cuthill_mckee_wavefront(dag);
    std::vector<unsigned> expected_cm_wavefront = {0, 3, 1, 2, 6, 4, 5, 7};
    BOOST_CHECK_EQUAL_COLLECTIONS(cm_wavefront.begin(), cm_wavefront.end(), expected_cm_wavefront.begin(),
                                  expected_cm_wavefront.end());

    cm_wavefront = cuthill_mckee_wavefront(dag, true);
    expected_cm_wavefront = {0, 2, 3, 1, 5, 6, 4, 7};

    BOOST_CHECK_EQUAL_COLLECTIONS(cm_wavefront.begin(), cm_wavefront.end(), expected_cm_wavefront.begin(),
                                  expected_cm_wavefront.end());

    std::vector<VertexType> cm_undirected;
    std::vector<unsigned> expected_cm_undirected;

    cm_undirected = cuthill_mckee_undirected(dag, true);
    expected_cm_undirected = {7, 3, 4, 0, 1, 2, 6, 5};
    BOOST_CHECK_EQUAL_COLLECTIONS(cm_undirected.begin(), cm_undirected.end(), expected_cm_undirected.begin(),
                                  expected_cm_undirected.end());

    cm_undirected = cuthill_mckee_undirected(dag, false);
    expected_cm_undirected = {0, 3, 1, 2, 7, 6, 4, 5};
    BOOST_CHECK_EQUAL_COLLECTIONS(cm_undirected.begin(), cm_undirected.end(), expected_cm_undirected.begin(),
                                  expected_cm_undirected.end());

    cm_undirected = cuthill_mckee_undirected(dag, true, true);
    expected_cm_undirected = {3, 4, 5, 1, 2, 7, 6, 0};
    BOOST_CHECK_EQUAL_COLLECTIONS(cm_undirected.begin(), cm_undirected.end(), expected_cm_undirected.begin(),
                                  expected_cm_undirected.end());

    std::vector<VertexType> top_sort;
    for (const auto &vertex : priority_vec_top_sort_view(dag, cm_undirected)) {
        top_sort.push_back(vertex);
    }
    std::vector<unsigned> expected_top_sort = {0, 2, 5, 1, 6, 4, 3, 7};

    BOOST_CHECK_EQUAL_COLLECTIONS(top_sort.begin(), top_sort.end(), expected_top_sort.begin(), expected_top_sort.end());

    cm_undirected = cuthill_mckee_undirected(dag, false, true);
    expected_cm_undirected = {0, 2, 3, 1, 6, 7, 5, 4};
    BOOST_CHECK_EQUAL_COLLECTIONS(cm_undirected.begin(), cm_undirected.end(), expected_cm_undirected.begin(),
                                  expected_cm_undirected.end());

    dag.add_edge(8, 9);
    dag.add_edge(9, 10);

    cm_undirected = cuthill_mckee_undirected(dag, true);
    expected_cm_undirected = {7, 3, 4, 0, 1, 2, 6, 5, 10, 9, 8};
    BOOST_CHECK_EQUAL_COLLECTIONS(cm_undirected.begin(), cm_undirected.end(), expected_cm_undirected.begin(),
                                  expected_cm_undirected.end());

    cm_undirected = cuthill_mckee_undirected(dag, false);
    expected_cm_undirected = {0, 3, 1, 2, 7, 6, 4, 5, 8, 9, 10};
    BOOST_CHECK_EQUAL_COLLECTIONS(cm_undirected.begin(), cm_undirected.end(), expected_cm_undirected.begin(),
                                  expected_cm_undirected.end());
};

bool is_permutation(const std::vector<VertexType> &vec) {
    std::vector<VertexType> sorted_vec = vec;
    std::sort(sorted_vec.begin(), sorted_vec.end());
    for (unsigned i = 0; i < sorted_vec.size(); ++i) {
        if (sorted_vec[i] != i) {
            return false;
        }
    }
    return true;
}

bool is_top_sort(const std::vector<VertexType> &vec, const ComputationalDag &dag) {
    std::unordered_map<VertexType, VertexType> position;
    for (VertexType i = 0; i < vec.size(); ++i) {
        position[vec[i]] = i;
    }

    for (const auto &vertex : dag.vertices()) {

        for (const auto &child : dag.children(vertex)) {
            if (position[vertex] > position[child]) {
                return false;
            }
        }
    }

    return true;
}

BOOST_AUTO_TEST_CASE(cuthill_mckee_2) {

    std::vector<std::string> filenames_graph = tiny_spaa_graphs();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filename_graph : filenames_graph) {

        ComputationalDag graph;
        auto status_graph = file_reader::readComputationalDagHyperdagFormat((cwd / filename_graph).string(), graph);

        if (!status_graph) {

            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        } else {
            std::cout << "File read:" << filename_graph << std::endl;
        }

        std::vector<VertexType> wavefront = cuthill_mckee_wavefront(graph);
        BOOST_CHECK(is_permutation(wavefront));

        wavefront = cuthill_mckee_wavefront(graph, true);
        BOOST_CHECK(is_permutation(wavefront));

        const auto cm_undirected = cuthill_mckee_undirected(graph, true, true);
        BOOST_CHECK(is_permutation(cm_undirected));

        std::vector<VertexType> top_sort;

        for (const auto &vertex : priority_vec_top_sort_view(graph, cm_undirected)) {
            top_sort.push_back(vertex);
        }

        BOOST_CHECK(is_permutation(top_sort));
        BOOST_CHECK(is_top_sort(top_sort, graph));
    }
};