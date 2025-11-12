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

#define BOOST_TEST_MODULE ApproxEdgeReduction

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <vector>

#include "osp/graph_algorithms/computational_dag_util.hpp"
#include "osp/graph_algorithms/directed_graph_edge_desc_util.hpp"
#include "osp/graph_algorithms/directed_graph_edge_desc_util_parallel.hpp"
#include "osp/graph_algorithms/directed_graph_edge_view.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "test_utils.hpp"
#include "test_graphs.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(longest_edge_triangle_parallel) {


    using graph_t = boost_graph_int_t;

    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenames_graph = large_spaa_graphs();

    const auto project_root = get_project_root();

    for (auto &filename_graph : filenames_graph) {
        graph_t graph;


        bool status_graph = file_reader::readComputationalDagHyperdagFormatDB((project_root / filename_graph).string(),
                                                                                graph);

        BOOST_CHECK(status_graph);

        auto start_time = std::chrono::high_resolution_clock::now();
        auto deleted_edges = long_edges_in_triangles(graph);
        auto finish_time = std::chrono::high_resolution_clock::now();

        std::cout << "\n" << filename_graph << std::endl;

        std::cout << "Time for long_edges_in_triangles: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_time).count() << "ms"
                  << std::endl;

        start_time = std::chrono::high_resolution_clock::now();
        auto deleted_edges_parallel = long_edges_in_triangles_parallel(graph);
        finish_time = std::chrono::high_resolution_clock::now();

        std::cout << "Time for long_edges_in_triangles_parallel: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_time).count() << "ms"
                  << std::endl;

        BOOST_CHECK_EQUAL(deleted_edges.size(), deleted_edges_parallel.size());

        for (const auto &edge : deleted_edges) {
            BOOST_CHECK(deleted_edges_parallel.find(edge) != deleted_edges_parallel.cend());
        }

        for (const auto &edge : deleted_edges_parallel) {
            BOOST_CHECK(deleted_edges.find(edge) != deleted_edges.cend());
        }
    }
}