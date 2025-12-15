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

#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/graph_algorithms/computational_dag_util.hpp"
#include "osp/graph_algorithms/directed_graph_edge_desc_util.hpp"
#include "osp/graph_algorithms/directed_graph_edge_desc_util_parallel.hpp"
#include "osp/graph_algorithms/directed_graph_edge_view.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"
#include "test_graphs.hpp"
#include "test_utils.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(LongestEdgeTriangleParallel) {
    using GraphT = BoostGraphIntT;

    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenamesGraph = LargeSpaaGraphs();

    const auto projectRoot = GetProjectRoot();

    for (auto &filenameGraph : filenamesGraph) {
        GraphT graph;

        bool statusGraph = file_reader::ReadComputationalDagHyperdagFormatDB((projectRoot / filenameGraph).string(), graph);

        BOOST_CHECK(statusGraph);

        auto startTime = std::chrono::high_resolution_clock::now();
        auto deletedEdges = long_edges_in_triangles(graph);
        auto finishTime = std::chrono::high_resolution_clock::now();

        std::cout << "\n" << filenameGraph << std::endl;

        std::cout << "Time for long_edges_in_triangles: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(finishTime - startTime).count() << "ms" << std::endl;

        startTime = std::chrono::high_resolution_clock::now();
        auto deletedEdgesParallel = long_edges_in_triangles_parallel(graph);
        finishTime = std::chrono::high_resolution_clock::now();

        std::cout << "Time for long_edges_in_triangles_parallel: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(finishTime - startTime).count() << "ms" << std::endl;

        BOOST_CHECK_EQUAL(deletedEdges.size(), deletedEdgesParallel.size());

        for (const auto &edge : deletedEdges) {
            BOOST_CHECK(deletedEdgesParallel.find(edge) != deletedEdgesParallel.cend());
        }

        for (const auto &edge : deletedEdgesParallel) {
            BOOST_CHECK(deletedEdges.find(edge) != deletedEdges.cend());
        }
    }
}
