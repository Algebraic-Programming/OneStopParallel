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
#include "osp/graph_algorithms/cuthill_mckee.hpp"

#include <boost/test/unit_test.hpp>
#include <filesystem>

#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"
#include "test_graphs.hpp"

using namespace osp;

using ComputationalDag = boost_graph_int_t;
using VertexType = VertexIdxT<ComputationalDag>;

BOOST_AUTO_TEST_CASE(CuthillMckee1) {
    ComputationalDag dag;

    dag.AddVertex(2, 9);
    dag.AddVertex(3, 8);
    dag.AddVertex(4, 7);
    dag.AddVertex(5, 6);
    dag.AddVertex(6, 5);
    dag.AddVertex(7, 4);
    dag.AddVertex(8, 3);
    dag.AddVertex(9, 2);

    dag.AddEdge(0, 1, 2);
    dag.AddEdge(0, 2, 3);
    dag.AddEdge(0, 3, 4);
    dag.AddEdge(1, 4, 5);
    dag.AddEdge(2, 4, 6);
    dag.AddEdge(2, 5, 7);
    dag.AddEdge(1, 6, 8);
    dag.AddEdge(4, 7, 9);
    dag.AddEdge(3, 7, 9);

    std::vector<VertexType> cmWavefront = cuthill_mckee_wavefront(dag);
    std::vector<unsigned> expectedCmWavefront = {0, 3, 1, 2, 6, 4, 5, 7};
    BOOST_CHECK_EQUAL_COLLECTIONS(cmWavefront.begin(), cmWavefront.end(), expectedCmWavefront.begin(), expectedCmWavefront.end());

    cmWavefront = cuthill_mckee_wavefront(dag, true);
    expectedCmWavefront = {0, 2, 3, 1, 5, 6, 4, 7};

    BOOST_CHECK_EQUAL_COLLECTIONS(cmWavefront.begin(), cmWavefront.end(), expectedCmWavefront.begin(), expectedCmWavefront.end());

    std::vector<VertexType> cmUndirected;
    std::vector<unsigned> expectedCmUndirected;

    cmUndirected = cuthill_mckee_undirected(dag, true);
    expectedCmUndirected = {7, 3, 4, 0, 1, 2, 6, 5};
    BOOST_CHECK_EQUAL_COLLECTIONS(
        cmUndirected.begin(), cmUndirected.end(), expectedCmUndirected.begin(), expectedCmUndirected.end());

    cmUndirected = cuthill_mckee_undirected(dag, false);
    expectedCmUndirected = {0, 3, 1, 2, 7, 6, 4, 5};
    BOOST_CHECK_EQUAL_COLLECTIONS(
        cmUndirected.begin(), cmUndirected.end(), expectedCmUndirected.begin(), expectedCmUndirected.end());

    cmUndirected = cuthill_mckee_undirected(dag, true, true);
    expectedCmUndirected = {3, 4, 5, 1, 2, 7, 6, 0};
    BOOST_CHECK_EQUAL_COLLECTIONS(
        cmUndirected.begin(), cmUndirected.end(), expectedCmUndirected.begin(), expectedCmUndirected.end());

    std::vector<VertexType> topSort;
    for (const auto &vertex : priority_vec_top_sort_view(dag, cmUndirected)) {
        topSort.push_back(vertex);
    }
    std::vector<unsigned> expectedTopSort = {0, 2, 5, 1, 6, 4, 3, 7};

    BOOST_CHECK_EQUAL_COLLECTIONS(topSort.begin(), topSort.end(), expectedTopSort.begin(), expectedTopSort.end());

    cmUndirected = cuthill_mckee_undirected(dag, false, true);
    expectedCmUndirected = {0, 2, 3, 1, 6, 7, 5, 4};
    BOOST_CHECK_EQUAL_COLLECTIONS(
        cmUndirected.begin(), cmUndirected.end(), expectedCmUndirected.begin(), expectedCmUndirected.end());

    dag.AddEdge(8, 9);
    dag.AddEdge(9, 10);

    cmUndirected = cuthill_mckee_undirected(dag, true);
    expectedCmUndirected = {7, 3, 4, 0, 1, 2, 6, 5, 10, 9, 8};
    BOOST_CHECK_EQUAL_COLLECTIONS(
        cmUndirected.begin(), cmUndirected.end(), expectedCmUndirected.begin(), expectedCmUndirected.end());

    cmUndirected = cuthill_mckee_undirected(dag, false);
    expectedCmUndirected = {0, 3, 1, 2, 7, 6, 4, 5, 8, 9, 10};
    BOOST_CHECK_EQUAL_COLLECTIONS(
        cmUndirected.begin(), cmUndirected.end(), expectedCmUndirected.begin(), expectedCmUndirected.end());
}

bool IsPermutation(const std::vector<VertexType> &vec) {
    std::vector<VertexType> sortedVec = vec;
    std::sort(sortedVec.begin(), sortedVec.end());
    for (unsigned i = 0; i < sortedVec.size(); ++i) {
        if (sortedVec[i] != i) {
            return false;
        }
    }
    return true;
}

bool IsTopSort(const std::vector<VertexType> &vec, const ComputationalDag &dag) {
    std::unordered_map<VertexType, VertexType> position;
    for (VertexType i = 0; i < vec.size(); ++i) {
        position[vec[i]] = i;
    }

    for (const auto &vertex : dag.Vertices()) {
        for (const auto &child : dag.Children(vertex)) {
            if (position[vertex] > position[child]) {
                return false;
            }
        }
    }

    return true;
}

BOOST_AUTO_TEST_CASE(CuthillMckee2) {
    std::vector<std::string> filenamesGraph = TinySpaaGraphs();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filenameGraph : filenamesGraph) {
        ComputationalDag graph;
        auto statusGraph = file_reader::ReadComputationalDagHyperdagFormatDB((cwd / filenameGraph).string(), graph);

        if (!statusGraph) {
            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        } else {
            std::cout << "File read:" << filenameGraph << std::endl;
        }

        std::vector<VertexType> wavefront = cuthill_mckee_wavefront(graph);
        BOOST_CHECK(IsPermutation(wavefront));

        wavefront = cuthill_mckee_wavefront(graph, true);
        BOOST_CHECK(IsPermutation(wavefront));

        const auto cmUndirected = cuthill_mckee_undirected(graph, true, true);
        BOOST_CHECK(IsPermutation(cmUndirected));

        std::vector<VertexType> topSort;

        for (const auto &vertex : priority_vec_top_sort_view(graph, cmUndirected)) {
            topSort.push_back(vertex);
        }

        BOOST_CHECK(IsPermutation(topSort));
        BOOST_CHECK(IsTopSort(topSort, graph));
    }
}
