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

#ifdef EIGEN_FOUND

#    define BOOST_TEST_MODULE SparseMatrixImpl

#    include <boost/test/unit_test.hpp>
#    include <iostream>
#    include <vector>

#    include "osp/auxiliary/io/general_file_reader.hpp"
#    include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#    include "osp/graph_algorithms/directed_graph_path_util.hpp"
#    include "osp/graph_algorithms/directed_graph_util.hpp"
#    include "osp/graph_implementations/adj_list_impl/compact_sparse_graph.hpp"
#    include "osp/graph_implementations/eigen_matrix_adapter/sparse_matrix.hpp"
#    include "test_graphs.hpp"
#    include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"

using namespace osp;

using SmCsr = Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t>;
using SmCsc = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>;
using Triplet = Eigen::Triplet<double>;

using VImpl1 = CDagVertexImpl<std::size_t, unsigned, unsigned, unsigned, unsigned>;

BOOST_AUTO_TEST_CASE(TestSparseMatrixAdapter1) {
    /*

           ---0
         /  / | \
      --|--1  2--|-\
      | |  |   \ |  |
      | |  4 <-- 3  /
      |  \ |       /
       \   5      /
        \  | /----
          \|/
           6


        j→  0     1     2     3     4     5     6
    i ↓
       -------------------------------------------
      0 |   0     0     0     0     0     0     0
      1 | 2.0     0     0     0     0     0     0
      2 | 3.0     0     0     0     0     0     0
      3 | 4.0     0   5.0     0     0     0     0
      4 | 0.0   6.0     0   7.0     0     0     0
      5 | 8.0     0     0     0   9.0     0     0
      6 | 0.0  10.0  11.0     0     0  12.0     0

    */
    const int size = 7;
    std::vector<Triplet> triplets;

    // Diagonal entries
    for (int i = 0; i < size; ++i) {
        triplets.emplace_back(i, i, 1.0);
    }

    // Dependencies (i depends on j if L(i,j) ≠ 0, j < i)
    triplets.emplace_back(1, 0, 2.0);     // x1 ← x0
    triplets.emplace_back(2, 0, 3.0);     // x2 ← x0
    triplets.emplace_back(3, 0, 4.0);     // x3 ← x0
    triplets.emplace_back(3, 2, 5.0);     // x3 ← x2
    triplets.emplace_back(4, 1, 6.0);     // x4 ← x1
    triplets.emplace_back(4, 3, 7.0);     // x4 ← x3
    triplets.emplace_back(5, 0, 8.0);     // x5 ← x0
    triplets.emplace_back(5, 4, 9.0);     // x5 ← x4
    triplets.emplace_back(6, 1, 10.0);    // x6 ← x1
    triplets.emplace_back(6, 2, 11.0);    // x6 ← x2
    triplets.emplace_back(6, 5, 12.0);    // x6 ← x5

    // Construct matrix
    SmCsr lCsr(size, size);
    lCsr.setFromTriplets(triplets.begin(), triplets.end());

    SparseMatrixImp<int32_t> graph;
    graph.SetCsr(&lCsr);
    SmCsc lCsc{};
    lCsc = lCsr;
    graph.SetCsc(&lCsc);

    BOOST_CHECK_EQUAL(graph.NumEdges(), 11);
    BOOST_CHECK_EQUAL(graph.NumVertices(), 7);
    BOOST_CHECK_EQUAL(graph.InDegree(0), 0);
    BOOST_CHECK_EQUAL(graph.InDegree(1), 1);
    BOOST_CHECK_EQUAL(graph.InDegree(3), 2);
    BOOST_CHECK_EQUAL(graph.InDegree(6), 3);
    BOOST_CHECK_EQUAL(graph.OutDegree(0), 4);
    BOOST_CHECK_EQUAL(graph.OutDegree(1), 2);
    BOOST_CHECK_EQUAL(graph.OutDegree(3), 1);
    BOOST_CHECK_EQUAL(graph.OutDegree(6), 0);

    using VertexIdx = int32_t;

    std::vector<VertexIdx> vertices{0, 1, 2, 3, 4, 5, 6};

    std::vector<std::vector<VertexIdx>> outNeighbors{
        {1, 2, 3, 5},
        {4, 6},
        {3, 6},
        {4},
        {5},
        {6},
        {}
    };

    std::vector<std::vector<VertexIdx>> inNeighbors{
        {},
        {0},
        {0},
        {0, 2},
        {1, 3},
        {0, 4},
        {1, 2, 5}
    };

    size_t idx = 0;

    for (const long unsigned int &v : graph.Vertices()) {
        BOOST_CHECK_EQUAL(v, vertices[idx++]);

        size_t i = 0;
        size_t cntr = 0;
        const size_t vi = static_cast<size_t>(v);

        for (const auto &e : graph.Children(v)) {
            ++cntr;
            BOOST_CHECK_EQUAL(e, outNeighbors[vi][i++]);
        }
        BOOST_CHECK_EQUAL(cntr, outNeighbors[vi].size());
        BOOST_CHECK_EQUAL(graph.OutDegree(v), outNeighbors[vi].size());

        i = 0;
        cntr = 0;
        for (const auto &e : graph.Parents(v)) {
            ++cntr;
            BOOST_CHECK_EQUAL(e, inNeighbors[vi][i++]);
        }
        BOOST_CHECK_EQUAL(cntr, inNeighbors[vi].size());
        BOOST_CHECK_EQUAL(graph.InDegree(v), inNeighbors[vi].size());

        i = 0;
        cntr = 0;
        for (const auto &e : OutEdges(v, graph)) {
            ++cntr;
            BOOST_CHECK_EQUAL(Target(e, graph), outNeighbors[vi][i++]);
        }
        BOOST_CHECK_EQUAL(cntr, outNeighbors[vi].size());
        BOOST_CHECK_EQUAL(graph.OutDegree(v), outNeighbors[vi].size());

        i = 0;
        cntr = 0;
        for (const auto &e : InEdges(v, graph)) {
            ++cntr;
            BOOST_CHECK_EQUAL(Source(e, graph), inNeighbors[vi][i++]);
        }
        BOOST_CHECK_EQUAL(cntr, inNeighbors[vi].size());
        BOOST_CHECK_EQUAL(graph.InDegree(v), inNeighbors[vi].size());
    }

    unsigned count = 0;
    for (const auto &e : Edges(graph)) {
        std::cout << e.source_ << " -> " << e.target_ << std::endl;
        count++;
    }
    BOOST_CHECK_EQUAL(count, 11);
}

BOOST_AUTO_TEST_CASE(TestSparseMatrixAdapter2) {
    std::vector<std::string> filenamesGraph = TestMTXGraphs();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filenameGraph : filenamesGraph) {
        std::string nameGraph = filenameGraph.substr(filenameGraph.find_last_of("/\\") + 1);
        nameGraph = nameGraph.substr(0, nameGraph.find_last_of("."));

        std::cout << "Graph: " << nameGraph << std::endl;

        ComputationalDagVectorImpl<VImpl1> graph1;

        const bool statusGraph = file_reader::ReadGraph((cwd / filenameGraph).string(), graph1);

        BOOST_CHECK(statusGraph);
        if (!statusGraph) {
            std::cout << "Reading files failed." << std::endl;
        }

        CompactSparseGraph<true, true, true, true, true> graph2(graph1);

        std::vector<Triplet> triplets;
        // Diagonal entries
        for (const auto &vert : graph1.Vertices()) {
            triplets.emplace_back(vert, vert, 1.0);
        }

        // Below Diagonal
        for (const auto &vert : graph1.Vertices()) {
            for (const auto &child : graph1.Children(vert)) {
                triplets.emplace_back(child, vert, 2.0);
            }
        }

        const int32_t nVert = static_cast<int32_t>(graph1.NumVertices());
        SmCsr lCsr(nVert, nVert);
        lCsr.setFromTriplets(triplets.begin(), triplets.end());
        SmCsc lCsc{};
        lCsc = lCsr;

        SparseMatrixImp<int32_t> graph;
        graph.SetCsr(&lCsr);
        graph.SetCsc(&lCsc);

        BOOST_CHECK_EQUAL(static_cast<std::size_t>(graph.NumVertices()), graph1.NumVertices());
        BOOST_CHECK_EQUAL(static_cast<std::size_t>(graph.NumVertices()), graph2.NumVertices());

        BOOST_CHECK_EQUAL(static_cast<std::size_t>(graph.NumEdges()), graph1.NumEdges());
        BOOST_CHECK_EQUAL(static_cast<std::size_t>(graph.NumEdges()), graph2.NumEdges());

        for (const auto &vert : graph2.Vertices()) {
            auto chldren = graph.Children(vert);
            auto chldren2 = graph2.Children(vert);
            auto it = chldren.begin();
            auto it_other = chldren.begin();
            const auto begin = chldren.begin();
            auto it2 = chldren2.begin();
            const auto end = chldren.end();
            const auto end_other = chldren.end();
            const auto end2 = chldren2.end();

            BOOST_CHECK(end == end_other);

            std::size_t cntr = 0;
            while ((it != end) && (it2 != end2)) {
                BOOST_CHECK_EQUAL(*it, *it2);
                BOOST_CHECK(it == it_other);
                BOOST_CHECK(cntr == 0U || it != begin);
                BOOST_CHECK(cntr == 0U || (not (it == begin)));
                BOOST_CHECK_EQUAL(it, it != end);

                ++cntr;
                ++it;
                ++it_other;
                ++it2;
            }
            BOOST_CHECK_EQUAL(cntr, graph.OutDegree(vert));
            BOOST_CHECK_EQUAL(cntr, graph1.OutDegree(vert));
            BOOST_CHECK_EQUAL(cntr, graph2.OutDegree(vert));
            BOOST_CHECK(it == end);
            BOOST_CHECK(it2 == end2);
            BOOST_CHECK_EQUAL(it, it != end);
        }

        for (const auto &vert : graph2.Vertices()) {
            auto parents = graph.Parents(vert);
            auto parents2 = graph2.Parents(vert);
            auto it = parents.begin();
            auto it_other = parents.begin();
            const auto begin = parents.begin();
            auto it2 = parents2.begin();
            const auto end = parents.end();
            const auto end_other = parents.end();
            const auto end2 = parents2.end();

            BOOST_CHECK(end == end_other);

            std::size_t cntr = 0;
            while ((it != end) && (it2 != end2)) {
                BOOST_CHECK_EQUAL(*it, *it2);
                BOOST_CHECK(cntr == 0U || it != begin);
                BOOST_CHECK(cntr == 0U || (not (it == begin)));
                BOOST_CHECK_EQUAL(it, it != end);

                ++cntr;
                ++it;
                ++it2;
            }
            BOOST_CHECK_EQUAL(cntr, graph.InDegree(vert));
            BOOST_CHECK_EQUAL(cntr, graph1.InDegree(vert));
            BOOST_CHECK_EQUAL(cntr, graph2.InDegree(vert));
            BOOST_CHECK(it == end);
            BOOST_CHECK(it2 == end2);
            BOOST_CHECK_EQUAL(it, it != end);
        }
    }
}

#endif
