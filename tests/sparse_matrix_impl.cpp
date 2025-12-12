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

#    include "osp/graph_algorithms/directed_graph_path_util.hpp"
#    include "osp/graph_algorithms/directed_graph_util.hpp"
#    include "osp/graph_implementations/eigen_matrix_adapter/sparse_matrix.hpp"

using namespace osp;

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
    using SmCsr = Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t>;
    using SmCsc = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>;
    using Triplet = Eigen::Triplet<double>;
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
    graph.setCSR(&lCsr);
    SmCsc lCsc{};
    lCsc = lCsr;
    graph.setCSC(&lCsc);

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

    for (const long unsigned int &v : graph.vertices()) {
        BOOST_CHECK_EQUAL(v, vertices[idx++]);

        size_t i = 0;
        const size_t vi = static_cast<size_t>(v);

        for (const auto &e : graph.Children(v)) {
            BOOST_CHECK_EQUAL(e, outNeighbors[vi][i++]);
        }

        i = 0;
        for (const auto &e : graph.Parents(v)) {
            BOOST_CHECK_EQUAL(e, inNeighbors[vi][i++]);
        }

        i = 0;
        for (const auto &e : OutEdges(v, graph)) {
            BOOST_CHECK_EQUAL(Traget(e, graph), outNeighbors[vi][i++]);
        }

        i = 0;
        for (const auto &e : InEdges(v, graph)) {
            BOOST_CHECK_EQUAL(Source(e, graph), inNeighbors[vi][i++]);
        }

        BOOST_CHECK_EQUAL(graph.InDegree(v), inNeighbors[vi].size());
        BOOST_CHECK_EQUAL(graph.OutDegree(v), outNeighbors[vi].size());
    }

    unsigned count = 0;
    for (const auto &e : Edges(graph)) {
        std::cout << e.source << " -> " << e.target << std::endl;
        count++;
    }
    BOOST_CHECK_EQUAL(count, 11);
}

#endif
