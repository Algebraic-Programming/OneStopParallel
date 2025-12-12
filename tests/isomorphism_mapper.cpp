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

#define BOOST_TEST_MODULE IsomorphismMapper
#include <boost/test/unit_test.hpp>
#include <numeric>
#include <set>
#include <unordered_map>

#include "osp/dag_divider/isomorphism_divider/IsomorphismMapper.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "test_graphs.hpp"

using namespace osp;

BOOST_AUTO_TEST_SUITE(isomorphism_mapper_test_suite)

using GraphT = computational_dag_vector_impl_def_t;
using ConstrGraphT = computational_dag_vector_impl_def_t;

BOOST_AUTO_TEST_CASE(MapperSimpleChain) {
    // Rep: 0 -> 1 -> 2
    ConstrGraphT repGraph;
    repGraph.add_vertex(10, 1, 1);
    repGraph.add_vertex(20, 1, 1);
    repGraph.add_vertex(30, 1, 1);
    repGraph.add_edge(0, 1);
    repGraph.add_edge(1, 2);
    std::vector<VertexIdxT<GraphT>> repMap = {100, 101, 102};

    // Current: 2 -> 0 -> 1 (isomorphic, but different local IDs)
    ConstrGraphT currentGraph;
    currentGraph.add_vertex(20, 1, 1);    // local 0 (work 20)
    currentGraph.add_vertex(30, 1, 1);    // local 1 (work 30)
    currentGraph.add_vertex(10, 1, 1);    // local 2 (work 10)
    currentGraph.add_edge(2, 0);
    currentGraph.add_edge(0, 1);
    std::vector<VertexIdxT<GraphT>> currentMap = {201, 202, 200};

    IsomorphismMapper<GraphT, ConstrGraphT> mapper(repGraph);
    auto resultMapLocal = mapper.find_mapping(currentGraph);

    // Translate local map to global map for the test
    std::unordered_map<VertexIdxT<GraphT>, VertexIdxT<GraphT>> resultMap;
    for (const auto &[curr_local, rep_local] : resultMapLocal) {
        resultMap[currentMap[curr_local]] = repMap[rep_local];
    }

    BOOST_REQUIRE_EQUAL(resultMap.size(), 3);
    // current global ID -> rep global ID
    // 200 (work 10) -> 100 (work 10)
    // 201 (work 20) -> 101 (work 20)
    // 202 (work 30) -> 102 (work 30)
    BOOST_CHECK_EQUAL(resultMap.at(200), 100);
    BOOST_CHECK_EQUAL(resultMap.at(201), 101);
    BOOST_CHECK_EQUAL(resultMap.at(202), 102);
}

BOOST_AUTO_TEST_CASE(MapperForkJoin) {
    // Rep: 0 -> {1,2} -> 3
    ConstrGraphT repGraph;
    repGraph.add_vertex(10, 1, 1);
    repGraph.add_vertex(20, 1, 1);
    repGraph.add_vertex(20, 1, 1);
    repGraph.add_vertex(30, 1, 1);
    repGraph.add_edge(0, 1);
    repGraph.add_edge(0, 2);
    repGraph.add_edge(1, 3);
    repGraph.add_edge(2, 3);
    std::vector<VertexIdxT<GraphT>> repMap = {10, 11, 12, 13};

    // Current: 3 -> {0,2} -> 1
    ConstrGraphT currentGraph;
    currentGraph.add_vertex(20, 1, 1);    // local 0
    currentGraph.add_vertex(30, 1, 1);    // local 1
    currentGraph.add_vertex(20, 1, 1);    // local 2
    currentGraph.add_vertex(10, 1, 1);    // local 3
    currentGraph.add_edge(3, 0);
    currentGraph.add_edge(3, 2);
    currentGraph.add_edge(0, 1);
    currentGraph.add_edge(2, 1);
    std::vector<VertexIdxT<GraphT>> currentMap = {21, 23, 22, 20};

    IsomorphismMapper<GraphT, ConstrGraphT> mapper(repGraph);
    auto resultMapLocal = mapper.find_mapping(currentGraph);

    std::unordered_map<VertexIdxT<GraphT>, VertexIdxT<GraphT>> resultMap;
    for (const auto &[curr_local, rep_local] : resultMapLocal) {
        resultMap[currentMap[curr_local]] = repMap[rep_local];
    }

    BOOST_REQUIRE_EQUAL(resultMap.size(), 4);
    // current global ID -> rep global ID
    // 20 (work 10) -> 10 (work 10)
    // 23 (work 30) -> 13 (work 30)
    BOOST_CHECK_EQUAL(resultMap.at(20), 10);
    BOOST_CHECK_EQUAL(resultMap.at(23), 13);

    // The two middle nodes are symmetric. The mapping could be either way.
    // current {21, 22} -> rep {11, 12}
    bool mapping1 = (resultMap.at(21) == 11 && resultMap.at(22) == 12);
    bool mapping2 = (resultMap.at(21) == 12 && resultMap.at(22) == 11);
    BOOST_CHECK(mapping1 || mapping2);
}

BOOST_AUTO_TEST_CASE(MapperDisconnectedComponents) {
    // Rep: {0->1}, {2->3}. Two identical but disconnected components.
    ConstrGraphT repGraph;
    repGraph.add_vertex(10, 1, 1);
    repGraph.add_vertex(20, 1, 1);    // 0, 1
    repGraph.add_vertex(10, 1, 1);
    repGraph.add_vertex(20, 1, 1);    // 2, 3
    repGraph.add_edge(0, 1);
    repGraph.add_edge(2, 3);
    std::vector<VertexIdxT<GraphT>> repMap = {10, 11, 12, 13};

    // Current: {2->3}, {0->1}. Same components, but different local IDs.
    ConstrGraphT currentGraph;
    currentGraph.add_vertex(10, 1, 1);
    currentGraph.add_vertex(20, 1, 1);    // 0, 1
    currentGraph.add_vertex(10, 1, 1);
    currentGraph.add_vertex(20, 1, 1);    // 2, 3
    currentGraph.add_edge(2, 3);
    currentGraph.add_edge(0, 1);
    std::vector<VertexIdxT<GraphT>> currentMap = {22, 23, 20, 21};

    IsomorphismMapper<GraphT, ConstrGraphT> mapper(repGraph);
    auto resultMapLocal = mapper.find_mapping(currentGraph);

    std::unordered_map<VertexIdxT<GraphT>, VertexIdxT<GraphT>> resultMap;
    for (const auto &[curr_local, rep_local] : resultMapLocal) {
        resultMap[currentMap[curr_local]] = repMap[rep_local];
    }

    BOOST_REQUIRE_EQUAL(resultMap.size(), 4);

    // The two components are symmetric. The mapping could be component {0,1} -> {0,1}
    // and {2,3} -> {2,3}, OR component {0,1} -> {2,3} and {2,3} -> {0,1}.

    // Mapping Option 1:
    // rep {10,11} -> current {20,21}
    // rep {12,13} -> current {22,23}
    bool mapping1 = (resultMap.at(20) == 12 && resultMap.at(21) == 13 && resultMap.at(22) == 10 && resultMap.at(23) == 11);

    // Mapping Option 2:
    // rep {10,11} -> current {22,23}
    // rep {12,13} -> current {20,21}
    bool mapping2 = (resultMap.at(22) == 12 && resultMap.at(23) == 13 && resultMap.at(20) == 10 && resultMap.at(21) == 11);

    BOOST_CHECK(mapping1 || mapping2);
}

BOOST_AUTO_TEST_CASE(MapperMultiPipeline) {
    // This test checks the mapping of a graph that is composed of multiple
    // isomorphic disconnected components (two parallel pipelines).

    // Rep: Two pipelines {0->1->2} and {3->4->5}
    // All nodes at the same stage have the same work weight.
    ConstrGraphT repGraph = construct_multi_pipeline_dag<ConstrGraphT>(2, 3);
    std::vector<VertexIdxT<GraphT>> repMap = {10, 11, 12, 20, 21, 22};

    // Current: Isomorphic to rep, but the pipelines are swapped and vertex IDs are shuffled.
    // Pipeline 1 (local IDs 0,1,2) corresponds to rep pipeline 2 (global 20,21,22)
    // Pipeline 2 (local IDs 3,4,5) corresponds to rep pipeline 1 (global 10,11,12)
    ConstrGraphT currentGraph;
    currentGraph.add_vertex(10, 1, 1);    // local 0, stage 0
    currentGraph.add_vertex(20, 1, 1);    // local 1, stage 1
    currentGraph.add_vertex(30, 1, 1);    // local 2, stage 2
    currentGraph.add_vertex(10, 1, 1);    // local 3, stage 0
    currentGraph.add_vertex(20, 1, 1);    // local 4, stage 1
    currentGraph.add_vertex(30, 1, 1);    // local 5, stage 2
    currentGraph.add_edge(0, 1);
    currentGraph.add_edge(1, 2);    // First pipeline
    currentGraph.add_edge(3, 4);
    currentGraph.add_edge(4, 5);    // Second pipeline
    std::vector<VertexIdxT<GraphT>> currentMap = {120, 121, 122, 110, 111, 112};

    IsomorphismMapper<GraphT, ConstrGraphT> mapper(repGraph);
    auto resultMapLocal = mapper.find_mapping(currentGraph);

    std::unordered_map<VertexIdxT<GraphT>, VertexIdxT<GraphT>> resultMap;
    for (const auto &[curr_local, rep_local] : resultMapLocal) {
        resultMap[currentMap[curr_local]] = repMap[rep_local];
    }

    BOOST_REQUIRE_EQUAL(resultMap.size(), 6);

    // The two pipelines are symmetric, so the mapping can go either way.

    // Mapping Option 1: current pipeline 1 -> rep pipeline 1, current pipeline 2 -> rep pipeline 2
    bool mapping1 = (resultMap.at(110) == 10 && resultMap.at(111) == 11 && resultMap.at(112) == 12 && resultMap.at(120) == 20
                     && resultMap.at(121) == 21 && resultMap.at(122) == 22);

    // Mapping Option 2: current pipeline 1 -> rep pipeline 2, current pipeline 2 -> rep pipeline 1
    bool mapping2 = (resultMap.at(110) == 20 && resultMap.at(111) == 21 && resultMap.at(112) == 22 && resultMap.at(120) == 10
                     && resultMap.at(121) == 11 && resultMap.at(122) == 12);

    BOOST_CHECK(mapping1 || mapping2);
}

BOOST_AUTO_TEST_CASE(MapperShuffledSymmetric) {
    // This test uses a symmetric graph (a ladder) and shuffles the vertex IDs
    // of the 'current' graph to ensure the mapper correctly finds the structural
    // isomorphism, not just a naive index-based mapping.

    // Rep: A ladder graph with 2 rungs.
    // Structure: {0,1} -> {2,3} -> {4,5}
    // Nodes {0,2,4} have work 10 (left side).
    // Nodes {1,3,5} have work 20 (right side).
    ConstrGraphT repGraph = construct_ladder_dag<ConstrGraphT>(2);
    std::vector<VertexIdxT<GraphT>> repMap = {10, 11, 12, 13, 14, 15};

    // Current: Isomorphic to rep, but with shuffled local IDs.
    // A naive mapping of local IDs (0->0, 1->1, etc.) would be incorrect
    // because the work weights would not match.
    ConstrGraphT currentGraph;
    currentGraph.add_vertex(20, 1, 1);    // local 0 (work 20, right)
    currentGraph.add_vertex(10, 1, 1);    // local 1 (work 10, left)
    currentGraph.add_vertex(20, 1, 1);    // local 2 (work 20, right)
    currentGraph.add_vertex(10, 1, 1);    // local 3 (work 10, left)
    currentGraph.add_vertex(20, 1, 1);    // local 4 (work 20, right)
    currentGraph.add_vertex(10, 1, 1);    // local 5 (work 10, left)
    // Edges for {5,0} -> {3,2} -> {1,4}
    currentGraph.add_edge(5, 3);
    currentGraph.add_edge(5, 2);    // Rung 1
    currentGraph.add_edge(0, 3);
    currentGraph.add_edge(0, 2);

    currentGraph.add_edge(3, 1);
    currentGraph.add_edge(3, 4);    // Rung 2
    currentGraph.add_edge(2, 1);
    currentGraph.add_edge(2, 4);

    std::vector<VertexIdxT<GraphT>> currentMap = {111, 114, 113, 112, 115, 110};

    IsomorphismMapper<GraphT, ConstrGraphT> mapper(repGraph);
    auto resultMapLocal = mapper.find_mapping(currentGraph);

    std::unordered_map<VertexIdxT<GraphT>, VertexIdxT<GraphT>> resultMap;
    for (const auto &[curr_local, rep_local] : resultMapLocal) {
        resultMap[currentMap[curr_local]] = repMap[rep_local];
    }

    BOOST_REQUIRE_EQUAL(resultMap.size(), 6);
    // Check that structurally identical nodes are mapped, regardless of their original IDs.
    // E.g., current global 110 (from local 5, work 10) must map to a rep node with work 10.
    BOOST_CHECK_EQUAL(resultMap.at(110), 10);    // current 5 (work 10) -> rep 0 (work 10)
    BOOST_CHECK_EQUAL(resultMap.at(111), 11);    // current 0 (work 20) -> rep 1 (work 20)
}

BOOST_AUTO_TEST_SUITE_END()
