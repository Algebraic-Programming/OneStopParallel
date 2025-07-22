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

#define BOOST_TEST_MODULE BSP_SCHEDULE_RECOMP
#include <boost/test/unit_test.hpp>

#include "osp/dag_divider/MerkleHashComputer.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include <filesystem>
#include <iostream>

using namespace osp;

BOOST_AUTO_TEST_CASE(BspScheduleRecomp_test)
{

    using graph_t = computational_dag_vector_impl_def_t;

    graph_t graph;

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    file_reader::readComputationalDagHyperdagFormat((cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(), graph);

    MerkleHashComputer<graph_t, default_node_hash_func<vertex_idx_t<graph_t>, 11>> m_hash(graph);

    BOOST_CHECK_EQUAL(m_hash.get_vertex_hashes().size(), graph.num_vertices());
    
    for (const auto& v : source_vertices_view(graph)) {
        BOOST_CHECK_EQUAL(m_hash.get_vertex_hash(v), 11);
    }

    size_t num = 0;
    for (const auto& pair : m_hash.get_orbits()) {

        num += pair.second.size();
        std::cout << "orbit " << pair.first << ": ";
        for (const auto& v : pair.second) {
            std::cout << v << ", ";
        } 
        std::cout << std::endl;
    }

    BOOST_CHECK_EQUAL(num, graph.num_vertices());

    BOOST_CHECK_EQUAL(m_hash.get_vertex_hash(41), m_hash.get_vertex_hash(47));
    BOOST_CHECK_EQUAL(m_hash.get_vertex_hash(28), m_hash.get_vertex_hash(18));
    BOOST_CHECK_EQUAL(m_hash.get_vertex_hash(43), m_hash.get_vertex_hash(48));
    BOOST_CHECK_EQUAL(m_hash.get_vertex_hash(29), m_hash.get_vertex_hash(22));
    BOOST_CHECK(m_hash.get_vertex_hash(3) != m_hash.get_vertex_hash(12));
    BOOST_CHECK(m_hash.get_vertex_hash(53) != m_hash.get_vertex_hash(29));


};