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

#define BOOST_TEST_MODULE wavefront_divider
#include <boost/test/unit_test.hpp>

#include <filesystem>
#include <iostream>

#include "dag_divider/WavefrontComponentDivider.hpp"
#include "dag_divider/WavefrontComponentScheduler.hpp"
#include "dag_divider/WavefrontParallelismDivider.hpp"
#include "io/dot_graph_file_reader.hpp"
#include "bsp/scheduler/GreedySchedulers/BspLocking.hpp"
#include "bsp/scheduler/LocalSearch/KernighanLin/kl_total_cut.hpp"
#include "bsp/scheduler/LocalSearch/KernighanLin/kl_total_comm.hpp"

#include "graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

using namespace osp;

std::vector<std::string> test_graphs_dot() {
    return { "data/dot/smpl_dot_graph_1.dot"
    };
}

template<typename Graph_t>
bool check_vertex_maps(const std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> &maps, const Graph_t &dag) {

    std::unordered_set<vertex_idx_t<Graph_t>> all_vertices;
    for (const auto &step : maps) {
        for (const auto &subgraph : step) {

            for (const auto &vertex : subgraph)
                all_vertices.insert(vertex);
        }
    }

    return all_vertices.size() == dag.num_vertices();
}


// BOOST_AUTO_TEST_CASE(wavefront_component_divider_2) {

//     std::vector<std::string> filenames_graph = test_graphs_dot();

//     // Getting root git directory
//     std::filesystem::path cwd = std::filesystem::current_path();
//     std::cout << cwd << std::endl;
//     while ((!cwd.empty()) && (cwd.filename() != "one-stop-parallel")) {
//         cwd = cwd.parent_path();
//         std::cout << cwd << std::endl;
//     }

//     for (auto &filename_graph : filenames_graph) {

//         auto [status_graph, graph] = FileReader::readComputationalDagDotFormat((cwd / filename_graph).string());

//         if (!status_graph) {

//             std::cout << "Reading files failed." << std::endl;
//             BOOST_CHECK(false);
//         } else {
//             std::cout << "File read:" << filename_graph << std::endl;
//         }

//         // for(const auto& vertex : graph.vertices()) {
//         //     graph.setNodeType(vertex, 0);
//         // }

//         WavefrontComponentDivider wavefront;
//         wavefront.set_split_method(WavefrontComponentDivider::SplitMethod::VARIANCE);

//         auto maps = wavefront.divide(graph);

//         BOOST_CHECK(check_vertex_maps(maps, graph));

//         GreedyBspLocking greedy;

//         WavefrontComponentScheduler scheduler(wavefront, greedy);
//         scheduler.set_check_isomorphism_groups(true);

//         BspArchitecture arch(90u, 1u, 1u);

//         arch.set_processors_consequ_types({30u, 60u}, {1000000, 500000});
//         // arch.print_architecture(std::cout);

//         BspInstance instance(graph, arch);
//         instance.setDiagonalCompatibilityMatrix(2);

//         auto [status, schedule] = scheduler.computeSchedule(instance);

//         BOOST_CHECK(status == RETURN_STATUS::SUCCESS);

//         BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
//         BOOST_CHECK(schedule.satisfiesNodeTypeConstraints());
//         BOOST_CHECK(schedule.hasValidCommSchedule());
//     }
// }

BOOST_AUTO_TEST_CASE(wavefront_component_divider_4) {

    std::vector<std::string> filenames_graph = test_graphs_dot();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    using graph_t = computational_dag_edge_idx_vector_impl_def_t;


    for (auto &filename_graph : filenames_graph) {


        BspInstance<graph_t> instance;
        auto& graph = instance.getComputationalDag();

        auto status_graph = file_reader::readComputationalDagDotFormat((cwd / filename_graph).string(), graph);

        if (!status_graph) {

            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        } else {
            std::cout << "File read:" << filename_graph << std::endl;
        }



        // for(const auto& vertex : graph.vertices()) {
        //     graph.setNodeType(vertex, 0);
        // }

        WavefrontComponentDivider<graph_t> wavefront;
        wavefront.set_split_method(WavefrontComponentDivider<graph_t>::MIN_DIFF);

        auto maps = wavefront.divide(graph);

        if (!maps.empty()) {

            BOOST_CHECK(check_vertex_maps(maps, graph));
        }

        BspLocking<graph_t> greedy;
        
        kl_total_cut<graph_t> kl_cut;
        

        WavefrontComponentScheduler<graph_t> scheduler(wavefront, greedy);
        scheduler.set_check_isomorphism_groups(true);

        // BspArchitecture arch(75u, 1u, 10000u);
        // arch.setMemoryConstraintType(LOCAL_INC_EDGES);
        // arch.set_processors_consequ_types({25u, 50u}, {100000000, 5000000});
   
        // instance.setDiagonalCompatibilityMatrix(2);

        BspSchedule<graph_t> schedule(instance);
        auto status = scheduler.computeSchedule(schedule);

        BOOST_CHECK(status == SUCCESS);

        BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
        // BOOST_CHECK(schedule.satisfiesNodeTypeConstraints());
        // BOOST_CHECK(schedule.satisfiesMemoryConstraints());

    }
}

