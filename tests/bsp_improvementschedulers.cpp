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

#define BOOST_TEST_MODULE BSP_IMPROVEMENTSCHEDULERS
#include <boost/test/unit_test.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include "osp/bsp/scheduler/ImprovementScheduler.hpp"


// std::vector<std::string> test_graphs() {
//     return {"data/spaa/small/instance_exp_N20_K4_nzP0d2.txt", "data/spaa/small/instance_kNN_N20_K5_nzP0d2.txt",
//             "data/spaa/small/instance_exp_N10_K8_nzP0d2.txt"};  //, "data/spaa/medium/instance_kNN_N50_K4_nzP0d18.txt"}; //,
// //            "data/spaa/large/instance_exp_N50_K12_nzP0d15.txt"};
// }

// std::vector<std::string> test_architectures() {
//     return {"data/machine_params/p3.txt", "data/machine_params/p8_g1_l5_numa2.txt"};
// }

// void print_bsp_schedule(const BspSchedule &bsp_schedule) {
//     std::vector<std::vector<std::vector<unsigned>>> schedule(
//         bsp_schedule.numberOfSupersteps(),
//         std::vector<std::vector<unsigned>>(bsp_schedule.getInstance().numberOfProcessors(), std::vector<unsigned>()));

//     for (size_t node = 0; node < bsp_schedule.getInstance().numberOfVertices(); node++) {
//         schedule[bsp_schedule.assignedSuperstep(node)][bsp_schedule.assignedProcessor(node)].push_back(node);
//     }

//     std::cout << std::endl << "Schedule:" << std::endl;
//     for (size_t i = 0; i < schedule.size(); i++) {
//         std::cout << std::endl << "Superstep " << i << std::endl;
//         for (size_t j = 0; j < schedule[i].size(); j++) {
//             std::cout << "Processor " << j << ": ";
//             for (auto &node : schedule[i][j]) {
//                 std::cout << node << ", ";
//             }
//             std::cout << std::endl;
//         }
//     }
// }

// void run_test(ImprovementScheduler *test_improver) {
//     // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
//     std::vector<std::string> filenames_graph = test_graphs();
//     std::vector<std::string> filenames_architectures = test_architectures();

//     // Getting root git directory
//     std::filesystem::path cwd = std::filesystem::current_path();
//     std::cout << cwd << std::endl;
//     while ((!cwd.empty()) && (cwd.filename() != "one-stop-parallel")) {
//         cwd = cwd.parent_path();
//         std::cout << cwd << std::endl;
//     }

//     for (auto &filename_graph : filenames_graph) {
//         for (auto &filename_machine : filenames_architectures) {
//             std::string name_graph =
//                 filename_graph.substr(filename_machine.find_last_of("/\\") + 1, filename_graph.find_last_of("."));
//             std::string name_machine = filename_machine.substr(filename_machine.find_last_of("/\\") + 1);
//             name_machine = name_machine.substr(0, name_machine.rfind("."));

//             std::cout << std::endl << "Graph: " << name_graph << std::endl;
//             std::cout << "Architecture: " << name_machine << std::endl;

//             auto [status_graph, graph] =
//                 FileReader::readComputationalDagHyperdagFormat((cwd / filename_graph).string());
//             auto [status_architecture, architecture] =
//                 FileReader::readBspArchitecture((cwd / filename_machine).string());

//             if (!status_graph || !status_architecture) {

//                 std::cout << "Reading files failed." << std::endl;
//                 BOOST_CHECK(false);
//             }

//             BspInstance instance(graph, architecture);


//             RandomBadGreedy test0;

//             std::pair<RETURN_STATUS, BspSchedule> result0 = test0.computeSchedule(instance);
//             test_improver->improveSchedule(result0.second);

//             print_bsp_schedule(result0.second);

//             BOOST_CHECK_EQUAL(SUCCESS, result0.first);

//             BOOST_CHECK(result0.second.satisfiesPrecedenceConstraints());
//             BOOST_CHECK(result0.second.hasValidCommSchedule());


//             BalDMixR test1;

//             std::pair<RETURN_STATUS, BspSchedule> result1 = test1.computeSchedule(instance);
//             test_improver->improveSchedule(result1.second);

//             print_bsp_schedule(result1.second);

//             BOOST_CHECK_EQUAL(SUCCESS, result1.first);

//             BOOST_CHECK(result1.second.satisfiesPrecedenceConstraints());
//             BOOST_CHECK(result1.second.hasValidCommSchedule());


//             HDagg_simple test2;

//             std::pair<RETURN_STATUS, BspSchedule> result2 = test2.computeSchedule(instance);
//             test_improver->improveSchedule(result2.second);

//             print_bsp_schedule(result2.second);

//             BOOST_CHECK_EQUAL(SUCCESS, result2.first);

//             BOOST_CHECK(result2.second.satisfiesPrecedenceConstraints());
//             BOOST_CHECK(result2.second.hasValidCommSchedule());

//         }
//     }
// };



BOOST_AUTO_TEST_CASE(Hungarian_alg_process_permuter_test) {
    // Hungarian_alg_process_permuter test;
    // run_test(&test);
}

// BOOST_AUTO_TEST_CASE(HillClimbingScheduler_test) {
//     HillClimbingScheduler test;
//     run_test(&test);
// }

// BOOST_AUTO_TEST_CASE(LKTotalCommScheduler_test) {
//     kl_total_comm test;
    
//     test.setTimeLimitSeconds(10);
//     test.set_compute_with_time_limit(true);
//     run_test(&test);
// }




