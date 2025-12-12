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

#define BOOST_TEST_MODULE kl
#include <boost/test/unit_test.hpp>
#include <filesystem>

#include "osp/auxiliary/io/arch_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_base.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_total_comm.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_total_cut.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/kl_improver_test.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/kl_include.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/kl_include_mt.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "test_graphs.hpp"

using namespace osp;

template <typename GraphT>
void AddMemWeights(GraphT &dag) {
    int memWeight = 1;
    int commWeight = 7;

    for (const auto &v : dag.vertices()) {
        dag.SetVertexWorkWeight(v, static_cast<VMemwT<GraphT>>(memWeight++ % 10 + 2));
        dag.SetVertexMemWeight(v, static_cast<VMemwT<GraphT>>(memWeight++ % 10 + 2));
        dag.SetVertexCommWeight(v, static_cast<VCommwT<GraphT>>(commWeight++ % 10 + 2));
    }
}

template <typename GraphT>
void AddNodeTypes(GraphT &dag) {
    unsigned nodeType = 0;

    for (const auto &v : dag.vertices()) {
        dag.SetVertexType(v, nodeType++ % 2);
    }
}

template <typename TableT>
void CheckEqualAffinityTable(TableT &table1, TableT &table2, const std::set<size_t> &nodes) {
    for (auto i : nodes) {
        BOOST_CHECK_EQUAL(table1[i].size(), table2[i].size());
        if (table1[i].size() != table2[i].size()) {
            continue;
        }
        for (size_t j = 0; j < table1[i].size(); ++j) {
            BOOST_CHECK_EQUAL(table1[i][j].size(), table2[i][j].size());
            if (table1[i][j].size() != table2[i][j].size()) {
                continue;
            }
            for (size_t k = 0; k < table1[i][j].size(); ++k) {
                BOOST_CHECK(std::abs(table1[i][j][k] - table2[i][j][k]) < 0.000001);

                if (std::abs(table1[i][j][k] - table2[i][j][k]) > 0.000001) {
                    std::cout << "Mismatch at [" << i << "][" << j << "][" << k << "]: table_1=" << table1[i][j][k]
                              << ", table_2=" << table2[i][j][k] << std::endl;
                }
            }
        }
    }
}

void CheckEqualLambdaMap(const std::vector<std::map<unsigned, unsigned>> &map1,
                         const std::vector<std::map<unsigned, unsigned>> &map2) {
    BOOST_CHECK_EQUAL(map1.size(), map2.size());
    if (map1.size() != map2.size()) {
        return;
    }

    for (size_t i = 0; i < map1.size(); ++i) {
        for (const auto &[key, value] : map1[i]) {
            BOOST_CHECK_EQUAL(value, map2[i].at(key));

            if (value != map2[i].at(key)) {
                std::cout << "Mismatch at [" << i << "][" << key << "]: map_1=" << value << ", map_2=" << map2[i].at(key)
                          << std::endl;
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(KlLambdaImproverWithNodeTypesTest) {
    std::vector<std::string> filenamesGraph = TestGraphs();

    using Graph = computational_dag_edge_idx_vector_impl_def_int_t;

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    GreedyBspScheduler<computational_dag_edge_idx_vector_impl_def_int_t> testScheduler;

    for (auto &filenameGraph : filenamesGraph) {
        BspInstance<Graph> instance;

        bool statusGraph
            = file_reader::readComputationalDagHyperdagFormatDB((cwd / filenameGraph).string(), instance.GetComputationalDag());

        instance.GetArchitecture().setSynchronisationCosts(5);
        instance.GetArchitecture().setCommunicationCosts(5);
        instance.GetArchitecture().setNumberOfProcessors(4);

        if (!statusGraph) {
            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        std::cout << "Instance: " << filenameGraph << std::endl;

        AddMemWeights(instance.GetComputationalDag());
        AddNodeTypes(instance.GetComputationalDag());

        instance.GetArchitecture().setProcessorsWithTypes({0, 0, 1, 1});

        instance.setDiagonalCompatibilityMatrix(2);

        BspSchedule<Graph> schedule(instance);
        const auto result = testScheduler.computeSchedule(schedule);

        BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, result);
        BOOST_CHECK_EQUAL(&schedule.GetInstance(), &instance);
        BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
        BOOST_CHECK(schedule.satisfiesNodeTypeConstraints());

        kl_total_lambda_comm_improver<Graph, no_local_search_memory_constraint, 1> kl;

        auto status = kl.improveSchedule(schedule);

        BOOST_CHECK(status == RETURN_STATUS::OSP_SUCCESS || status == RETURN_STATUS::BEST_FOUND);
        BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
        BOOST_CHECK(schedule.satisfiesNodeTypeConstraints());
    }
}

BOOST_AUTO_TEST_CASE(KlLambdaImproverOnTestGraphs) {
    std::vector<std::string> filenamesGraph = TestGraphs();

    using Graph = computational_dag_edge_idx_vector_impl_def_int_t;

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    GreedyBspScheduler<computational_dag_edge_idx_vector_impl_def_int_t> testScheduler;

    for (auto &filenameGraph : filenamesGraph) {
        BspInstance<Graph> instance;

        bool statusGraph
            = file_reader::readComputationalDagHyperdagFormatDB((cwd / filenameGraph).string(), instance.GetComputationalDag());

        instance.GetArchitecture().setSynchronisationCosts(5);
        instance.GetArchitecture().setCommunicationCosts(5);
        instance.GetArchitecture().setNumberOfProcessors(4);

        if (!statusGraph) {
            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        std::cout << "Instance: " << filenameGraph << std::endl;

        AddMemWeights(instance.GetComputationalDag());

        BspSchedule<Graph> schedule(instance);
        const auto result = testScheduler.computeSchedule(schedule);

        BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, result);
        BOOST_CHECK_EQUAL(&schedule.GetInstance(), &instance);
        BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

        kl_total_lambda_comm_improver<Graph, no_local_search_memory_constraint, 1> kl;

        auto status = kl.improveSchedule(schedule);

        BOOST_CHECK(status == RETURN_STATUS::OSP_SUCCESS || status == RETURN_STATUS::BEST_FOUND);
        BOOST_CHECK_EQUAL(schedule.satisfiesPrecedenceConstraints(), true);
    }
}

// BOOST_AUTO_TEST_CASE(kl_lambda_improver_incremental_update_test) {

//     using graph = computational_dag_edge_idx_vector_impl_def_int_t;
//     using VertexType = graph::vertex_idx;
//     using kl_move = kl_move_struct<double, VertexType>;

//     graph dag;

//     const VertexType v1 = dag.add_vertex(2, 9, 2);
//     const VertexType v2 = dag.add_vertex(3, 8, 4);
//     const VertexType v3 = dag.add_vertex(4, 7, 3);
//     const VertexType v4 = dag.add_vertex(5, 6, 2);
//     const VertexType v5 = dag.add_vertex(6, 5, 6);
//     const VertexType v6 = dag.add_vertex(7, 4, 2);
//     const VertexType v7 = dag.add_vertex(8, 3, 4);
//     const VertexType v8 = dag.add_vertex(9, 2, 1);

//     dag.add_edge(v1, v2, 2);
//     dag.add_edge(v1, v3, 2);
//     dag.add_edge(v1, v4, 2);
//     dag.add_edge(v1, v5, 2);
//     dag.add_edge(v1, v8, 2);
//     dag.add_edge(v2, v5, 12);
//     dag.add_edge(v2, v6, 12);
//     dag.add_edge(v2, v8, 12);
//     dag.add_edge(v3, v5, 6);
//     dag.add_edge(v3, v6, 7);
//     dag.add_edge(v5, v8, 9);
//     dag.add_edge(v4, v8, 9);

//     BspArchitecture<graph> arch;

//     BspInstance<graph> instance(dag, arch);

//     BspSchedule schedule(instance);

//     schedule.setAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});
//     schedule.setAssignedSupersteps({0, 0, 1, 1, 2, 2, 3, 3});

//     schedule.updateNumberOfSupersteps();

//     using cost_f = kl_hyper_total_comm_cost_function<graph, double, no_local_search_memory_constraint, 1>;
//     using kl_improver_test = kl_improver_test<graph, cost_f, no_local_search_memory_constraint, 1, double>;
//     kl_improver_test kl;

//     kl.setup_schedule(schedule);

//     auto &kl_active_schedule = kl.get_active_schedule();

//     BOOST_CHECK_EQUAL(kl_active_schedule.work_datastructures.step_max_work(0), 5.0);
//     BOOST_CHECK_EQUAL(kl_active_schedule.work_datastructures.step_second_max_work(0), 0.0);
//     BOOST_CHECK_EQUAL(kl_active_schedule.work_datastructures.step_max_work(1), 9.0);
//     BOOST_CHECK_EQUAL(kl_active_schedule.work_datastructures.step_second_max_work(1), 0.0);
//     BOOST_CHECK_EQUAL(kl_active_schedule.work_datastructures.step_max_work(2), 7.0);
//     BOOST_CHECK_EQUAL(kl_active_schedule.work_datastructures.step_second_max_work(2), 6.0);
//     BOOST_CHECK_EQUAL(kl_active_schedule.work_datastructures.step_max_work(3), 9.0);
//     BOOST_CHECK_EQUAL(kl_active_schedule.work_datastructures.step_second_max_work(3), 8.0);

//     BOOST_CHECK_EQUAL(kl_active_schedule.num_steps(), 4);
//     BOOST_CHECK_EQUAL(kl_active_schedule.is_feasible(), true);

//     auto node_selection = kl.insert_gain_heap_test({0, 1, 2, 3, 4, 5, 6, 7});

//     std::set<VertexType> nodes_to_check = {0, 1, 2, 3, 4, 5, 6, 7};
//     auto& affinity = kl.get_affinity_table();
//     auto& lambda_map = kl.get_comm_cost_f().node_lambda_map;

//     kl_move move_1(v7, 0.0, 0, 3, 0, 2);
//     kl.update_affinity_table_test(move_1, node_selection);

//     BspSchedule<graph> test_sched_1(instance);
//     kl.get_active_schedule_test(test_sched_1);
//     kl_improver_test kl_1;
//     kl_1.setup_schedule(test_sched_1);
//     kl_1.insert_gain_heap_test({0, 1, 2, 3, 4, 5, 6, 7});

//     nodes_to_check.erase(v7);

//     check_equal_lambda_map(lambda_map, kl_1.get_comm_cost_f().node_lambda_map);
//     check_equal_affinity_table(affinity, kl_1.get_affinity_table(), nodes_to_check);

//     kl_move move_2(v4, 0.0, 0, 1 , 0, 2);
//     kl.update_affinity_table_test(move_2, node_selection);

//     BspSchedule<graph> test_sched_2(instance);
//     kl.get_active_schedule_test(test_sched_2);
//     kl_improver_test kl_2;
//     kl_2.setup_schedule(test_sched_2);
//     kl_2.insert_gain_heap_test({0, 1, 2, 3, 4, 5, 6, 7});

//     nodes_to_check.erase(v4);

//     check_equal_lambda_map(lambda_map, kl_2.get_comm_cost_f().node_lambda_map);
//     check_equal_affinity_table(affinity, kl_2.get_affinity_table(), nodes_to_check);

//     kl_move move_3(v2, 0.0, 1, 0 , 0, 0);
//     kl.update_affinity_table_test(move_3, node_selection);

//     BspSchedule<graph> test_sched_3(instance);
//     kl.get_active_schedule_test(test_sched_3);
//     kl_improver_test kl_3;
//     kl_3.setup_schedule(test_sched_3);
//     kl_3.insert_gain_heap_test({0, 1, 2, 3, 4, 5, 6, 7});

//     nodes_to_check.erase(v2);

//     check_equal_lambda_map(lambda_map, kl_3.get_comm_cost_f().node_lambda_map);
//     check_equal_affinity_table(affinity, kl_3.get_affinity_table(), nodes_to_check);

//     kl_move move_4(v6, 0.0, 0, 2 , 1, 3);
//     kl.update_affinity_table_test(move_4, node_selection);

//     BspSchedule<graph> test_sched_4(instance);
//     kl.get_active_schedule_test(test_sched_4);
//     kl_improver_test kl_4;
//     kl_4.setup_schedule(test_sched_4);
//     kl_4.insert_gain_heap_test({0, 1, 2, 3, 4, 5, 6, 7});

//     nodes_to_check.erase(v6);

//     check_equal_lambda_map(lambda_map, kl_4.get_comm_cost_f().node_lambda_map);
//     check_equal_affinity_table(affinity, kl_4.get_affinity_table(), nodes_to_check);

//     kl_move move_5(v8, 0.0, 1, 3 , 0, 2);
//     kl.update_affinity_table_test(move_5, node_selection);

//     BspSchedule<graph> test_sched_5(instance);
//     kl.get_active_schedule_test(test_sched_5);
//     kl_improver_test kl_5;
//     kl_5.setup_schedule(test_sched_5);
//     kl_5.insert_gain_heap_test({0, 1, 2, 3, 4, 5, 6, 7});

//     nodes_to_check.erase(v8);

//     check_equal_lambda_map(lambda_map, kl_5.get_comm_cost_f().node_lambda_map);
//     check_equal_affinity_table(affinity, kl_5.get_affinity_table(), nodes_to_check);

//     kl_move move_6(v3, 0.0, 0, 1 , 1, 1);
//     kl.update_affinity_table_test(move_6, node_selection);

//     BspSchedule<graph> test_sched_6(instance);
//     kl.get_active_schedule_test(test_sched_6);
//     kl_improver_test kl_6;
//     kl_6.setup_schedule(test_sched_6);
//     kl_6.insert_gain_heap_test({0, 1, 2, 3, 4, 5, 6, 7});

//     nodes_to_check.erase(v3);

//     check_equal_lambda_map(lambda_map, kl_6.get_comm_cost_f().node_lambda_map);
//     check_equal_affinity_table(affinity, kl_6.get_affinity_table(), nodes_to_check);

// };

// BOOST_AUTO_TEST_CASE(kl_lambda_improver_inner_loop_test) {
//     using graph = computational_dag_edge_idx_vector_impl_def_int_t;
//     using VertexType = graph::vertex_idx;
//     using kl_move = kl_move_struct<double, VertexType>;

//     graph dag;

//     const VertexType v1 = dag.add_vertex(2, 9, 2);
//     const VertexType v2 = dag.add_vertex(3, 8, 4);
//     const VertexType v3 = dag.add_vertex(4, 7, 3);
//     const VertexType v4 = dag.add_vertex(5, 6, 2);
//     const VertexType v5 = dag.add_vertex(6, 5, 6);
//     const VertexType v6 = dag.add_vertex(7, 4, 2);
//     dag.add_vertex(8, 3, 4);
//     const VertexType v8 = dag.add_vertex(9, 2, 1);

//     dag.add_edge(v1, v2, 2);
//     dag.add_edge(v1, v3, 2);
//     dag.add_edge(v1, v4, 2);
//     dag.add_edge(v2, v5, 12);
//     dag.add_edge(v3, v5, 6);
//     dag.add_edge(v3, v6, 7);
//     dag.add_edge(v5, v8, 9);
//     dag.add_edge(v4, v8, 9);

//     BspArchitecture<graph> arch;

//     BspInstance<graph> instance(dag, arch);

//     BspSchedule schedule(instance);

//     schedule.setAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});
//     schedule.setAssignedSupersteps({0, 0, 1, 1, 2, 2, 3, 3});

//     schedule.updateNumberOfSupersteps();

//     using cost_f = kl_hyper_total_comm_cost_function<graph, double, no_local_search_memory_constraint, 1>;
//     using kl_improver_test = kl_improver_test<graph, cost_f, no_local_search_memory_constraint, 1, double>;
//     kl_improver_test kl;

//     kl.setup_schedule(schedule);

//     auto &kl_active_schedule = kl.get_active_schedule();

//     BOOST_CHECK_EQUAL(kl_active_schedule.work_datastructures.step_max_work(0), 5.0);
//     BOOST_CHECK_EQUAL(kl_active_schedule.work_datastructures.step_second_max_work(0), 0.0);
//     BOOST_CHECK_EQUAL(kl_active_schedule.work_datastructures.step_max_work(1), 9.0);
//     BOOST_CHECK_EQUAL(kl_active_schedule.work_datastructures.step_second_max_work(1), 0.0);
//     BOOST_CHECK_EQUAL(kl_active_schedule.work_datastructures.step_max_work(2), 7.0);
//     BOOST_CHECK_EQUAL(kl_active_schedule.work_datastructures.step_second_max_work(2), 6.0);
//     BOOST_CHECK_EQUAL(kl_active_schedule.work_datastructures.step_max_work(3), 9.0);
//     BOOST_CHECK_EQUAL(kl_active_schedule.work_datastructures.step_second_max_work(3), 8.0);

//      auto node_selection = kl.insert_gain_heap_test({0, 1, 2, 3, 4, 5, 6, 7});

//     std::set<VertexType> nodes_to_check = {0, 1, 2, 3, 4, 5, 6, 7};
//     auto& affinity = kl.get_affinity_table();
//     auto& lambda_map = kl.get_comm_cost_f().node_lambda_map;

//     kl_move move_2(v4, 0.0, 0, 1 , 1, 2);
//     kl.update_affinity_table_test(move_2, node_selection);

//     BspSchedule<graph> test_sched_2(instance);
//     kl.get_active_schedule_test(test_sched_2);
//     kl_improver_test kl_2;
//     kl_2.setup_schedule(test_sched_2);
//     kl_2.insert_gain_heap_test_penalty({0, 1, 2, 3, 4, 5, 6, 7});

//     nodes_to_check.erase(v4);

//     check_equal_lambda_map(lambda_map, kl_2.get_comm_cost_f().node_lambda_map);
//     check_equal_affinity_table(affinity, kl_2.get_affinity_table(), nodes_to_check);

//     kl_move move_3(v2, 0.0, 1, 0 , 0, 1);
//     kl.update_affinity_table_test(move_3, node_selection);

//     BspSchedule<graph> test_sched_3(instance);
//     kl.get_active_schedule_test(test_sched_3);
//     kl_improver_test kl_3;
//     kl_3.setup_schedule(test_sched_3);
//     kl_3.insert_gain_heap_test_penalty({0, 1, 2, 3, 4, 5, 6, 7});

//     nodes_to_check.erase(v2);

//     check_equal_lambda_map(lambda_map, kl_3.get_comm_cost_f().node_lambda_map);
//     check_equal_affinity_table(affinity, kl_3.get_affinity_table(), nodes_to_check);

//     kl_move move_4(v6, 0.0, 0, 2 , 1, 3);
//     kl.update_affinity_table_test(move_4, node_selection);

//     BspSchedule<graph> test_sched_4(instance);
//     kl.get_active_schedule_test(test_sched_4);
//     kl_improver_test kl_4;
//     kl_4.setup_schedule(test_sched_4);
//     kl_4.insert_gain_heap_test_penalty({0, 1, 2, 3, 4, 5, 6, 7});

//     nodes_to_check.erase(v6);

//     check_equal_lambda_map(lambda_map, kl_4.get_comm_cost_f().node_lambda_map);
//     check_equal_affinity_table(affinity, kl_4.get_affinity_table(), nodes_to_check);

//     kl_move move_5(v8, 0.0, 1, 3 , 0, 3);
//     kl.update_affinity_table_test(move_5, node_selection);

//     BspSchedule<graph> test_sched_5(instance);
//     kl.get_active_schedule_test(test_sched_5);
//     kl_improver_test kl_5;
//     kl_5.setup_schedule(test_sched_5);
//     kl_5.insert_gain_heap_test_penalty({0, 1, 2, 3, 4, 5, 6, 7});

//     nodes_to_check.erase(v8);

//     check_equal_lambda_map(lambda_map, kl_5.get_comm_cost_f().node_lambda_map);
//     check_equal_affinity_table(affinity, kl_5.get_affinity_table(), nodes_to_check);

//     kl_move move_6(v3, 0.0, 0, 1 , 1, 1);
//     kl.update_affinity_table_test(move_6, node_selection);

//     BspSchedule<graph> test_sched_6(instance);
//     kl.get_active_schedule_test(test_sched_6);
//     kl_improver_test kl_6;
//     kl_6.setup_schedule(test_sched_6);
//     kl_6.insert_gain_heap_test_penalty({0, 1, 2, 3, 4, 5, 6, 7});

//     nodes_to_check.erase(v3);

//     check_equal_lambda_map(lambda_map, kl_6.get_comm_cost_f().node_lambda_map);
//     check_equal_affinity_table(affinity, kl_6.get_affinity_table(), nodes_to_check);

// };

BOOST_AUTO_TEST_CASE(KlLambdaImproverInnerLoopPenaltyTest) {
    using Graph = computational_dag_edge_idx_vector_impl_def_int_t;
    using VertexType = Graph::vertex_idx;

    Graph dag;

    const VertexType v1 = dag.add_vertex(2, 9, 2);
    const VertexType v2 = dag.add_vertex(3, 8, 4);
    const VertexType v3 = dag.add_vertex(4, 7, 3);
    const VertexType v4 = dag.add_vertex(5, 6, 2);
    const VertexType v5 = dag.add_vertex(6, 5, 6);
    const VertexType v6 = dag.add_vertex(7, 4, 2);
    dag.add_vertex(8, 3, 4);
    const VertexType v8 = dag.add_vertex(9, 2, 1);

    dag.add_edge(v1, v2, 2);
    dag.add_edge(v1, v3, 2);
    dag.add_edge(v1, v4, 2);
    dag.add_edge(v2, v5, 12);
    dag.add_edge(v3, v5, 6);
    dag.add_edge(v3, v6, 7);
    dag.add_edge(v5, v8, 9);
    dag.add_edge(v4, v8, 9);

    BspArchitecture<Graph> arch;

    BspInstance<Graph> instance(dag, arch);

    BspSchedule schedule(instance);

    schedule.setAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});
    schedule.setAssignedSupersteps({0, 0, 1, 1, 2, 2, 3, 3});

    schedule.updateNumberOfSupersteps();

    using CostF = kl_hyper_total_comm_cost_function<Graph, double, no_local_search_memory_constraint, 1>;
    using KlImproverTest = kl_improver_test<Graph, CostF, no_local_search_memory_constraint, 1, double>;
    KlImproverTest kl;

    kl.setup_schedule(schedule);

    auto &klActiveSchedule = kl.get_active_schedule();

    BOOST_CHECK_EQUAL(klActiveSchedule.work_datastructures.step_max_work(0), 5.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.work_datastructures.step_second_max_work(0), 0.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.work_datastructures.step_max_work(1), 9.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.work_datastructures.step_second_max_work(1), 0.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.work_datastructures.step_max_work(2), 7.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.work_datastructures.step_second_max_work(2), 6.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.work_datastructures.step_max_work(3), 9.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.work_datastructures.step_second_max_work(3), 8.0);

    BOOST_CHECK_EQUAL(klActiveSchedule.num_steps(), 4);
    BOOST_CHECK_EQUAL(klActiveSchedule.is_feasible(), true);

    auto nodeSelection = kl.insert_gain_heap_test_penalty({2, 3});

    auto recomputeMaxGain = kl.run_inner_iteration_test();    // best move 3
    std::cout << "------------------------recompute max_gain: { ";
    for (const auto &[key, value] : recomputeMaxGain) {
        std::cout << key << " ";
    }
    std::cout << "}" << std::endl;

    BOOST_CHECK_CLOSE(kl.get_comm_cost_f().compute_schedule_cost_test(), kl.get_current_cost(), 0.00001);

    recomputeMaxGain = kl.run_inner_iteration_test();    // best move 0
    std::cout << "recompute max_gain: { ";
    for (const auto &[key, value] : recomputeMaxGain) {
        std::cout << key << " ";
    }
    std::cout << "}" << std::endl;

    BOOST_CHECK_CLOSE(kl.get_comm_cost_f().compute_schedule_cost_test(), kl.get_current_cost(), 0.00001);

    recomputeMaxGain = kl.run_inner_iteration_test();    // best move 1
    std::cout << "recompute max_gain: { ";
    for (const auto &[key, value] : recomputeMaxGain) {
        std::cout << key << " ";
    }
    std::cout << "}" << std::endl;

    BOOST_CHECK_CLOSE(kl.get_comm_cost_f().compute_schedule_cost_test(), kl.get_current_cost(), 0.00001);

    recomputeMaxGain = kl.run_inner_iteration_test();
    std::cout << "recompute max_gain: { ";
    for (const auto &[key, value] : recomputeMaxGain) {
        std::cout << key << " ";
    }
    std::cout << "}" << std::endl;

    BOOST_CHECK_CLOSE(kl.get_comm_cost_f().compute_schedule_cost_test(), kl.get_current_cost(), 0.00001);
}

BOOST_AUTO_TEST_CASE(KlLambdaImproverInnerLoopLambdaMapTest) {
    using Graph = computational_dag_edge_idx_vector_impl_def_int_t;
    using VertexType = Graph::vertex_idx;

    Graph dag;

    const VertexType v1 = dag.add_vertex(2, 9, 2);
    const VertexType v2 = dag.add_vertex(3, 8, 4);
    const VertexType v3 = dag.add_vertex(4, 7, 3);
    const VertexType v4 = dag.add_vertex(5, 6, 2);
    const VertexType v5 = dag.add_vertex(6, 5, 6);
    const VertexType v6 = dag.add_vertex(7, 4, 2);
    const VertexType v7 = dag.add_vertex(8, 3, 4);
    const VertexType v8 = dag.add_vertex(9, 2, 1);

    dag.add_edge(v1, v2, 2);
    dag.add_edge(v1, v3, 2);
    dag.add_edge(v1, v4, 2);
    dag.add_edge(v2, v5, 12);
    dag.add_edge(v3, v5, 6);
    dag.add_edge(v3, v6, 7);
    dag.add_edge(v5, v8, 9);
    dag.add_edge(v4, v8, 9);

    BspArchitecture<Graph> arch;

    BspInstance<Graph> instance(dag, arch);

    BspSchedule schedule(instance);

    schedule.setAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});
    schedule.setAssignedSupersteps({0, 0, 1, 1, 2, 2, 3, 3});

    schedule.updateNumberOfSupersteps();

    using CostF = kl_hyper_total_comm_cost_function<Graph, double, no_local_search_memory_constraint, 1>;
    using KlImproverTest = kl_improver_test<Graph, CostF, no_local_search_memory_constraint, 1, double>;
    KlImproverTest kl;

    kl.setup_schedule(schedule);

    BOOST_CHECK_CLOSE(kl.get_comm_cost_f().compute_schedule_cost_test(), kl.get_current_cost(), 0.00001);

    auto nodeSelection = kl.insert_gain_heap_test_penalty({7});

    auto recomputeMaxGain = kl.run_inner_iteration_test();
    std::cout << "-----------recompute max_gain: { ";
    for (const auto &[key, value] : recomputeMaxGain) {
        std::cout << key << " ";
    }
    std::cout << "}" << std::endl;

    BOOST_CHECK_CLOSE(kl.get_comm_cost_f().compute_schedule_cost_test(), kl.get_current_cost(), 0.00001);

    auto &lambdaMap = kl.get_comm_cost_f().node_lambda_map;

    BOOST_CHECK(lambdaMap.get_proc_entry(v1, 0) == 2);
    BOOST_CHECK(lambdaMap.get_proc_entry(v1, 1) == 1);
    BOOST_CHECK(lambdaMap.has_no_proc_entry(v2, 0));
    BOOST_CHECK(lambdaMap.get_proc_entry(v2, 1) == 1);
    BOOST_CHECK(lambdaMap.get_proc_entry(v3, 0) == 1);
    BOOST_CHECK(lambdaMap.get_proc_entry(v3, 1) == 1);
    BOOST_CHECK(lambdaMap.has_no_proc_entry(v4, 0));
    BOOST_CHECK(lambdaMap.get_proc_entry(v4, 1) == 1);
    BOOST_CHECK(lambdaMap.has_no_proc_entry(v5, 0));
    BOOST_CHECK(lambdaMap.get_proc_entry(v5, 1) == 1);
    BOOST_CHECK(lambdaMap.has_no_proc_entry(v6, 0));
    BOOST_CHECK(lambdaMap.has_no_proc_entry(v6, 0));
    BOOST_CHECK(lambdaMap.has_no_proc_entry(v7, 0));
    BOOST_CHECK(lambdaMap.has_no_proc_entry(v7, 0));
    BOOST_CHECK(lambdaMap.has_no_proc_entry(v8, 0));
    BOOST_CHECK(lambdaMap.has_no_proc_entry(v8, 0));

    recomputeMaxGain = kl.run_inner_iteration_test();
    std::cout << "recompute max_gain: { ";
    for (const auto &[key, value] : recomputeMaxGain) {
        std::cout << key << " ";
    }
    std::cout << "}" << std::endl;

    BOOST_CHECK_CLOSE(kl.get_comm_cost_f().compute_schedule_cost_test(), kl.get_current_cost(), 0.00001);

    recomputeMaxGain = kl.run_inner_iteration_test();
    std::cout << "recompute max_gain: { ";
    for (const auto &[key, value] : recomputeMaxGain) {
        std::cout << key << " ";
    }
    std::cout << "}" << std::endl;

    BOOST_CHECK_CLOSE(kl.get_comm_cost_f().compute_schedule_cost_test(), kl.get_current_cost(), 0.00001);

    recomputeMaxGain = kl.run_inner_iteration_test();
    std::cout << "recompute max_gain: { ";
    for (const auto &[key, value] : recomputeMaxGain) {
        std::cout << key << " ";
    }
    std::cout << "}" << std::endl;

    BOOST_CHECK_CLOSE(kl.get_comm_cost_f().compute_schedule_cost_test(), kl.get_current_cost(), 0.00001);
}

// BOOST_AUTO_TEST_CASE(kl_lambda_total_comm_large_test_graphs) {
//     std::vector<std::string> filenames_graph = large_spaa_graphs();
//     using graph = computational_dag_edge_idx_vector_impl_def_int_t;
//     // Getting root git directory
//     std::filesystem::path cwd = std::filesystem::current_path();
//     std::cout << cwd << std::endl;
//     while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
//         cwd = cwd.parent_path();
//         std::cout << cwd << std::endl;
//     }

//     for (auto &filename_graph : filenames_graph) {
//         GreedyBspScheduler<computational_dag_edge_idx_vector_impl_def_int_t> test_scheduler;
//         BspInstance<graph> instance;
//         bool status_graph = file_reader::readComputationalDagHyperdagFormatDB((cwd / filename_graph).string(),
//                                                                             instance.GetComputationalDag());

//         instance.GetArchitecture().setSynchronisationCosts(500);
//         instance.GetArchitecture().setCommunicationCosts(5);
//         instance.GetArchitecture().setNumberOfProcessors(4);

//         std::vector<std::vector<int>> send_cost = {{0,1,4,4},
//                                                    {1,0,4,4},
//                                                    {4,4,0,1},
//                                                    {4,4,1,0}};

//         instance.GetArchitecture().SetSendCosts(send_cost);

//         if (!status_graph) {

//             std::cout << "Reading files failed." << std::endl;
//             BOOST_CHECK(false);
//         }

//         add_mem_weights(instance.GetComputationalDag());

//         BspSchedule<graph> schedule(instance);
//         const auto result = test_scheduler.computeSchedule(schedule);

//         schedule.updateNumberOfSupersteps();

//         std::cout << "initial scedule with costs: " << schedule.computeTotalLambdaCosts() << " and " <<
//         schedule.NumberOfSupersteps() << " number of supersteps"<< std::endl;

//         BspSchedule<graph> schedule_2(schedule);

//         BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, result);
//         BOOST_CHECK_EQUAL(&schedule.GetInstance(), &instance);
//         BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

//         kl_total_lambda_comm_improver<graph,no_local_search_memory_constraint,1> kl;
//         auto start_time = std::chrono::high_resolution_clock::now();
//         auto status = kl.improveSchedule(schedule);
//         auto finish_time = std::chrono::high_resolution_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

//         std::cout << "kl new finished in " << duration << " seconds, costs: " << schedule.computeTotalLambdaCosts() << " with "
//         << schedule.NumberOfSupersteps() << " number of supersteps"<< std::endl;

//         BOOST_CHECK(status == RETURN_STATUS::OSP_SUCCESS || status == RETURN_STATUS::BEST_FOUND);
//         BOOST_CHECK_EQUAL(schedule.satisfiesPrecedenceConstraints(), true);

//         // kl_total_comm_test<graph> kl_old;

//         // start_time = std::chrono::high_resolution_clock::now();
//         // status = kl_old.improve_schedule_test_2(schedule_2);
//         // finish_time = std::chrono::high_resolution_clock::now();

//         // duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

//         // std::cout << "kl old finished in " << duration << " seconds, costs: " << schedule_2.computeTotalCosts() << " with "
//         << schedule_2.NumberOfSupersteps() << " number of supersteps"<< std::endl;

//         // BOOST_CHECK(status == RETURN_STATUS::OSP_SUCCESS || status == RETURN_STATUS::BEST_FOUND);
//         // BOOST_CHECK_EQUAL(schedule_2.satisfiesPrecedenceConstraints(), true);

//     }
// }

// BOOST_AUTO_TEST_CASE(kl_lambda_total_comm_large_test_graphs_mt) {
//     std::vector<std::string> filenames_graph = large_spaa_graphs();
//     using graph = computational_dag_edge_idx_vector_impl_def_int_t;
//     // Getting root git directory
//     std::filesystem::path cwd = std::filesystem::current_path();
//     std::cout << cwd << std::endl;
//     while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
//         cwd = cwd.parent_path();
//         std::cout << cwd << std::endl;
//     }

//     for (auto &filename_graph : filenames_graph) {
//         GreedyBspScheduler<computational_dag_edge_idx_vector_impl_def_int_t> test_scheduler;
//         BspInstance<graph> instance;
//         bool status_graph = file_reader::readComputationalDagHyperdagFormatDB((cwd / filename_graph).string(),
//                                                                             instance.GetComputationalDag());

//         instance.GetArchitecture().setSynchronisationCosts(500);
//         instance.GetArchitecture().setCommunicationCosts(5);
//         instance.GetArchitecture().setNumberOfProcessors(4);

//         std::vector<std::vector<int>> send_cost = {{0,1,4,4},
//                                                    {1,0,4,4},
//                                                    {4,4,0,1},
//                                                    {4,4,1,0}};

//         instance.GetArchitecture().SetSendCosts(send_cost);

//         if (!status_graph) {

//             std::cout << "Reading files failed." << std::endl;
//             BOOST_CHECK(false);
//         }

//         add_mem_weights(instance.GetComputationalDag());

//         BspSchedule<graph> schedule(instance);
//         const auto result = test_scheduler.computeSchedule(schedule);

//         schedule.updateNumberOfSupersteps();

//         std::cout << "initial scedule with costs: " << schedule.computeTotalLambdaCosts() << " and " <<
//         schedule.NumberOfSupersteps() << " number of supersteps"<< std::endl;

//         BspSchedule<graph> schedule_2(schedule);

//         BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, result);
//         BOOST_CHECK_EQUAL(&schedule.GetInstance(), &instance);
//         BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

//         kl_total_lambda_comm_improver_mt<graph,no_local_search_memory_constraint,1> kl;
//         auto start_time = std::chrono::high_resolution_clock::now();
//         auto status = kl.improveSchedule(schedule);
//         auto finish_time = std::chrono::high_resolution_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

//         std::cout << "kl new finished in " << duration << " seconds, costs: " << schedule.computeTotalLambdaCosts() << " with "
//         << schedule.NumberOfSupersteps() << " number of supersteps"<< std::endl;

//         BOOST_CHECK(status == RETURN_STATUS::OSP_SUCCESS || status == RETURN_STATUS::BEST_FOUND);
//         BOOST_CHECK_EQUAL(schedule.satisfiesPrecedenceConstraints(), true);

//         // kl_total_comm_test<graph> kl_old;

//         // start_time = std::chrono::high_resolution_clock::now();
//         // status = kl_old.improve_schedule_test_2(schedule_2);
//         // finish_time = std::chrono::high_resolution_clock::now();

//         // duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

//         // std::cout << "kl old finished in " << duration << " seconds, costs: " << schedule_2.computeTotalCosts() << " with "
//         << schedule_2.NumberOfSupersteps() << " number of supersteps"<< std::endl;

//         // BOOST_CHECK(status == RETURN_STATUS::OSP_SUCCESS || status == RETURN_STATUS::BEST_FOUND);
//         // BOOST_CHECK_EQUAL(schedule_2.satisfiesPrecedenceConstraints(), true);

//     }
// }
