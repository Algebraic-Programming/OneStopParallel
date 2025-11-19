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

#define BOOST_TEST_MODULE COARSER_TEST
#include <boost/test/unit_test.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>

#include "osp/bsp/scheduler/CoarseAndSchedule.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/coarser/BspScheduleCoarser.hpp"
#include "osp/coarser/coarser_util.hpp"
#include "osp/coarser/funnel/FunnelBfs.hpp"
#include "osp/coarser/hdagg/hdagg_coarser.hpp"
#include "osp/coarser/Sarkar/Sarkar.hpp"
#include "osp/coarser/Sarkar/SarkarMul.hpp"
#include "osp/coarser/SquashA/SquashA.hpp"
#include "osp/coarser/SquashA/SquashAMul.hpp"
#include "osp/coarser/top_order/top_order_coarser.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "osp/graph_implementations/adj_list_impl/compact_sparse_graph.hpp"
#include "osp/graph_implementations/adj_list_impl/compact_sparse_graph_edge_desc.hpp"
#include "osp/auxiliary/io/arch_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/auxiliary/io/general_file_reader.hpp"
#include "test_graphs.hpp"

using namespace osp;

using VertexType = vertex_idx_t<computational_dag_edge_idx_vector_impl_def_t>;

bool check_vertex_map(std::vector<std::vector<VertexType>> &map, std::size_t size) {

    std::unordered_set<VertexType> vertices;

    for (auto &v : map) {
        for (auto &v2 : v) {
            if (vertices.find(v2) != vertices.end()) {
                return false;
            }
            vertices.insert(v2);
        }
    }

    return vertices.size() == size;
}

template<typename ComputationalDag>
bool check_vertex_map_constraints(std::vector<std::vector<VertexType>> &map, ComputationalDag &dag,
                                  v_type_t<ComputationalDag> size_threshold,
                                  v_memw_t<ComputationalDag> memory_threshold,
                                  v_workw_t<ComputationalDag> work_threshold,
                                  v_commw_t<ComputationalDag> communication_threshold) {

    std::unordered_set<VertexType> vertices;

    for (auto &super_node : map) {

        v_memw_t<ComputationalDag> memory = 0;
        v_workw_t<ComputationalDag> work = 0;
        v_commw_t<ComputationalDag> communication = 0;

        if (super_node.size() > size_threshold) {
            return false;
        }

        if (super_node.size() == 0) {
            return false;
        }

        for (auto &v : super_node) {

            memory += dag.vertex_mem_weight(v);
            work += dag.vertex_work_weight(v);
            communication += dag.vertex_comm_weight(v);

            if (dag.vertex_type(v) != dag.vertex_type(super_node[0])) {
                return false;
            }
        }

        if (memory > memory_threshold || work > work_threshold || communication > communication_threshold) {
            return false;
        }
    }
    return true;
}

BOOST_AUTO_TEST_CASE(coarser_hdagg_test) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenames_graph = tiny_spaa_graphs();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filename_graph : filenames_graph) {

        std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
        name_graph = name_graph.substr(0, name_graph.find_last_of("."));

        std::cout << std::endl << "Graph: " << name_graph << std::endl;

        using graph_t = computational_dag_edge_idx_vector_impl_def_t;

        BspInstance<graph_t> instance;

        bool status_graph = file_reader::readGraph((cwd / filename_graph).string(),
                                                                            instance.getComputationalDag());

        bool status_architecture = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(),
                                                                    instance.getArchitecture());

        if (!status_graph || !status_architecture) {

            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        BspInstance<graph_t> coarse_instance;
        coarse_instance.setArchitecture(instance.getArchitecture());
        std::vector<std::vector<VertexType>> vertex_map;
        std::vector<VertexType> reverse_vertex_map;

        hdagg_coarser<graph_t, graph_t> coarser;

        BOOST_CHECK_EQUAL(coarser.getCoarserName(), "hdagg_coarser");

        coarser.coarsenDag(instance.getComputationalDag(), coarse_instance.getComputationalDag(), reverse_vertex_map);

        vertex_map = coarser_util::invert_vertex_contraction_map<graph_t, graph_t>(reverse_vertex_map);

        BOOST_CHECK(check_vertex_map(vertex_map, instance.getComputationalDag().num_vertices()));

        GreedyBspScheduler<graph_t> scheduler;
        BspSchedule<graph_t> schedule(coarse_instance);

        const auto status_sched = scheduler.computeSchedule(schedule);

        BOOST_CHECK(status_sched == RETURN_STATUS::OSP_SUCCESS);
        BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

        BspSchedule<graph_t> schedule_out(instance);

        BOOST_CHECK_EQUAL(coarser_util::pull_back_schedule(schedule, vertex_map, schedule_out), true);
        BOOST_CHECK(schedule_out.satisfiesPrecedenceConstraints());

        CoarseAndSchedule<graph_t, graph_t> coarse_and_schedule(coarser, scheduler);
        BspSchedule<graph_t> schedule2(instance);

        const auto status = coarse_and_schedule.computeSchedule(schedule2);
        BOOST_CHECK(status == RETURN_STATUS::OSP_SUCCESS || status == RETURN_STATUS::BEST_FOUND);
        BOOST_CHECK(schedule2.satisfiesPrecedenceConstraints());
    }
}

BOOST_AUTO_TEST_CASE(coarser_hdagg_test_diff_graph_impl) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenames_graph = tiny_spaa_graphs();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filename_graph : filenames_graph) {

        std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
        name_graph = name_graph.substr(0, name_graph.find_last_of("."));

        std::cout << std::endl << "Graph: " << name_graph << std::endl;

        using graph_t1 = computational_dag_edge_idx_vector_impl_def_t;
        using graph_t2 = computational_dag_vector_impl_def_t;

        BspInstance<graph_t1> instance;

        bool status_graph = file_reader::readGraph((cwd / filename_graph).string(),
                                                                            instance.getComputationalDag());

        bool status_architecture = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(),
                                                                    instance.getArchitecture());

        if (!status_graph || !status_architecture) {

            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        BspInstance<graph_t2> coarse_instance;
        BspArchitecture<graph_t2> architecture_t2(instance.getArchitecture());
        coarse_instance.setArchitecture(architecture_t2);
        std::vector<std::vector<VertexType>> vertex_map;
        std::vector<VertexType> reverse_vertex_map;

        hdagg_coarser<graph_t1, graph_t2> coarser;

        coarser.coarsenDag(instance.getComputationalDag(), coarse_instance.getComputationalDag(), reverse_vertex_map);

        vertex_map = coarser_util::invert_vertex_contraction_map<graph_t1, graph_t2>(reverse_vertex_map);

        BOOST_CHECK(check_vertex_map(vertex_map, instance.getComputationalDag().num_vertices()));

        GreedyBspScheduler<graph_t2> scheduler;
        BspSchedule<graph_t2> schedule(coarse_instance);

        auto status_sched = scheduler.computeSchedule(schedule);

        BOOST_CHECK(status_sched == RETURN_STATUS::OSP_SUCCESS);
        BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

        BspSchedule<graph_t1> schedule_out(instance);

        BOOST_CHECK_EQUAL(coarser_util::pull_back_schedule(schedule, vertex_map, schedule_out), true);
        BOOST_CHECK(schedule_out.satisfiesPrecedenceConstraints());

        CoarseAndSchedule<graph_t1, graph_t2> coarse_and_schedule(coarser, scheduler);
        BspSchedule<graph_t1> schedule2(instance);

        auto status = coarse_and_schedule.computeSchedule(schedule2);
        BOOST_CHECK(status == RETURN_STATUS::OSP_SUCCESS || status == RETURN_STATUS::BEST_FOUND);
        BOOST_CHECK(schedule2.satisfiesPrecedenceConstraints());
    }
}

BOOST_AUTO_TEST_CASE(coarser_bspschedule_test) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenames_graph = tiny_spaa_graphs();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filename_graph : filenames_graph) {

        std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
        name_graph = name_graph.substr(0, name_graph.find_last_of("."));

        std::cout << std::endl << "Graph: " << name_graph << std::endl;

        using graph_t = computational_dag_edge_idx_vector_impl_def_t;

        BspInstance<graph_t> instance;

        bool status_graph = file_reader::readGraph((cwd / filename_graph).string(),
                                                                            instance.getComputationalDag());

        bool status_architecture = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(),
                                                                    instance.getArchitecture());

        if (!status_graph || !status_architecture) {

            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        BspInstance<graph_t> coarse_instance;
        coarse_instance.setArchitecture(instance.getArchitecture());
        std::vector<std::vector<VertexType>> vertex_map;
        std::vector<VertexType> reverse_vertex_map;

        GreedyBspScheduler<graph_t> scheduler;
        BspSchedule<graph_t> schedule_orig(instance);

        const auto status_sched_orig = scheduler.computeSchedule(schedule_orig);

        BOOST_CHECK(status_sched_orig == RETURN_STATUS::OSP_SUCCESS);
        BOOST_CHECK(schedule_orig.satisfiesPrecedenceConstraints());

        BspScheduleCoarser<graph_t, graph_t> coarser(schedule_orig);

        coarser.coarsenDag(instance.getComputationalDag(), coarse_instance.getComputationalDag(), reverse_vertex_map);

        vertex_map = coarser_util::invert_vertex_contraction_map<graph_t, graph_t>(reverse_vertex_map);

        BOOST_CHECK(check_vertex_map(vertex_map, instance.getComputationalDag().num_vertices()));

        BspSchedule<graph_t> schedule(coarse_instance);

        const auto status_sched = scheduler.computeSchedule(schedule);

        BOOST_CHECK(status_sched == RETURN_STATUS::OSP_SUCCESS);
        BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

        BspSchedule<graph_t> schedule_out(instance);

        BOOST_CHECK_EQUAL(coarser_util::pull_back_schedule(schedule, vertex_map, schedule_out), true);
        BOOST_CHECK(schedule_out.satisfiesPrecedenceConstraints());

        CoarseAndSchedule<graph_t, graph_t> coarse_and_schedule(coarser, scheduler);
        BspSchedule<graph_t> schedule2(instance);

        const auto status = coarse_and_schedule.computeSchedule(schedule2);
        BOOST_CHECK(status == RETURN_STATUS::OSP_SUCCESS || status == RETURN_STATUS::BEST_FOUND);
        BOOST_CHECK(schedule2.satisfiesPrecedenceConstraints());
    }
}

template<typename graph_t>
void test_coarser_same_graph(Coarser<graph_t, graph_t> &coarser) {

    // BOOST_AUTO_TEST_CASE(coarser_bspschedule_test) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenames_graph = tiny_spaa_graphs();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filename_graph : filenames_graph) {

        std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
        name_graph = name_graph.substr(0, name_graph.find_last_of("."));

        std::cout << std::endl << "Graph: " << name_graph << std::endl;

        BspInstance<graph_t> instance;

        bool status_graph = file_reader::readGraph((cwd / filename_graph).string(),
                                                                            instance.getComputationalDag());

        bool status_architecture = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(),
                                                                    instance.getArchitecture());

        if (!status_graph || !status_architecture) {

            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        BspInstance<graph_t> coarse_instance;
        coarse_instance.setArchitecture(instance.getArchitecture());
        std::vector<std::vector<VertexType>> vertex_map;
        std::vector<VertexType> reverse_vertex_map;

        GreedyBspScheduler<graph_t> scheduler;


        bool coarse_success = coarser.coarsenDag(instance.getComputationalDag(), coarse_instance.getComputationalDag(), reverse_vertex_map);
        BOOST_CHECK(coarse_success);


        vertex_map = coarser_util::invert_vertex_contraction_map<graph_t, graph_t>(reverse_vertex_map);

        BOOST_CHECK(check_vertex_map(vertex_map, instance.getComputationalDag().num_vertices()));

        BspSchedule<graph_t> schedule(coarse_instance);

        const auto status_sched = scheduler.computeSchedule(schedule);

        BOOST_CHECK(status_sched == RETURN_STATUS::OSP_SUCCESS);
        BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

        BspSchedule<graph_t> schedule_out(instance);

        BOOST_CHECK_EQUAL(coarser_util::pull_back_schedule(schedule, vertex_map, schedule_out), true);
        BOOST_CHECK(schedule_out.satisfiesPrecedenceConstraints());

        CoarseAndSchedule<graph_t, graph_t> coarse_and_schedule(coarser, scheduler);
        BspSchedule<graph_t> schedule2(instance);

        const auto status = coarse_and_schedule.computeSchedule(schedule2);
        BOOST_CHECK(status == RETURN_STATUS::OSP_SUCCESS || status == RETURN_STATUS::BEST_FOUND);
        BOOST_CHECK(schedule2.satisfiesPrecedenceConstraints());
    }
}

BOOST_AUTO_TEST_CASE(coarser_funndel_bfs_test) {

    using graph_t = computational_dag_edge_idx_vector_impl_def_t;
    FunnelBfs<graph_t, graph_t> coarser;

    test_coarser_same_graph<graph_t>(coarser);

    FunnelBfs<graph_t, graph_t>::FunnelBfs_parameters params{std::numeric_limits<v_workw_t<graph_t>>::max(),
                                                             std::numeric_limits<v_memw_t<graph_t>>::max(),
                                                             std::numeric_limits<unsigned>::max(), false, true};

    FunnelBfs<graph_t, graph_t> coarser_params(params);

    test_coarser_same_graph<graph_t>(coarser_params);

    params.max_depth = 2;
    FunnelBfs<graph_t, graph_t> coarser_params_2(params);

    test_coarser_same_graph<graph_t>(coarser_params_2);
}

BOOST_AUTO_TEST_CASE(coarser_top_sort_test) {

    using graph_t = computational_dag_edge_idx_vector_impl_def_t;
    top_order_coarser<graph_t, graph_t, GetTopOrder> coarser;

    test_coarser_same_graph<graph_t>(coarser);

    top_order_coarser<graph_t, graph_t, GetTopOrderMaxChildren> coarser_2;

    test_coarser_same_graph<graph_t>(coarser_2);

    top_order_coarser<graph_t, graph_t, GetTopOrderGorder> coarser_3;

    test_coarser_same_graph<graph_t>(coarser_3);
}

BOOST_AUTO_TEST_CASE(squashA_test) {
    using graph_t = computational_dag_edge_idx_vector_impl_def_t;
    // using graph_t = computational_dag_vector_impl_def_t;

    SquashAParams::Parameters params;
    params.mode = SquashAParams::Mode::EDGE_WEIGHT;
    params.use_structured_poset = false;

    SquashA<graph_t, graph_t> coarser(params);

    test_coarser_same_graph<graph_t>(coarser);
    
    
    params.mode = SquashAParams::Mode::TRIANGLES;
    params.use_structured_poset = true;
    params.use_top_poset = true;
    coarser.setParams(params);
    
    test_coarser_same_graph<graph_t>(coarser);

    params.use_top_poset = false;
    coarser.setParams(params);
    
    test_coarser_same_graph<graph_t>(coarser);
}







BOOST_AUTO_TEST_CASE(coarser_SquashA_test_diff_graph_impl_CSG) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenames_graph = tiny_spaa_graphs();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filename_graph : filenames_graph) {

        std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
        name_graph = name_graph.substr(0, name_graph.find_last_of("."));

        std::cout << std::endl << "Graph: " << name_graph << std::endl;

        using graph_t1 = computational_dag_edge_idx_vector_impl_def_t;
        using graph_t2 = CSG;

        BspInstance<graph_t1> instance;

        bool status_graph = file_reader::readGraph((cwd / filename_graph).string(),
                                                                            instance.getComputationalDag());

        bool status_architecture = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(),
                                                                    instance.getArchitecture());

        if (!status_graph || !status_architecture) {

            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        BspInstance<graph_t2> coarse_instance;
        BspArchitecture<graph_t2> architecture_t2(instance.getArchitecture());
        coarse_instance.setArchitecture(architecture_t2);
        std::vector<std::vector<VertexType>> vertex_map;
        std::vector<VertexType> reverse_vertex_map;

        SquashAParams::Parameters params;
        params.mode = SquashAParams::Mode::EDGE_WEIGHT;
        params.use_structured_poset = false;

        SquashA<graph_t1, graph_t2> coarser(params);

        coarser.coarsenDag(instance.getComputationalDag(), coarse_instance.getComputationalDag(), reverse_vertex_map);

        vertex_map = coarser_util::invert_vertex_contraction_map<graph_t1, graph_t2>(reverse_vertex_map);

        BOOST_CHECK(check_vertex_map(vertex_map, instance.getComputationalDag().num_vertices()));

        GreedyBspScheduler<graph_t2> scheduler;
        BspSchedule<graph_t2> schedule(coarse_instance);

        auto status_sched = scheduler.computeSchedule(schedule);

        BOOST_CHECK(status_sched == RETURN_STATUS::OSP_SUCCESS);
        BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

        BspSchedule<graph_t1> schedule_out(instance);

        BOOST_CHECK_EQUAL(coarser_util::pull_back_schedule(schedule, vertex_map, schedule_out), true);
        BOOST_CHECK(schedule_out.satisfiesPrecedenceConstraints());

        CoarseAndSchedule<graph_t1, graph_t2> coarse_and_schedule(coarser, scheduler);
        BspSchedule<graph_t1> schedule2(instance);

        auto status = coarse_and_schedule.computeSchedule(schedule2);
        BOOST_CHECK(status == RETURN_STATUS::OSP_SUCCESS || status == RETURN_STATUS::BEST_FOUND);
        BOOST_CHECK(schedule2.satisfiesPrecedenceConstraints());
    }
}

BOOST_AUTO_TEST_CASE(coarser_SquashA_test_diff_graph_impl_CSGE) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenames_graph = tiny_spaa_graphs();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filename_graph : filenames_graph) {

        std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
        name_graph = name_graph.substr(0, name_graph.find_last_of("."));

        std::cout << std::endl << "Graph: " << name_graph << std::endl;

        using graph_t1 = computational_dag_edge_idx_vector_impl_def_t;
        using graph_t2 = CSGE;

        BspInstance<graph_t1> instance;

        bool status_graph = file_reader::readGraph((cwd / filename_graph).string(),
                                                                            instance.getComputationalDag());

        bool status_architecture = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(),
                                                                    instance.getArchitecture());

        if (!status_graph || !status_architecture) {

            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        BspInstance<graph_t2> coarse_instance;
        BspArchitecture<graph_t2> architecture_t2(instance.getArchitecture());
        coarse_instance.setArchitecture(architecture_t2);
        std::vector<std::vector<VertexType>> vertex_map;
        std::vector<VertexType> reverse_vertex_map;

        SquashAParams::Parameters params;
        params.mode = SquashAParams::Mode::EDGE_WEIGHT;
        params.use_structured_poset = false;

        SquashA<graph_t1, graph_t2> coarser(params);

        coarser.coarsenDag(instance.getComputationalDag(), coarse_instance.getComputationalDag(), reverse_vertex_map);

        vertex_map = coarser_util::invert_vertex_contraction_map<graph_t1, graph_t2>(reverse_vertex_map);

        BOOST_CHECK(check_vertex_map(vertex_map, instance.getComputationalDag().num_vertices()));

        GreedyBspScheduler<graph_t2> scheduler;
        BspSchedule<graph_t2> schedule(coarse_instance);

        auto status_sched = scheduler.computeSchedule(schedule);

        BOOST_CHECK(status_sched == RETURN_STATUS::OSP_SUCCESS);
        BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

        BspSchedule<graph_t1> schedule_out(instance);

        BOOST_CHECK_EQUAL(coarser_util::pull_back_schedule(schedule, vertex_map, schedule_out), true);
        BOOST_CHECK(schedule_out.satisfiesPrecedenceConstraints());

        CoarseAndSchedule<graph_t1, graph_t2> coarse_and_schedule(coarser, scheduler);
        BspSchedule<graph_t1> schedule2(instance);

        auto status = coarse_and_schedule.computeSchedule(schedule2);
        BOOST_CHECK(status == RETURN_STATUS::OSP_SUCCESS || status == RETURN_STATUS::BEST_FOUND);
        BOOST_CHECK(schedule2.satisfiesPrecedenceConstraints());
    }
}








BOOST_AUTO_TEST_CASE(Sarkar_test) {
    using graph_t = computational_dag_edge_idx_vector_impl_def_t;
    // using graph_t = computational_dag_vector_impl_def_t;

    SarkarParams::Parameters<v_workw_t<graph_t>> params;
    params.mode = SarkarParams::Mode::LINES;
    params.commCost = 100;
    params.useTopPoset = true;

    Sarkar<graph_t, graph_t> coarser(params);

    test_coarser_same_graph<graph_t>(coarser);

    
    params.useTopPoset = false;
    coarser.setParameters(params);
    test_coarser_same_graph<graph_t>(coarser);
    
    
    params.mode = SarkarParams::Mode::FAN_IN_FULL;
    coarser.setParameters(params);
    test_coarser_same_graph<graph_t>(coarser);

    
    params.mode = SarkarParams::Mode::FAN_IN_PARTIAL;
    coarser.setParameters(params);
    test_coarser_same_graph<graph_t>(coarser);

    
    params.mode = SarkarParams::Mode::FAN_OUT_FULL;
    coarser.setParameters(params);
    test_coarser_same_graph<graph_t>(coarser);


    params.mode = SarkarParams::Mode::FAN_OUT_PARTIAL;
    coarser.setParameters(params);
    test_coarser_same_graph<graph_t>(coarser);


    params.mode = SarkarParams::Mode::LEVEL_EVEN;
    coarser.setParameters(params);
    test_coarser_same_graph<graph_t>(coarser);
    
    
    params.mode = SarkarParams::Mode::LEVEL_ODD;
    coarser.setParameters(params);
    test_coarser_same_graph<graph_t>(coarser);


    params.mode = SarkarParams::Mode::FAN_IN_BUFFER;
    coarser.setParameters(params);
    test_coarser_same_graph<graph_t>(coarser);


    params.mode = SarkarParams::Mode::FAN_OUT_BUFFER;
    coarser.setParameters(params);
    test_coarser_same_graph<graph_t>(coarser);


    params.mode = SarkarParams::Mode::HOMOGENEOUS_BUFFER;
    coarser.setParameters(params);
    test_coarser_same_graph<graph_t>(coarser);
}


BOOST_AUTO_TEST_CASE(SarkarML_test) {
    using graph_t = computational_dag_edge_idx_vector_impl_def_t;
    // using graph_t = computational_dag_vector_impl_def_t;

    SarkarParams::MulParameters<v_workw_t<graph_t>> params;
    params.commCostVec = {100};

    SarkarMul<graph_t, graph_t> coarser;
    coarser.setParameters(params);

    test_coarser_same_graph<graph_t>(coarser);
}

BOOST_AUTO_TEST_CASE(SquashAML_test) {
    using graph_t = computational_dag_edge_idx_vector_impl_def_t;
    // using graph_t = computational_dag_vector_impl_def_t;

    SquashAMul<graph_t, graph_t> coarser;
    
    test_coarser_same_graph<graph_t>(coarser);
}