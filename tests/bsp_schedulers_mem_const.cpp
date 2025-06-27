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

#define BOOST_TEST_MODULE BSP_SCHEDULERS
#include <boost/test/unit_test.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include "bsp/scheduler/LoadBalanceScheduler/VariancePartitioner.hpp"
#include "bsp/scheduler/GreedySchedulers/BspLocking.hpp"
#include "bsp/scheduler/GreedySchedulers/GrowLocalAutoCores.hpp"
#include "bsp/scheduler/GreedySchedulers/EtfScheduler.hpp"
#include "bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "bsp/scheduler/GreedySchedulers/VarianceFillup.hpp"
#include "bsp/scheduler/LoadBalanceScheduler/LightEdgeVariancePartitioner.hpp"
#include "bsp/scheduler/Serial.hpp"
#include "graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "io/arch_file_reader.hpp"
#include "io/hdag_graph_file_reader.hpp"
#include "bsp/scheduler/LocalSearch/KernighanLin/kl_total_comm.hpp"
#include "bsp/scheduler/LocalSearch/KernighanLin/kl_total_cut.hpp"


using namespace osp;

std::vector<std::string> test_graphs() {
    return {"data/spaa/tiny/instance_bicgstab.hdag", "data/spaa/tiny/instance_CG_N2_K2_nzP0d75.hdag"};
}

std::vector<std::string> test_architectures() { return {"data/machine_params/p3.arch"}; }

template<typename Graph_t>
void add_mem_weights(Graph_t &dag) {

    int mem_weight = 1;
    int comm_weight = 1;

    for (const auto &v : dag.vertices()) {

        dag.set_vertex_mem_weight(v, static_cast<v_memw_t<Graph_t>>(mem_weight++ % 3 + 1));
        dag.set_vertex_comm_weight(v, static_cast<v_commw_t<Graph_t>>(comm_weight++ % 3 + 1));
    }
}

template<typename Graph_t>
void run_test_local_memory(Scheduler<Graph_t> *test_scheduler) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenames_graph = test_graphs();
    std::vector<std::string> filenames_architectures = test_architectures();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filename_graph : filenames_graph) {
        for (auto &filename_machine : filenames_architectures) {
            std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
            name_graph = name_graph.substr(0, name_graph.find_last_of("."));
            std::string name_machine = filename_machine.substr(filename_machine.find_last_of("/\\") + 1);
            name_machine = name_machine.substr(0, name_machine.rfind("."));

            std::cout << std::endl << "Scheduler: " << test_scheduler->getScheduleName() << std::endl;
            std::cout << "Graph: " << name_graph << std::endl;
            std::cout << "Architecture: " << name_machine << std::endl;

            BspInstance<Graph_t> instance;

            bool status_graph = file_reader::readComputationalDagHyperdagFormat((cwd / filename_graph).string(),
                                                                                instance.getComputationalDag());
            bool status_architecture = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(),
                                                                        instance.getArchitecture());

            add_mem_weights(instance.getComputationalDag());
            instance.getArchitecture().setMemoryConstraintType(LOCAL);
            std::cout << "Memory constraint type: LOCAL" << std::endl;

            if (!status_graph || !status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            const std::vector<v_memw_t<Graph_t>> bounds_to_test = {10, 20, 50, 100};

            for (const auto &bound : bounds_to_test) {

                instance.getArchitecture().setMemoryBound(bound);

                BspSchedule<Graph_t> schedule(instance);
                const auto result = test_scheduler->computeSchedule(schedule);

                BOOST_CHECK(SUCCESS == result || BEST_FOUND == result);
                BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
                BOOST_CHECK(schedule.satisfiesMemoryConstraints());
            }
        }
    }
};

template<typename Graph_t>
void run_test_persistent_transient_memory(Scheduler<Graph_t> *test_scheduler) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenames_graph = test_graphs();
    std::vector<std::string> filenames_architectures = test_architectures();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filename_graph : filenames_graph) {
        for (auto &filename_machine : filenames_architectures) {
            std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
            name_graph = name_graph.substr(0, name_graph.find_last_of("."));
            std::string name_machine = filename_machine.substr(filename_machine.find_last_of("/\\") + 1);
            name_machine = name_machine.substr(0, name_machine.rfind("."));

            std::cout << std::endl << "Scheduler: " << test_scheduler->getScheduleName() << std::endl;
            std::cout << "Graph: " << name_graph << std::endl;
            std::cout << "Architecture: " << name_machine << std::endl;

            BspInstance<Graph_t> instance;

            bool status_graph = file_reader::readComputationalDagHyperdagFormat((cwd / filename_graph).string(),
                                                                                instance.getComputationalDag());
            bool status_architecture = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(),
                                                                        instance.getArchitecture());

            add_mem_weights(instance.getComputationalDag());
            instance.getArchitecture().setMemoryConstraintType(PERSISTENT_AND_TRANSIENT);
            std::cout << "Memory constraint type: PERSISTENT_AND_TRANSIENT" << std::endl;

            if (!status_graph || !status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            const std::vector<v_memw_t<Graph_t>> bounds_to_test = {50, 100};

            for (const auto &bound : bounds_to_test) {

                instance.getArchitecture().setMemoryBound(bound);

                BspSchedule<Graph_t> schedule(instance);
                const auto result = test_scheduler->computeSchedule(schedule);

                BOOST_CHECK_EQUAL(SUCCESS, result);
                BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
                BOOST_CHECK(schedule.satisfiesMemoryConstraints());
            }
        }
    }
};

template<typename Graph_t>
void run_test_local_in_out_memory(Scheduler<Graph_t> *test_scheduler) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenames_graph = test_graphs();
    std::vector<std::string> filenames_architectures = test_architectures();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filename_graph : filenames_graph) {
        for (auto &filename_machine : filenames_architectures) {
            std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
            name_graph = name_graph.substr(0, name_graph.find_last_of("."));
            std::string name_machine = filename_machine.substr(filename_machine.find_last_of("/\\") + 1);
            name_machine = name_machine.substr(0, name_machine.rfind("."));

            std::cout << std::endl << "Scheduler: " << test_scheduler->getScheduleName() << std::endl;
            std::cout << "Graph: " << name_graph << std::endl;
            std::cout << "Architecture: " << name_machine << std::endl;

            BspInstance<Graph_t> instance;

            bool status_graph = file_reader::readComputationalDagHyperdagFormat((cwd / filename_graph).string(),
                                                                                instance.getComputationalDag());
            bool status_architecture = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(),
                                                                        instance.getArchitecture());

            add_mem_weights(instance.getComputationalDag());
            instance.getArchitecture().setMemoryConstraintType(LOCAL_IN_OUT);
            std::cout << "Memory constraint type: LOCAL_IN_OUT" << std::endl;

            if (!status_graph || !status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            const std::vector<v_memw_t<Graph_t>> bounds_to_test = {10, 20, 50, 100};

            for (const auto &bound : bounds_to_test) {

                instance.getArchitecture().setMemoryBound(bound);

                BspSchedule<Graph_t> schedule(instance);
                const auto result = test_scheduler->computeSchedule(schedule);

                BOOST_CHECK_EQUAL(SUCCESS, result);
                BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
                BOOST_CHECK(schedule.satisfiesMemoryConstraints());
            }
        }
    }
};

template<typename Graph_t>
void run_test_local_inc_edges_memory(Scheduler<Graph_t> *test_scheduler) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenames_graph = test_graphs();
    std::vector<std::string> filenames_architectures = test_architectures();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filename_graph : filenames_graph) {
        for (auto &filename_machine : filenames_architectures) {
            std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
            name_graph = name_graph.substr(0, name_graph.find_last_of("."));
            std::string name_machine = filename_machine.substr(filename_machine.find_last_of("/\\") + 1);
            name_machine = name_machine.substr(0, name_machine.rfind("."));

            std::cout << std::endl << "Scheduler: " << test_scheduler->getScheduleName() << std::endl;
            std::cout << "Graph: " << name_graph << std::endl;
            std::cout << "Architecture: " << name_machine << std::endl;

            BspInstance<Graph_t> instance;

            bool status_graph = file_reader::readComputationalDagHyperdagFormat((cwd / filename_graph).string(),
                                                                                instance.getComputationalDag());
            bool status_architecture = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(),
                                                                        instance.getArchitecture());

            add_mem_weights(instance.getComputationalDag());
            instance.getArchitecture().setMemoryConstraintType(LOCAL_INC_EDGES);
            std::cout << "Memory constraint type: LOCAL_INC_EDGES" << std::endl;

            if (!status_graph || !status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            const std::vector<v_memw_t<Graph_t>> bounds_to_test = {50, 100};

            for (const auto &bound : bounds_to_test) {

                instance.getArchitecture().setMemoryBound(bound);

                BspSchedule<Graph_t> schedule(instance);
                const auto result = test_scheduler->computeSchedule(schedule);

                BOOST_CHECK_EQUAL(SUCCESS, result);
                BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
                BOOST_CHECK(schedule.satisfiesMemoryConstraints());
            }
        }
    }
};

template<typename Graph_t>
void run_test_local_inc_edges_2_memory(Scheduler<Graph_t> *test_scheduler) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenames_graph = test_graphs();
    std::vector<std::string> filenames_architectures = test_architectures();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filename_graph : filenames_graph) {
        for (auto &filename_machine : filenames_architectures) {
            std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
            name_graph = name_graph.substr(0, name_graph.find_last_of("."));
            std::string name_machine = filename_machine.substr(filename_machine.find_last_of("/\\") + 1);
            name_machine = name_machine.substr(0, name_machine.rfind("."));

            std::cout << std::endl << "Scheduler: " << test_scheduler->getScheduleName() << std::endl;
            std::cout << "Graph: " << name_graph << std::endl;
            std::cout << "Architecture: " << name_machine << std::endl;

            BspInstance<Graph_t> instance;

            bool status_graph = file_reader::readComputationalDagHyperdagFormat((cwd / filename_graph).string(),
                                                                                instance.getComputationalDag());
            bool status_architecture = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(),
                                                                        instance.getArchitecture());

            add_mem_weights(instance.getComputationalDag());
            instance.getArchitecture().setMemoryConstraintType(LOCAL_SOURCES_INC_EDGES);
            std::cout << "Memory constraint type: LOCAL_SOURCES_INC_EDGES" << std::endl;

            if (!status_graph || !status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            const std::vector<v_memw_t<Graph_t>> bounds_to_test = {20, 50, 100};

            for (const auto &bound : bounds_to_test) {

                instance.getArchitecture().setMemoryBound(bound);

                BspSchedule<Graph_t> schedule(instance);
                const auto result = test_scheduler->computeSchedule(schedule);

                BOOST_CHECK_EQUAL(SUCCESS, result);
                BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
                BOOST_CHECK(schedule.satisfiesMemoryConstraints());
            }
        }
    }
};

BOOST_AUTO_TEST_CASE(GreedyBspScheduler_local_test) {

    using graph_impl_t = computational_dag_edge_idx_vector_impl_def_int_t;

    GreedyBspScheduler<graph_impl_t, local_memory_constraint<graph_impl_t>> test_1;
    run_test_local_memory(&test_1);

    GreedyBspScheduler<graph_impl_t, local_in_out_memory_constraint<graph_impl_t>> test_2;
    run_test_local_in_out_memory(&test_2);

    GreedyBspScheduler<graph_impl_t, local_inc_edges_memory_constraint<graph_impl_t>> test_3;
    run_test_local_inc_edges_memory(&test_3);

    GreedyBspScheduler<graph_impl_t, local_sources_inc_edges_memory_constraint<graph_impl_t>> test_4;
    run_test_local_inc_edges_2_memory(&test_4);
};

BOOST_AUTO_TEST_CASE(GrowLocalAutoCores_local_test) {

    using graph_impl_t = computational_dag_edge_idx_vector_impl_def_int_t;

    GrowLocalAutoCores<graph_impl_t, local_memory_constraint<graph_impl_t>> test_1;
    run_test_local_memory(&test_1);

    GrowLocalAutoCores<graph_impl_t, local_in_out_memory_constraint<graph_impl_t>> test_2;
    run_test_local_in_out_memory(&test_2);

    GrowLocalAutoCores<graph_impl_t, local_inc_edges_memory_constraint<graph_impl_t>> test_3;
    run_test_local_inc_edges_memory(&test_3);

    GrowLocalAutoCores<graph_impl_t, local_sources_inc_edges_memory_constraint<graph_impl_t>> test_4;
    run_test_local_inc_edges_2_memory(&test_4);
};

BOOST_AUTO_TEST_CASE(BspLocking_local_test) {

    using graph_impl_t = computational_dag_edge_idx_vector_impl_def_t;

    BspLocking<graph_impl_t, local_memory_constraint<graph_impl_t>> test_1;
    run_test_local_memory(&test_1);

    BspLocking<graph_impl_t, local_in_out_memory_constraint<graph_impl_t>> test_2;
    run_test_local_in_out_memory(&test_2);

    BspLocking<graph_impl_t, local_inc_edges_memory_constraint<graph_impl_t>> test_3;
    run_test_local_inc_edges_memory(&test_3);

    BspLocking<graph_impl_t, local_sources_inc_edges_memory_constraint<graph_impl_t>> test_4;
    run_test_local_inc_edges_2_memory(&test_4);
};

BOOST_AUTO_TEST_CASE(variance_local_test) {

    VarianceFillup<computational_dag_edge_idx_vector_impl_def_t,
                   local_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>>
        test;
    run_test_local_memory(&test);
};

// BOOST_AUTO_TEST_CASE(kl_local_test) {

//     VarianceFillup<computational_dag_edge_idx_vector_impl_def_t,
//                    local_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>>
//         test;
    
//     kl_total_comm<computational_dag_edge_idx_vector_impl_def_t, local_search_local_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>> kl;
    
//     ComboScheduler<computational_dag_edge_idx_vector_impl_def_t> combo_test(test, kl);
    
//     run_test_local_memory(&combo_test);
// };


BOOST_AUTO_TEST_CASE(GreedyBspScheduler_persistent_transient_test) {

    GreedyBspScheduler<computational_dag_edge_idx_vector_impl_def_t,
                       persistent_transient_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>>
        test;
    run_test_persistent_transient_memory(&test);
};

BOOST_AUTO_TEST_CASE(EtfScheduler_persistent_transient_test) {

    EtfScheduler<computational_dag_edge_idx_vector_impl_def_t,
                 persistent_transient_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>>
        test;
    run_test_persistent_transient_memory(&test);
};


BOOST_AUTO_TEST_CASE(VariancePartitioner_test) {
    VariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, linear_interpolation, local_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>> test_linear;
    run_test_local_memory(&test_linear);

    VariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, flat_spline_interpolation, local_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>> test_flat;
    run_test_local_memory(&test_flat);

    VariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, superstep_only_interpolation, local_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>> test_superstep;
    run_test_local_memory(&test_superstep);

    VariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, global_only_interpolation, local_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>> test_global;
    run_test_local_memory(&test_global);

    VariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, linear_interpolation, persistent_transient_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>> test_linear_tp;
    run_test_persistent_transient_memory(&test_linear_tp);

    VariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, flat_spline_interpolation, persistent_transient_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>> test_flat_tp;
    run_test_persistent_transient_memory(&test_flat_tp);

    VariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, superstep_only_interpolation, persistent_transient_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>> test_superstep_tp;
    run_test_persistent_transient_memory(&test_superstep_tp);

    VariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, global_only_interpolation, persistent_transient_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>> test_global_tp;
    run_test_persistent_transient_memory(&test_global_tp);

}


BOOST_AUTO_TEST_CASE(LightEdgeVariancePartitioner_test) {
    LightEdgeVariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, linear_interpolation, local_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>> test_linear;
    run_test_local_memory(&test_linear);

    LightEdgeVariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, flat_spline_interpolation, local_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>> test_flat;
    run_test_local_memory(&test_flat);

    LightEdgeVariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, superstep_only_interpolation, local_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>> test_superstep;
    run_test_local_memory(&test_superstep);

    LightEdgeVariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, global_only_interpolation, local_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>> test_global;
    run_test_local_memory(&test_global);

    LightEdgeVariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, linear_interpolation, persistent_transient_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>> test_linear_tp;
    run_test_persistent_transient_memory(&test_linear_tp);

    LightEdgeVariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, flat_spline_interpolation, persistent_transient_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>> test_flat_tp;
    run_test_persistent_transient_memory(&test_flat_tp);

    LightEdgeVariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, superstep_only_interpolation, persistent_transient_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>> test_superstep_tp;
    run_test_persistent_transient_memory(&test_superstep_tp);

    LightEdgeVariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, global_only_interpolation, persistent_transient_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>> test_global_tp;
    run_test_persistent_transient_memory(&test_global_tp);

}
