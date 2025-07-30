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

#include "osp/bsp/scheduler/MultilevelCoarseAndSchedule.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/BspLocking.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/CilkScheduler.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/EtfScheduler.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyChildren.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GrowLocalAutoCores.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GrowLocalAutoCoresParallel.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/RandomGreedy.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/VarianceFillup.hpp"
#include "osp/bsp/scheduler/LoadBalanceScheduler/LightEdgeVariancePartitioner.hpp"
#include "osp/bsp/scheduler/LoadBalanceScheduler/VariancePartitioner.hpp"
#include "osp/bsp/scheduler/LocalSearch/HillClimbing/hill_climbing.hpp"
#include "osp/bsp/scheduler/Serial.hpp"
#include "osp/coarser/Sarkar/SarkarMul.hpp"
#include "osp/coarser/SquashA/SquashAMul.hpp"
#include "osp/graph_implementations/adj_list_impl/compact_sparse_graph.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/auxiliary/io/arch_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"

using namespace osp;

std::vector<std::string> tiny_spaa_graphs() {
    return {"data/spaa/tiny/instance_bicgstab.hdag",
            "data/spaa/tiny/instance_CG_N2_K2_nzP0d75.hdag",
            "data/spaa/tiny/instance_CG_N3_K1_nzP0d5.hdag",
            "data/spaa/tiny/instance_CG_N4_K1_nzP0d35.hdag",
            "data/spaa/tiny/instance_exp_N4_K2_nzP0d5.hdag",
            "data/spaa/tiny/instance_exp_N5_K3_nzP0d4.hdag",
            "data/spaa/tiny/instance_exp_N6_K4_nzP0d25.hdag",
            "data/spaa/tiny/instance_k-means.hdag",
            "data/spaa/tiny/instance_k-NN_3_gyro_m.hdag",
            "data/spaa/tiny/instance_kNN_N4_K3_nzP0d5.hdag",
            "data/spaa/tiny/instance_kNN_N5_K3_nzP0d3.hdag",
            "data/spaa/tiny/instance_kNN_N6_K4_nzP0d2.hdag",
            "data/spaa/tiny/instance_pregel.hdag",
            "data/spaa/tiny/instance_spmv_N6_nzP0d4.hdag",
            "data/spaa/tiny/instance_spmv_N7_nzP0d35.hdag",
            "data/spaa/tiny/instance_spmv_N10_nzP0d25.hdag"};
}

std::vector<std::string> test_architectures() { return {"data/machine_params/p3.arch"}; }

template<typename Graph_t>
void run_test(Scheduler<Graph_t> *test_scheduler) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenames_graph = tiny_spaa_graphs();
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

            if (!status_graph || !status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            BspSchedule<Graph_t> schedule(instance);
            const auto result = test_scheduler->computeSchedule(schedule);

            BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, result);
            BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
        }
    }
};

template<typename Graph_t>
void run_test_2(Scheduler<Graph_t> *test_scheduler) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenames_graph = tiny_spaa_graphs();
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

            computational_dag_edge_idx_vector_impl_def_t graph;
            BspArchitecture<Graph_t> arch;

            bool status_graph = file_reader::readComputationalDagHyperdagFormat((cwd / filename_graph).string(), graph);
            bool status_architecture =
                file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(), arch);

            if (!status_graph || !status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            BspInstance<Graph_t> instance(graph, arch);

            BspSchedule<Graph_t> schedule(instance);
            const auto result = test_scheduler->computeSchedule(schedule);

            BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, result);
            BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
        }
    }
};

BOOST_AUTO_TEST_CASE(GreedyBspScheduler_test) {

    GreedyBspScheduler<computational_dag_vector_impl_def_t> test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(GreedyBspScheduler_test_2) {

    GreedyBspScheduler<computational_dag_edge_idx_vector_impl_def_t> test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(Serial_test) {

    Serial<computational_dag_vector_impl_def_t> test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(cilk_test_1) {

    CilkScheduler<computational_dag_vector_impl_def_t> test;
    test.setMode(CILK);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(cilk_test_2) {

    CilkScheduler<computational_dag_vector_impl_def_t> test;
    test.setMode(SJF);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(etf_test) {

    EtfScheduler<computational_dag_vector_impl_def_t> test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(random_test) {

    RandomGreedy<computational_dag_vector_impl_def_t> test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(children_test) {

    GreedyChildren<computational_dag_vector_impl_def_t> test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(locking_test) {

    BspLocking<computational_dag_vector_impl_def_int_t> test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(variancefillup_test) {

    VarianceFillup<computational_dag_vector_impl_def_t> test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(etf_test_edge_desc_impl) {

    EtfScheduler<computational_dag_edge_idx_vector_impl_def_t> test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(grow_local_auto_test_edge_desc_impl) {

    GrowLocalAutoCores<computational_dag_edge_idx_vector_impl_def_t> test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(grow_local_auto_parallel_top_test_1) {
    {
        using Graph_t = computational_dag_vector_impl_def_t;
        GrowLocalAutoCoresParallel_Params<vertex_idx_t<Graph_t>, v_workw_t<Graph_t>> params;

        params.numThreads = 1;

        GrowLocalAutoCoresParallel<Graph_t> test(params);
        run_test(&test);
    }
}

BOOST_AUTO_TEST_CASE(grow_local_auto_parallel_top_test_2) {
    {
        using Graph_t = computational_dag_vector_impl_def_t;
        GrowLocalAutoCoresParallel_Params<vertex_idx_t<Graph_t>, v_workw_t<Graph_t>> params;

        params.numThreads = 2;

        GrowLocalAutoCoresParallel<Graph_t> test(params);
        run_test(&test);
    }
}

BOOST_AUTO_TEST_CASE(grow_local_auto_parallel_top_test_5) {
    {
        using Graph_t = computational_dag_vector_impl_def_t;
        GrowLocalAutoCoresParallel_Params<vertex_idx_t<Graph_t>, v_workw_t<Graph_t>> params;

        params.numThreads = 5;

        GrowLocalAutoCoresParallel<Graph_t> test(params);
        run_test(&test);
    }
}

BOOST_AUTO_TEST_CASE(grow_local_auto_parallel_test_1) {
    {
        using Graph_t = Compact_Sparse_Graph<true, true>;
        GrowLocalAutoCoresParallel_Params<vertex_idx_t<Graph_t>, v_workw_t<Graph_t>> params;

        params.numThreads = 1;

        GrowLocalAutoCoresParallel<Graph_t> test(params);
        run_test_2(&test);
    }
}

BOOST_AUTO_TEST_CASE(grow_local_auto_parallel_test_2) {
    {
        using Graph_t = Compact_Sparse_Graph<true, true>;
        GrowLocalAutoCoresParallel_Params<vertex_idx_t<Graph_t>, v_workw_t<Graph_t>> params;

        params.numThreads = 2;

        GrowLocalAutoCoresParallel<Graph_t> test(params);
        run_test_2(&test);
    }
}

BOOST_AUTO_TEST_CASE(grow_local_auto_parallel_test_5) {
    {
        using Graph_t = Compact_Sparse_Graph<true, true>;
        GrowLocalAutoCoresParallel_Params<vertex_idx_t<Graph_t>, v_workw_t<Graph_t>> params;

        params.numThreads = 5;

        GrowLocalAutoCoresParallel<Graph_t> test(params);
        run_test_2(&test);
    }
}

BOOST_AUTO_TEST_CASE(VariancePartitioner_test) {
    VariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, linear_interpolation> test_linear;
    run_test(&test_linear);

    VariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, flat_spline_interpolation> test_flat;
    run_test(&test_flat);

    VariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, superstep_only_interpolation> test_superstep;
    run_test(&test_superstep);

    VariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, global_only_interpolation> test_global;
    run_test(&test_global);
}

BOOST_AUTO_TEST_CASE(LightEdgeVariancePartitioner_test) {
    LightEdgeVariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, linear_interpolation> test_linear;
    run_test(&test_linear);

    LightEdgeVariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, flat_spline_interpolation> test_flat;
    run_test(&test_flat);

    LightEdgeVariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, superstep_only_interpolation>
        test_superstep;
    run_test(&test_superstep);

    LightEdgeVariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, global_only_interpolation> test_global;
    run_test(&test_global);
}

BOOST_AUTO_TEST_CASE(SquashAMul_test) {
    GreedyBspScheduler<computational_dag_edge_idx_vector_impl_def_t> sched;

    SquashAMul<computational_dag_edge_idx_vector_impl_def_t, computational_dag_edge_idx_vector_impl_def_t> ml_coarsen;
    MultilevelCoarseAndSchedule<computational_dag_edge_idx_vector_impl_def_t, computational_dag_edge_idx_vector_impl_def_t> coarsen_test(sched, ml_coarsen);
    
    run_test(&coarsen_test);
}

BOOST_AUTO_TEST_CASE(SquashAMul_improver_test) {
    GreedyBspScheduler<computational_dag_edge_idx_vector_impl_def_t> sched;
    HillClimbingScheduler<computational_dag_edge_idx_vector_impl_def_t> improver;

    SquashAMul<computational_dag_edge_idx_vector_impl_def_t, computational_dag_edge_idx_vector_impl_def_t> ml_coarsen;
    MultilevelCoarseAndSchedule<computational_dag_edge_idx_vector_impl_def_t, computational_dag_edge_idx_vector_impl_def_t> coarsen_test(sched, improver, ml_coarsen);
    
    
    run_test(&coarsen_test);
}


BOOST_AUTO_TEST_CASE(SarkarMul_test) {
    GreedyBspScheduler<computational_dag_edge_idx_vector_impl_def_t> sched;

    SarkarMul<computational_dag_edge_idx_vector_impl_def_t, computational_dag_edge_idx_vector_impl_def_t> ml_coarsen;
    MultilevelCoarseAndSchedule<computational_dag_edge_idx_vector_impl_def_t, computational_dag_edge_idx_vector_impl_def_t> coarsen_test(sched, ml_coarsen);
    
    run_test(&coarsen_test);
}

BOOST_AUTO_TEST_CASE(SarkarMul_improver_test) {
    GreedyBspScheduler<computational_dag_edge_idx_vector_impl_def_t> sched;
    HillClimbingScheduler<computational_dag_edge_idx_vector_impl_def_t> improver;

    SarkarMul<computational_dag_edge_idx_vector_impl_def_t, computational_dag_edge_idx_vector_impl_def_t> ml_coarsen;
    MultilevelCoarseAndSchedule<computational_dag_edge_idx_vector_impl_def_t, computational_dag_edge_idx_vector_impl_def_t> coarsen_test(sched, improver, ml_coarsen);
    
    run_test(&coarsen_test);
}