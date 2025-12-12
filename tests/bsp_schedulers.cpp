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

#include "osp/auxiliary/io/arch_file_reader.hpp"
#include "osp/auxiliary/io/general_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
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
#include "osp/bsp/scheduler/MultilevelCoarseAndSchedule.hpp"
#include "osp/bsp/scheduler/Serial.hpp"
#include "osp/coarser/Sarkar/SarkarMul.hpp"
#include "osp/coarser/SquashA/SquashAMul.hpp"
#include "osp/graph_implementations/adj_list_impl/compact_sparse_graph.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "test_graphs.hpp"

using namespace osp;

std::vector<std::string> TestArchitectures() { return {"data/machine_params/p3.arch"}; }

template <typename GraphT>
void RunTest(Scheduler<GraphT> *testScheduler) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenamesGraph = TinySpaaGraphs();
    std::vector<std::string> filenamesArchitectures = TestArchitectures();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filenameGraph : filenamesGraph) {
        for (auto &filenameMachine : filenamesArchitectures) {
            std::string nameGraph = filenameGraph.substr(filenameGraph.find_last_of("/\\") + 1);
            nameGraph = nameGraph.substr(0, nameGraph.find_last_of("."));
            std::string nameMachine = filenameMachine.substr(filenameMachine.find_last_of("/\\") + 1);
            nameMachine = nameMachine.substr(0, nameMachine.rfind("."));

            std::cout << std::endl << "Scheduler: " << testScheduler->getScheduleName() << std::endl;
            std::cout << "Graph: " << nameGraph << std::endl;
            std::cout << "Architecture: " << nameMachine << std::endl;

            BspInstance<GraphT> instance;

            bool statusGraph = file_reader::readGraph((cwd / filenameGraph).string(), instance.GetComputationalDag());
            bool statusArchitecture
                = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(), instance.GetArchitecture());

            if (!statusGraph || !statusArchitecture) {
                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            BspSchedule<GraphT> schedule(instance);
            const auto result = testScheduler->computeSchedule(schedule);

            BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
            BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
        }
    }
}

template <typename GraphT>
void RunTest2(Scheduler<GraphT> *testScheduler) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenamesGraph = TinySpaaGraphs();
    std::vector<std::string> filenamesArchitectures = TestArchitectures();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filenameGraph : filenamesGraph) {
        for (auto &filenameMachine : filenamesArchitectures) {
            std::string nameGraph = filenameGraph.substr(filenameGraph.find_last_of("/\\") + 1);
            nameGraph = nameGraph.substr(0, nameGraph.find_last_of("."));
            std::string nameMachine = filenameMachine.substr(filenameMachine.find_last_of("/\\") + 1);
            nameMachine = nameMachine.substr(0, nameMachine.rfind("."));

            std::cout << std::endl << "Scheduler: " << testScheduler->getScheduleName() << std::endl;
            std::cout << "Graph: " << nameGraph << std::endl;
            std::cout << "Architecture: " << nameMachine << std::endl;

            computational_dag_edge_idx_vector_impl_def_t graph;
            BspArchitecture<GraphT> arch;

            bool statusGraph = file_reader::readGraph((cwd / filenameGraph).string(), graph);
            bool statusArchitecture = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(), arch);

            if (!statusGraph || !statusArchitecture) {
                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            BspInstance<GraphT> instance(graph, arch);

            BspSchedule<GraphT> schedule(instance);
            const auto result = testScheduler->computeSchedule(schedule);

            BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
            BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
        }
    }
}

BOOST_AUTO_TEST_CASE(GreedyBspSchedulerTest) {
    GreedyBspScheduler<computational_dag_vector_impl_def_t> test;
    RunTest(&test);
}

BOOST_AUTO_TEST_CASE(GreedyBspSchedulerTest2) {
    GreedyBspScheduler<computational_dag_edge_idx_vector_impl_def_t> test;
    RunTest(&test);
}

BOOST_AUTO_TEST_CASE(SerialTest) {
    Serial<computational_dag_vector_impl_def_t> test;
    RunTest(&test);
}

BOOST_AUTO_TEST_CASE(CilkTest1) {
    CilkScheduler<computational_dag_vector_impl_def_t> test;
    test.setMode(CILK);
    RunTest(&test);
}

BOOST_AUTO_TEST_CASE(CilkTest2) {
    CilkScheduler<computational_dag_vector_impl_def_t> test;
    test.setMode(SJF);
    RunTest(&test);
}

BOOST_AUTO_TEST_CASE(EtfTest) {
    EtfScheduler<computational_dag_vector_impl_def_t> test;
    RunTest(&test);
}

BOOST_AUTO_TEST_CASE(RandomTest) {
    RandomGreedy<computational_dag_vector_impl_def_t> test;
    RunTest(&test);
}

BOOST_AUTO_TEST_CASE(ChildrenTest) {
    GreedyChildren<computational_dag_vector_impl_def_t> test;
    RunTest(&test);
}

BOOST_AUTO_TEST_CASE(LockingTest) {
    BspLocking<computational_dag_vector_impl_def_int_t> test;
    RunTest(&test);
}

BOOST_AUTO_TEST_CASE(VariancefillupTest) {
    VarianceFillup<computational_dag_vector_impl_def_t> test;
    RunTest(&test);
}

BOOST_AUTO_TEST_CASE(EtfTestEdgeDescImpl) {
    EtfScheduler<computational_dag_edge_idx_vector_impl_def_t> test;
    RunTest(&test);
}

BOOST_AUTO_TEST_CASE(GrowLocalAutoTestEdgeDescImpl) {
    GrowLocalAutoCores<computational_dag_edge_idx_vector_impl_def_t> test;
    RunTest(&test);
}

BOOST_AUTO_TEST_CASE(GrowLocalAutoParallelTopTest1) {
    {
        using GraphT = computational_dag_vector_impl_def_t;
        GrowLocalAutoCoresParallel_Params<VertexIdxT<GraphT>, VWorkwT<GraphT>> params;

        params.numThreads = 1;

        GrowLocalAutoCoresParallel<GraphT> Test(params);
        RunTest(&test);
    }
}

BOOST_AUTO_TEST_CASE(GrowLocalAutoParallelTopTest2) {
    {
        using GraphT = computational_dag_vector_impl_def_t;
        GrowLocalAutoCoresParallel_Params<VertexIdxT<GraphT>, VWorkwT<GraphT>> params;

        params.numThreads = 2;

        GrowLocalAutoCoresParallel<GraphT> Test(params);
        RunTest(&test);
    }
}

BOOST_AUTO_TEST_CASE(GrowLocalAutoParallelTopTest5) {
    {
        using GraphT = computational_dag_vector_impl_def_t;
        GrowLocalAutoCoresParallel_Params<VertexIdxT<GraphT>, VWorkwT<GraphT>> params;

        params.numThreads = 5;

        GrowLocalAutoCoresParallel<GraphT> Test(params);
        RunTest(&test);
    }
}

BOOST_AUTO_TEST_CASE(GrowLocalAutoParallelTest1) {
    {
        using GraphT = CompactSparseGraph<true, true>;
        GrowLocalAutoCoresParallel_Params<VertexIdxT<GraphT>, VWorkwT<GraphT>> params;

        params.numThreads = 1;

        GrowLocalAutoCoresParallel<GraphT> Test(params);
        RunTest2(&test);
    }
}

BOOST_AUTO_TEST_CASE(GrowLocalAutoParallelTest2) {
    {
        using GraphT = CompactSparseGraph<true, true>;
        GrowLocalAutoCoresParallel_Params<VertexIdxT<GraphT>, VWorkwT<GraphT>> params;

        params.numThreads = 2;

        GrowLocalAutoCoresParallel<GraphT> Test(params);
        RunTest2(&test);
    }
}

BOOST_AUTO_TEST_CASE(GrowLocalAutoParallelTest5) {
    {
        using GraphT = CompactSparseGraph<true, true>;
        GrowLocalAutoCoresParallel_Params<VertexIdxT<GraphT>, VWorkwT<GraphT>> params;

        params.numThreads = 5;

        GrowLocalAutoCoresParallel<GraphT> Test(params);
        RunTest2(&test);
    }
}

BOOST_AUTO_TEST_CASE(VariancePartitionerTest) {
    VariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, linear_interpolation> testLinear;
    RunTest(&testLinear);

    VariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, flat_spline_interpolation> testFlat;
    RunTest(&testFlat);

    VariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, superstep_only_interpolation> testSuperstep;
    RunTest(&testSuperstep);

    VariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, global_only_interpolation> testGlobal;
    RunTest(&testGlobal);
}

BOOST_AUTO_TEST_CASE(LightEdgeVariancePartitionerTest) {
    LightEdgeVariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, linear_interpolation> testLinear;
    RunTest(&testLinear);

    LightEdgeVariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, flat_spline_interpolation> testFlat;
    RunTest(&testFlat);

    LightEdgeVariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, superstep_only_interpolation> testSuperstep;
    RunTest(&testSuperstep);

    LightEdgeVariancePartitioner<computational_dag_edge_idx_vector_impl_def_t, global_only_interpolation> testGlobal;
    RunTest(&testGlobal);
}

BOOST_AUTO_TEST_CASE(SquashAMulTest) {
    GreedyBspScheduler<computational_dag_edge_idx_vector_impl_def_t> sched;

    SquashAMul<computational_dag_edge_idx_vector_impl_def_t, computational_dag_edge_idx_vector_impl_def_t> mlCoarsen;
    MultilevelCoarseAndSchedule<computational_dag_edge_idx_vector_impl_def_t, computational_dag_edge_idx_vector_impl_def_t> coarsenTest(
        sched, mlCoarsen);

    RunTest(&coarsenTest);
}

BOOST_AUTO_TEST_CASE(SquashAMulImproverTest) {
    GreedyBspScheduler<computational_dag_edge_idx_vector_impl_def_t> sched;
    HillClimbingScheduler<computational_dag_edge_idx_vector_impl_def_t> improver;

    SquashAMul<computational_dag_edge_idx_vector_impl_def_t, computational_dag_edge_idx_vector_impl_def_t> mlCoarsen;
    MultilevelCoarseAndSchedule<computational_dag_edge_idx_vector_impl_def_t, computational_dag_edge_idx_vector_impl_def_t> coarsenTest(
        sched, improver, mlCoarsen);

    RunTest(&coarsenTest);
}

BOOST_AUTO_TEST_CASE(SarkarMulTest) {
    GreedyBspScheduler<computational_dag_edge_idx_vector_impl_def_t> sched;

    SarkarMul<computational_dag_edge_idx_vector_impl_def_t, computational_dag_edge_idx_vector_impl_def_t> mlCoarsen;
    MultilevelCoarseAndSchedule<computational_dag_edge_idx_vector_impl_def_t, computational_dag_edge_idx_vector_impl_def_t> coarsenTest(
        sched, mlCoarsen);

    RunTest(&coarsenTest);
}

BOOST_AUTO_TEST_CASE(SarkarMulImproverTest) {
    GreedyBspScheduler<computational_dag_edge_idx_vector_impl_def_t> sched;
    HillClimbingScheduler<computational_dag_edge_idx_vector_impl_def_t> improver;

    SarkarMul<computational_dag_edge_idx_vector_impl_def_t, computational_dag_edge_idx_vector_impl_def_t> mlCoarsen;
    MultilevelCoarseAndSchedule<computational_dag_edge_idx_vector_impl_def_t, computational_dag_edge_idx_vector_impl_def_t> coarsenTest(
        sched, improver, mlCoarsen);

    RunTest(&coarsenTest);
}
