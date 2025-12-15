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
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/BspLocking.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/EtfScheduler.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GrowLocalAutoCores.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/VarianceFillup.hpp"
#include "osp/bsp/scheduler/LoadBalanceScheduler/LightEdgeVariancePartitioner.hpp"
#include "osp/bsp/scheduler/LoadBalanceScheduler/VariancePartitioner.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_total_comm.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_total_cut.hpp"
#include "osp/bsp/scheduler/Serial.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "test_graphs.hpp"

using namespace osp;

std::vector<std::string> TestArchitectures() { return {"data/machine_params/p3.arch"}; }

template <typename GraphT>
void AddMemWeights(GraphT &dag) {
    int memWeight = 1;
    int commWeight = 1;

    for (const auto &v : dag.Vertices()) {
        dag.SetVertexMemWeight(v, static_cast<VMemwT<GraphT>>(memWeight++ % 3 + 1));
        dag.SetVertexCommWeight(v, static_cast<VCommwT<GraphT>>(commWeight++ % 3 + 1));
    }
}

template <typename GraphT>
void RunTestLocalMemory(Scheduler<GraphT> *testScheduler) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenamesGraph = TestGraphs();
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

            std::cout << std::endl << "Scheduler: " << testScheduler->GetScheduleName() << std::endl;
            std::cout << "Graph: " << nameGraph << std::endl;
            std::cout << "Architecture: " << nameMachine << std::endl;

            BspInstance<GraphT> instance;

            bool statusGraph = file_reader::readComputationalDagHyperdagFormatDB((cwd / filenameGraph).string(),
                                                                                 instance.GetComputationalDag());
            bool statusArchitecture
                = file_reader::ReadBspArchitecture((cwd / "data/machine_params/p3.arch").string(), instance.GetArchitecture());

            AddMemWeights(instance.GetComputationalDag());
            instance.GetArchitecture().SetMemoryConstraintType(MemoryConstraintType::LOCAL);
            std::cout << "Memory constraint type: LOCAL" << std::endl;

            if (!statusGraph || !statusArchitecture) {
                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            const std::vector<VMemwT<GraphT>> boundsToTest = {10, 20, 50, 100};

            for (const auto &bound : boundsToTest) {
                instance.GetArchitecture().setMemoryBound(bound);

                BspSchedule<GraphT> schedule(instance);
                const auto result = testScheduler->ComputeSchedule(schedule);

                BOOST_CHECK(ReturnStatus::OSP_SUCCESS == result || ReturnStatus::BEST_FOUND == result);
                BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
                BOOST_CHECK(schedule.satisfiesMemoryConstraints());
            }
        }
    }
}

template <typename GraphT>
void RunTestPersistentTransientMemory(Scheduler<GraphT> *testScheduler) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenamesGraph = TestGraphs();
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

            std::cout << std::endl << "Scheduler: " << testScheduler->GetScheduleName() << std::endl;
            std::cout << "Graph: " << nameGraph << std::endl;
            std::cout << "Architecture: " << nameMachine << std::endl;

            BspInstance<GraphT> instance;

            bool statusGraph = file_reader::readComputationalDagHyperdagFormatDB((cwd / filenameGraph).string(),
                                                                                 instance.GetComputationalDag());
            bool statusArchitecture
                = file_reader::ReadBspArchitecture((cwd / "data/machine_params/p3.arch").string(), instance.GetArchitecture());

            AddMemWeights(instance.GetComputationalDag());
            instance.GetArchitecture().SetMemoryConstraintType(MemoryConstraintType::PERSISTENT_AND_TRANSIENT);
            std::cout << "Memory constraint type: PERSISTENT_AND_TRANSIENT" << std::endl;

            if (!statusGraph || !statusArchitecture) {
                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            const std::vector<VMemwT<GraphT>> boundsToTest = {50, 100};

            for (const auto &bound : boundsToTest) {
                instance.GetArchitecture().setMemoryBound(bound);

                BspSchedule<GraphT> schedule(instance);
                const auto result = testScheduler->ComputeSchedule(schedule);

                BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
                BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
                BOOST_CHECK(schedule.satisfiesMemoryConstraints());
            }
        }
    }
}

template <typename GraphT>
void RunTestLocalInOutMemory(Scheduler<GraphT> *testScheduler) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenamesGraph = TestGraphs();
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

            std::cout << std::endl << "Scheduler: " << testScheduler->GetScheduleName() << std::endl;
            std::cout << "Graph: " << nameGraph << std::endl;
            std::cout << "Architecture: " << nameMachine << std::endl;

            BspInstance<GraphT> instance;

            bool statusGraph = file_reader::readComputationalDagHyperdagFormatDB((cwd / filenameGraph).string(),
                                                                                 instance.GetComputationalDag());
            bool statusArchitecture
                = file_reader::ReadBspArchitecture((cwd / "data/machine_params/p3.arch").string(), instance.GetArchitecture());

            AddMemWeights(instance.GetComputationalDag());
            instance.GetArchitecture().SetMemoryConstraintType(MemoryConstraintType::LOCAL_IN_OUT);
            std::cout << "Memory constraint type: LOCAL_IN_OUT" << std::endl;

            if (!statusGraph || !statusArchitecture) {
                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            const std::vector<VMemwT<GraphT>> boundsToTest = {10, 20, 50, 100};

            for (const auto &bound : boundsToTest) {
                instance.GetArchitecture().setMemoryBound(bound);

                BspSchedule<GraphT> schedule(instance);
                const auto result = testScheduler->ComputeSchedule(schedule);

                BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
                BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
                BOOST_CHECK(schedule.satisfiesMemoryConstraints());
            }
        }
    }
}

template <typename GraphT>
void RunTestLocalIncEdgesMemory(Scheduler<GraphT> *testScheduler) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenamesGraph = TestGraphs();
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

            std::cout << std::endl << "Scheduler: " << testScheduler->GetScheduleName() << std::endl;
            std::cout << "Graph: " << nameGraph << std::endl;
            std::cout << "Architecture: " << nameMachine << std::endl;

            BspInstance<GraphT> instance;

            bool statusGraph = file_reader::readComputationalDagHyperdagFormatDB((cwd / filenameGraph).string(),
                                                                                 instance.GetComputationalDag());
            bool statusArchitecture
                = file_reader::ReadBspArchitecture((cwd / "data/machine_params/p3.arch").string(), instance.GetArchitecture());

            AddMemWeights(instance.GetComputationalDag());
            instance.GetArchitecture().SetMemoryConstraintType(MemoryConstraintType::LOCAL_INC_EDGES);
            std::cout << "Memory constraint type: LOCAL_INC_EDGES" << std::endl;

            if (!statusGraph || !statusArchitecture) {
                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            const std::vector<VMemwT<GraphT>> boundsToTest = {50, 100};

            for (const auto &bound : boundsToTest) {
                instance.GetArchitecture().setMemoryBound(bound);

                BspSchedule<GraphT> schedule(instance);
                const auto result = testScheduler->ComputeSchedule(schedule);

                BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
                BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
                BOOST_CHECK(schedule.satisfiesMemoryConstraints());
            }
        }
    }
}

template <typename GraphT>
void RunTestLocalIncEdges2Memory(Scheduler<GraphT> *testScheduler) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenamesGraph = TestGraphs();
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

            std::cout << std::endl << "Scheduler: " << testScheduler->GetScheduleName() << std::endl;
            std::cout << "Graph: " << nameGraph << std::endl;
            std::cout << "Architecture: " << nameMachine << std::endl;

            BspInstance<GraphT> instance;

            bool statusGraph = file_reader::readComputationalDagHyperdagFormatDB((cwd / filenameGraph).string(),
                                                                                 instance.GetComputationalDag());
            bool statusArchitecture
                = file_reader::ReadBspArchitecture((cwd / "data/machine_params/p3.arch").string(), instance.GetArchitecture());

            AddMemWeights(instance.GetComputationalDag());
            instance.GetArchitecture().SetMemoryConstraintType(MemoryConstraintType::LOCAL_SOURCES_INC_EDGES);
            std::cout << "Memory constraint type: LOCAL_SOURCES_INC_EDGES" << std::endl;

            if (!statusGraph || !statusArchitecture) {
                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            const std::vector<VMemwT<GraphT>> boundsToTest = {20, 50, 100};

            for (const auto &bound : boundsToTest) {
                instance.GetArchitecture().setMemoryBound(bound);

                BspSchedule<GraphT> schedule(instance);
                const auto result = testScheduler->ComputeSchedule(schedule);

                BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
                BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
                BOOST_CHECK(schedule.satisfiesMemoryConstraints());
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(GreedyBspSchedulerLocalTest) {
    using GraphImplT = computational_dag_edge_idx_vector_impl_def_int_t;

    GreedyBspScheduler<GraphImplT, local_memory_constraint<GraphImplT>> test1;
    RunTestLocalMemory(&test1);

    GreedyBspScheduler<GraphImplT, local_in_out_memory_constraint<GraphImplT>> test2;
    RunTestLocalInOutMemory(&test2);

    GreedyBspScheduler<GraphImplT, local_inc_edges_memory_constraint<GraphImplT>> test3;
    RunTestLocalIncEdgesMemory(&test3);

    GreedyBspScheduler<GraphImplT, local_sources_inc_edges_memory_constraint<GraphImplT>> test4;
    RunTestLocalIncEdges2Memory(&test4);
}

BOOST_AUTO_TEST_CASE(GrowLocalAutoCoresLocalTest) {
    using GraphImplT = computational_dag_edge_idx_vector_impl_def_int_t;

    GrowLocalAutoCores<GraphImplT, local_memory_constraint<GraphImplT>> test1;
    RunTestLocalMemory(&test1);

    GrowLocalAutoCores<GraphImplT, local_in_out_memory_constraint<GraphImplT>> test2;
    RunTestLocalInOutMemory(&test2);

    GrowLocalAutoCores<GraphImplT, local_inc_edges_memory_constraint<GraphImplT>> test3;
    RunTestLocalIncEdgesMemory(&test3);

    GrowLocalAutoCores<GraphImplT, local_sources_inc_edges_memory_constraint<GraphImplT>> test4;
    RunTestLocalIncEdges2Memory(&test4);
}

BOOST_AUTO_TEST_CASE(BspLockingLocalTest) {
    using GraphImplT = ComputationalDagEdgeIdxVectorImplDefT;

    BspLocking<GraphImplT, local_memory_constraint<GraphImplT>> test1;
    RunTestLocalMemory(&test1);

    BspLocking<GraphImplT, local_in_out_memory_constraint<GraphImplT>> test2;
    RunTestLocalInOutMemory(&test2);

    BspLocking<GraphImplT, local_inc_edges_memory_constraint<GraphImplT>> test3;
    RunTestLocalIncEdgesMemory(&test3);

    BspLocking<GraphImplT, local_sources_inc_edges_memory_constraint<GraphImplT>> test4;
    RunTestLocalIncEdges2Memory(&test4);
}

BOOST_AUTO_TEST_CASE(VarianceLocalTest) {
    VarianceFillup<ComputationalDagEdgeIdxVectorImplDefT, local_memory_constraint<ComputationalDagEdgeIdxVectorImplDefT>>
        test;
    RunTestLocalMemory(&test);
}

// BOOST_AUTO_TEST_CASE(kl_local_test) {

//     VarianceFillup<ComputationalDagEdgeIdxVectorImplDefT,
//                    local_memory_constraint<ComputationalDagEdgeIdxVectorImplDefT>>
//         test;

//     kl_total_comm<ComputationalDagEdgeIdxVectorImplDefT,
//     local_search_local_memory_constraint<ComputationalDagEdgeIdxVectorImplDefT>> kl;

//     ComboScheduler<ComputationalDagEdgeIdxVectorImplDefT> combo_test(test, kl);

//     run_test_local_memory(&combo_test);
// };

BOOST_AUTO_TEST_CASE(GreedyBspSchedulerPersistentTransientTest) {
    GreedyBspScheduler<ComputationalDagEdgeIdxVectorImplDefT,
                       persistent_transient_memory_constraint<ComputationalDagEdgeIdxVectorImplDefT>>
        test;
    RunTestPersistentTransientMemory(&test);
}

BOOST_AUTO_TEST_CASE(EtfSchedulerPersistentTransientTest) {
    EtfScheduler<ComputationalDagEdgeIdxVectorImplDefT,
                 persistent_transient_memory_constraint<ComputationalDagEdgeIdxVectorImplDefT>>
        test;
    RunTestPersistentTransientMemory(&test);
}

BOOST_AUTO_TEST_CASE(VariancePartitionerTest) {
    VariancePartitioner<ComputationalDagEdgeIdxVectorImplDefT,
                        linear_interpolation,
                        local_memory_constraint<ComputationalDagEdgeIdxVectorImplDefT>>
        testLinear;
    RunTestLocalMemory(&testLinear);

    VariancePartitioner<ComputationalDagEdgeIdxVectorImplDefT,
                        flat_spline_interpolation,
                        local_memory_constraint<ComputationalDagEdgeIdxVectorImplDefT>>
        testFlat;
    RunTestLocalMemory(&testFlat);

    VariancePartitioner<ComputationalDagEdgeIdxVectorImplDefT,
                        superstep_only_interpolation,
                        local_memory_constraint<ComputationalDagEdgeIdxVectorImplDefT>>
        testSuperstep;
    RunTestLocalMemory(&testSuperstep);

    VariancePartitioner<ComputationalDagEdgeIdxVectorImplDefT,
                        global_only_interpolation,
                        local_memory_constraint<ComputationalDagEdgeIdxVectorImplDefT>>
        testGlobal;
    RunTestLocalMemory(&testGlobal);

    VariancePartitioner<ComputationalDagEdgeIdxVectorImplDefT,
                        linear_interpolation,
                        persistent_transient_memory_constraint<ComputationalDagEdgeIdxVectorImplDefT>>
        testLinearTp;
    RunTestPersistentTransientMemory(&testLinearTp);

    VariancePartitioner<ComputationalDagEdgeIdxVectorImplDefT,
                        flat_spline_interpolation,
                        persistent_transient_memory_constraint<ComputationalDagEdgeIdxVectorImplDefT>>
        testFlatTp;
    RunTestPersistentTransientMemory(&testFlatTp);

    VariancePartitioner<ComputationalDagEdgeIdxVectorImplDefT,
                        superstep_only_interpolation,
                        persistent_transient_memory_constraint<ComputationalDagEdgeIdxVectorImplDefT>>
        testSuperstepTp;
    RunTestPersistentTransientMemory(&testSuperstepTp);

    VariancePartitioner<ComputationalDagEdgeIdxVectorImplDefT,
                        global_only_interpolation,
                        persistent_transient_memory_constraint<ComputationalDagEdgeIdxVectorImplDefT>>
        testGlobalTp;
    RunTestPersistentTransientMemory(&testGlobalTp);
}

BOOST_AUTO_TEST_CASE(LightEdgeVariancePartitionerTest) {
    LightEdgeVariancePartitioner<ComputationalDagEdgeIdxVectorImplDefT,
                                 linear_interpolation,
                                 local_memory_constraint<ComputationalDagEdgeIdxVectorImplDefT>>
        testLinear;
    RunTestLocalMemory(&testLinear);

    LightEdgeVariancePartitioner<ComputationalDagEdgeIdxVectorImplDefT,
                                 flat_spline_interpolation,
                                 local_memory_constraint<ComputationalDagEdgeIdxVectorImplDefT>>
        testFlat;
    RunTestLocalMemory(&testFlat);

    LightEdgeVariancePartitioner<ComputationalDagEdgeIdxVectorImplDefT,
                                 superstep_only_interpolation,
                                 local_memory_constraint<ComputationalDagEdgeIdxVectorImplDefT>>
        testSuperstep;
    RunTestLocalMemory(&testSuperstep);

    LightEdgeVariancePartitioner<ComputationalDagEdgeIdxVectorImplDefT,
                                 global_only_interpolation,
                                 local_memory_constraint<ComputationalDagEdgeIdxVectorImplDefT>>
        testGlobal;
    RunTestLocalMemory(&testGlobal);

    LightEdgeVariancePartitioner<ComputationalDagEdgeIdxVectorImplDefT,
                                 linear_interpolation,
                                 persistent_transient_memory_constraint<ComputationalDagEdgeIdxVectorImplDefT>>
        testLinearTp;
    RunTestPersistentTransientMemory(&testLinearTp);

    LightEdgeVariancePartitioner<ComputationalDagEdgeIdxVectorImplDefT,
                                 flat_spline_interpolation,
                                 persistent_transient_memory_constraint<ComputationalDagEdgeIdxVectorImplDefT>>
        testFlatTp;
    RunTestPersistentTransientMemory(&testFlatTp);

    LightEdgeVariancePartitioner<ComputationalDagEdgeIdxVectorImplDefT,
                                 superstep_only_interpolation,
                                 persistent_transient_memory_constraint<ComputationalDagEdgeIdxVectorImplDefT>>
        testSuperstepTp;
    RunTestPersistentTransientMemory(&testSuperstepTp);

    LightEdgeVariancePartitioner<ComputationalDagEdgeIdxVectorImplDefT,
                                 global_only_interpolation,
                                 persistent_transient_memory_constraint<ComputationalDagEdgeIdxVectorImplDefT>>
        testGlobalTp;
    RunTestPersistentTransientMemory(&testGlobalTp);
}
