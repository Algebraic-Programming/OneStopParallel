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

@author Toni Boehnlein, Christos Matzoros, Pal Andras Papp, Raphael S. Steiner
*/

#define BOOST_TEST_MODULE BSP_SCHEDULERS
#include <boost/test/unit_test.hpp>
#include <filesystem>
#include <string>
#include <vector>

#include "osp/auxiliary/io/arch_file_reader.hpp"
#include "osp/auxiliary/io/general_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyVarianceSspScheduler.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GrowLocalMaxBsp.hpp"
#include "osp/bsp/scheduler/MaxBspScheduler.hpp"
#include "osp/graph_implementations/adj_list_impl/compact_sparse_graph.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "test_graphs.hpp"

using namespace osp;

using VImpl1 = CDagVertexImpl<std::size_t, unsigned, unsigned, unsigned, unsigned>;
using VImpl2 = CDagVertexImpl<uint32_t, unsigned, unsigned, unsigned, unsigned>;

std::vector<std::string> TestArchitectures() { return {"data/machine_params/p3.arch"}; }

template <typename GraphT>
void checkPrecedenceContraints(const BspSchedule<GraphT> &schedule, const unsigned staleness) {
    for (const auto &v : schedule.GetInstance().GetComputationalDag().Vertices()) {
        BOOST_CHECK_LT(schedule.AssignedSuperstep(v), schedule.NumberOfSupersteps());

        for (const auto &chld : schedule.GetInstance().GetComputationalDag().Children(v)) {
            const unsigned sameProcessors = (schedule.AssignedProcessor(v) == schedule.AssignedProcessor(chld)) ? 0U : staleness;

            BOOST_CHECK_LE(schedule.AssignedSuperstep(v) + sameProcessors, schedule.AssignedSuperstep(chld));
            if (schedule.AssignedSuperstep(v) + sameProcessors > schedule.AssignedSuperstep(chld)) {
                std::cout << "Vertex: " << v << " (S:" << schedule.AssignedSuperstep(v) << " P:" << schedule.AssignedProcessor(v)
                          << ")" << " Child: " << chld << " (S:" << schedule.AssignedSuperstep(chld)
                          << " P:" << schedule.AssignedProcessor(chld) << ")" << '\n';
            }
        }
    }
}

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

            std::cout << std::endl << "Scheduler: " << testScheduler->GetScheduleName() << std::endl;
            std::cout << "Graph: " << nameGraph << std::endl;
            std::cout << "Architecture: " << nameMachine << std::endl;

            ComputationalDagVectorImpl<VImpl1> graph;
            BspArchitecture<GraphT> arch;

            bool statusGraph = file_reader::ReadGraph((cwd / filenameGraph).string(), graph);
            bool statusArchitecture = file_reader::ReadBspArchitecture((cwd / filenameMachine).string(), arch);

            if (!statusGraph || !statusArchitecture) {
                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            BspInstance<GraphT> instance(graph, arch);

            BspSchedule<GraphT> schedule(instance);
            const auto result = testScheduler->ComputeSchedule(schedule);

            BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
            BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
            checkPrecedenceContraints(schedule, 1U);
        }
    }
}

template <typename GraphT>
void RunTestMaxBsp(MaxBspScheduler<GraphT> *testScheduler) {
    std::vector<std::string> filenamesGraph = TinySpaaGraphs();
    std::vector<std::string> filenamesArchitectures = TestArchitectures();

    // Locate project root
    std::filesystem::path cwd = std::filesystem::current_path();
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
    }

    for (auto &filenameGraph : filenamesGraph) {
        for (auto &filenameMachine : filenamesArchitectures) {
            std::string nameGraph = filenameGraph.substr(filenameGraph.find_last_of("/\\") + 1);
            nameGraph = nameGraph.substr(0, nameGraph.find_last_of("."));
            std::string nameMachine = filenameMachine.substr(filenameMachine.find_last_of("/\\") + 1);
            nameMachine = nameMachine.substr(0, nameMachine.rfind("."));

            std::cout << std::endl
                      << "Scheduler (MaxBsp): " << testScheduler->GetScheduleName() << std::endl
                      << "Graph: " << nameGraph << std::endl
                      << "Architecture: " << nameMachine << std::endl;

            ComputationalDagVectorImpl<VImpl1> graph;
            BspArchitecture<GraphT> arch;

            bool statusGraph = file_reader::ReadGraph((cwd / filenameGraph).string(), graph);
            bool statusArchitecture = file_reader::ReadBspArchitecture((cwd / filenameMachine).string(), arch);

            BOOST_REQUIRE_MESSAGE(statusGraph, "Failed to read graph: " << filenameGraph);
            BOOST_REQUIRE_MESSAGE(statusArchitecture, "Failed to read architecture: " << filenameMachine);

            BspInstance<GraphT> instance(graph, arch);

            MaxBspSchedule<GraphT> schedule(instance);

            const auto result = testScheduler->ComputeSchedule(schedule);

            BOOST_CHECK_EQUAL(result, ReturnStatus::OSP_SUCCESS);
            BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
            checkPrecedenceContraints(schedule, 2U);
        }
    }
}

// Tests ComputeSchedule(BspSchedule&) → staleness = 1
BOOST_AUTO_TEST_CASE(GreedyVarianceSspSchedulerTestVectorImpl) {
    GreedyVarianceSspScheduler<ComputationalDagVectorImpl<VImpl1>> test;
    RunTest(&test);
}

// Tests ComputeSchedule(BspSchedule&) → staleness = 1 (different graph impl)
BOOST_AUTO_TEST_CASE(GreedyVarianceSspSchedulerTestEdgeIdxImpl) {
    GreedyVarianceSspScheduler<ComputationalDagVectorImpl<VImpl2>> test;
    RunTest(&test);
}

// Tests ComputeSchedule(MaxBspSchedule&) → staleness = 2
BOOST_AUTO_TEST_CASE(GreedyVarianceSspSchedulerMaxBspScheduleLargeTest) {
    GreedyVarianceSspScheduler<ComputationalDagVectorImpl<VImpl1>> test;
    RunTestMaxBsp(&test);
}

// Tests ComputeSchedule(BspSchedule&) → staleness = 1
BOOST_AUTO_TEST_CASE(GrowLocalSSPBspScheduleLargeTest) {
    GrowLocalSSP<CompactSparseGraph<false, true, true, true, true>> test;
    RunTest(&test);
}

// Tests ComputeSchedule(MaxBspSchedule&) → staleness = 2
BOOST_AUTO_TEST_CASE(GrowLocalSSPMaxBspScheduleLargeTest) {
    GrowLocalSSP<CompactSparseGraph<false, true, true, true, true>> test;
    RunTestMaxBsp(&test);
}
