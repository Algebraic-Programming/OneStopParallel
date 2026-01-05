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
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_include.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "test_graphs.hpp"

using namespace osp;

template <typename GraphT>
void AddMemWeights(GraphT &dag) {
    int memWeight = 1;
    int commWeight = 1;

    for (const auto &v : dag.Vertices()) {
        dag.SetVertexMemWeight(v, static_cast<VMemwT<GraphT>>(memWeight++ % 3 + 1));
        dag.SetVertexCommWeight(v, static_cast<VCommwT<GraphT>>(commWeight++ % 3 + 1));
    }
}

BOOST_AUTO_TEST_CASE(KlLocalMemconst) {
    std::vector<std::string> filenamesGraph = TestGraphs();

    using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    GreedyBspScheduler<Graph, LocalMemoryConstraint<Graph>> testScheduler;

    for (auto &filenameGraph : filenamesGraph) {
        std::cout << filenameGraph << std::endl;
        BspInstance<Graph> instance;

        bool statusGraph
            = file_reader::ReadComputationalDagHyperdagFormatDB((cwd / filenameGraph).string(), instance.GetComputationalDag());
        instance.GetArchitecture().SetSynchronisationCosts(10);
        instance.GetArchitecture().SetCommunicationCosts(5);
        instance.GetArchitecture().SetNumberOfProcessors(4);
        instance.GetArchitecture().SetMemoryConstraintType(MemoryConstraintType::LOCAL);
        instance.GetArchitecture().SetSynchronisationCosts(0);

        const std::vector<int> boundsToTest = {10, 20};

        AddMemWeights(instance.GetComputationalDag());

        if (!statusGraph) {
            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        for (const auto &bound : boundsToTest) {
            instance.GetArchitecture().SetMemoryBound(bound);

            BspSchedule<Graph> schedule(instance);
            const auto result = testScheduler.ComputeSchedule(schedule);

            BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
            BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
            BOOST_CHECK(schedule.SatisfiesMemoryConstraints());

            KlTotalCommImproverLocalMemConstr<Graph> kl;

            auto status = kl.ImproveSchedule(schedule);

            BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
            BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
            BOOST_CHECK(schedule.SatisfiesMemoryConstraints());
        }
    }
}
