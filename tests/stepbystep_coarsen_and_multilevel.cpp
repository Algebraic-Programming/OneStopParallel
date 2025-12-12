
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

#define BOOST_TEST_MODULE STEPBYSTEP_AND_MULTILEVEL
#include <boost/test/unit_test.hpp>
#include <filesystem>
#include <string>
#include <vector>

#include "osp/auxiliary/io/arch_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/bsp/scheduler/CoarsenRefineSchedulers/MultiLevelHillClimbing.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/coarser/StepByStep/StepByStepCoarser.hpp"
#include "osp/coarser/coarser_util.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(StepByStepCoarserTest) {
    using Graph = boost_graph_uint_t;
    StepByStepCoarser<Graph> test;

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    Graph dag;

    bool status = file_reader::readComputationalDagHyperdagFormatDB(
        (cwd / "data/spaa/tiny/instance_spmv_N10_nzP0d25.hdag").string(), dag);

    BOOST_CHECK(status);

    StepByStepCoarser<Graph> coarser;

    coarser.setTargetNumberOfNodes(static_cast<unsigned>(dag.NumVertices()) / 2);

    Graph coarsenedDag1, coarsenedDag2;
    std::vector<std::vector<VertexIdxT<Graph>>> oldVertexIds;
    std::vector<VertexIdxT<Graph>> newVertexId;

    coarser.coarsenDag(dag, coarsenedDag1, newVertexId);
    oldVertexIds = coarser_util::invert_vertex_contraction_map<Graph, Graph>(newVertexId);

    coarser.setTargetNumberOfNodes(static_cast<unsigned>(dag.NumVertices()) * 2 / 3);
    coarser.coarsenForPebbling(dag, coarsenedDag2, newVertexId);
    oldVertexIds = coarser_util::invert_vertex_contraction_map<Graph, Graph>(newVertexId);
}

BOOST_AUTO_TEST_CASE(MultilevelTest) {
    using Graph = boost_graph_uint_t;
    StepByStepCoarser<Graph> test;

    BspInstance<Graph> instance;
    instance.setNumberOfProcessors(2);
    instance.setCommunicationCosts(3);
    instance.setSynchronisationCosts(5);

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    bool status = file_reader::readComputationalDagHyperdagFormatDB((cwd / "data/spaa/tiny/instance_pregel.hdag").string(),
                                                                    instance.GetComputationalDag());

    BOOST_CHECK(status);

    MultiLevelHillClimbingScheduler<Graph> multi1, multi2;
    BspSchedule<Graph> schedule1(instance), schedule2(instance);

    multi1.setContractionRate(0.3);
    multi1.useLinearRefinementSteps(5);

    auto result = multi1.computeSchedule(schedule1);
    BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
    BOOST_CHECK(schedule1.satisfiesPrecedenceConstraints());

    multi2.setContractionRate(0.3);
    multi2.useExponentialRefinementPoints(1.2);

    result = multi2.computeSchedule(schedule2);
    BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
    BOOST_CHECK(schedule2.satisfiesPrecedenceConstraints());
}
