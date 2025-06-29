
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

#include "coarser/StepByStep/StepByStepCoarser.hpp"
#include "bsp/scheduler/Scheduler.hpp"
#include "bsp/scheduler/CoarsenRefineSchedulers/MultiLevelHillClimbing.hpp"
#include "auxiliary/io/hdag_graph_file_reader.hpp"
#include "auxiliary/io/arch_file_reader.hpp"
#include "coarser/coarser_util.hpp"

#include "graph_implementations/boost_graphs/boost_graph.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(StepByStepCoarser_test) {

    using graph = boost_graph_uint_t;
    StepByStepCoarser<graph> test;

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    graph DAG;

    bool status = file_reader::readComputationalDagHyperdagFormat(
        (cwd / "data/spaa/tiny/instance_spmv_N10_nzP0d25.hdag").string(), DAG);

    BOOST_CHECK(status);

    StepByStepCoarser<graph> coarser;

    coarser.setTargetNumberOfNodes(static_cast<unsigned>(DAG.num_vertices())/2);

    graph coarsened_dag1, coarsened_dag2;
    std::vector<std::vector<vertex_idx_t<graph>>> old_vertex_ids;
    std::vector<vertex_idx_t<graph>> new_vertex_id;

    coarser.coarsenDag(DAG, coarsened_dag1, new_vertex_id);
    old_vertex_ids = coarser_util::invert_vertex_contraction_map<graph, graph>(new_vertex_id);

    coarser.setTargetNumberOfNodes(static_cast<unsigned>(DAG.num_vertices())*2/3);
    coarser.coarsenForPebbling(DAG, coarsened_dag2, new_vertex_id);
    old_vertex_ids = coarser_util::invert_vertex_contraction_map<graph, graph>(new_vertex_id);

};

BOOST_AUTO_TEST_CASE(Multilevel_test) {

    using graph = boost_graph_uint_t;
    StepByStepCoarser<graph> test;

    BspInstance<graph> instance;
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

    bool status = file_reader::readComputationalDagHyperdagFormat(
        (cwd / "data/spaa/tiny/instance_pregel.hdag").string(), instance.getComputationalDag());

    BOOST_CHECK(status);


    MultiLevelHillClimbingScheduler<graph> multi1, multi2;
    BspSchedule<graph> schedule1(instance), schedule2(instance);

    multi1.setTargetNumberOfNodes(static_cast<unsigned>(instance.getComputationalDag().num_vertices())/3);
    multi1.setLinearRefinementPoints(static_cast<vertex_idx_t<graph> >(instance.getComputationalDag().num_vertices()), 5);

    auto result = multi1.computeSchedule(schedule1);
    BOOST_CHECK_EQUAL(SUCCESS, result);
    BOOST_CHECK(schedule1.satisfiesPrecedenceConstraints());

    multi2.setTargetNumberOfNodes(static_cast<unsigned>(instance.getComputationalDag().num_vertices())/3);
    multi2.setExponentialRefinementPoints(static_cast<vertex_idx_t<graph> >(instance.getComputationalDag().num_vertices()), 1.2);

    result = multi2.computeSchedule(schedule2);
    BOOST_CHECK_EQUAL(SUCCESS, result);
    BOOST_CHECK(schedule2.satisfiesPrecedenceConstraints());

};