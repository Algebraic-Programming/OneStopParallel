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

#define BOOST_TEST_MODULE Bsp_Architecture
#include <boost/test/unit_test.hpp>

#include "graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "graph_implementations/boost_graphs/boost_graph.hpp"
#include "io/arch_file_reader.hpp"
#include "io/hdag_graph_file_reader.hpp"
#include <filesystem>
#include <iostream>

#include "pebbling/pebblers/pebblingILP/MultiProcessorPebbling.hpp"
#include "pebbling/pebblers/pebblingILP/PebblingPartialILP.hpp"
#include "bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(test_full) {

    using graph = computational_dag_vector_impl_def_t;

    BspInstance<graph> instance;
    instance.setNumberOfProcessors(4);
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
        (cwd / "data/spaa/tiny/instance_spmv_N6_nzP0d4.hdag").string(), instance.getComputationalDag());

    BOOST_CHECK(status);

    GreedyBspScheduler<graph> greedy;
    BspSchedule<graph> bsp_initial(instance);
    BOOST_CHECK_EQUAL(SUCCESS, greedy.computeSchedule(bsp_initial));

    std::vector<v_memw_t<graph> > minimum_memory_required_vector = PebblingSchedule<graph>::minimumMemoryRequiredPerNodeType(instance);
    v_memw_t<graph> max_required = *std::max_element(minimum_memory_required_vector.begin(), minimum_memory_required_vector.end());
    instance.getArchitecture().setMemoryBound(max_required);

    PebblingSchedule<graph> initial_sol(bsp_initial, PebblingSchedule<graph>::CACHE_EVICTION_STRATEGY::FORESIGHT);
    BOOST_CHECK(initial_sol.isValid());

    MultiProcessorPebbling<graph> mpp;
    mpp.setTimeLimitSeconds(10);
    auto status_and_solution = mpp.computePebblingWithInitialSolution(instance, initial_sol);
    status_and_solution.second.cleanSchedule();
    BOOST_CHECK(status_and_solution.second.isValid());

};

BOOST_AUTO_TEST_CASE(test_partial) {

    using graph = computational_dag_vector_impl_def_t;

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
        (cwd / "data/spaa/tiny/instance_spmv_N10_nzP0d25.hdag").string(), instance.getComputationalDag());

    BOOST_CHECK(status);

    std::vector<v_memw_t<graph> > minimum_memory_required_vector = PebblingSchedule<graph>::minimumMemoryRequiredPerNodeType(instance);
    v_memw_t<graph> max_required = *std::max_element(minimum_memory_required_vector.begin(), minimum_memory_required_vector.end());
    instance.getArchitecture().setMemoryBound(max_required);

    PebblingPartialILP<graph> mpp;
    mpp.setMinSize(15);
    mpp.setSecondsForSubILP(5);
    auto status_and_solution = mpp.computePebbling(instance);
    BOOST_CHECK(status_and_solution.second.isValid());

};