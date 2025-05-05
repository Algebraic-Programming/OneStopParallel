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

#include "bsp/model/BspInstance.hpp"
#include "bsp/model/BspSchedule.hpp"
#include "bsp/model/BspScheduleCS.hpp"
#include "graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "io/arch_file_reader.hpp"
#include "io/graph_file_reader.hpp"
#include <filesystem>
#include <iostream>

#include "bsp/scheduler/GreedySchedulers/BspLocking.hpp"
#include "bsp/scheduler/GreedySchedulers/CilkScheduler.hpp"
#include "bsp/scheduler/GreedySchedulers/EtfScheduler.hpp"
#include "bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "bsp/scheduler/GreedySchedulers/GreedyChildren.hpp"
#include "bsp/scheduler/GreedySchedulers/GrowLocalAutoCores.hpp"
#include "bsp/scheduler/GreedySchedulers/RandomGreedy.hpp"
#include "bsp/scheduler/GreedySchedulers/VarianceFillup.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(test_instance_bicgstab) {

    using graph = computational_dag_edge_idx_vector_impl_def_t;

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
        (cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(), instance.getComputationalDag());

    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(instance.getComputationalDag().num_vertices(), 54);
    BOOST_CHECK_EQUAL(instance.getComputationalDag().num_vertex_types(), 1);

    std::vector<Scheduler<graph> *> schedulers = {new BspLocking<graph>(),                                      
                                                  new EtfScheduler<graph>(),   
                                                  new GreedyBspScheduler<graph>(),
                                                  new GreedyChildren<graph>(), 
                                                  new GrowLocalAutoCores<graph>(), 
                                                  new VarianceFillup<graph>()};

    std::vector<int> expected_bsp_costs = {92, 108, 100, 108, 102,  110};
    std::vector<double> expected_total_costs = {74, 87, 84.25, 80.25, 91.25, 86.75};
    std::vector<int> expected_buffered_sending_costs = {92, 111, 103, 105, 102, 113};
    std::vector<unsigned> expected_supersteps = {6, 7, 7, 5, 3, 7};

    std::vector<int> expected_bsp_cs_costs = {86, 99, 97, 99, 102, 107};


    size_t i = 0;
    for (auto &scheduler : schedulers) {

        std::pair<RETURN_STATUS, BspSchedule<graph>> result = scheduler->computeSchedule(instance);

        BOOST_CHECK_EQUAL(SUCCESS, result.first);
        BOOST_CHECK_EQUAL(&result.second.getInstance(), &instance);
        BOOST_CHECK(result.second.satisfiesPrecedenceConstraints());

        BOOST_CHECK_EQUAL(result.second.computeCosts(), expected_bsp_costs[i]);
        BOOST_CHECK_EQUAL(result.second.computeTotalCosts(), expected_total_costs[i]);
        BOOST_CHECK_EQUAL(result.second.computeBufferedSendingCosts(), expected_buffered_sending_costs[i]);
        BOOST_CHECK_EQUAL(result.second.numberOfSupersteps(), expected_supersteps[i]);


        const auto result_cs = scheduler->computeScheduleCS(instance);
        BOOST_CHECK_EQUAL(SUCCESS, result_cs.first);

        BOOST_CHECK(result_cs.second.hasValidCommSchedule());

        BOOST_CHECK_EQUAL(result_cs.second.computeCosts(), expected_bsp_cs_costs[i]);
        

        i++;

        delete scheduler;
    }
};

