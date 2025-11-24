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

#define BOOST_TEST_MODULE MAXBSP_SCHEDULERS
#include <boost/test/unit_test.hpp>

#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/BspToMaxBspConverter.hpp"
#include "osp/bsp/scheduler/LocalSearch/HillClimbing/hill_climbing.hpp"
#include "osp/bsp/scheduler/LocalSearch/HillClimbing/hill_climbing_for_comm_schedule.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include <filesystem>
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"

using namespace osp;


BOOST_AUTO_TEST_CASE(maxbsp_scheduling) {

    using graph = computational_dag_vector_impl_def_t;

    BspInstance<graph> instance;
    instance.setNumberOfProcessors(4);
    instance.setCommunicationCosts(3);
    instance.setSynchronisationCosts(3);

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    bool status = file_reader::readComputationalDagHyperdagFormatDB(
        (cwd / "data/spaa/tiny/instance_CG_N4_K1_nzP0d35.hdag").string(), instance.getComputationalDag());

    BOOST_CHECK(status);

    GreedyBspScheduler<graph> greedy;
    BspSchedule<graph> bsp_initial(instance);
    BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, greedy.computeSchedule(bsp_initial));
    BOOST_CHECK(bsp_initial.satisfiesPrecedenceConstraints());


    // PART I: from BspSchedule to MaxBspSchedule conversion

    std::cout<<"Original Bsp Cost: "<<bsp_initial.computeCosts()<<std::endl;
    GreedyBspToMaxBspConverter<graph> converter;
    MaxBspSchedule<graph> maxbsp = converter.Convert(bsp_initial);
    BOOST_CHECK(maxbsp.satisfiesPrecedenceConstraints());
    auto cost_conversion = maxbsp.computeCosts();
    std::cout<<"Cost after maxBsp conversion: "<<cost_conversion<<std::endl;

    // hill climbing

    HillClimbingScheduler<graph> HC;
    HC.improveSchedule(maxbsp);
    BOOST_CHECK(maxbsp.satisfiesPrecedenceConstraints());
    auto cost_hc = maxbsp.computeCosts();
    std::cout<<"Cost after Hill Climbing: "<<cost_hc<<std::endl;
    BOOST_CHECK(cost_hc <= cost_conversion);

    
    // PART II: from BspScheduleCS to MaxBspScheduleCS conversion

    BspScheduleCS<graph> bsp_initial_cs(bsp_initial);
    BOOST_CHECK(bsp_initial_cs.hasValidCommSchedule());
    std::cout<<"Original BspCS Cost: "<<bsp_initial_cs.computeCosts()<<std::endl;

    MaxBspScheduleCS<graph> maxbsp_cs = converter.Convert(bsp_initial_cs);
    BOOST_CHECK(maxbsp_cs.satisfiesPrecedenceConstraints());
    BOOST_CHECK(maxbsp_cs.hasValidCommSchedule());
    auto cost_conversion_cs = maxbsp_cs.computeCosts();
    std::cout<<"Cost after maxBsp(CS) conversion: "<<cost_conversion_cs<<std::endl;

    // hill climbing for comm. schedule

    HillClimbingForCommSteps<graph> HCcs;
    HCcs.improveSchedule(maxbsp_cs);
    BOOST_CHECK(maxbsp_cs.satisfiesPrecedenceConstraints());
    BOOST_CHECK(maxbsp_cs.hasValidCommSchedule());
    auto cost_hccs = maxbsp_cs.computeCosts();
    std::cout<<"Cost after comm. sched. hill climbing: "<<cost_hccs<<std::endl;
    BOOST_CHECK(cost_hccs <= cost_conversion_cs);


    // PART III: same for larger DAG

    status = file_reader::readComputationalDagHyperdagFormatDB(
        (cwd / "data/spaa/large/instance_CG_N24_K22_nzP0d2.hdag").string(), instance.getComputationalDag());

    BOOST_CHECK(status);
    instance.setSynchronisationCosts(7);

    BspSchedule<graph> bsp_initial_large(instance);
    BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, greedy.computeSchedule(bsp_initial_large));

    BspScheduleCS<graph> bsp_initial_large_cs(bsp_initial_large);
    BOOST_CHECK(bsp_initial_large_cs.hasValidCommSchedule());
    std::cout<<"Original Bsp Cost on large DAG: "<<bsp_initial_large_cs.computeCosts()<<std::endl;

    MaxBspScheduleCS<graph> maxbsp_cs_large = converter.Convert(bsp_initial_large_cs);
    BOOST_CHECK(maxbsp_cs_large.satisfiesPrecedenceConstraints());
    BOOST_CHECK(maxbsp_cs_large.hasValidCommSchedule());
    auto cost_maxbsp_cs_large = maxbsp_cs_large.computeCosts();
    std::cout<<"Cost after maxBsp conversion on large DAG: "<<cost_maxbsp_cs_large<<std::endl;

    HCcs.improveSchedule(maxbsp_cs_large);
    BOOST_CHECK(maxbsp_cs_large.satisfiesPrecedenceConstraints());
    BOOST_CHECK(maxbsp_cs_large.hasValidCommSchedule());
    auto cost_hccs_large = maxbsp_cs_large.computeCosts();
    std::cout<<"Cost after comm. sched. hill climbing on large DAG: "<<cost_hccs_large<<std::endl;
    BOOST_CHECK(cost_hccs_large <= cost_maxbsp_cs_large);
}
