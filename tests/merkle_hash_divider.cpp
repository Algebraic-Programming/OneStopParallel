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

#define BOOST_TEST_MODULE BSP_SCHEDULE_RECOMP
#include <boost/test/unit_test.hpp>

#include "osp/dag_divider/wavefront_divider/ScanWavefrontDivider.hpp"
#include "osp/dag_divider/wavefront_divider/RecursiveWavefrontDivider.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/auxiliary/io/dot_graph_file_reader.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/BspLocking.hpp"
#include "osp/dag_divider/IsomorphicWavefrontComponentScheduler.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/kl_include.hpp"

#include <filesystem>
#include <iostream>

using namespace osp;

BOOST_AUTO_TEST_CASE(BspScheduleRecomp_test)
{

    using graph_t = computational_dag_vector_impl_def_t;

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    BspInstance<graph_t> instance;
    file_reader::readComputationalDagDotFormat(".dot", instance.getComputationalDag());

    instance.getArchitecture().setProcessorsWithTypes({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 , 1, 1, 1, 1, 1, 1, 1, 1 , 1, 1, 1, 1, 1, 1, 1, 1 , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    instance.setDiagonalCompatibilityMatrix(2);
    //instance.setSynchronisationCosts(800000);

    BspLocking<graph_t> greedy;
    // kl_total_lambda_comm_improver<graph_t> kl;
    // ComboScheduler<graph_t> combo(greedy, kl);

    RecursiveWavefrontDivider<graph_t> wavefront;
    wavefront.use_threshold_scan_splitter(8.0, 8.0, 2);
    // ScanWavefrontDivider<graph_t> wavefront;
    //wavefront.set_algorithm(SplitAlgorithm::THRESHOLD_SCAN);
    //wavefront.set_threshold_scan_params(8.0, 8.0);

    IsomorphicWavefrontComponentScheduler<graph_t, graph_t> scheduler(wavefront, greedy);
  
    BspSchedule<graph_t> schedule(instance);

    scheduler.computeSchedule(schedule);

    BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
    BOOST_CHECK(schedule.satisfiesNodeTypeConstraints());

    // WavefrontMerkleDivider<graph_t> divider; 


    // auto maps = wavefront.divide(graph);

    // IsomorphismGroups<graph_t, graph_t> iso_groups;
    // iso_groups.compute_isomorphism_groups(maps, graph);

    
    // auto other_graph = iso_groups.get_isomorphism_groups_subgraphs()[1][0];

    // auto other_maps = wavefront.divide(other_graph);

    // IsomorphismGroups<graph_t, graph_t> other_iso_groups;
    // other_iso_groups.compute_isomorphism_groups(other_maps, other_graph);


};