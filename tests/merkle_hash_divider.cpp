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

#define BOOST_TEST_MODULE IsomorphicComponentDivider
#include <boost/test/unit_test.hpp>

#include "osp/dag_divider/wavefront_divider/ScanWavefrontDivider.hpp"
#include "osp/dag_divider/wavefront_divider/RecursiveWavefrontDivider.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/graph_implementations/adj_list_impl/dag_vector_adapter.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/auxiliary/io/dot_graph_file_reader.hpp"
#include "osp/auxiliary/io/DotFileWriter.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/BspLocking.hpp"
#include "osp/dag_divider/IsomorphicWavefrontComponentScheduler.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/kl_include_mt.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_total_comm.hpp"
#include "osp/coarser/coarser_util.hpp"
#include "osp/dag_divider/isomorphism_divider/EftSubgraphScheduler.hpp"
#include "osp/dag_divider/isomorphism_divider/IsomorphicSubgraphScheduler.hpp"

#include "test_utils.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(BspScheduleRecomp_test)
{
    using graph_t = computational_dag_vector_impl_def_t;
    using graph_t2 = graph_t;
    //using graph_t2 = dag_vector_adapter<cdag_vertex_impl_unsigned,int>;
   
    BspInstance<graph_t2> instance;
    file_reader::readComputationalDagDotFormat("", instance.getComputationalDag());

    for (const auto& v : instance.vertices()) {

        instance.getComputationalDag().set_vertex_comm_weight(v, instance.getComputationalDag().vertex_comm_weight(v) / 1064 + 1);
        instance.getComputationalDag().set_vertex_work_weight(v, instance.getComputationalDag().vertex_work_weight(v) / 1000 + 1);
    }

    instance.getArchitecture().setProcessorsWithTypes({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 , 1, 1, 1, 1, 1, 1, 1, 1 , 1, 1, 1, 1, 1, 1, 1, 1 , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    instance.setDiagonalCompatibilityMatrix(2);
    instance.setSynchronisationCosts(1000);
    instance.setCommunicationCosts(1);

    BspLocking<graph_t> greedy;
    kl_total_comm_improver_mt<graph_t> kl;
    ComboScheduler<graph_t> combo(greedy, kl);


    // // //RecursiveWavefrontDivider<graph_t> wavefront;
    // // //wavefront.use_threshold_scan_splitter(8.0, 8.0, 2);

    // wavefront.discover_isomorphic_groups(instance.getComputationalDag());

    // DotFileWriter writer;

    // writer.write_colored_graph("graph.dot", instance.getComputationalDag(), wavefront.get_vertex_color_map());

    // subgrah_scheduler_input<graph_t> input;
    // input.prepare_subgraph_scheduling_input(instance, wavefront.get_finalized_subgraphs(), wavefront.get_isomorphic_groups());

    // bool acyc = is_acyclic(input.instance.getComputationalDag());
    // std::cout << "Coarse dag is " << (acyc ? "acyclic." : "not acyclic.");
    
    // writer.write_graph("graph_2.dot", input.instance.getComputationalDag());

    // EftSubgraphScheduler<graph_t> pre_scheduler;
    
    // auto schedule = pre_scheduler.run(input.instance, input.multiplicities, input.required_proc_types);

    IsomorphicSubgraphScheduler<graph_t2, graph_t> iso_scheduler(combo);
    iso_scheduler.set_symmetry(2);
    iso_scheduler.set_plot_dot_graphs(true);

    auto partition = iso_scheduler.compute_partition(instance);

    graph_t corase_graph;
    coarser_util::construct_coarse_dag(instance.getComputationalDag(), corase_graph,
                                            partition);

    bool acyc = is_acyclic(corase_graph);

    DotFileWriter writer;

    writer.write_graph("graph.dot", corase_graph);

    std::cout << "Coarse dag is " << (acyc ? "acyclic." : "not acyclic.");
    BOOST_CHECK(acyc);



    // IsomorphicWavefrontComponentScheduler<graph_t, graph_t> scheduler(wavefront, combo);
  
    // BspSchedule<graph_t> schedule(instance);

    // scheduler.computeSchedule(schedule);

    // BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
    // BOOST_CHECK(schedule.satisfiesNodeTypeConstraints());

    // WavefrontMerkleDivider<graph_t> divider; 


    // auto maps = wavefront.divide(graph);

    // IsomorphismGroups<graph_t, graph_t> iso_groups;
    // iso_groups.compute_isomorphism_groups(maps, graph);

    
    // auto other_graph = iso_groups.get_isomorphism_groups_subgraphs()[1][0];

    // auto other_maps = wavefront.divide(other_graph);

    // IsomorphismGroups<graph_t, graph_t> other_iso_groups;
    // other_iso_groups.compute_isomorphism_groups(other_maps, other_graph);


};