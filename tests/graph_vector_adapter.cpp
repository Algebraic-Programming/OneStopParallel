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

#define BOOST_TEST_MODULE ApproxEdgeReduction

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <vector>

#include "osp/graph_algorithms/directed_graph_util.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/graph_implementations/adj_list_impl/dag_vector_adapter.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"
#include "osp/graph_implementations/adj_list_impl/compact_sparse_graph.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/BspLocking.hpp"
#include "osp/bsp/scheduler/Serial.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyMetaScheduler.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyChildren.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GrowLocalAutoCores.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/kl_include_mt.hpp"
#include "osp/coarser/coarser_util.hpp"
#include "osp/dag_divider/isomorphism_divider/IsomorphicSubgraphScheduler.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/coarser/Sarkar/Sarkar.hpp"
#include "osp/coarser/Sarkar/SarkarMul.hpp"

using namespace osp;


BOOST_AUTO_TEST_CASE(test_dag_vector_adapter) {

    std::vector<std::vector<int>> out_neighbors{{1, 2, 3}, {4, 6}, {4, 5}, {7}, {7}, {}, {}, {}};

    std::vector<std::vector<int>> in_neighbors{{}, {0}, {0}, {0}, {1, 2}, {2}, {1}, {4, 3}};

    using v_impl = cdag_vertex_impl<unsigned, int, int, int, unsigned>;
    using graph_t = dag_vector_adapter<v_impl,int>;
    using graph_constr_t = computational_dag_vector_impl<v_impl>;
    using CoarseGraphType = Compact_Sparse_Graph<true, true, true, true, true, vertex_idx_t<graph_t>, std::size_t, v_workw_t<graph_t>, v_workw_t<graph_t>, v_workw_t<graph_t>, v_type_t<graph_t>>;

    graph_t graph(out_neighbors, in_neighbors);
    
    for (auto v : graph.vertices()) {
        graph.set_vertex_work_weight(v, 10);
    }

    BspInstance<graph_t> instance;
    instance.getComputationalDag() = graph;

    instance.getArchitecture().setProcessorsWithTypes({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 , 1, 1, 1, 1, 1, 1, 1, 1 , 1, 1, 1, 1, 1, 1, 1, 1 , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    instance.setDiagonalCompatibilityMatrix(2);
    instance.setSynchronisationCosts(1000);
    instance.setCommunicationCosts(1);



    // Set up the scheduler
    GrowLocalAutoCores<graph_constr_t> growlocal;
    BspLocking<graph_constr_t> locking;
    GreedyChildren<graph_constr_t> children;
    kl_total_lambda_comm_improver<graph_constr_t> kl(42);
    kl.setSuperstepRemoveStrengthParameter(2.0);
    kl.setTimeQualityParameter(5.0);
    ComboScheduler<graph_constr_t> growlocal_kl(growlocal, kl);
    ComboScheduler<graph_constr_t> locking_kl(locking, kl);
    ComboScheduler<graph_constr_t> children_kl(children, kl);

    GreedyMetaScheduler<graph_constr_t> scheduler;
   // scheduler.addScheduler(growlocal_kl);
    scheduler.addScheduler(locking_kl);
    scheduler.addScheduler(children_kl);
    scheduler.addSerialScheduler();

    IsomorphicSubgraphScheduler<graph_t, graph_constr_t> iso_scheduler(scheduler);

    auto partition = iso_scheduler.compute_partition(instance);

    graph_constr_t corase_graph;
    coarser_util::construct_coarse_dag(instance.getComputationalDag(), corase_graph, partition);
    bool acyc = is_acyclic(corase_graph);
    BOOST_CHECK(acyc);

    SarkarMul<graph_t, CoarseGraphType> coarser;

    CoarseGraphType coarse_dag;
    std::vector<unsigned> reverse_vertex_map;
    coarser.coarsenDag(graph, coarse_dag, reverse_vertex_map);

    acyc = is_acyclic(coarse_dag);
    BOOST_CHECK(acyc);
}