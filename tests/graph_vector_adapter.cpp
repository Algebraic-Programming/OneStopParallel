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

#include "osp/bsp/scheduler/GreedySchedulers/BspLocking.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyChildren.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyMetaScheduler.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GrowLocalAutoCores.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_include_mt.hpp"
#include "osp/bsp/scheduler/Serial.hpp"
#include "osp/coarser/Sarkar/Sarkar.hpp"
#include "osp/coarser/Sarkar/SarkarMul.hpp"
#include "osp/coarser/coarser_util.hpp"
#include "osp/dag_divider/isomorphism_divider/IsomorphicSubgraphScheduler.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"
#include "osp/graph_implementations/adj_list_impl/compact_sparse_graph.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/graph_implementations/adj_list_impl/dag_vector_adapter.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(TestDagVectorAdapterEdge) {
    std::vector<std::vector<int>> outNeighbors{
        {1, 2, 3},
        {4, 6},
        {4, 5},
        {7},
        {7},
        {},
        {},
        {}
    };

    std::vector<std::vector<int>> inNeighbors{
        {},
        {0},
        {0},
        {0},
        {1, 2},
        {2},
        {1},
        {4, 3}
    };

    using VImpl = CDagVertexImpl<unsigned, int, int, int, unsigned>;
    using GraphT = DagVectorAdapter<VImpl, int>;
    using GraphConstrT = ComputationalDagEdgeIdxVectorImpl<VImpl, CDagEdgeImplInt>;
    using CoarseGraphType = CompactSparseGraph<true,
                                               true,
                                               true,
                                               true,
                                               true,
                                               VertexIdxT<GraphT>,
                                               std::size_t,
                                               VWorkwT<GraphT>,
                                               VWorkwT<GraphT>,
                                               VWorkwT<GraphT>,
                                               VTypeT<GraphT>>;

    GraphT graph(outNeighbors, inNeighbors);

    for (auto v : graph.Vertices()) {
        graph.SetVertexWorkWeight(v, 10);
    }

    BspInstance<GraphT> instance;
    instance.GetComputationalDag() = graph;

    instance.GetArchitecture().SetProcessorsWithTypes({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    instance.SetDiagonalCompatibilityMatrix(2);
    instance.SetSynchronisationCosts(1000);
    instance.SetCommunicationCosts(1);

    // Set up the scheduler
    GrowLocalAutoCores<GraphConstrT> growlocal;
    BspLocking<GraphConstrT> locking;
    GreedyChildren<GraphConstrT> children;
    KlTotalLambdaCommImprover<GraphConstrT> kl(42);
    kl.SetSuperstepRemoveStrengthParameter(2.0);
    kl.SetTimeQualityParameter(5.0);
    ComboScheduler<GraphConstrT> growlocalKl(growlocal, kl);
    ComboScheduler<GraphConstrT> lockingKl(locking, kl);
    ComboScheduler<GraphConstrT> childrenKl(children, kl);

    GreedyMetaScheduler<GraphConstrT> scheduler;
    scheduler.AddScheduler(lockingKl);
    scheduler.AddScheduler(childrenKl);
    scheduler.AddSerialScheduler();

    IsomorphicSubgraphScheduler<GraphT, GraphConstrT> isoScheduler(scheduler);

    auto partition = isoScheduler.ComputePartition(instance);

    GraphConstrT coraseGraph;
    coarser_util::ConstructCoarseDag(instance.GetComputationalDag(), coraseGraph, partition);
    bool acyc = IsAcyclic(coraseGraph);
    BOOST_CHECK(acyc);

    SarkarMul<GraphT, CoarseGraphType> coarser;

    CoarseGraphType coarseDag;
    std::vector<unsigned> reverseVertexMap;
    coarser.CoarsenDag(graph, coarseDag, reverseVertexMap);

    acyc = IsAcyclic(coarseDag);
    BOOST_CHECK(acyc);
}

BOOST_AUTO_TEST_CASE(TestDagVectorAdapter) {
    std::vector<std::vector<int>> outNeighbors{
        {1, 2, 3},
        {4, 6},
        {4, 5},
        {7},
        {7},
        {},
        {},
        {}
    };

    std::vector<std::vector<int>> inNeighbors{
        {},
        {0},
        {0},
        {0},
        {1, 2},
        {2},
        {1},
        {4, 3}
    };

    using VImpl = CDagVertexImpl<unsigned, int, int, int, unsigned>;
    using GraphT = DagVectorAdapter<VImpl, int>;
    using GraphConstrT = computational_dag_vector_impl<VImpl>;
    using CoarseGraphType = CompactSparseGraph<true,
                                               true,
                                               true,
                                               true,
                                               true,
                                               VertexIdxT<GraphT>,
                                               std::size_t,
                                               VWorkwT<GraphT>,
                                               VWorkwT<GraphT>,
                                               VWorkwT<GraphT>,
                                               VTypeT<GraphT>>;

    GraphT graph(outNeighbors, inNeighbors);

    for (auto v : graph.Vertices()) {
        graph.SetVertexWorkWeight(v, 10);
    }

    BspInstance<GraphT> instance;
    instance.GetComputationalDag() = graph;

    instance.GetArchitecture().SetProcessorsWithTypes({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    instance.SetDiagonalCompatibilityMatrix(2);
    instance.SetSynchronisationCosts(1000);
    instance.SetCommunicationCosts(1);

    // Set up the scheduler
    GrowLocalAutoCores<GraphConstrT> growlocal;
    BspLocking<GraphConstrT> locking;
    GreedyChildren<GraphConstrT> children;
    KlTotalLambdaCommImprover<GraphConstrT> kl(42);
    kl.SetSuperstepRemoveStrengthParameter(2.0);
    kl.SetTimeQualityParameter(5.0);
    ComboScheduler<GraphConstrT> growlocalKl(growlocal, kl);
    ComboScheduler<GraphConstrT> lockingKl(locking, kl);
    ComboScheduler<GraphConstrT> childrenKl(children, kl);

    GreedyMetaScheduler<GraphConstrT> scheduler;
    scheduler.AddScheduler(lockingKl);
    scheduler.AddScheduler(childrenKl);
    scheduler.AddSerialScheduler();

    IsomorphicSubgraphScheduler<GraphT, GraphConstrT> isoScheduler(scheduler);

    auto partition = isoScheduler.ComputePartition(instance);

    GraphConstrT coraseGraph;
    coarser_util::ConstructCoarseDag(instance.GetComputationalDag(), coraseGraph, partition);
    bool acyc = IsAcyclic(coraseGraph);
    BOOST_CHECK(acyc);

    SarkarMul<GraphT, CoarseGraphType> coarser;

    CoarseGraphType coarseDag;
    std::vector<unsigned> reverseVertexMap;
    coarser.CoarsenDag(graph, coarseDag, reverseVertexMap);

    acyc = IsAcyclic(coarseDag);
    BOOST_CHECK(acyc);
}
