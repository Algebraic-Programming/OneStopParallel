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

#define BOOST_TEST_MODULE HYPERGRAPH_AND_PARTITION
#include <boost/test/unit_test.hpp>
#include <filesystem>
#include <string>
#include <vector>

#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/auxiliary/io/mtx_hypergraph_file_reader.hpp"
#include "osp/auxiliary/io/partitioning_file_writer.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/partitioning/model/hypergraph_utility.hpp"
#include "osp/partitioning/model/partitioning.hpp"
#include "osp/partitioning/model/partitioning_replication.hpp"
#include "osp/partitioning/partitioners/generic_FM.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(HypergraphAndPartitionTest) {
    using Graph = ComputationalDagVectorImplDefIntT;
    using HypergraphImpl = HypergraphDefT;

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    Graph dag;

    bool status = file_reader::ReadComputationalDagHyperdagFormatDB((cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(), dag);

    BOOST_CHECK(status);

    HypergraphImpl hgraph;

    // Matrix format, one hyperedge for each row/column
    status = file_reader::ReadHypergraphMartixMarketFormat((cwd / "data/mtx_tests/ErdosRenyi_8_19_A.mtx").string(), hgraph);
    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(hgraph.NumVertices(), 27);
    BOOST_CHECK_EQUAL(hgraph.NumHyperedges(), 16);

    // DAG format, all hyperedges have size 2
    hgraph = ConvertFromCdagAsDag<HypergraphImpl, Graph>(dag);
    BOOST_CHECK_EQUAL(dag.NumVertices(), hgraph.NumVertices());
    BOOST_CHECK_EQUAL(dag.NumEdges(), hgraph.NumHyperedges());
    BOOST_CHECK_EQUAL(dag.NumEdges() * 2, hgraph.NumPins());

    // HyperDAG format, one hypredge for each non-sink node
    unsigned nrOfNonSinks = 0;
    for (const auto &node : dag.Vertices()) {
        if (dag.OutDegree(node) > 0) {
            ++nrOfNonSinks;
        }
    }

    hgraph = convert_from_cdag_as_hyperdag<HypergraphImpl, Graph>(dag);
    BOOST_CHECK_EQUAL(dag.NumVertices(), hgraph.NumVertices());
    BOOST_CHECK_EQUAL(nrOfNonSinks, hgraph.NumHyperedges());
    BOOST_CHECK_EQUAL(dag.NumEdges() + nrOfNonSinks, hgraph.NumPins());

    // Dummy partitioning

    PartitioningProblem instance(hgraph, 3, 30);

    Partitioning partition(instance);
    for (unsigned node = 0; node < hgraph.NumVertices(); ++node) {
        partition.SetAssignedPartition(node, node % 3);
    }

    BOOST_CHECK(partition.SatisfiesBalanceConstraint());
    int cutNetCost = partition.ComputeCutNetCost();
    int connectivityCost = partition.ComputeConnectivityCost();
    BOOST_CHECK(connectivityCost >= cutNetCost);

    for (unsigned node = 0; node < hgraph.NumVertices(); ++node) {
        instance.GetHypergraph().SetVertexWorkWeight(node, 1);
    }

    instance.SetMaxWorkWeightViaImbalanceFactor(0);
    BOOST_CHECK(partition.SatisfiesBalanceConstraint());

    instance.SetNumberOfPartitions(5);
    instance.SetMaxWorkWeightViaImbalanceFactor(0);
    BOOST_CHECK(!partition.SatisfiesBalanceConstraint());

    for (unsigned node = 0; node < hgraph.NumVertices(); ++node) {
        partition.SetAssignedPartition(node, node % 5);
    }

    BOOST_CHECK(partition.SatisfiesBalanceConstraint());
    BOOST_CHECK(partition.ComputeConnectivityCost() >= partition.ComputeCutNetCost());

    for (unsigned node = 0; node < hgraph.NumVertices(); ++node) {
        instance.GetHypergraph().SetVertexMemoryWeight(node, 1);
    }
    instance.SetMaxMemoryWeightExplicitly(10);
    BOOST_CHECK(partition.SatisfiesBalanceConstraint() == false);
    instance.SetMaxMemoryWeightExplicitly(std::numeric_limits<int>::max());

    file_writer::WriteTxt(std::cout, partition);

    // Dummy partitioning with replication

    instance.setHypergraph(convert_from_cdag_as_hyperdag<HypergraphImpl, Graph>(dag));
    instance.SetNumberOfPartitions(3);
    instance.SetMaxWorkWeightExplicitly(30);
    PartitioningWithReplication partitionWithRep(instance);
    for (unsigned node = 0; node < hgraph.NumVertices(); ++node) {
        partitionWithRep.SetAssignedPartitions(node, {node % 3});
    }

    BOOST_CHECK(partitionWithRep.SatisfiesBalanceConstraint());
    BOOST_CHECK(partitionWithRep.ComputeCutNetCost() == cutNetCost);
    BOOST_CHECK(partitionWithRep.ComputeConnectivityCost() == connectivityCost);

    instance.SetMaxWorkWeightExplicitly(60);
    for (unsigned node = 0; node < hgraph.NumVertices(); ++node) {
        partitionWithRep.SetAssignedPartitions(node, {node % 3, (node + 1) % 3});
    }

    BOOST_CHECK(partitionWithRep.SatisfiesBalanceConstraint());
    BOOST_CHECK(partitionWithRep.ComputeConnectivityCost() >= partitionWithRep.ComputeCutNetCost());

    instance.SetMaxWorkWeightExplicitly(ComputeTotalVertexWorkWeight(hgraph));
    for (unsigned node = 0; node < hgraph.NumVertices(); ++node) {
        partitionWithRep.SetAssignedPartitions(node, {0, 1, 2});
    }

    BOOST_CHECK(partitionWithRep.SatisfiesBalanceConstraint());
    BOOST_CHECK(partitionWithRep.ComputeConnectivityCost() == 0);
    BOOST_CHECK(partitionWithRep.ComputeCutNetCost() == 0);

    file_writer::WriteTxt(std::cout, partitionWithRep);

    // Generic FM

    instance.SetNumberOfPartitions(2);
    instance.SetMaxWorkWeightExplicitly(35);
    for (unsigned node = 0; node < instance.GetHypergraph().NumVertices(); ++node) {
        instance.GetHypergraph().SetVertexWorkWeight(node, 1);
    }

    Partitioning partitionToImprove(instance);
    for (unsigned node = 0; node < instance.GetHypergraph().NumVertices(); ++node) {
        partitionToImprove.SetAssignedPartition(node, node % 2);
    }

    int originalCost = partitionToImprove.ComputeConnectivityCost();

    GenericFM<HypergraphImpl> fm;
    fm.ImprovePartitioning(partitionToImprove);
    int newCost = partitionToImprove.ComputeConnectivityCost();

    BOOST_CHECK(partitionToImprove.SatisfiesBalanceConstraint());
    BOOST_CHECK(newCost <= originalCost);
    std::cout << originalCost << " --> " << newCost << std::endl;

    Graph largerDag;
    file_reader::ReadComputationalDagHyperdagFormatDB((cwd / "data/spaa/large/instance_CG_N24_K22_nzP0d2.hdag").string(),
                                                      largerDag);
    instance.setHypergraph(convert_from_cdag_as_hyperdag<HypergraphImpl, Graph>(largerDag));

    instance.SetMaxWorkWeightExplicitly(4000);
    for (unsigned node = 0; node < instance.GetHypergraph().NumVertices(); ++node) {
        instance.GetHypergraph().SetVertexWorkWeight(node, 1);
    }

    partitionToImprove.ResetPartition();
    for (unsigned node = 0; node < instance.GetHypergraph().NumVertices(); ++node) {
        partitionToImprove.SetAssignedPartition(node, node % 2);
    }

    originalCost = partitionToImprove.ComputeConnectivityCost();

    fm.SetMaxNodesInPart(0);
    fm.ImprovePartitioning(partitionToImprove);
    newCost = partitionToImprove.ComputeConnectivityCost();

    BOOST_CHECK(partitionToImprove.SatisfiesBalanceConstraint());
    BOOST_CHECK(newCost <= originalCost);
    std::cout << originalCost << " --> " << newCost << std::endl;

    // Recursive FM
    instance.SetNumberOfPartitions(16);
    instance.SetMaxWorkWeightViaImbalanceFactor(0.3);

    for (unsigned node = 0; node < instance.GetHypergraph().NumVertices(); ++node) {
        partitionToImprove.SetAssignedPartition(node, node % 16);
    }

    originalCost = partitionToImprove.ComputeConnectivityCost();

    fm.SetMaxNodesInPart(0);
    fm.RecursiveFM(partitionToImprove);
    newCost = partitionToImprove.ComputeConnectivityCost();

    BOOST_CHECK(partitionToImprove.SatisfiesBalanceConstraint());
    BOOST_CHECK(newCost <= originalCost);
    std::cout << originalCost << " --> " << newCost << std::endl;
}
