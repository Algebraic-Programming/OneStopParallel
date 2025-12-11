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
    using Graph = computational_dag_vector_impl_def_int_t;
    using Hypergraph = Hypergraph_def_t;

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    Graph dag;

    bool status = file_reader::readComputationalDagHyperdagFormatDB((cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(), dag);

    BOOST_CHECK(status);

    Hypergraph hgraph;

    // Matrix format, one hyperedge for each row/column
    status = file_reader::readHypergraphMartixMarketFormat((cwd / "data/mtx_tests/ErdosRenyi_8_19_A.mtx").string(), hgraph);
    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(hgraph.num_vertices(), 27);
    BOOST_CHECK_EQUAL(hgraph.num_hyperedges(), 16);

    // DAG format, all hyperedges have size 2
    hgraph = convert_from_cdag_as_dag<Hypergraph, Graph>(dag);
    BOOST_CHECK_EQUAL(dag.num_vertices(), hgraph.num_vertices());
    BOOST_CHECK_EQUAL(dag.num_edges(), hgraph.num_hyperedges());
    BOOST_CHECK_EQUAL(dag.num_edges() * 2, hgraph.num_pins());

    // HyperDAG format, one hypredge for each non-sink node
    unsigned nrOfNonSinks = 0;
    for (const auto &node : dag.vertices()) {
        if (dag.out_degree(node) > 0) {
            ++nrOfNonSinks;
        }
    }

    hgraph = convert_from_cdag_as_hyperdag<Hypergraph, Graph>(dag);
    BOOST_CHECK_EQUAL(dag.num_vertices(), hgraph.num_vertices());
    BOOST_CHECK_EQUAL(nrOfNonSinks, hgraph.num_hyperedges());
    BOOST_CHECK_EQUAL(dag.num_edges() + nrOfNonSinks, hgraph.num_pins());

    // Dummy partitioning

    PartitioningProblem instance(hgraph, 3, 30);

    Partitioning partition(instance);
    for (unsigned node = 0; node < hgraph.num_vertices(); ++node) {
        partition.setAssignedPartition(node, node % 3);
    }

    BOOST_CHECK(partition.satisfiesBalanceConstraint());
    int cutNetCost = partition.computeCutNetCost();
    int connectivityCost = partition.computeConnectivityCost();
    BOOST_CHECK(connectivityCost >= cutNetCost);

    for (unsigned node = 0; node < hgraph.num_vertices(); ++node) {
        instance.getHypergraph().set_vertex_work_weight(node, 1);
    }

    instance.setMaxWorkWeightViaImbalanceFactor(0);
    BOOST_CHECK(partition.satisfiesBalanceConstraint());

    instance.setNumberOfPartitions(5);
    instance.setMaxWorkWeightViaImbalanceFactor(0);
    BOOST_CHECK(!partition.satisfiesBalanceConstraint());

    for (unsigned node = 0; node < hgraph.num_vertices(); ++node) {
        partition.setAssignedPartition(node, node % 5);
    }

    BOOST_CHECK(partition.satisfiesBalanceConstraint());
    BOOST_CHECK(partition.computeConnectivityCost() >= partition.computeCutNetCost());

    for (unsigned node = 0; node < hgraph.num_vertices(); ++node) {
        instance.getHypergraph().set_vertex_memory_weight(node, 1);
    }
    instance.setMaxMemoryWeightExplicitly(10);
    BOOST_CHECK(partition.satisfiesBalanceConstraint() == false);
    instance.setMaxMemoryWeightExplicitly(std::numeric_limits<int>::max());

    file_writer::write_txt(std::cout, partition);

    // Dummy partitioning with replication

    instance.setHypergraph(convert_from_cdag_as_hyperdag<Hypergraph, Graph>(dag));
    instance.setNumberOfPartitions(3);
    instance.setMaxWorkWeightExplicitly(30);
    PartitioningWithReplication partitionWithRep(instance);
    for (unsigned node = 0; node < hgraph.num_vertices(); ++node) {
        partitionWithRep.setAssignedPartitions(node, {node % 3});
    }

    BOOST_CHECK(partitionWithRep.satisfiesBalanceConstraint());
    BOOST_CHECK(partitionWithRep.computeCutNetCost() == cutNetCost);
    BOOST_CHECK(partitionWithRep.computeConnectivityCost() == connectivityCost);

    instance.setMaxWorkWeightExplicitly(60);
    for (unsigned node = 0; node < hgraph.num_vertices(); ++node) {
        partitionWithRep.setAssignedPartitions(node, {node % 3, (node + 1) % 3});
    }

    BOOST_CHECK(partitionWithRep.satisfiesBalanceConstraint());
    BOOST_CHECK(partitionWithRep.computeConnectivityCost() >= partitionWithRep.computeCutNetCost());

    instance.setMaxWorkWeightExplicitly(compute_total_vertex_work_weight(hgraph));
    for (unsigned node = 0; node < hgraph.num_vertices(); ++node) {
        partitionWithRep.setAssignedPartitions(node, {0, 1, 2});
    }

    BOOST_CHECK(partitionWithRep.satisfiesBalanceConstraint());
    BOOST_CHECK(partitionWithRep.computeConnectivityCost() == 0);
    BOOST_CHECK(partitionWithRep.computeCutNetCost() == 0);

    file_writer::write_txt(std::cout, partitionWithRep);

    // Generic FM

    instance.setNumberOfPartitions(2);
    instance.setMaxWorkWeightExplicitly(35);
    for (unsigned node = 0; node < instance.getHypergraph().num_vertices(); ++node) {
        instance.getHypergraph().set_vertex_work_weight(node, 1);
    }

    Partitioning partitionToImprove(instance);
    for (unsigned node = 0; node < instance.getHypergraph().num_vertices(); ++node) {
        partitionToImprove.setAssignedPartition(node, node % 2);
    }

    int originalCost = partitionToImprove.computeConnectivityCost();

    GenericFM<Hypergraph> fm;
    fm.ImprovePartitioning(partitionToImprove);
    int newCost = partitionToImprove.computeConnectivityCost();

    BOOST_CHECK(partitionToImprove.satisfiesBalanceConstraint());
    BOOST_CHECK(newCost <= originalCost);
    std::cout << originalCost << " --> " << newCost << std::endl;

    Graph largerDag;
    file_reader::readComputationalDagHyperdagFormatDB((cwd / "data/spaa/large/instance_CG_N24_K22_nzP0d2.hdag").string(),
                                                      largerDag);
    instance.setHypergraph(convert_from_cdag_as_hyperdag<Hypergraph, Graph>(largerDag));

    instance.setMaxWorkWeightExplicitly(4000);
    for (unsigned node = 0; node < instance.getHypergraph().num_vertices(); ++node) {
        instance.getHypergraph().set_vertex_work_weight(node, 1);
    }

    partitionToImprove.resetPartition();
    for (unsigned node = 0; node < instance.getHypergraph().num_vertices(); ++node) {
        partitionToImprove.setAssignedPartition(node, node % 2);
    }

    originalCost = partitionToImprove.computeConnectivityCost();

    fm.setMaxNodesInPart(0);
    fm.ImprovePartitioning(partitionToImprove);
    newCost = partitionToImprove.computeConnectivityCost();

    BOOST_CHECK(partitionToImprove.satisfiesBalanceConstraint());
    BOOST_CHECK(newCost <= originalCost);
    std::cout << originalCost << " --> " << newCost << std::endl;

    // Recursive FM
    instance.setNumberOfPartitions(16);
    instance.setMaxWorkWeightViaImbalanceFactor(0.3);

    for (unsigned node = 0; node < instance.getHypergraph().num_vertices(); ++node) {
        partitionToImprove.setAssignedPartition(node, node % 16);
    }

    originalCost = partitionToImprove.computeConnectivityCost();

    fm.setMaxNodesInPart(0);
    fm.RecursiveFM(partitionToImprove);
    newCost = partitionToImprove.computeConnectivityCost();

    BOOST_CHECK(partitionToImprove.satisfiesBalanceConstraint());
    BOOST_CHECK(newCost <= originalCost);
    std::cout << originalCost << " --> " << newCost << std::endl;
}
