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

BOOST_AUTO_TEST_CASE(Hypergraph_and_Partition_test) {
    using graph = computational_dag_vector_impl_def_int_t;
    using hypergraph = Hypergraph_def_t;

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    graph DAG;

    bool status = file_reader::readComputationalDagHyperdagFormatDB((cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(), DAG);

    BOOST_CHECK(status);

    hypergraph Hgraph;

    // Matrix format, one hyperedge for each row/column
    status = file_reader::readHypergraphMartixMarketFormat((cwd / "data/mtx_tests/ErdosRenyi_8_19_A.mtx").string(), Hgraph);
    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(Hgraph.num_vertices(), 27);
    BOOST_CHECK_EQUAL(Hgraph.num_hyperedges(), 16);

    // DAG format, all hyperedges have size 2
    Hgraph = convert_from_cdag_as_dag<hypergraph, graph>(DAG);
    BOOST_CHECK_EQUAL(DAG.num_vertices(), Hgraph.num_vertices());
    BOOST_CHECK_EQUAL(DAG.num_edges(), Hgraph.num_hyperedges());
    BOOST_CHECK_EQUAL(DAG.num_edges() * 2, Hgraph.num_pins());

    // HyperDAG format, one hypredge for each non-sink node
    unsigned nr_of_non_sinks = 0;
    for (const auto &node : DAG.vertices()) {
        if (DAG.out_degree(node) > 0) { ++nr_of_non_sinks; }
    }

    Hgraph = convert_from_cdag_as_hyperdag<hypergraph, graph>(DAG);
    BOOST_CHECK_EQUAL(DAG.num_vertices(), Hgraph.num_vertices());
    BOOST_CHECK_EQUAL(nr_of_non_sinks, Hgraph.num_hyperedges());
    BOOST_CHECK_EQUAL(DAG.num_edges() + nr_of_non_sinks, Hgraph.num_pins());

    // Dummy partitioning

    PartitioningProblem instance(Hgraph, 3, 30);

    Partitioning partition(instance);
    for (unsigned node = 0; node < Hgraph.num_vertices(); ++node) { partition.setAssignedPartition(node, node % 3); }

    BOOST_CHECK(partition.satisfiesBalanceConstraint());
    int cutNetCost = partition.computeCutNetCost();
    int connectivityCost = partition.computeConnectivityCost();
    BOOST_CHECK(connectivityCost >= cutNetCost);

    for (unsigned node = 0; node < Hgraph.num_vertices(); ++node) { instance.getHypergraph().set_vertex_work_weight(node, 1); }

    instance.setMaxWorkWeightViaImbalanceFactor(0);
    BOOST_CHECK(partition.satisfiesBalanceConstraint());

    instance.setNumberOfPartitions(5);
    instance.setMaxWorkWeightViaImbalanceFactor(0);
    BOOST_CHECK(!partition.satisfiesBalanceConstraint());

    for (unsigned node = 0; node < Hgraph.num_vertices(); ++node) { partition.setAssignedPartition(node, node % 5); }

    BOOST_CHECK(partition.satisfiesBalanceConstraint());
    BOOST_CHECK(partition.computeConnectivityCost() >= partition.computeCutNetCost());

    for (unsigned node = 0; node < Hgraph.num_vertices(); ++node) { instance.getHypergraph().set_vertex_memory_weight(node, 1); }
    instance.setMaxMemoryWeightExplicitly(10);
    BOOST_CHECK(partition.satisfiesBalanceConstraint() == false);
    instance.setMaxMemoryWeightExplicitly(std::numeric_limits<int>::max());

    file_writer::write_txt(std::cout, partition);

    // Dummy partitioning with replication

    instance.setHypergraph(convert_from_cdag_as_hyperdag<hypergraph, graph>(DAG));
    instance.setNumberOfPartitions(3);
    instance.setMaxWorkWeightExplicitly(30);
    PartitioningWithReplication partition_with_rep(instance);
    for (unsigned node = 0; node < Hgraph.num_vertices(); ++node) { partition_with_rep.setAssignedPartitions(node, {node % 3}); }

    BOOST_CHECK(partition_with_rep.satisfiesBalanceConstraint());
    BOOST_CHECK(partition_with_rep.computeCutNetCost() == cutNetCost);
    BOOST_CHECK(partition_with_rep.computeConnectivityCost() == connectivityCost);

    instance.setMaxWorkWeightExplicitly(60);
    for (unsigned node = 0; node < Hgraph.num_vertices(); ++node) {
        partition_with_rep.setAssignedPartitions(node, {node % 3, (node + 1) % 3});
    }

    BOOST_CHECK(partition_with_rep.satisfiesBalanceConstraint());
    BOOST_CHECK(partition_with_rep.computeConnectivityCost() >= partition_with_rep.computeCutNetCost());

    instance.setMaxWorkWeightExplicitly(compute_total_vertex_work_weight(Hgraph));
    for (unsigned node = 0; node < Hgraph.num_vertices(); ++node) { partition_with_rep.setAssignedPartitions(node, {0, 1, 2}); }

    BOOST_CHECK(partition_with_rep.satisfiesBalanceConstraint());
    BOOST_CHECK(partition_with_rep.computeConnectivityCost() == 0);
    BOOST_CHECK(partition_with_rep.computeCutNetCost() == 0);

    file_writer::write_txt(std::cout, partition_with_rep);

    // Generic FM

    instance.setNumberOfPartitions(2);
    instance.setMaxWorkWeightExplicitly(35);
    for (unsigned node = 0; node < instance.getHypergraph().num_vertices(); ++node) {
        instance.getHypergraph().set_vertex_work_weight(node, 1);
    }

    Partitioning partition_to_improve(instance);
    for (unsigned node = 0; node < instance.getHypergraph().num_vertices(); ++node) {
        partition_to_improve.setAssignedPartition(node, node % 2);
    }

    int original_cost = partition_to_improve.computeConnectivityCost();

    GenericFM<hypergraph> fm;
    fm.ImprovePartitioning(partition_to_improve);
    int new_cost = partition_to_improve.computeConnectivityCost();

    BOOST_CHECK(partition_to_improve.satisfiesBalanceConstraint());
    BOOST_CHECK(new_cost <= original_cost);
    std::cout << original_cost << " --> " << new_cost << std::endl;

    graph larger_DAG;
    file_reader::readComputationalDagHyperdagFormatDB((cwd / "data/spaa/large/instance_CG_N24_K22_nzP0d2.hdag").string(),
                                                      larger_DAG);
    instance.setHypergraph(convert_from_cdag_as_hyperdag<hypergraph, graph>(larger_DAG));

    instance.setMaxWorkWeightExplicitly(4000);
    for (unsigned node = 0; node < instance.getHypergraph().num_vertices(); ++node) {
        instance.getHypergraph().set_vertex_work_weight(node, 1);
    }

    partition_to_improve.resetPartition();
    for (unsigned node = 0; node < instance.getHypergraph().num_vertices(); ++node) {
        partition_to_improve.setAssignedPartition(node, node % 2);
    }

    original_cost = partition_to_improve.computeConnectivityCost();

    fm.setMaxNodesInPart(0);
    fm.ImprovePartitioning(partition_to_improve);
    new_cost = partition_to_improve.computeConnectivityCost();

    BOOST_CHECK(partition_to_improve.satisfiesBalanceConstraint());
    BOOST_CHECK(new_cost <= original_cost);
    std::cout << original_cost << " --> " << new_cost << std::endl;

    // Recursive FM
    instance.setNumberOfPartitions(16);
    instance.setMaxWorkWeightViaImbalanceFactor(0.3);

    for (unsigned node = 0; node < instance.getHypergraph().num_vertices(); ++node) {
        partition_to_improve.setAssignedPartition(node, node % 16);
    }

    original_cost = partition_to_improve.computeConnectivityCost();

    fm.setMaxNodesInPart(0);
    fm.RecursiveFM(partition_to_improve);
    new_cost = partition_to_improve.computeConnectivityCost();

    BOOST_CHECK(partition_to_improve.satisfiesBalanceConstraint());
    BOOST_CHECK(new_cost <= original_cost);
    std::cout << original_cost << " --> " << new_cost << std::endl;
}
