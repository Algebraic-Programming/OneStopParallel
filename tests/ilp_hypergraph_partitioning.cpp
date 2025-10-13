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

#define BOOST_TEST_MODULE HYPERGRAPH_PARTITIONING_ILP
#include <boost/test/unit_test.hpp>

#include <filesystem>
#include "osp/partitioning/partitioners/partitioning_ILP.hpp"
#include "osp/partitioning/partitioners/partitioning_ILP_replication.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"


using namespace osp;

BOOST_AUTO_TEST_CASE(test_full) {

    using graph = boost_graph_uint_t;

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    } 

    graph DAG;

    bool status = file_reader::readComputationalDagHyperdagFormatDB(
        (cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(), DAG);

    BOOST_CHECK(status);

    Hypergraph Hgraph;

    Hgraph.convert_from_cdag_as_hyperdag(DAG);
    BOOST_CHECK_EQUAL(DAG.num_vertices(), Hgraph.num_vertices());

    PartitioningProblem instance(Hgraph, 3, 35);
    Partitioning partition(instance);


    // ILP without replication
    
    HypergraphPartitioningILP partitioner;
    partitioner.setTimeLimitSeconds(60);
    partitioner.computePartitioning(partition);

    BOOST_CHECK(partition.satisfiesBalanceConstraint());
    BOOST_CHECK(partition.computeConnectivityCost() >= partition.computeCutNetCost());

    for(unsigned node = 0; node < Hgraph.num_vertices(); ++node)
        partition.setAssignedPartition(node, node % 3);

    partitioner.setUseInitialSolution(true);
    partitioner.computePartitioning(partition);

    BOOST_CHECK(partition.satisfiesBalanceConstraint());
    int cutNetCost = partition.computeCutNetCost(), connectivityCost = partition.computeConnectivityCost();
    BOOST_CHECK(connectivityCost >= cutNetCost);

    instance.setMaxMemoryWeightExplicitly(37);
    partitioner.computePartitioning(partition);
    BOOST_CHECK(partition.satisfiesBalanceConstraint());
    BOOST_CHECK(cutNetCost == partition.computeCutNetCost());
    BOOST_CHECK(connectivityCost == partition.computeConnectivityCost());
    instance.setMaxMemoryWeightExplicitly(std::numeric_limits<int>::max());

    // ILP with replication

    HypergraphPartitioningILPWithReplication partitioner_rep;
    PartitioningWithReplication partition_rep(instance);

    partitioner_rep.setTimeLimitSeconds(60);
    partitioner_rep.computePartitioning(partition_rep);

    BOOST_CHECK(partition_rep.satisfiesBalanceConstraint());
    BOOST_CHECK(partition_rep.computeConnectivityCost() == 0);

    partitioner_rep.setUseInitialSolution(true);
    for(unsigned node = 0; node < Hgraph.num_vertices(); ++node)
        partition_rep.setAssignedPartitions(node, {node % 3});

    partitioner_rep.computePartitioning(partition_rep);
    BOOST_CHECK(partition_rep.satisfiesBalanceConstraint());
    BOOST_CHECK(partition_rep.computeConnectivityCost() == 0);

    instance.setMaxWorkWeightExplicitly(60);
    for(unsigned node = 0; node < Hgraph.num_vertices(); ++node)
        partition_rep.setAssignedPartitions(node, {node % 3, (node+1)%3});

    partitioner_rep.computePartitioning(partition_rep);
    BOOST_CHECK(partition_rep.satisfiesBalanceConstraint());
    BOOST_CHECK(partition_rep.computeConnectivityCost() == 0);

    // same tests with other replication formulation
    instance.setMaxWorkWeightExplicitly(35);
    partitioner_rep.setReplicationModel(HypergraphPartitioningILPWithReplication::REPLICATION_MODEL_IN_ILP::GENERAL);
    partitioner_rep.setUseInitialSolution(false);
    partitioner_rep.computePartitioning(partition_rep);

    BOOST_CHECK(partition_rep.satisfiesBalanceConstraint());
    BOOST_CHECK(partition_rep.computeConnectivityCost() == 0);

    partitioner_rep.setUseInitialSolution(true);
    for(unsigned node = 0; node < Hgraph.num_vertices(); ++node)
        partition_rep.setAssignedPartitions(node, {node % 3});

    partitioner_rep.computePartitioning(partition_rep);
    BOOST_CHECK(partition_rep.satisfiesBalanceConstraint());
    BOOST_CHECK(partition_rep.computeConnectivityCost() == 0);

    instance.setMaxWorkWeightExplicitly(60);
    for(unsigned node = 0; node < Hgraph.num_vertices(); ++node)
        partition_rep.setAssignedPartitions(node, {node % 3, (node+1)%3});

    partitioner_rep.computePartitioning(partition_rep);
    BOOST_CHECK(partition_rep.satisfiesBalanceConstraint());
    BOOST_CHECK(partition_rep.computeConnectivityCost() == 0);

};