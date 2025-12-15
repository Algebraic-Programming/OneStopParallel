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

#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/partitioning/model/hypergraph_utility.hpp"
#include "osp/partitioning/partitioners/partitioning_ILP.hpp"
#include "osp/partitioning/partitioners/partitioning_ILP_replication.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(TestFull) {
    using graph = ComputationalDagVectorImplDefIntT;
    using HypergraphImpl = HypergraphDefT;

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    graph dag;

    bool status = file_reader::ReadComputationalDagHyperdagFormatDB((cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(), DAG);

    BOOST_CHECK(status);

    HypergraphImpl hgraph = convert_from_cdag_as_hyperdag<HypergraphImpl, graph>(DAG);
    BOOST_CHECK_EQUAL(DAG.NumVertices(), Hgraph.NumVertices());

    PartitioningProblem instance(hgraph, 3, 35);
    Partitioning partition(instance);

    // ILP without replication

    HypergraphPartitioningILP<HypergraphImpl> partitioner;
    partitioner.SetTimeLimitSeconds(60);
    partitioner.ComputePartitioning(partition);

    BOOST_CHECK(partition.satisfiesBalanceConstraint());
    BOOST_CHECK(partition.computeConnectivityCost() >= partition.computeCutNetCost());

    for (unsigned node = 0; node < hgraph.NumVertices(); ++node) {
        partition.setAssignedPartition(node, node % 3);
    }

    partitioner.SetUseInitialSolution(true);
    partitioner.ComputePartitioning(partition);

    BOOST_CHECK(partition.satisfiesBalanceConstraint());
    int cutNetCost = partition.computeCutNetCost(), connectivityCost = partition.computeConnectivityCost();
    BOOST_CHECK(connectivityCost >= cutNetCost);

    instance.setMaxMemoryWeightExplicitly(37);
    partitioner.ComputePartitioning(partition);
    BOOST_CHECK(partition.satisfiesBalanceConstraint());
    BOOST_CHECK(cutNetCost == partition.computeCutNetCost());
    BOOST_CHECK(connectivityCost == partition.computeConnectivityCost());
    instance.setMaxMemoryWeightExplicitly(std::numeric_limits<int>::max());

    // ILP with replication

    HypergraphPartitioningILPWithReplication<HypergraphImpl> partitionerRep;
    PartitioningWithReplication partitionRep(instance);

    partitionerRep.SetTimeLimitSeconds(60);
    partitionerRep.ComputePartitioning(partition_rep);

    BOOST_CHECK(partitionRep.satisfiesBalanceConstraint());
    BOOST_CHECK(partitionRep.computeConnectivityCost() == 0);

    partitionerRep.SetUseInitialSolution(true);
    for (unsigned node = 0; node < hgraph.NumVertices(); ++node) {
        partitionRep.SetAssignedPartitions(node, {node % 3});
    }

    partitionerRep.ComputePartitioning(partition_rep);
    BOOST_CHECK(partitionRep.satisfiesBalanceConstraint());
    BOOST_CHECK(partitionRep.computeConnectivityCost() == 0);

    instance.setMaxWorkWeightExplicitly(60);
    for (unsigned node = 0; node < hgraph.NumVertices(); ++node) {
        partitionRep.SetAssignedPartitions(node, {node % 3, (node + 1) % 3});
    }

    partitionerRep.ComputePartitioning(partition_rep);
    BOOST_CHECK(partitionRep.satisfiesBalanceConstraint());
    BOOST_CHECK(partitionRep.computeConnectivityCost() == 0);

    // same tests with other replication formulation
    instance.setMaxWorkWeightExplicitly(35);
    partitioner_rep.SetReplicationModel(HypergraphPartitioningILPWithReplication<HypergraphImpl>::REPLICATION_MODEL_IN_ILP::GENERAL);
    partitionerRep.SetUseInitialSolution(false);
    partitionerRep.ComputePartitioning(partition_rep);

    BOOST_CHECK(partitionRep.satisfiesBalanceConstraint());
    BOOST_CHECK(partitionRep.computeConnectivityCost() == 0);

    partitionerRep.SetUseInitialSolution(true);
    for (unsigned node = 0; node < hgraph.NumVertices(); ++node) {
        partitionRep.SetAssignedPartitions(node, {node % 3});
    }

    partitionerRep.ComputePartitioning(partition_rep);
    BOOST_CHECK(partitionRep.satisfiesBalanceConstraint());
    BOOST_CHECK(partitionRep.computeConnectivityCost() == 0);

    instance.setMaxWorkWeightExplicitly(60);
    for (unsigned node = 0; node < hgraph.NumVertices(); ++node) {
        partitionRep.SetAssignedPartitions(node, {node % 3, (node + 1) % 3});
    }

    partitionerRep.ComputePartitioning(partition_rep);
    BOOST_CHECK(partitionRep.satisfiesBalanceConstraint());
    BOOST_CHECK(partitionRep.computeConnectivityCost() == 0);
};
