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

#include <iostream>

#include "osp/auxiliary/io/DotFileWriter.hpp"
#include "osp/auxiliary/io/dot_graph_file_reader.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/BspLocking.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyChildren.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyMetaScheduler.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GrowLocalAutoCores.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_include_mt.hpp"
#include "osp/bsp/scheduler/Serial.hpp"
#include "osp/coarser/coarser_util.hpp"
#include "osp/dag_divider/isomorphism_divider/IsomorphicSubgraphScheduler.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"

using namespace osp;

template <typename GraphT>
void CheckPartitionTypeHomogeneity(const GraphT &dag, const std::vector<VertexIdxT<GraphT>> &partition) {
    // Group partitions by their ID
    std::map<VertexIdxT<GraphT>, std::vector<VertexIdxT<GraphT>>> partitions;
    for (VertexIdxT<GraphT> i = 0; i < dag.NumVertices(); ++i) {
        partitions[partition[i]].push_back(i);
    }

    // For each partition, check that all vertices have the same type
    for (const auto &[part_id, vertices] : partitions) {
        if (vertices.empty()) {
            continue;
        }
        const auto firstNodeType = dag.VertexType(vertices[0]);
        for (const auto &vertex : vertices) {
            if (dag.VertexType(vertex) != firstNodeType) {
                std::cerr << "Partition " << part_id << " contains vertices with different types." << std::endl;
                return;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_dot_file>" << std::endl;
        return 1;
    }

    std::string dotFilePath = argv[1];

    using GraphT = computational_dag_vector_impl_def_t;
    using GraphT2 = GraphT;

    BspInstance<GraphT2> instance;
    if (!file_reader::readComputationalDagDotFormat(dotFilePath, instance.GetComputationalDag())) {
        std::cerr << "Failed to read graph from " << dotFilePath << std::endl;
        return 1;
    }

    std::cout << "Graph loaded successfully. " << instance.NumberOfVertices() << " vertices." << std::endl;

    for (auto v : instance.GetComputationalDag().Vertices()) {
        instance.GetComputationalDag().SetVertexCommWeight(
            v, static_cast<VCommwT<GraphT2>>(instance.GetComputationalDag().VertexCommWeight(v) * 0.01));
    }

    // Set up architecture
    instance.GetArchitecture().SetProcessorsConsequTypes({24, 48}, {100, 100});
    instance.setDiagonalCompatibilityMatrix(2);
    instance.setSynchronisationCosts(2000);
    instance.setCommunicationCosts(1);

    // Set up the scheduler
    GrowLocalAutoCores<GraphT> growlocal;
    BspLocking<GraphT> locking;
    GreedyChildren<GraphT> children;
    kl_total_lambda_comm_improver<GraphT> kl(42);
    kl.setSuperstepRemoveStrengthParameter(1.0);
    kl.setTimeQualityParameter(1.0);
    ComboScheduler<GraphT> growlocalKl(growlocal, kl);
    ComboScheduler<GraphT> lockingKl(locking, kl);
    ComboScheduler<GraphT> childrenKl(children, kl);

    GreedyMetaScheduler<GraphT> scheduler;
    // scheduler.addScheduler(growlocal_kl);
    scheduler.addScheduler(lockingKl);
    scheduler.addScheduler(childrenKl);
    scheduler.addSerialScheduler();

    IsomorphicSubgraphScheduler<GraphT2, GraphT> isoScheduler(scheduler);
    isoScheduler.setMergeDifferentTypes(false);
    isoScheduler.setWorkThreshold(100);
    isoScheduler.setCriticalPathThreshold(500);
    isoScheduler.setOrbitLockRatio(0.5);
    isoScheduler.setAllowTrimmedScheduler(false);
    isoScheduler.set_plot_dot_graphs(true);    // Enable plotting for debug

    std::cout << "Starting partition computation..." << std::endl;

    // This is the call that is expected to throw the exception
    auto partition = isoScheduler.compute_partition(instance);

    CheckPartitionTypeHomogeneity(instance.GetComputationalDag(), partition);

    GraphT coraseGraph;
    coarser_util::ConstructCoarseDag(instance.GetComputationalDag(), coraseGraph, partition);
    bool acyc = is_acyclic(coraseGraph);
    std::cout << "Partition is " << (acyc ? "acyclic." : "not acyclic.");

    std::cout << "Partition computation finished." << std::endl;
    std::cout << "Generated " << std::set<VertexIdxT<GraphT>>(partition.begin(), partition.end()).size() << " partitions."
              << std::endl;

    return 0;
}
