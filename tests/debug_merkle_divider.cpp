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
#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/kl_include_mt.hpp"
#include "osp/bsp/scheduler/Serial.hpp"
#include "osp/coarser/coarser_util.hpp"
#include "osp/dag_divider/isomorphism_divider/IsomorphicSubgraphScheduler.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"

using namespace osp;

template <typename GraphT>
void check_partition_type_homogeneity(const GraphT &dag, const std::vector<vertex_idx_t<GraphT>> &partition) {
    // Group partitions by their ID
    std::map<vertex_idx_t<GraphT>, std::vector<vertex_idx_t<GraphT>>> partitions;
    for (vertex_idx_t<GraphT> i = 0; i < dag.num_vertices(); ++i) {
        partitions[partition[i]].push_back(i);
    }

    // For each partition, check that all vertices have the same type
    for (const auto &[part_id, vertices] : partitions) {
        if (vertices.empty()) {
            continue;
        }
        const auto first_node_type = dag.vertex_type(vertices[0]);
        for (const auto &vertex : vertices) {
            if (dag.vertex_type(vertex) != first_node_type) {
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

    std::string dot_file_path = argv[1];

    using graph_t = computational_dag_vector_impl_def_t;
    using graph_t2 = graph_t;

    BspInstance<graph_t2> instance;
    if (!file_reader::readComputationalDagDotFormat(dot_file_path, instance.getComputationalDag())) {
        std::cerr << "Failed to read graph from " << dot_file_path << std::endl;
        return 1;
    }

    std::cout << "Graph loaded successfully. " << instance.numberOfVertices() << " vertices." << std::endl;

    for (auto v : instance.getComputationalDag().vertices()) {
        instance.getComputationalDag().set_vertex_comm_weight(
            v, static_cast<v_commw_t<graph_t2>>(instance.getComputationalDag().vertex_comm_weight(v) * 0.01));
    }

    // Set up architecture
    instance.getArchitecture().SetProcessorsConsequTypes({24, 48}, {100, 100});
    instance.setDiagonalCompatibilityMatrix(2);
    instance.setSynchronisationCosts(2000);
    instance.setCommunicationCosts(1);

    // Set up the scheduler
    GrowLocalAutoCores<graph_t> growlocal;
    BspLocking<graph_t> locking;
    GreedyChildren<graph_t> children;
    kl_total_lambda_comm_improver<graph_t> kl(42);
    kl.setSuperstepRemoveStrengthParameter(1.0);
    kl.setTimeQualityParameter(1.0);
    ComboScheduler<graph_t> growlocal_kl(growlocal, kl);
    ComboScheduler<graph_t> locking_kl(locking, kl);
    ComboScheduler<graph_t> children_kl(children, kl);

    GreedyMetaScheduler<graph_t> scheduler;
    // scheduler.addScheduler(growlocal_kl);
    scheduler.addScheduler(locking_kl);
    scheduler.addScheduler(children_kl);
    scheduler.addSerialScheduler();

    IsomorphicSubgraphScheduler<graph_t2, graph_t> iso_scheduler(scheduler);
    iso_scheduler.setMergeDifferentTypes(false);
    iso_scheduler.setWorkThreshold(100);
    iso_scheduler.setCriticalPathThreshold(500);
    iso_scheduler.setOrbitLockRatio(0.5);
    iso_scheduler.setAllowTrimmedScheduler(false);
    iso_scheduler.set_plot_dot_graphs(true);    // Enable plotting for debug

    std::cout << "Starting partition computation..." << std::endl;

    // This is the call that is expected to throw the exception
    auto partition = iso_scheduler.compute_partition(instance);

    check_partition_type_homogeneity(instance.getComputationalDag(), partition);

    graph_t corase_graph;
    coarser_util::construct_coarse_dag(instance.getComputationalDag(), corase_graph, partition);
    bool acyc = is_acyclic(corase_graph);
    std::cout << "Partition is " << (acyc ? "acyclic." : "not acyclic.");

    std::cout << "Partition computation finished." << std::endl;
    std::cout << "Generated " << std::set<vertex_idx_t<graph_t>>(partition.begin(), partition.end()).size() << " partitions."
              << std::endl;

    return 0;
}
