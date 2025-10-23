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
#include "osp/auxiliary/io/dot_graph_file_reader.hpp"
#include "osp/auxiliary/io/DotFileWriter.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/BspLocking.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/kl_include_mt.hpp"
#include "osp/coarser/coarser_util.hpp"
#include "osp/dag_divider/isomorphism_divider/IsomorphicSubgraphScheduler.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"

using namespace osp;

int main(int argc, char* argv[]) {
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

    // Set up architecture
    instance.getArchitecture().setProcessorsWithTypes({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 , 1, 1, 1, 1, 1, 1, 1, 1 , 1, 1, 1, 1, 1, 1, 1, 1 , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    instance.setDiagonalCompatibilityMatrix(2);
    instance.setSynchronisationCosts(20000);
    instance.setCommunicationCosts(10);

    // Set up the scheduler
    BspLocking<graph_t> greedy;
    kl_total_lambda_comm_improver_mt<graph_t> kl;
    ComboScheduler<graph_t> combo(greedy, kl);

    IsomorphicSubgraphScheduler<graph_t2, graph_t> iso_scheduler(combo);
    iso_scheduler.set_symmetry(4);
    iso_scheduler.set_plot_dot_graphs(true); // Enable plotting for debug

    std::cout << "Starting partition computation..." << std::endl;

    // This is the call that is expected to throw the exception
    auto partition = iso_scheduler.compute_partition(instance);

    std::cout << "Partition computation finished." << std::endl;
    std::cout << "Generated " << std::set<vertex_idx_t<graph_t>>(partition.begin(), partition.end()).size() << " partitions." << std::endl;

    return 0;
}
