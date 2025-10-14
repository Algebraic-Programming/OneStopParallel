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

#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "osp/auxiliary/misc.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"
#include "osp/auxiliary/io/general_file_reader.hpp"
#include "osp/partitioning/partitioners/partitioning_ILP.hpp"
#include "osp/partitioning/partitioners/partitioning_ILP_replication.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/auxiliary/io/mtx_hypergraph_file_reader.hpp"
#include "osp/auxiliary/io/partitioning_file_writer.hpp"
    

using namespace osp;

using graph = computational_dag_vector_impl_def_int_t;

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <nr_parts> <imbalance> <optional:part_repl|full_repl>"
                  << std::endl;
        return 1;
    }

    std::string filename_hgraph = argv[1];
    std::string name_hgraph = filename_hgraph.substr(0, filename_hgraph.rfind("."));
    std::string file_ending = filename_hgraph.substr(filename_hgraph.rfind(".") + 1);
    if (!file_reader::isPathSafe(filename_hgraph)) {
        std::cerr << "Error: Unsafe file path (possible traversal or invalid type).\n";
        return 1;
    }

    std::cout << name_hgraph << std::endl;

    int nr_parts = std::stoi(argv[2]);
    if (nr_parts < 2 || nr_parts > 32) {
        std::cerr << "Argument nr_parts must be an integer between 2 and 32: " << nr_parts << std::endl;
        return 1;
    }

    float imbalance = std::stof(argv[3]);
    if (imbalance < 0.01 || imbalance > .99) {
        std::cerr << "Argument imbalance must be a float between 0.01 and 0.99: " << imbalance << std::endl;
        return 1;
    }

    unsigned replicate = 0;

    if (argc > 4 && std::string(argv[4]) == "part_repl") {
        replicate = 1;
    } else if (argc > 4 && std::string(argv[4]) == "full_repl") {
        replicate = 2;
    } else if (argc > 4) {
        std::cerr << "Unknown argument: " << argv[4] << ". Expected 'part_repl' or 'full_repl' for replication." << std::endl;
        return 1;
    }

    Hypergraph hgraph;

    bool file_status = true;    
    if (file_ending == "hdag") {
        graph dag;
        file_status = file_reader::readComputationalDagHyperdagFormatDB(filename_hgraph, dag);
        if(file_status)
            hgraph.convert_from_cdag_as_hyperdag(dag);
    } else if (file_ending == "mtx") {
        file_status = file_reader::readHypergraphMartixMarketFormat(filename_hgraph, hgraph);
    } else {
        std::cout << "Unknown file extension." << std::endl;
        return 1;
    }
    if (!file_status) {

        std::cout << "Reading input file failed." << std::endl;
        return 1;
    }

    PartitioningProblem instance(hgraph, static_cast<unsigned>(nr_parts));
    instance.setMaxWorkWeightViaImbalanceFactor(imbalance);

    if (replicate > 0) {

        PartitioningWithReplication partition(instance);
        HypergraphPartitioningILPWithReplication partitioner;

        for(size_t node = 0; node < hgraph.num_vertices(); ++node)
            partition.setAssignedPartitions(node, {static_cast<unsigned>(node % static_cast<size_t>(nr_parts))});
        if(partition.satisfiesBalanceConstraint())
            partitioner.setUseInitialSolution(true);

        partitioner.setTimeLimitSeconds(600);
        if(replicate == 2)
            partitioner.setReplicationModel(HypergraphPartitioningILPWithReplication<>::REPLICATION_MODEL_IN_ILP::GENERAL);

        auto solve_status = partitioner.computePartitioning(partition);

        if (solve_status == RETURN_STATUS::OSP_SUCCESS || solve_status == RETURN_STATUS::BEST_FOUND) {
            file_writer::write_txt(name_hgraph + "_" + std::to_string(nr_parts) + "_" + std::to_string(imbalance) +
                "_ILP_rep" + std::to_string(replicate) + ".txt", partition);
            std::cout << "Partitioning (with replicaiton) computed with costs: " << partition.computeConnectivityCost() << std::endl;
        } else {
            std::cout << "Computing partition failed." << std::endl;
            return 1;
        }

    } else {

        Partitioning partition(instance);
        HypergraphPartitioningILP partitioner;

        for(size_t node = 0; node < hgraph.num_vertices(); ++node)
            partition.setAssignedPartition(node, static_cast<unsigned>(node % static_cast<size_t>(nr_parts)));
        if(partition.satisfiesBalanceConstraint())
            partitioner.setUseInitialSolution(true);

        partitioner.setTimeLimitSeconds(600);

        auto solve_status = partitioner.computePartitioning(partition);

        if (solve_status == RETURN_STATUS::OSP_SUCCESS || solve_status == RETURN_STATUS::BEST_FOUND) {
            file_writer::write_txt(name_hgraph + "_" + std::to_string(nr_parts) + "_" + std::to_string(imbalance) +
                "_ILP_rep" + std::to_string(replicate) + ".txt", partition);
            std::cout << "Partitioning computed with costs: " << partition.computeConnectivityCost() << std::endl;
        } else {
            std::cout << "Computing partition failed." << std::endl;
            return 1;
        }
    }
    return 0;
}
