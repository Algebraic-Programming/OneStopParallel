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

#include "osp/auxiliary/io/general_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/auxiliary/io/mtx_hypergraph_file_reader.hpp"
#include "osp/auxiliary/io/partitioning_file_writer.hpp"
#include "osp/auxiliary/misc.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/partitioning/model/hypergraph_utility.hpp"
#include "osp/partitioning/partitioners/generic_FM.hpp"
#include "osp/partitioning/partitioners/partitioning_ILP.hpp"
#include "osp/partitioning/partitioners/partitioning_ILP_replication.hpp"

using namespace osp;

using Graph = computational_dag_vector_impl_def_int_t;
using Hypergraph = Hypergraph_def_t;

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <nr_parts> <imbalance> <optional:part_repl|full_repl>" << std::endl;
        return 1;
    }

    std::string filenameHgraph = argv[1];
    std::string nameHgraph = filenameHgraph.substr(0, filenameHgraph.rfind("."));
    std::string fileEnding = filenameHgraph.substr(filenameHgraph.rfind(".") + 1);
    if (!file_reader::isPathSafe(filenameHgraph)) {
        std::cerr << "Error: Unsafe file path (possible traversal or invalid type).\n";
        return 1;
    }

    std::cout << nameHgraph << std::endl;

    int nrParts = std::stoi(argv[2]);
    if (nrParts < 2 || nrParts > 32) {
        std::cerr << "Argument nr_parts must be an integer between 2 and 32: " << nrParts << std::endl;
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

    PartitioningProblem<Hypergraph> instance;

    bool fileStatus = true;
    if (fileEnding == "hdag") {
        Graph dag;
        fileStatus = file_reader::readComputationalDagHyperdagFormatDB(filenameHgraph, dag);
        if (fileStatus) {
            instance.getHypergraph() = convert_from_cdag_as_hyperdag<Hypergraph, Graph>(dag);
        }
    } else if (fileEnding == "mtx") {
        fileStatus = file_reader::readHypergraphMartixMarketFormat(filenameHgraph, instance.getHypergraph());
    } else {
        std::cout << "Unknown file extension." << std::endl;
        return 1;
    }
    if (!fileStatus) {
        std::cout << "Reading input file failed." << std::endl;
        return 1;
    }

    instance.setNumberOfPartitions(static_cast<unsigned>(nrParts));
    instance.setMaxWorkWeightViaImbalanceFactor(imbalance);

    Partitioning<Hypergraph> initialPartition(instance);
    GenericFM<Hypergraph> fm;
    for (size_t node = 0; node < instance.getHypergraph().num_vertices(); ++node) {
        initialPartition.setAssignedPartition(node, static_cast<unsigned>(node % static_cast<size_t>(nrParts)));
    }
    if (nrParts == 2) {
        fm.ImprovePartitioning(initialPartition);
    }
    if (nrParts == 4 || nrParts == 8 || nrParts == 16 || nrParts == 32) {
        fm.RecursiveFM(initialPartition);
    }

    if (replicate > 0) {
        PartitioningWithReplication<Hypergraph> partition(instance);
        HypergraphPartitioningILPWithReplication<Hypergraph> partitioner;

        for (size_t node = 0; node < instance.getHypergraph().num_vertices(); ++node) {
            partition.setAssignedPartitions(node, {initialPartition.assignedPartition(node)});
        }
        if (partition.satisfiesBalanceConstraint()) {
            partitioner.setUseInitialSolution(true);
        }

        partitioner.setTimeLimitSeconds(600);
        if (replicate == 2) {
            partitioner.setReplicationModel(
                HypergraphPartitioningILPWithReplication<Hypergraph>::REPLICATION_MODEL_IN_ILP::GENERAL);
        }

        auto solveStatus = partitioner.computePartitioning(partition);

        if (solveStatus == RETURN_STATUS::OSP_SUCCESS || solveStatus == RETURN_STATUS::BEST_FOUND) {
            file_writer::write_txt(nameHgraph + "_" + std::to_string(nrParts) + "_" + std::to_string(imbalance) + "_ILP_rep"
                                       + std::to_string(replicate) + ".txt",
                                   partition);
            std::cout << "Partitioning (with replicaiton) computed with costs: " << partition.computeConnectivityCost()
                      << std::endl;
        } else {
            std::cout << "Computing partition failed." << std::endl;
            return 1;
        }

    } else {
        Partitioning<Hypergraph> partition(instance);
        HypergraphPartitioningILP<Hypergraph> partitioner;

        for (size_t node = 0; node < instance.getHypergraph().num_vertices(); ++node) {
            partition.setAssignedPartition(node, initialPartition.assignedPartition(node));
        }
        if (partition.satisfiesBalanceConstraint()) {
            partitioner.setUseInitialSolution(true);
        }

        partitioner.setTimeLimitSeconds(600);

        auto solveStatus = partitioner.computePartitioning(partition);

        if (solveStatus == RETURN_STATUS::OSP_SUCCESS || solveStatus == RETURN_STATUS::BEST_FOUND) {
            file_writer::write_txt(nameHgraph + "_" + std::to_string(nrParts) + "_" + std::to_string(imbalance) + "_ILP_rep"
                                       + std::to_string(replicate) + ".txt",
                                   partition);
            std::cout << "Partitioning computed with costs: " << partition.computeConnectivityCost() << std::endl;
        } else {
            std::cout << "Computing partition failed." << std::endl;
            return 1;
        }
    }
    return 0;
}
