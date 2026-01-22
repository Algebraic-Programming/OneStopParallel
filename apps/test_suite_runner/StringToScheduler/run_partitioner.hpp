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

#pragma once

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <filesystem>
#include <iostream>
#include <string>

#include "../ConfigParser.hpp"
#include "osp/partitioning/partitioners/partitioning_ILP.hpp"
#include "osp/partitioning/partitioners/partitioning_ILP_replication.hpp"

namespace osp {

const std::set<std::string> GetAvailablePartitionerNames() { return {"ILP", "ILP_dupl", "ILP_repl"}; }


template <typename IndexType, typename WorkwType, typename MemwType, typename CommwType>
ReturnStatus RunPartitioner(const ConfigParser &parser,
                                   const boost::property_tree::ptree &algorithm,
                                   const PartitioningProblem<Hypergraph<IndexType, WorkwType, MemwType, CommwType> > &instance,
                                   std::pair<CommwType, CommwType> &cost) {
    using Hgraph = Hypergraph<IndexType, WorkwType, MemwType, CommwType>;

    std::cout << "Running algorithm: " << algorithm.get_child("name").get_value<std::string>() << std::endl;

    if (algorithm.get_child("name").get_value<std::string>() == "ILP") {
        HypergraphPartitioningILP<Hgraph> partitioner;
        Partitioning<Hgraph> solution(instance);

        const unsigned timeLimit = parser.globalParams_.get_child("timeLimit").get_value<unsigned>();
        partitioner.SetTimeLimitSeconds(timeLimit);
        ReturnStatus status = partitioner.ComputePartitioning(solution);
        cost = {solution.ComputeConnectivityCost(), solution.ComputeCutNetCost()};
        return status;
    
    } else if (algorithm.get_child("name").get_value<std::string>() == "ILP_dupl"
                || algorithm.get_child("name").get_value<std::string>() == "ILP_repl") {
        HypergraphPartitioningILPWithReplication<Hgraph> partitioner;
        PartitioningWithReplication<Hgraph> solution(instance);

        const unsigned timeLimit = parser.globalParams_.get_child("timeLimit").get_value<unsigned>();
        partitioner.SetTimeLimitSeconds(timeLimit);
        if (algorithm.get_child("name").get_value<std::string>() == "ILP_repl") {
            partitioner.SetReplicationModel(HypergraphPartitioningILPWithReplication<Hgraph>::ReplicationModelInIlp::GENERAL);
        }

        ReturnStatus status = partitioner.ComputePartitioning(solution);
        cost = {solution.ComputeConnectivityCost(), solution.ComputeCutNetCost()};
        return status;

    } else {
        throw std::invalid_argument("Parameter error: Unknown algorithm.\n");
    }
}

}    // namespace osp