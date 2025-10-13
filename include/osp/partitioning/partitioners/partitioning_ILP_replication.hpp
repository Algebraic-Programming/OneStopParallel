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

#include <callbackbase.h>
#include <coptcpp_pch.h>

#include "osp/partitioning/partitioners/partitioning_ILP_base.hpp"
#include "osp/partitioning/model/partitioning_replication.hpp"

namespace osp{

class HypergraphPartitioningILPWithReplication : public HypergraphPartitioningILPBase {

  public:
    enum class REPLICATION_MODEL_IN_ILP { ONLY_TWICE, GENERAL };

  protected:
    std::vector<unsigned> readCoptAssignment(const PartitioningProblem &instance, Model& model);

    void setupExtraVariablesConstraints(const PartitioningProblem &instance, Model& model);

    void setInitialSolution(const PartitioningWithReplication &partition, Model& model);

    REPLICATION_MODEL_IN_ILP replication_model = REPLICATION_MODEL_IN_ILP::ONLY_TWICE;

  public:

    virtual ~HypergraphPartitioningILPWithReplication() = default;

    RETURN_STATUS computePartitioning(PartitioningWithReplication& result);

    virtual std::string getAlgorithmName() const override { return "HypergraphPartitioningILPWithReplication"; }

    void setReplicationModel(REPLICATION_MODEL_IN_ILP replication_model_) { replication_model = replication_model_; }
};

RETURN_STATUS HypergraphPartitioningILPWithReplication::computePartitioning(PartitioningWithReplication& result)
{
    Envr env;
    Model model = env.CreateModel("HypergraphPartRepl");

    setupFundamentalVariablesConstraintsObjective(result.getInstance(), model);
    setupExtraVariablesConstraints(result.getInstance(), model);

    if(use_initial_solution)
        setInitialSolution(result, model);

    solveILP(model);

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {

        result.setAssignedPartitionVectors(readAllCoptAssignments(result.getInstance(), model));
        return RETURN_STATUS::OSP_SUCCESS;

    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {

        return RETURN_STATUS::ERROR;

    } else {

        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {

            result.setAssignedPartitionVectors(readAllCoptAssignments(result.getInstance(), model));
            return RETURN_STATUS::OSP_SUCCESS;

        } else {
            return RETURN_STATUS::ERROR;
        }
    }
}

void HypergraphPartitioningILPWithReplication::setupExtraVariablesConstraints(const PartitioningProblem &instance, Model& model) {

    const unsigned numberOfParts = instance.getNumberOfPartitions();
    const unsigned numberOfVertices = instance.getHypergraph().num_vertices();

    if(replication_model == REPLICATION_MODEL_IN_ILP::GENERAL)
    {
        // create variables for each pin+partition combination
        std::map<std::pair<unsigned, unsigned>, unsigned> pin_ID_map;
        unsigned nr_of_pins = 0;
        for (unsigned node = 0; node < numberOfVertices; node++)
            for (const unsigned& hyperedge : instance.getHypergraph().get_incident_hyperedges(node))
                pin_ID_map[std::make_pair(node, hyperedge)] = nr_of_pins++;
        
        std::vector<VarArray> pin_covered_by_partition = std::vector<VarArray>(nr_of_pins);

        for (unsigned pin = 0; pin < nr_of_pins; pin++)
            pin_covered_by_partition[pin] = model.AddVars(static_cast<int>(numberOfParts), COPT_BINARY, "pin_covered_by_partition");

        //  each pin covered exactly once
        for (unsigned pin = 0; pin < nr_of_pins; pin++) {

            Expr expr;
            for (unsigned part = 0; part < numberOfParts; part++)
                expr += pin_covered_by_partition[pin][static_cast<int>(part)];

            model.AddConstr(expr == 1);
        }

        // pin covering requires node assignment
        for (unsigned part = 0; part < numberOfParts; part++)
            for (unsigned node = 0; node < numberOfVertices; node++)
                for (const unsigned& hyperedge : instance.getHypergraph().get_incident_hyperedges(node))
                    model.AddConstr(node_in_partition[node][static_cast<int>(part)] >= pin_covered_by_partition[pin_ID_map[std::make_pair(node, hyperedge)]][static_cast<int>(part)]);

        // pin covering requires hyperedge use
        for (unsigned part = 0; part < numberOfParts; part++)
            for (unsigned node = 0; node < numberOfVertices; node++)
                for (const unsigned& hyperedge : instance.getHypergraph().get_incident_hyperedges(node))
                    model.AddConstr(hyperedge_uses_partition[hyperedge][static_cast<int>(part)] >= pin_covered_by_partition[pin_ID_map[std::make_pair(node, hyperedge)]][static_cast<int>(part)]);

    }
    else if(replication_model == REPLICATION_MODEL_IN_ILP::ONLY_TWICE)
    {
        // each node has one or two copies
        VarArray node_replicated = model.AddVars(static_cast<int>(numberOfVertices), COPT_BINARY, "node_replicated");
        
        for (unsigned node = 0; node < numberOfVertices; node++) {

            Expr expr = -1;
            for (unsigned part = 0; part < numberOfParts; part++)
                expr += node_in_partition[node][static_cast<int>(part)];

            model.AddConstr(expr == node_replicated[static_cast<int>(node)]);
        }

        // hyperedge indicators if node is not replicated
        for (unsigned part = 0; part < numberOfParts; part++)
            for (unsigned node = 0; node < numberOfVertices; node++)
                for (const unsigned& hyperedge : instance.getHypergraph().get_incident_hyperedges(node))
                    model.AddConstr(hyperedge_uses_partition[hyperedge][static_cast<int>(part)] >= node_in_partition[node][static_cast<int>(part)] - node_replicated[static_cast<int>(node)]);
        
        // hyperedge indicators if node is replicated
        for (unsigned node = 0; node < numberOfVertices; node++)
            for (const unsigned& hyperedge : instance.getHypergraph().get_incident_hyperedges(node))
                for (unsigned part1 = 0; part1 < numberOfParts; part1++)
                    for (unsigned part2 = part1+1; part2 < numberOfParts; part2++)
                        model.AddConstr(hyperedge_uses_partition[hyperedge][static_cast<int>(part1)] + hyperedge_uses_partition[hyperedge][static_cast<int>(part2)] >=
                                        node_in_partition[node][static_cast<int>(part1)] + node_in_partition[node][static_cast<int>(part2)] - 1);
    }
             
};

void HypergraphPartitioningILPWithReplication::setInitialSolution(const PartitioningWithReplication &partition,  Model& model)
{
    const std::vector<std::vector<unsigned> >& assignments = partition.assignedPartitions();
    const unsigned& numPartitions = partition.getInstance().getNumberOfPartitions();
    if(assignments.size() != partition.getInstance().getHypergraph().num_vertices())
        return;

    for(unsigned node = 0; node < assignments.size(); ++node)
    {
        std::vector<bool> assingedToPart(numPartitions, false);
        for(unsigned part : assignments[node])
            if(part < numPartitions)
                assingedToPart[part] = true;
        
        for(unsigned part = 0; part < numPartitions; ++part)
            model.SetMipStart(node_in_partition[node][static_cast<int>(part)], static_cast<int>(assingedToPart[part]));
    }
    model.LoadMipStart();
}

} // namespace osp