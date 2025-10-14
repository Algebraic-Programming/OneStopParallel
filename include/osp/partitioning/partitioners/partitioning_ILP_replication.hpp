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

template<typename index_type = size_t, typename workw_type = int, typename memw_type = int, typename commw_type = int>
class HypergraphPartitioningILPWithReplication : public HypergraphPartitioningILPBase<index_type, workw_type, memw_type, commw_type> {

  public:
    enum class REPLICATION_MODEL_IN_ILP { ONLY_TWICE, GENERAL };

  protected:
    void setupExtraVariablesConstraints(const PartitioningProblem<index_type, workw_type, memw_type, commw_type> &instance, Model& model);

    void setInitialSolution(const PartitioningWithReplication<index_type, workw_type, memw_type, commw_type> &partition, Model& model);

    REPLICATION_MODEL_IN_ILP replication_model = REPLICATION_MODEL_IN_ILP::ONLY_TWICE;

  public:

    virtual ~HypergraphPartitioningILPWithReplication() = default;

    RETURN_STATUS computePartitioning(PartitioningWithReplication<index_type, workw_type, memw_type, commw_type>& result);

    virtual std::string getAlgorithmName() const override { return "HypergraphPartitioningILPWithReplication"; }

    void setReplicationModel(REPLICATION_MODEL_IN_ILP replication_model_) { replication_model = replication_model_; }
};

template<typename index_type, typename workw_type, typename memw_type, typename commw_type>
RETURN_STATUS HypergraphPartitioningILPWithReplication<index_type, workw_type, memw_type, commw_type>::computePartitioning(PartitioningWithReplication<index_type, workw_type, memw_type, commw_type>& result)
{
    Envr env;
    Model model = env.CreateModel("HypergraphPartRepl");

    this->setupFundamentalVariablesConstraintsObjective(result.getInstance(), model);
    setupExtraVariablesConstraints(result.getInstance(), model);

    if(this->use_initial_solution)
        setInitialSolution(result, model);

    this->solveILP(model);

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {

        result.setAssignedPartitionVectors(this->readAllCoptAssignments(result.getInstance(), model));
        return RETURN_STATUS::OSP_SUCCESS;

    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {

        return RETURN_STATUS::ERROR;

    } else {

        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {

            result.setAssignedPartitionVectors(this->readAllCoptAssignments(result.getInstance(), model));
            return RETURN_STATUS::OSP_SUCCESS;

        } else {
            return RETURN_STATUS::ERROR;
        }
    }
}

template<typename index_type, typename workw_type, typename memw_type, typename commw_type>
void HypergraphPartitioningILPWithReplication<index_type, workw_type, memw_type, commw_type>::setupExtraVariablesConstraints(const PartitioningProblem<index_type, workw_type, memw_type, commw_type> &instance, Model& model) {

    const index_type numberOfParts = instance.getNumberOfPartitions();
    const index_type numberOfVertices = instance.getHypergraph().num_vertices();

    if(replication_model == REPLICATION_MODEL_IN_ILP::GENERAL)
    {
        // create variables for each pin+partition combination
        std::map<std::pair<index_type, unsigned>, index_type> pin_ID_map;
        index_type nr_of_pins = 0;
        for (index_type node = 0; node < numberOfVertices; node++)
            for (const index_type& hyperedge : instance.getHypergraph().get_incident_hyperedges(node))
                pin_ID_map[std::make_pair(node, hyperedge)] = nr_of_pins++;
        
        std::vector<VarArray> pin_covered_by_partition = std::vector<VarArray>(nr_of_pins);

        for (index_type pin = 0; pin < nr_of_pins; pin++)
            pin_covered_by_partition[pin] = model.AddVars(static_cast<int>(numberOfParts), COPT_BINARY, "pin_covered_by_partition");

        //  each pin covered exactly once
        for (index_type pin = 0; pin < nr_of_pins; pin++) {

            Expr expr;
            for (unsigned part = 0; part < numberOfParts; part++)
                expr += pin_covered_by_partition[pin][static_cast<int>(part)];

            model.AddConstr(expr == 1);
        }

        // pin covering requires node assignment
        for (unsigned part = 0; part < numberOfParts; part++)
            for (index_type node = 0; node < numberOfVertices; node++)
                for (const index_type& hyperedge : instance.getHypergraph().get_incident_hyperedges(node))
                    model.AddConstr(this->node_in_partition[node][static_cast<int>(part)] >= pin_covered_by_partition[pin_ID_map[std::make_pair(node, hyperedge)]][static_cast<int>(part)]);

        // pin covering requires hyperedge use
        for (unsigned part = 0; part < numberOfParts; part++)
            for (index_type node = 0; node < numberOfVertices; node++)
                for (const index_type& hyperedge : instance.getHypergraph().get_incident_hyperedges(node))
                    model.AddConstr(this->hyperedge_uses_partition[hyperedge][static_cast<int>(part)] >= pin_covered_by_partition[pin_ID_map[std::make_pair(node, hyperedge)]][static_cast<int>(part)]);

    }
    else if(replication_model == REPLICATION_MODEL_IN_ILP::ONLY_TWICE)
    {
        // each node has one or two copies
        VarArray node_replicated = model.AddVars(static_cast<int>(numberOfVertices), COPT_BINARY, "node_replicated");
        
        for (index_type node = 0; node < numberOfVertices; node++) {

            Expr expr = -1;
            for (unsigned part = 0; part < numberOfParts; part++)
                expr += this->node_in_partition[node][static_cast<int>(part)];

            model.AddConstr(expr == node_replicated[static_cast<int>(node)]);
        }

        // hyperedge indicators if node is not replicated
        for (unsigned part = 0; part < numberOfParts; part++)
            for (index_type node = 0; node < numberOfVertices; node++)
                for (const index_type& hyperedge : instance.getHypergraph().get_incident_hyperedges(node))
                    model.AddConstr(this->hyperedge_uses_partition[hyperedge][static_cast<int>(part)] >= this->node_in_partition[node][static_cast<int>(part)] - node_replicated[static_cast<int>(node)]);
        
        // hyperedge indicators if node is replicated
        for (index_type node = 0; node < numberOfVertices; node++)
            for (const index_type& hyperedge : instance.getHypergraph().get_incident_hyperedges(node))
                for (unsigned part1 = 0; part1 < numberOfParts; part1++)
                    for (unsigned part2 = part1+1; part2 < numberOfParts; part2++)
                        model.AddConstr(this->hyperedge_uses_partition[hyperedge][static_cast<int>(part1)] + this->hyperedge_uses_partition[hyperedge][static_cast<int>(part2)] >=
                                        this->node_in_partition[node][static_cast<int>(part1)] + this->node_in_partition[node][static_cast<int>(part2)] - 1);
    }
             
};

template<typename index_type, typename workw_type, typename memw_type, typename commw_type>
void HypergraphPartitioningILPWithReplication<index_type, workw_type, memw_type, commw_type>::setInitialSolution(const PartitioningWithReplication<index_type, workw_type, memw_type, commw_type> &partition,  Model& model)
{
    const std::vector<std::vector<unsigned> >& assignments = partition.assignedPartitions();
    const unsigned& numPartitions = partition.getInstance().getNumberOfPartitions();
    if(assignments.size() != partition.getInstance().getHypergraph().num_vertices())
        return;

    for(index_type node = 0; node < assignments.size(); ++node)
    {
        std::vector<bool> assingedToPart(numPartitions, false);
        for(unsigned part : assignments[node])
            if(part < numPartitions)
                assingedToPart[part] = true;
        
        for(unsigned part = 0; part < numPartitions; ++part)
            model.SetMipStart(this->node_in_partition[node][static_cast<int>(part)], static_cast<int>(assingedToPart[part]));
    }
    model.LoadMipStart();
}

} // namespace osp