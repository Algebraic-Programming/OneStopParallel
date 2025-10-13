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
#include "osp/partitioning/model/partitioning.hpp"

namespace osp{

class HypergraphPartitioningILP : public HypergraphPartitioningILPBase {

  protected:
    std::vector<unsigned> readCoptAssignment(const PartitioningProblem &instance, Model& model);

    void setupExtraVariablesConstraints(const PartitioningProblem &instance, Model& model);

    void setInitialSolution(const Partitioning &partition, Model& model);

  public:

    virtual ~HypergraphPartitioningILP() = default;

    RETURN_STATUS computePartitioning(Partitioning& result);

    virtual std::string getAlgorithmName() const override { return "HypergraphPartitioningILP"; }
};

RETURN_STATUS HypergraphPartitioningILP::computePartitioning(Partitioning& result)
{
    Envr env;
    Model model = env.CreateModel("HypergraphPart");

    setupFundamentalVariablesConstraintsObjective(result.getInstance(), model);
    setupExtraVariablesConstraints(result.getInstance(), model);

    if(use_initial_solution)
        setInitialSolution(result, model);

    solveILP(model);

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {

        result.setAssignedPartitions(readCoptAssignment(result.getInstance(), model));
        return RETURN_STATUS::OSP_SUCCESS;

    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {

        return RETURN_STATUS::ERROR;

    } else {

        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {

            result.setAssignedPartitions(readCoptAssignment(result.getInstance(), model));
            return RETURN_STATUS::OSP_SUCCESS;

        } else {
            return RETURN_STATUS::ERROR;
        }
    }
}

void HypergraphPartitioningILP::setupExtraVariablesConstraints(const PartitioningProblem &instance, Model& model) {

    const unsigned numberOfParts = instance.getNumberOfPartitions();
    const unsigned numberOfVertices = instance.getHypergraph().num_vertices();

    // Constraints

    // each node assigned to exactly one partition
    for (unsigned node = 0; node < numberOfVertices; node++) {

        Expr expr;
        for (unsigned part = 0; part < numberOfParts; part++)
            expr += node_in_partition[node][static_cast<int>(part)];

        model.AddConstr(expr == 1);
    }

    // hyperedge indicators match node variables
    for (unsigned part = 0; part < numberOfParts; part++)
        for (unsigned node = 0; node < numberOfVertices; node++)
            for (const unsigned& hyperedge : instance.getHypergraph().get_incident_hyperedges(node))
                model.AddConstr(hyperedge_uses_partition[hyperedge][static_cast<int>(part)] >= node_in_partition[node][static_cast<int>(part)]);
             
};

// convert generic one-to-many assingment (of base class function) to one-to-one
std::vector<unsigned> HypergraphPartitioningILP::readCoptAssignment(const PartitioningProblem &instance, Model& model)
{
    std::vector<unsigned> node_to_partition(instance.getHypergraph().num_vertices(), UINT_MAX);
    std::vector<std::vector<unsigned> > assignmentsGenericForm = readAllCoptAssignments(instance, model);

    for (unsigned node = 0; node < instance.getHypergraph().num_vertices(); node++)
        node_to_partition[node] = assignmentsGenericForm[node].front();

    return node_to_partition;
}

void HypergraphPartitioningILP::setInitialSolution(const Partitioning &partition,  Model& model)
{
    const std::vector<unsigned>& assignment = partition.assignedPartitions();
    const unsigned& numPartitions = partition.getInstance().getNumberOfPartitions();
    if(assignment.size() != partition.getInstance().getHypergraph().num_vertices())
        return;

    for(unsigned node = 0; node < assignment.size(); ++node)
    {
        if(assignment[node] >= numPartitions)
            continue;
        
        for(unsigned part = 0; part < numPartitions; ++part)
            model.SetMipStart(node_in_partition[node][static_cast<int>(part)], static_cast<int>(assignment[node] == part));
    }
    model.LoadMipStart();
}

} // namespace osp