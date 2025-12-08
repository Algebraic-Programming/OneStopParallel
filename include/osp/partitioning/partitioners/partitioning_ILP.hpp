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

#include "osp/auxiliary/return_status.hpp"
#include "osp/partitioning/model/partitioning.hpp"
#include "osp/partitioning/partitioners/partitioning_ILP_base.hpp"

namespace osp {

template<typename hypergraph_t>
class HypergraphPartitioningILP : public HypergraphPartitioningILPBase<hypergraph_t> {

  protected:
    std::vector<unsigned> readCoptAssignment(const PartitioningProblem<hypergraph_t> &instance, Model &model);

    void setupExtraVariablesConstraints(const PartitioningProblem<hypergraph_t> &instance, Model &model);

    void setInitialSolution(const Partitioning<hypergraph_t> &partition, Model &model);

  public:
    virtual ~HypergraphPartitioningILP() override = default;

    RETURN_STATUS computePartitioning(Partitioning<hypergraph_t> &result);

    virtual std::string getAlgorithmName() const override { return "HypergraphPartitioningILP"; }
};

template<typename hypergraph_t>
RETURN_STATUS HypergraphPartitioningILP<hypergraph_t>::computePartitioning(Partitioning<hypergraph_t> &result) {
    Envr env;
    Model model = env.CreateModel("HypergraphPart");

    this->setupFundamentalVariablesConstraintsObjective(result.getInstance(), model);
    setupExtraVariablesConstraints(result.getInstance(), model);

    if (this->use_initial_solution)
        setInitialSolution(result, model);

    this->solveILP(model);

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

template<typename hypergraph_t>
void HypergraphPartitioningILP<hypergraph_t>::setupExtraVariablesConstraints(const PartitioningProblem<hypergraph_t> &instance, Model &model) {

    using index_type = typename hypergraph_t::vertex_idx;

    const index_type numberOfParts = instance.getNumberOfPartitions();
    const index_type numberOfVertices = instance.getHypergraph().num_vertices();

    // Constraints

    // each node assigned to exactly one partition
    for (index_type node = 0; node < numberOfVertices; node++) {

        Expr expr;
        for (unsigned part = 0; part < numberOfParts; part++)
            expr += this->node_in_partition[node][static_cast<int>(part)];

        model.AddConstr(expr == 1);
    }

    // hyperedge indicators match node variables
    for (unsigned part = 0; part < numberOfParts; part++)
        for (index_type node = 0; node < numberOfVertices; node++)
            for (const index_type &hyperedge : instance.getHypergraph().get_incident_hyperedges(node))
                model.AddConstr(this->hyperedge_uses_partition[hyperedge][static_cast<int>(part)] >= this->node_in_partition[node][static_cast<int>(part)]);
}

// convert generic one-to-many assingment (of base class function) to one-to-one
template<typename hypergraph_t>
std::vector<unsigned> HypergraphPartitioningILP<hypergraph_t>::readCoptAssignment(const PartitioningProblem<hypergraph_t> &instance, Model &model) {
    using index_type = typename hypergraph_t::vertex_idx;

    std::vector<unsigned> node_to_partition(instance.getHypergraph().num_vertices(), std::numeric_limits<unsigned>::max());
    std::vector<std::vector<unsigned>> assignmentsGenericForm = this->readAllCoptAssignments(instance, model);

    for (index_type node = 0; node < instance.getHypergraph().num_vertices(); node++)
        node_to_partition[node] = assignmentsGenericForm[node].front();

    return node_to_partition;
}

template<typename hypergraph_t>
void HypergraphPartitioningILP<hypergraph_t>::setInitialSolution(const Partitioning<hypergraph_t> &partition, Model &model) {
    using index_type = typename hypergraph_t::vertex_idx;

    const std::vector<unsigned> &assignment = partition.assignedPartitions();
    const unsigned &numPartitions = partition.getInstance().getNumberOfPartitions();
    if (assignment.size() != partition.getInstance().getHypergraph().num_vertices())
        return;

    for (index_type node = 0; node < assignment.size(); ++node) {
        if (assignment[node] >= numPartitions)
            continue;

        for (unsigned part = 0; part < numPartitions; ++part)
            model.SetMipStart(this->node_in_partition[node][static_cast<int>(part)], static_cast<int>(assignment[node] == part));
    }
    model.LoadMipStart();
}

} // namespace osp