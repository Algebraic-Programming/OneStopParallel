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

#include "osp/partitioning/model/partitioning_problem.hpp"
#include "osp/bsp/model/BspInstance.hpp" // for return statuses (stati?)

namespace osp{

class HypergraphPartitioningILPBase {

  protected:
    std::vector<VarArray> node_in_partition;
    std::vector<VarArray> hyperedge_uses_partition;

    unsigned time_limit_seconds = 3600;
    bool use_initial_solution = false;

    std::vector<std::vector<unsigned> > readAllCoptAssignments(const PartitioningProblem &instance, Model& model);

    void setupFundamentalVariablesConstraintsObjective(const PartitioningProblem &instance, Model& model);

    void solveILP(Model& model);

  public:

    virtual std::string getAlgorithmName() const = 0;

    inline unsigned getTimeLimitSeconds() const { return time_limit_seconds; }
    inline void setTimeLimitSeconds(unsigned limit_) { time_limit_seconds = limit_; }
    inline void setUseInitialSolution(bool use_) { use_initial_solution = use_; }
};

void HypergraphPartitioningILPBase::solveILP(Model& model) {

    model.SetIntParam(COPT_INTPARAM_LOGTOCONSOLE, 0);

    model.SetDblParam(COPT_DBLPARAM_TIMELIMIT, time_limit_seconds);
    model.SetIntParam(COPT_INTPARAM_THREADS, 128);

    model.SetIntParam(COPT_INTPARAM_STRONGBRANCHING, 1);
    model.SetIntParam(COPT_INTPARAM_LPMETHOD, 1);
    model.SetIntParam(COPT_INTPARAM_ROUNDINGHEURLEVEL, 1);

    model.SetIntParam(COPT_INTPARAM_SUBMIPHEURLEVEL, 1);
    // model.SetIntParam(COPT_INTPARAM_PRESOLVE, 1);
    // model.SetIntParam(COPT_INTPARAM_CUTLEVEL, 0);
    model.SetIntParam(COPT_INTPARAM_TREECUTLEVEL, 2);
    // model.SetIntParam(COPT_INTPARAM_DIVINGHEURLEVEL, 2);

    model.Solve();
}

void HypergraphPartitioningILPBase::setupFundamentalVariablesConstraintsObjective(const PartitioningProblem &instance, Model& model) {

    const unsigned numberOfParts = instance.getNumberOfPartitions();
    const unsigned numberOfVertices = instance.getHypergraph().num_vertices();
    const unsigned numberOfHyperedges = instance.getHypergraph().num_hyperedges();

    // Variables

    node_in_partition = std::vector<VarArray>(numberOfVertices);

    for (unsigned node = 0; node < numberOfVertices; node++)
        node_in_partition[node] = model.AddVars(static_cast<int>(numberOfParts), COPT_BINARY, "node_in_partition");

    hyperedge_uses_partition = std::vector<VarArray>(numberOfHyperedges);

    for (unsigned hyperedge = 0; hyperedge < numberOfHyperedges; hyperedge++)
        hyperedge_uses_partition[hyperedge] = model.AddVars(static_cast<int>(numberOfParts), COPT_BINARY, "hyperedge_uses_partition");
    
    // partition size constraints
    for (unsigned part = 0; part < numberOfParts; part++)
    {
        Expr expr;
        for (unsigned node = 0; node < numberOfVertices; node++)
            expr += instance.getHypergraph().get_vertex_weight(node) * node_in_partition[node][static_cast<int>(part)];

        model.AddConstr(expr <= instance.getMaxWeightPerPartition());
    }    

    // set objective
    Expr expr;
    for (unsigned hyperedge = 0; hyperedge < numberOfHyperedges; hyperedge++)
    {
        expr -= instance.getHypergraph().get_hyperedge_weight(hyperedge);
        for (unsigned part = 0; part < numberOfParts; part++)
            expr += instance.getHypergraph().get_hyperedge_weight(hyperedge) * hyperedge_uses_partition[hyperedge][static_cast<int>(part)];
    }

    model.SetObjective(expr, COPT_MINIMIZE);
             
};

std::vector<std::vector<unsigned> > HypergraphPartitioningILPBase::readAllCoptAssignments(const PartitioningProblem &instance, Model& model)
{
    std::vector<std::vector<unsigned> > node_to_partitions(instance.getHypergraph().num_vertices());

    std::set<unsigned> nonempty_partition_ids;
    for (unsigned node = 0; node < instance.getHypergraph().num_vertices(); node++)
        for(unsigned part = 0; part < instance.getNumberOfPartitions(); part++)
            if(node_in_partition[node][static_cast<int>(part)].Get(COPT_DBLINFO_VALUE) >= .99)
            {
                node_to_partitions[node].push_back(part);
                nonempty_partition_ids.insert(part);
            }

    for(std::vector<unsigned>& chosen_partitions : node_to_partitions)
        if(chosen_partitions.empty())
        {
            std::cout<<"Error: partitioning returned by ILP seems incomplete!"<<std::endl;
            chosen_partitions.push_back(UINT_MAX);
        }
    
    unsigned current_index = 0;
    std::map<unsigned, unsigned> new_part_index;
    for(unsigned part_index : nonempty_partition_ids)
    {
        new_part_index[part_index] = current_index;
        ++current_index;
    }

    for(unsigned node = 0; node < instance.getHypergraph().num_vertices(); node++)
        for(unsigned entry_idx = 0; entry_idx < node_to_partitions[node].size(); ++entry_idx)
            node_to_partitions[node][entry_idx] = new_part_index[node_to_partitions[node][entry_idx]];

    std::cout<<"Hypergraph partitioning ILP best solution value: "<<model.GetDblAttr(COPT_DBLATTR_BESTOBJ)<<", best lower bound: "<<model.GetDblAttr(COPT_DBLATTR_BESTBND)<<std::endl;

    return node_to_partitions;
}

} // namespace osp