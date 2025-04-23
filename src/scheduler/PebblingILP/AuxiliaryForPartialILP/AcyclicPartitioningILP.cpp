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


#include "scheduler/PebblingILP/AuxiliaryForPartialILP/AcyclicPartitioningILP.hpp"
#include <stdexcept>

void AcyclicPartitioningILP::solveILP(const BspInstance &instance) {

    model.SetIntParam(COPT_INTPARAM_LOGTOCONSOLE, 0);

    model.SetDblParam(COPT_DBLPARAM_TIMELIMIT, timeLimitSeconds);
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


std::pair<RETURN_STATUS, std::vector<unsigned> > AcyclicPartitioningILP::computePartitioning(const BspInstance &instance)
{
    if(numberOfParts == 0)
    {
        numberOfParts = std::floor((double) instance.numberOfVertices()  / (double) minPartitionSize);
        std::cout<<"ILP nr parts: "<<numberOfParts<<std::endl;
    }

    setupVariablesConstraintsObjective(instance);

    solveILP(instance);

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {

        return {SUCCESS, returnAssignment(instance)};

    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {

        return {ERROR, std::vector<unsigned>(instance.numberOfVertices(), UINT_MAX)};

    } else {

        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {

            return {BEST_FOUND, returnAssignment(instance)};

        } else {
            return {TIMEOUT, std::vector<unsigned>(instance.numberOfVertices(), UINT_MAX)};
        }
    }
}

void AcyclicPartitioningILP::setupVariablesConstraintsObjective(const BspInstance &instance) {

    // Variables

    node_in_partition = std::vector<VarArray>(instance.numberOfVertices());

    for (unsigned node = 0; node < instance.numberOfVertices(); node++)
        node_in_partition[node] = model.AddVars(numberOfParts, COPT_BINARY, "node_in_partition");

    
    std::map<unsigned, unsigned> node_to_hyperedge_index;
    unsigned numberOfHyperedges = 0;
    for (unsigned node = 0; node < instance.numberOfVertices(); node++)
        if(instance.getComputationalDag().numberOfChildren(node) > 0)
        {
            node_to_hyperedge_index[node] = numberOfHyperedges;
            ++numberOfHyperedges;
        }

    hyperedge_intersects_partition = std::vector<VarArray>(numberOfHyperedges);

    for (unsigned hyperedge = 0; hyperedge < numberOfHyperedges; hyperedge++)
        hyperedge_intersects_partition[hyperedge] = model.AddVars(numberOfParts, COPT_BINARY, "hyperedge_intersects_partition");

    // Constraints

    // each node assigned to exactly one partition
    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        Expr expr;
        for (unsigned part = 0; part < numberOfParts; part++) {

            expr += node_in_partition[node][part];
        }
        model.AddConstr(expr == 1);
    }

    // hyperedge indicators match node variables
    for (unsigned part = 0; part < numberOfParts; part++)
        for (unsigned node = 0; node < instance.numberOfVertices(); node++)
        {
            if(instance.getComputationalDag().numberOfChildren(node) == 0)
                continue;

            model.AddConstr(hyperedge_intersects_partition[node_to_hyperedge_index[node]][part] >= node_in_partition[node][part]);
            for (const auto &succ : instance.getComputationalDag().children(node))
                model.AddConstr(hyperedge_intersects_partition[node_to_hyperedge_index[node]][part] >= node_in_partition[succ][part]);
        }
    
    // partition size constraints
    for (unsigned part = 0; part < numberOfParts; part++)
    {
        Expr expr;
        for (unsigned node = 0; node < instance.numberOfVertices(); node++)
            if(!ignore_sources_for_constraint || is_original_source.empty() || !is_original_source[node])
                expr += node_in_partition[node][part];

        model.AddConstr(expr <= maxPartitionSize);
        model.AddConstr(expr >= minPartitionSize);
    }

    // acyclicity constraints
    for (unsigned from_part = 0; from_part < numberOfParts; from_part++)
        for (unsigned to_part = 0; to_part < from_part; to_part++)
            for (unsigned node = 0; node < instance.numberOfVertices(); node++)
                for (const auto &succ : instance.getComputationalDag().children(node))
                    model.AddConstr(node_in_partition[node][from_part] + node_in_partition[succ][to_part] <= 1);
    

    // set objective
    Expr expr;
    for (unsigned node = 0; node < instance.numberOfVertices(); node++)
        if(instance.getComputationalDag().numberOfChildren(node) > 0)
        {
            expr -= instance.getComputationalDag().nodeCommunicationWeight(node);
            for (unsigned part = 0; part < numberOfParts; part++)
                expr += instance.getComputationalDag().nodeCommunicationWeight(node) * hyperedge_intersects_partition[node_to_hyperedge_index[node]][part];
        }

    model.SetObjective(expr, COPT_MINIMIZE);
             
};

void AcyclicPartitioningILP::WriteSolutionCallback::callback() {

    if (Where() == COPT_CBCONTEXT_MIPSOL && counter < max_number_solution && GetIntInfo(COPT_CBINFO_HASINCUMBENT)) {

        try {

            if (GetDblInfo(COPT_CBINFO_BESTOBJ) < best_obj && 0.0 < GetDblInfo(COPT_CBINFO_BESTBND)) {

                best_obj = GetDblInfo(COPT_CBINFO_BESTOBJ);

                //    auto sched = constructBspScheduleFromCallback();
                //    BspScheduleWriter sched_writer(sched);
                //    sched_writer.write_dot(write_solutions_path_cb + "intmed_sol_" + solution_file_prefix_cb + "_"
                //    +
                //                           std::to_string(counter) + "_schedule.dot");
                counter++;
            }

        } catch (const std::exception &e) {
        }
    }
};

std::vector<unsigned> AcyclicPartitioningILP::returnAssignment(const BspInstance &instance)
{
    std::vector<unsigned> node_to_partition(instance.numberOfVertices(), UINT_MAX);

    std::set<unsigned> nonempty_partition_ids;
    for (unsigned node = 0; node < instance.numberOfVertices(); node++)
        for(unsigned part = 0; part < numberOfParts; part++)
            if(node_in_partition[node][part].Get(COPT_DBLINFO_VALUE) >= .99)
            {
                node_to_partition[node] = part;
                nonempty_partition_ids.insert(part);
            }

    for(unsigned chosen_partition : node_to_partition)
        if(chosen_partition == UINT_MAX)
            std::cout<<"Error: partitioning returned by ILP seems incomplete!"<<std::endl;
    
    unsigned current_index = 0;
    std::map<unsigned, unsigned> new_index;
    for(unsigned part_index : nonempty_partition_ids)
    {
        new_index[part_index] = current_index;
        ++current_index;
    }

    for(unsigned node = 0; node < instance.numberOfVertices(); node++)
        node_to_partition[node] = new_index[node_to_partition[node]];

    std::cout<<"Acyclic partitioning ILP best solution value: "<<model.GetDblAttr(COPT_DBLATTR_BESTOBJ)<<", best lower bound: "<<model.GetDblAttr(COPT_DBLATTR_BESTBND)<<std::endl;

    return node_to_partition;
}

std::pair<RETURN_STATUS, BspSchedule> AcyclicPartitioningILP::computeSchedule(const BspInstance &instance) {
    return {ERROR, BspSchedule()};
}
