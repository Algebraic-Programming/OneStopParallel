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

#include "algorithms/IlpSchedulers/CoptCommunicationScheduleOptimizer.hpp"


std::map<KeyTriple, unsigned> CoptCommunicationScheduleOptimizer::computeCommunicationSchedule(const IBspSchedule &sched) {

    setupVariablesConstraintsObjective(sched, false);

    coptModel.SetDblParam(COPT_DBLPARAM_TIMELIMIT, timeLimitSeconds);
    coptModel.SetIntParam(COPT_INTPARAM_THREADS, 128);

    coptModel.Solve();

    if (coptModel.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {

        return constructCommScheduleFromSolution(sched);

    } else {

        return std::map<KeyTriple, unsigned>();
    }
}

std::pair<RETURN_STATUS, BspSchedule> CoptCommunicationScheduleOptimizer::constructImprovedSchedule(const BspSchedule& initial_schedule) {

    setupVariablesConstraintsObjective(initial_schedule, true);

    coptModel.SetDblParam(COPT_DBLPARAM_TIMELIMIT, timeLimitSeconds);
    coptModel.SetIntParam(COPT_INTPARAM_THREADS, 128);

    coptModel.Solve();

    if (coptModel.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {

        VectorSchedule schedule(initial_schedule);
        std::map<KeyTriple, unsigned int> cs;

        if (numberOfSuperstepsChanged(initial_schedule)) {
            cs = reduceNumberOfSuperstepsAndAddCommScheduleConFromSolution(schedule, initial_schedule);

        } else {
            cs = constructCommScheduleFromSolution(initial_schedule);

        }

        return {SUCCESS, schedule.buildBspSchedule(cs)};
    } else {
        return {TIMEOUT, BspSchedule()};
    }
}

std::map<KeyTriple, unsigned int> CoptCommunicationScheduleOptimizer::reduceNumberOfSuperstepsAndAddCommScheduleConFromSolution(
    VectorSchedule &schedule, const BspSchedule &initial_schedule) {

    std::map<KeyTriple, unsigned int> cs = constructCommScheduleFromSolution(initial_schedule);
    unsigned num_reduction = 0;
    for (unsigned step = 0; step < initial_schedule.numberOfSupersteps() - 1; step++) {

        if (superstep_used_var[step].Get(COPT_DBLINFO_VALUE) <= 0.01) {
            num_reduction += 1;
            for (unsigned node = 0; node < initial_schedule.getInstance().numberOfVertices(); node++) {

                if (schedule.assignedSuperstep(node) > step) {
                    schedule.setAssignedSuperstep(node, schedule.assignedSuperstep(node) - 1);
                }
            }

            for (auto const &[key, val] : cs) {
                if (val > step) {

                    cs[key] = val - 1;
                }
            }
        }
    }
    
    schedule.number_of_supersteps = schedule.number_of_supersteps - num_reduction;
    

    return cs;
}

bool CoptCommunicationScheduleOptimizer::numberOfSuperstepsChanged(const BspSchedule& initial_schedule) {

    for (unsigned step = 0; step < initial_schedule.numberOfSupersteps() - 1; step++) {

        if (superstep_used_var[step].Get(COPT_DBLINFO_VALUE) <= 0.01)
            return true;
    }
    return false;
}

RETURN_STATUS CoptCommunicationScheduleOptimizer::improveSchedule(BspSchedule& initial_schedule) {

    setupVariablesConstraintsObjective(initial_schedule, true);

    coptModel.Solve();

    if (coptModel.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
        auto cs = constructCommScheduleFromSolution(initial_schedule);
        initial_schedule.setCommunicationSchedule(cs);
        return SUCCESS;
    } else {
        return TIMEOUT;
    }
}

std::map<KeyTriple, unsigned int> CoptCommunicationScheduleOptimizer::constructCommScheduleFromSolution(const IBspSchedule& initial_schedule) {

    std::map<KeyTriple, unsigned int> cs;


    for (unsigned int node = 0; node < initial_schedule.getInstance().numberOfVertices(); node++) {

        for (unsigned int p_from = 0; p_from < initial_schedule.getInstance().numberOfProcessors(); p_from++) {
            for (unsigned int p_to = 0; p_to < initial_schedule.getInstance().numberOfProcessors(); p_to++) {
                if (p_from != p_to) {
                    for (unsigned int step = 0; step < initial_schedule.numberOfSupersteps(); step++) {
                        if (comm_processor_to_processor_superstep_node_var[p_from][p_to][step][node].Get(
                                COPT_DBLINFO_VALUE) >= .99) {
                            cs[std::make_tuple(node, p_from, p_to)] = step;
                        }
                    }
                }
            }
        }
    }

    return cs;
}

void CoptCommunicationScheduleOptimizer::setupVariablesConstraintsObjective(const IBspSchedule& initial_schedule, bool num_supersteps_can_change) {

    const unsigned &max_number_supersteps = initial_schedule.numberOfSupersteps();

    // variables indicating if superstep is used at all
    if (num_supersteps_can_change) {
        superstep_used_var = coptModel.AddVars(max_number_supersteps, COPT_BINARY, "superstep_used");
    }

    max_comm_superstep_var = coptModel.AddVars(max_number_supersteps, COPT_INTEGER, "max_comm_superstep");

    // communicate node from p1 to p2 at superstep

    comm_processor_to_processor_superstep_node_var = std::vector<std::vector<std::vector<VarArray>>>(
        initial_schedule.getInstance().numberOfProcessors(),
        std::vector<std::vector<VarArray>>(initial_schedule.getInstance().numberOfProcessors(),
                                           std::vector<VarArray>(max_number_supersteps)));

    for (unsigned int p1 = 0; p1 < initial_schedule.getInstance().numberOfProcessors(); p1++) {

        for (unsigned int p2 = 0; p2 < initial_schedule.getInstance().numberOfProcessors(); p2++) {

            for (unsigned int step = 0; step < max_number_supersteps; step++) {

                comm_processor_to_processor_superstep_node_var[p1][p2][step] = coptModel.AddVars(
                    initial_schedule.getInstance().numberOfVertices(), COPT_BINARY, "comm_processor_to_processor_superstep_node");
            }
        }
    }

    if (num_supersteps_can_change) {
        unsigned M = initial_schedule.getInstance().numberOfProcessors() * initial_schedule.getInstance().numberOfProcessors() * initial_schedule.getInstance().numberOfVertices();
        for (unsigned int step = 0; step < max_number_supersteps; step++) {

            Expr expr;

            for (unsigned int p1 = 0; p1 < initial_schedule.getInstance().numberOfProcessors(); p1++) {

                for (unsigned int p2 = 0; p2 < initial_schedule.getInstance().numberOfProcessors(); p2++) {

                    if (p1 != p2) {
                        for (unsigned node = 0; node < initial_schedule.getInstance().numberOfVertices(); node++) {

                            expr += comm_processor_to_processor_superstep_node_var[p1][p2][step][node];
                        }
                    }
                }
            }

            coptModel.AddConstr(expr <= M * superstep_used_var[step]);
        }
    }
    // precedence constraint: if task is computed then all of its predecessors must have been present
    // and vertex is present where it was computed
    for (unsigned int node = 0; node < initial_schedule.getInstance().numberOfVertices(); node++) {

        const unsigned &processor = initial_schedule.assignedProcessor(node);
        const unsigned &super_step = initial_schedule.assignedSuperstep(node);
        Expr expr;
        unsigned num_com_edges = 0;
        for (const auto &pred : initial_schedule.getInstance().getComputationalDag().parents(node)) {

            if (initial_schedule.assignedProcessor(node) != initial_schedule.assignedProcessor(pred)) {
                num_com_edges += 1;
                expr += comm_processor_to_processor_superstep_node_var[processor][processor][super_step][pred];

                coptModel.AddConstr(
                    comm_processor_to_processor_superstep_node_var[initial_schedule.assignedProcessor(pred)]
                                                                  [initial_schedule.assignedProcessor(pred)]
                                                                  [initial_schedule.assignedSuperstep(pred)][pred] ==
                    1);
            }
        }

        if (num_com_edges > 0)
            coptModel.AddConstr(expr >= num_com_edges);
    }

    // combines two constraints: node can only be communicated if it is present; and node is present if it was computed
    // or communicated
    for (unsigned int step = 0; step < max_number_supersteps; step++) {
        for (unsigned int processor = 0; processor < initial_schedule.getInstance().numberOfProcessors(); processor++) {
            for (unsigned int node = 0; node < initial_schedule.getInstance().numberOfVertices(); node++) {

                Expr expr1, expr2;
                if (step > 0) {

                    for (unsigned int p_from = 0; p_from < initial_schedule.getInstance().numberOfProcessors(); p_from++) {
                        expr1 += comm_processor_to_processor_superstep_node_var[p_from][processor][step - 1][node];
                    }
                }

                if (processor == initial_schedule.assignedProcessor(node) &&
                    step == initial_schedule.assignedSuperstep(node)) {
                    expr1 += 1;
                } else {
                    expr1 += 0;
                }

                for (unsigned int p_to = 0; p_to < initial_schedule.getInstance().numberOfProcessors(); p_to++) {
                    expr2 += comm_processor_to_processor_superstep_node_var[processor][p_to][step][node];
                }

                coptModel.AddConstr(initial_schedule.getInstance().numberOfProcessors() * (expr1) >= expr2);
            }
        }
    }

    for (unsigned int step = 0; step < max_number_supersteps; step++) {
        for (unsigned int processor = 0; processor < initial_schedule.getInstance().numberOfProcessors(); processor++) {

            Expr expr;
            for (unsigned int node = 0; node < initial_schedule.getInstance().numberOfVertices(); node++) {
                for (unsigned int p_to = 0; p_to < initial_schedule.getInstance().numberOfProcessors(); p_to++) {
                    if (processor != p_to) {
                        expr += initial_schedule.getInstance().getComputationalDag().nodeCommunicationWeight(node) *
                                initial_schedule.getInstance().sendCosts(processor, p_to) *
                                comm_processor_to_processor_superstep_node_var[processor][p_to][step][node];
                    }
                }
            }

            coptModel.AddConstr(max_comm_superstep_var[step] >= expr);
        }
    }

    for (unsigned int step = 0; step < max_number_supersteps; step++) {
        for (unsigned int processor = 0; processor < initial_schedule.getInstance().numberOfProcessors(); processor++) {

            Expr expr;
            for (unsigned int node = 0; node < initial_schedule.getInstance().numberOfVertices(); node++) {
                for (unsigned int p_from = 0; p_from < initial_schedule.getInstance().numberOfProcessors(); p_from++) {
                    if (processor != p_from) {
                        expr += initial_schedule.getInstance().getComputationalDag().nodeCommunicationWeight(node) *
                                initial_schedule.getInstance().sendCosts(p_from, processor) *
                                comm_processor_to_processor_superstep_node_var[p_from][processor][step][node];
                    }
                }
            }

            coptModel.AddConstr(max_comm_superstep_var[step] >= expr);
        }
    }

    /*
    Objective function
      */
    Expr expr;

    if (num_supersteps_can_change) {

        for (unsigned int step = 0; step < max_number_supersteps; step++) {
            expr += initial_schedule.getInstance().communicationCosts() * max_comm_superstep_var[step] +
                    initial_schedule.getInstance().synchronisationCosts() * superstep_used_var[step];
        }
    } else {

        for (unsigned int step = 0; step < max_number_supersteps; step++) {
            expr += initial_schedule.getInstance().communicationCosts() * max_comm_superstep_var[step];
        }
    }
    coptModel.SetObjective(expr - initial_schedule.getInstance().synchronisationCosts(), COPT_MINIMIZE);
}
