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

#include "algorithms/IlpSchedulers/CoptPartialScheduler.hpp"
#include <stdexcept>

std::pair<RETURN_STATUS, BspSchedule> CoptPartialScheduler::constructImprovedSchedule(const BspSchedule& initial_schedule) {

    setupVertexMaps(initial_schedule);

    setupPartialVariablesConstraintsObjective(initial_schedule);

    coptModel.SetDblParam(COPT_DBLPARAM_TIMELIMIT, timeLimitSeconds);
    coptModel.SetIntParam(COPT_INTPARAM_THREADS, 128);

    coptModel.Solve();

    if (coptModel.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {

        std::vector<unsigned> superstep_assignment = initial_schedule.assignedSupersteps();
        std::vector<unsigned> processor_assignment = initial_schedule.assignedProcessors();
        std::map<KeyTriple, unsigned> commSchedule;
        constructBspScheduleFromSolution(initial_schedule, processor_assignment, superstep_assignment, commSchedule);

        return {SUCCESS,
                BspSchedule(initial_schedule.getInstance(), processor_assignment, superstep_assignment, commSchedule)};

    } else {
        return {TIMEOUT, BspSchedule()};
    }
};

RETURN_STATUS CoptPartialScheduler::improveSchedule(BspSchedule& initial_schedule) {

    setupVertexMaps(initial_schedule);

    setupPartialVariablesConstraintsObjective(initial_schedule);

    coptModel.SetDblParam(COPT_DBLPARAM_TIMELIMIT, timeLimitSeconds);
    coptModel.SetIntParam(COPT_INTPARAM_THREADS, 128);

    coptModel.Solve();

    if (coptModel.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {

        std::vector<unsigned> superstep_assignment = initial_schedule.assignedSupersteps();
        std::vector<unsigned> processor_assignment = initial_schedule.assignedProcessors();
        std::map<KeyTriple, unsigned> commSchedule;
        constructBspScheduleFromSolution(initial_schedule, processor_assignment, superstep_assignment, commSchedule);

        initial_schedule.setAssignedProcessors(processor_assignment);
        initial_schedule.setAssignedSupersteps(superstep_assignment);
        initial_schedule.setCommunicationSchedule(commSchedule);

        return SUCCESS;

    } else {
        return TIMEOUT;
    }
}

unsigned CoptPartialScheduler::constructBspScheduleFromSolution(const BspSchedule& initial_schedule, std::vector<unsigned> &processor_assignment,
                                                                std::vector<unsigned> &superstep_assignment,
                                                                std::map<KeyTriple, unsigned> &commSchedule) {

    unsigned number_of_supersteps = 0;

    while (number_of_supersteps < max_number_supersteps &&
           superstep_used_var[number_of_supersteps].Get(COPT_DBLINFO_VALUE) >= .99) {
        number_of_supersteps++;
    }

    // std::vector<unsigned> superstep_assignment = initial_schedule->assignedSupersteps();
    const int offset = number_of_supersteps - (end_superstep - start_superstep + 1);

    for (unsigned node = 0; node < initial_schedule.getInstance().numberOfVertices(); node++) {

        if (superstep_assignment[node] > end_superstep) {
            superstep_assignment[node] = superstep_assignment[node] + offset;
        }
    }

    // std::vector<unsigned> processor_assignment = initial_schedule->assignedProcessors();

    for (unsigned node = 0; node < num_vertices; node++) {

        for (unsigned processor = 0; processor < initial_schedule.getInstance().numberOfProcessors(); processor++) {

            for (unsigned step = 0; step < max_number_supersteps; step++) {

                if (node_to_processor_superstep_var[node][processor][step].Get(COPT_DBLINFO_VALUE) >= .99) {
                    superstep_assignment[vertex_map[node]] = start_superstep + step;
                    processor_assignment[vertex_map[node]] = processor;
                }
            }
        }
    }

    // std::map<KeyTriple, unsigned> commSchedule;

    for (const auto &[key, val] : initial_schedule.getCommunicationSchedule()) {

        if (initial_schedule.assignedSuperstep(get<0>(key)) < start_superstep ||
            initial_schedule.assignedSuperstep(get<0>(key)) > end_superstep) {
            if (backward_source_map.find(get<0>(key)) == backward_source_map.end()) {
                if (val >= start_superstep) {
                    commSchedule[key] = val + offset;

                } else {
                    commSchedule[key] = val;
                }
            }
        }
    }

    for (unsigned int node = 0; node < num_vertices; node++) {

        for (unsigned int p_from = 0; p_from < initial_schedule.getInstance().numberOfProcessors(); p_from++) {
            for (unsigned int p_to = 0; p_to < initial_schedule.getInstance().numberOfProcessors(); p_to++) {
                if (p_from != p_to) {
                    for (unsigned int step = 0; step < max_number_supersteps; step++) {
                        if (comm_processor_to_processor_superstep_node_var[p_from][p_to][step][node].Get(
                                COPT_DBLINFO_VALUE) >= .99) {
                            commSchedule[std::make_tuple(vertex_map[node], p_from, p_to)] = start_superstep + step;
                        }
                    }
                }
            }
        }
    }

    for (unsigned source_node = 0; source_node < num_source_vertices; source_node++) {

        for (unsigned int p_from = 0; p_from < initial_schedule.getInstance().numberOfProcessors(); p_from++) {
            for (unsigned int p_to = 0; p_to < initial_schedule.getInstance().numberOfProcessors(); p_to++) {
                if (p_from != p_to) {
                    for (unsigned int step = 0; step < max_number_supersteps; step++) {
                        if (comm_processor_to_processor_superstep_source_var[p_from][p_to][step][source_node].Get(
                                COPT_DBLINFO_VALUE) >= .99) {
                            commSchedule[std::make_tuple(source_map[source_node], p_from, p_to)] =
                                start_superstep - 1 + step;
                        }
                    }
                }
            }
        }
    }

    for (auto node : target_vertices_set) {

        for (const auto &pred : initial_schedule.getInstance().getComputationalDag().parents(node)) {
            if (backward_source_map.find(pred) != backward_source_map.end() ||
                (initial_schedule.assignedSuperstep(pred) >= start_superstep &&
                 initial_schedule.assignedSuperstep(pred) <= end_superstep)) {

                commSchedule[std::make_tuple(pred, processor_assignment[pred], processor_assignment[node])] =
                    superstep_assignment[node] - 1;
            }
        }
    }

    // BspSchedule schedule(*instance, initial_schedule->numberOfSupersteps() + offset, processor_assignment,
    //                      superstep_assignment, commSchedule);

    return initial_schedule.numberOfSupersteps() + offset;
};

void CoptPartialScheduler::setupVertexMaps(const BspSchedule& initial_schedule) {

    num_source_vertices = 0;
    num_vertices = 0;
    for (unsigned node = 0; node < initial_schedule.getInstance().numberOfVertices(); node++) {

        if (initial_schedule.assignedSuperstep(node) >= start_superstep &&
            initial_schedule.assignedSuperstep(node) <= end_superstep) {

            vertex_map.push_back(node);
            backward_vertex_map[node] = num_vertices;
            num_vertices += 1;

            std::vector<unsigned> source_preds_;
            std::vector<unsigned> preds_;
            for (const auto &pred : initial_schedule.getInstance().getComputationalDag().parents(node)) {

                if (initial_schedule.assignedSuperstep(pred) < start_superstep) {

                    if (backward_source_map.find(pred) == backward_source_map.end()) {
                        source_map.push_back(pred);
                        backward_source_map[pred] = num_source_vertices;
                        source_preds_.push_back(num_source_vertices);
                        num_source_vertices += 1;
                    } else {

                        source_preds_.push_back(backward_source_map[pred]);
                    }

                } else if (initial_schedule.assignedSuperstep(pred) <= end_superstep) {

                    preds_.push_back(pred);

                } else {

                    throw std::invalid_argument("Initial Schedule might be invalid?!");
                }
            }
            source_predecessors.push_back(source_preds_);
            vertex_predecessors.push_back(preds_);

            for (const auto &succ : initial_schedule.getInstance().getComputationalDag().children(node)) {

                if (initial_schedule.assignedSuperstep(succ) > end_superstep) {
                    target_vertices_set.insert(succ);
                }
            }
        }
    }
};

void CoptPartialScheduler::setupPartialVariablesConstraintsObjective(const BspSchedule& initial_schedule) {

    node_to_processor_superstep_var = std::vector<std::vector<VarArray>>(
       initial_schedule.getInstance().numberOfVertices(), std::vector<VarArray>(initial_schedule.getInstance().numberOfProcessors()));

    /*
    Variables
    */
    // variables indicating if superstep is used at all
    superstep_used_var = coptModel.AddVars(max_number_supersteps, COPT_BINARY, "superstep_used");

    // variables for assigments of nodes to processor and superstep
    for (unsigned int node = 0; node < num_vertices; node++) {

        for (unsigned int processor = 0; processor < initial_schedule.getInstance().numberOfProcessors(); processor++) {

            node_to_processor_superstep_var[node][processor] =
                coptModel.AddVars(max_number_supersteps, COPT_BINARY, "node_to_processor_superstep");
        }
    }

    // communicate node from p1 to p2 at superstep

    comm_processor_to_processor_superstep_node_var = std::vector<std::vector<std::vector<VarArray>>>(
        initial_schedule.getInstance().numberOfProcessors(),
        std::vector<std::vector<VarArray>>(initial_schedule.getInstance().numberOfProcessors(),
                                           std::vector<VarArray>(max_number_supersteps)));

    for (unsigned int p1 = 0; p1 < initial_schedule.getInstance().numberOfProcessors(); p1++) {

        for (unsigned int p2 = 0; p2 < initial_schedule.getInstance().numberOfProcessors(); p2++) {
            for (unsigned int step = 0; step < max_number_supersteps; step++) {

                comm_processor_to_processor_superstep_node_var[p1][p2][step] =
                    coptModel.AddVars(num_vertices, COPT_BINARY, "comm_processor_to_processor_superstep_node");
            }
        }
    }

    // communicate nodes in supersteps smaller than start_superstep
    comm_processor_to_processor_superstep_source_var = std::vector<std::vector<std::vector<VarArray>>>(
        initial_schedule.getInstance().numberOfProcessors(),
        std::vector<std::vector<VarArray>>(initial_schedule.getInstance().numberOfProcessors(),
                                           std::vector<VarArray>(max_number_supersteps + 1)));

    for (unsigned int p1 = 0; p1 < initial_schedule.getInstance().numberOfProcessors(); p1++) {

        for (unsigned int p2 = 0; p2 < initial_schedule.getInstance().numberOfProcessors(); p2++) {
            for (unsigned int step = 0; step < max_number_supersteps + 1; step++) {

                comm_processor_to_processor_superstep_source_var[p1][p2][step] =
                    coptModel.AddVars(num_source_vertices, COPT_BINARY, "comm_processor_to_processor_superstep_node");
            }
        }
    }

    max_comm_superstep_var = coptModel.AddVars(max_number_supersteps + 1, COPT_INTEGER, "max_comm_superstep");

    max_work_superstep_var = coptModel.AddVars(max_number_supersteps, COPT_INTEGER, "max_work_superstep");

    /*
    Constraints
      */

    //  use consecutive supersteps starting from 0
    coptModel.AddConstr(superstep_used_var[0] == 1);

    for (unsigned int step = 0; step < max_number_supersteps - 1; step++) {
        coptModel.AddConstr(superstep_used_var[step] >= superstep_used_var[step + 1]);
    }

    // superstep is used at all
    for (unsigned int step = 0; step < max_number_supersteps; step++) {

        Expr expr;
        for (unsigned int node = 0; node < num_vertices; node++) {

            for (unsigned int processor = 0; processor < initial_schedule.getInstance().numberOfProcessors(); processor++) {
                expr += node_to_processor_superstep_var[node][processor][step];
            }
        }
        coptModel.AddConstr(expr <= num_vertices * initial_schedule.getInstance().numberOfProcessors() * superstep_used_var[step]);
    }

    // nodes are assigend depending on whether recomputation is allowed or not
    for (unsigned int node = 0; node < num_vertices; node++) {

        Expr expr;
        for (unsigned int processor = 0; processor < initial_schedule.getInstance().numberOfProcessors(); processor++) {

            for (unsigned int step = 0; step < max_number_supersteps; step++) {
                expr += node_to_processor_superstep_var[node][processor][step];
            }
        }

        //coptModel.AddConstr(initial_schedule.getInstance().allowRecomputation() ? expr >= 1 : expr == 1);
        coptModel.AddConstr(expr == 1);
    }

    // precedence constraint: if task is computed then all of its predecessors must have been present
    for (unsigned int node = 0; node < num_vertices; node++) {

        const unsigned &size_ = vertex_predecessors[node].size() + source_predecessors[node].size();
        if (size_ > 0) {
            for (unsigned int step = 0; step < max_number_supersteps; step++) {
                for (unsigned int processor = 0; processor < initial_schedule.getInstance().numberOfProcessors(); processor++) {

                    Expr expr;
                    for (const unsigned &pred : vertex_predecessors[node]) {

                        expr += comm_processor_to_processor_superstep_node_var[processor][processor][step]
                                                                              [backward_vertex_map[pred]];
                    }
                    for (const unsigned &pred : source_predecessors[node]) {
                        expr += comm_processor_to_processor_superstep_source_var[processor][processor][step + 1][pred];
                    }

                    coptModel.AddConstr(expr >= size_ * node_to_processor_superstep_var[node][processor][step]);
                }
            }
        }
    }

    // combines two constraints: node can only be communicated if it is present; and node is present if it was computed
    // or communicated
    for (unsigned int step = 0; step < max_number_supersteps; step++) {
        for (unsigned int processor = 0; processor < initial_schedule.getInstance().numberOfProcessors(); processor++) {
            for (unsigned int node = 0; node < num_vertices; node++) {

                Expr expr1, expr2;
                if (step > 0) {

                    for (unsigned int p_from = 0; p_from < initial_schedule.getInstance().numberOfProcessors(); p_from++) {
                        expr1 += comm_processor_to_processor_superstep_node_var[p_from][processor][step - 1][node];
                    }
                }

                expr1 += node_to_processor_superstep_var[node][processor][step];

                for (unsigned int p_to = 0; p_to < initial_schedule.getInstance().numberOfProcessors(); p_to++) {
                    expr2 += comm_processor_to_processor_superstep_node_var[processor][p_to][step][node];
                }

                coptModel.AddConstr(initial_schedule.getInstance().numberOfProcessors() * (expr1) >= expr2);
            }
        }
    }

    // combines two constraints: node can only be communicated if it is present; and node is present if it was computed
    // or communicated
    for (unsigned int step = 0; step < max_number_supersteps + 1; step++) {
        for (unsigned int processor = 0; processor < initial_schedule.getInstance().numberOfProcessors(); processor++) {
            for (unsigned int source_node = 0; source_node < num_source_vertices; source_node++) {

                Expr expr1, expr2;
                if (step > 0) {

                    for (unsigned int p_from = 0; p_from < initial_schedule.getInstance().numberOfProcessors(); p_from++) {
                        expr1 +=
                            comm_processor_to_processor_superstep_source_var[p_from][processor][step - 1][source_node];
                    }
                }

                if (processor == initial_schedule.assignedProcessor(source_map[source_node]) && step == 0) {
                    expr1 += 1;
                } else {
                    expr1 += 0;
                }

                for (unsigned int p_to = 0; p_to < initial_schedule.getInstance().numberOfProcessors(); p_to++) {
                    expr2 += comm_processor_to_processor_superstep_source_var[processor][p_to][step][source_node];
                }

                coptModel.AddConstr(initial_schedule.getInstance().numberOfProcessors() * (expr1) >= expr2);
            }
        }
    }

    for (unsigned int step = 0; step < max_number_supersteps; step++) {
        for (unsigned int processor = 0; processor < initial_schedule.getInstance().numberOfProcessors(); processor++) {

            Expr expr;
            for (unsigned int node = 0; node < num_vertices; node++) {
                expr += initial_schedule.getInstance().getComputationalDag().nodeWorkWeight(vertex_map[node]) *
                        node_to_processor_superstep_var[node][processor][step];
            }

            coptModel.AddConstr(max_work_superstep_var[step] >= expr);
        }
    }

    for (unsigned int step = 0; step < max_number_supersteps; step++) {
        for (unsigned int processor = 0; processor < initial_schedule.getInstance().numberOfProcessors(); processor++) {

            Expr expr;
            for (unsigned int node = 0; node < num_vertices; node++) {
                for (unsigned int p_to = 0; p_to < initial_schedule.getInstance().numberOfProcessors(); p_to++) {
                    if (processor != p_to) {
                        expr += initial_schedule.getInstance().getComputationalDag().nodeCommunicationWeight(vertex_map[node]) *
                                initial_schedule.getInstance().sendCosts(processor, p_to) *
                                comm_processor_to_processor_superstep_node_var[processor][p_to][step][node];
                    }
                }
            }

            for (unsigned int source_node = 0; source_node < num_source_vertices; source_node++) {
                for (unsigned int p_to = 0; p_to < initial_schedule.getInstance().numberOfProcessors(); p_to++) {
                    if (processor != p_to) {
                        expr +=
                            initial_schedule.getInstance().getComputationalDag().nodeCommunicationWeight(source_map[source_node]) *
                            initial_schedule.getInstance().sendCosts(processor, p_to) *
                            comm_processor_to_processor_superstep_source_var[processor][p_to][step + 1][source_node];
                    }
                }
            }

            coptModel.AddConstr(max_comm_superstep_var[step + 1] >= expr);
        }
    }

    for (unsigned int step = 0; step < max_number_supersteps; step++) {
        for (unsigned int processor = 0; processor < initial_schedule.getInstance().numberOfProcessors(); processor++) {

            Expr expr;
            for (unsigned int node = 0; node < num_vertices; node++) {
                for (unsigned int p_from = 0; p_from < initial_schedule.getInstance().numberOfProcessors(); p_from++) {
                    if (processor != p_from) {
                        expr += initial_schedule.getInstance().getComputationalDag().nodeCommunicationWeight(vertex_map[node]) *
                                initial_schedule.getInstance().sendCosts(p_from, processor) *
                                comm_processor_to_processor_superstep_node_var[p_from][processor][step][node];
                    }
                }
            }

            for (unsigned int source_node = 0; source_node < num_source_vertices; source_node++) {
                for (unsigned int p_from = 0; p_from < initial_schedule.getInstance().numberOfProcessors(); p_from++) {
                    if (processor != p_from) {
                        expr +=
                            initial_schedule.getInstance().getComputationalDag().nodeCommunicationWeight(source_map[source_node]) *
                            initial_schedule.getInstance().sendCosts(p_from, processor) *
                            comm_processor_to_processor_superstep_source_var[p_from][processor][step + 1][source_node];
                    }
                }
            }

            coptModel.AddConstr(max_comm_superstep_var[step + 1] >= expr);
        }
    }

    for (unsigned int processor = 0; processor < initial_schedule.getInstance().numberOfProcessors(); processor++) {
        Expr expr1;
        for (unsigned int source_node = 0; source_node < num_source_vertices; source_node++) {
            for (unsigned int p_to = 0; p_to < initial_schedule.getInstance().numberOfProcessors(); p_to++) {
                if (processor != p_to) {
                    expr1 += initial_schedule.getInstance().getComputationalDag().nodeCommunicationWeight(source_map[source_node]) *
                             initial_schedule.getInstance().sendCosts(processor, p_to) *
                             comm_processor_to_processor_superstep_source_var[processor][p_to][0][source_node];
                }
            }
        }
        coptModel.AddConstr(max_comm_superstep_var[0] >= expr1);

        Expr expr2;
        for (unsigned int source_node = 0; source_node < num_source_vertices; source_node++) {
            for (unsigned int p_from = 0; p_from < initial_schedule.getInstance().numberOfProcessors(); p_from++) {
                if (processor != p_from) {
                    expr2 += initial_schedule.getInstance().getComputationalDag().nodeCommunicationWeight(source_map[source_node]) *
                             initial_schedule.getInstance().sendCosts(p_from, processor) *
                             comm_processor_to_processor_superstep_source_var[p_from][processor][0][source_node];
                }
            }
        }
        coptModel.AddConstr(max_comm_superstep_var[0] >= expr2);
    }

    /*
    Objective function
    */
    Expr expr;

    for (unsigned int step = 0; step < max_number_supersteps; step++) {
        expr += max_work_superstep_var[step] + initial_schedule.getInstance().communicationCosts() * max_comm_superstep_var[step + 1] +
                initial_schedule.getInstance().synchronisationCosts() * superstep_used_var[step];
    }

    expr += initial_schedule.getInstance().communicationCosts() * max_comm_superstep_var[0];

    if (initial_schedule.numberOfSupersteps() == end_superstep) {
        expr -= initial_schedule.getInstance().synchronisationCosts();
    }

    coptModel.SetObjective(expr, COPT_MINIMIZE);
};
