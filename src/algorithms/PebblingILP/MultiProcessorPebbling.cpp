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

#include "algorithms/PebblingILP/MultiProcessorPebbling.hpp"
#include <stdexcept>

void MultiProcessorPebbling::solveILP(const BspInstance &instance) {

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

std::pair<RETURN_STATUS, BspSchedule> MultiProcessorPebbling::computeSchedule(const BspInstance &instance) {

    max_time = 2 * instance.numberOfVertices();

    setupBaseVariablesConstraints(instance);
    setupSyncPhaseVariablesConstraints(instance);
    setupBspVariablesConstraintsObjective(instance);

    solveILP(instance);

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {

        return {SUCCESS, BspSchedule()};

    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {

        return {ERROR, BspSchedule()};

    } else {

        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {

            return {BEST_FOUND, BspSchedule()};

        } else {
            return {TIMEOUT, BspSchedule()};
        }
    }
};

std::pair<RETURN_STATUS, PebblingStrategy> MultiProcessorPebbling::computeSynchPebbling(const BspInstance &instance) {

    max_time = 2 * instance.numberOfVertices();

    setupBaseVariablesConstraints(instance);
    setupSyncPhaseVariablesConstraints(instance);
    setupSyncObjective(instance);

    solveILP(instance);

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {

        return {SUCCESS, PebblingStrategy()};

    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {

        return {ERROR, PebblingStrategy()};

    } else {

        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {

            return {BEST_FOUND, PebblingStrategy()};

        } else {
            return {TIMEOUT, PebblingStrategy()};
        }
    }
}

std::pair<RETURN_STATUS, PebblingStrategy> MultiProcessorPebbling::computeAsyncPebbling(const BspInstance &instance) {

    max_time = 2 * instance.numberOfVertices();

    setupBaseVariablesConstraints(instance);
    setupAsyncVariablesConstraintsObjective(instance);

    solveILP(instance);

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {

        return {SUCCESS, PebblingStrategy()};

    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {

        return {ERROR, PebblingStrategy()};

    } else {

        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {

            return {BEST_FOUND, PebblingStrategy()};

        } else {
            return {TIMEOUT, PebblingStrategy()};
        }
    }
}

void MultiProcessorPebbling::setupBaseVariablesConstraints(const BspInstance &instance) {

    /*
        Variables
    */
    compute = std::vector<std::vector<VarArray>>(instance.numberOfVertices(),
                                                 std::vector<VarArray>(instance.numberOfProcessors()));

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            compute[node][processor] = model.AddVars(max_time, COPT_BINARY, "node_processor_time");
        }
    }

    send_up = std::vector<std::vector<VarArray>>(instance.numberOfVertices(),
                                                 std::vector<VarArray>(instance.numberOfProcessors()));

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            send_up[node][processor] = model.AddVars(max_time, COPT_BINARY, "send_up");
        }
    }

    send_down = std::vector<std::vector<VarArray>>(instance.numberOfVertices(),
                                                   std::vector<VarArray>(instance.numberOfProcessors()));

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            send_down[node][processor] = model.AddVars(max_time, COPT_BINARY, "send_down");
        }
    }

    has_blue = std::vector<VarArray>(instance.numberOfVertices());

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        has_blue[node] = model.AddVars(max_time, COPT_BINARY, "blue_pebble");
    }

    has_red = std::vector<std::vector<VarArray>>(instance.numberOfVertices(),
                                                 std::vector<VarArray>(instance.numberOfProcessors()));

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            has_red[node][processor] = model.AddVars(max_time, COPT_BINARY, "red_pebble");
        }
    }

    /*
        Constraints
    */
    for (unsigned t = 0; t < max_time; t++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            Expr expr;
            for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

                expr += compute[node][processor][t] + send_up[node][processor][t] + send_down[node][processor][t];
            }
            model.AddConstr(expr <= 1);
        }
    }

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        for (unsigned t = 1; t < max_time; t++) {

            Expr expr;

            for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

                expr += send_up[node][processor][t - 1];
            }
            model.AddConstr(has_blue[node][t] <= has_blue[node][t - 1] + expr);
        }
    }

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            for (unsigned t = 1; t < max_time; t++) {

                model.AddConstr(has_red[node][processor][t] <= has_red[node][processor][t - 1] +
                                                                   send_down[node][processor][t - 1] +
                                                                   compute[node][processor][t - 1]);
            }
        }
    }

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            for (unsigned t = 0; t < max_time; t++) {

                for (const auto &source : instance.getComputationalDag().parents(node)) {

                    model.AddConstr(compute[node][processor][t] <= has_red[source][processor][t]);
                }
            }
        }
    }

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            for (unsigned t = 0; t < max_time; t++) {

                model.AddConstr(send_up[node][processor][t] <= has_red[node][processor][t]);
            }
        }
    }

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            for (unsigned t = 0; t < max_time; t++) {

                model.AddConstr(send_down[node][processor][t] <= has_blue[node][t]);
            }
        }
    }

    for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

        for (unsigned t = 0; t < max_time; t++) {
            Expr expr;
            for (unsigned node = 0; node < instance.numberOfVertices(); node++) {
                expr += has_red[node][processor][t];
            }

            model.AddConstr(expr <= memory_bound);
        }
    }

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            model.AddConstr(has_red[node][processor][0] == 0);
        }
    }

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        model.AddConstr(has_blue[node][0] == 0);
    }

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        if (instance.getComputationalDag().numberOfChildren(node) == 0) {

            for (unsigned t = 0; t < max_time; t++) {

                model.AddConstr(has_blue[node][t] >= 1);
            }
        }
    }
};

void MultiProcessorPebbling::setupSyncPhaseVariablesConstraints(const BspInstance &instance) {

    comp_phase = model.AddVars(max_time, COPT_BINARY, "comp_phase");

    send_up_phase = model.AddVars(max_time, COPT_BINARY, "send_up_phase");

    send_down_phase = model.AddVars(max_time, COPT_BINARY, "send_down_phase");

    const unsigned M = instance.numberOfProcessors() * instance.numberOfVertices();

    for (unsigned t = 0; t < max_time; t++) {

        Expr expr_comp, expr_send_up, expr_send_down;
        for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

            for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
                expr_comp += compute[node][processor][t];
                expr_send_up += send_up[node][processor][t];
                expr_send_down += send_down[node][processor][t];
            }
        }

        model.AddConstr(M * comp_phase[t] >= expr_comp);
        model.AddConstr(M * send_up_phase[t] >= expr_send_up);
        model.AddConstr(M * send_down_phase[t] >= expr_send_down);

        model.AddConstr(comp_phase[t] + send_up_phase[t] + send_down_phase[t] <= 1);
    }
};

void MultiProcessorPebbling::setupBspVariablesConstraintsObjective(const BspInstance &instance) {

    comp_phase_ends = model.AddVars(max_time, COPT_BINARY, "comp_phase_ends");

    comm_phase_ends = model.AddVars(max_time, COPT_BINARY, "comm_phase_ends");

    VarArray work_induced = model.AddVars(max_time, COPT_INTEGER, "work_induced");

    VarArray comm_induced = model.AddVars(max_time, COPT_INTEGER, "comm_induced");

    std::vector<VarArray> work_step_until(instance.numberOfProcessors());

    std::vector<VarArray> send_up_step_until(instance.numberOfProcessors());

    std::vector<VarArray> send_down_step_until(instance.numberOfProcessors());

    for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
        work_step_until[processor] = model.AddVars(max_time, COPT_INTEGER, "work_step_until");
        send_up_step_until[processor] = model.AddVars(max_time, COPT_INTEGER, "send_up_step_until");
        send_down_step_until[processor] = model.AddVars(max_time, COPT_INTEGER, "send_up_step_until");
    }

    for (unsigned t = 0; t < max_time - 1; t++) {

        model.AddConstr(comp_phase_ends[t] >= comp_phase[t] + send_down_phase[t + 1] + send_up_phase[t + 1] - 1);

        model.AddConstr(comm_phase_ends[t] >= comp_phase[t + 1] + send_down_phase[t] + send_up_phase[t] - 1);
    }

    const unsigned M = instance.numberOfProcessors() * instance.getComputationalDag().sumOfVerticesWorkWeights(
                                                           instance.getComputationalDag().vertices().begin(),
                                                           instance.getComputationalDag().vertices().end());

    for (unsigned t = 1; t < max_time; t++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            Expr expr_work;
            Expr expr_send_up;
            Expr expr_send_down;
            for (unsigned node = 0; node < instance.numberOfVertices(); node++) {
                expr_work += instance.getComputationalDag().nodeWorkWeight(node) * compute[node][processor][t];
                expr_send_up += instance.communicationCosts() * send_up[node][processor][t];
                expr_send_down += instance.communicationCosts() * send_down[node][processor][t];
            }

            model.AddConstr(M * (1 - comm_phase_ends[t]) + work_step_until[processor][t] >=
                            work_step_until[processor][t - 1] + expr_work);

            model.AddConstr(M * (1 - comp_phase_ends[t]) + send_up_step_until[processor][t] >=
                            send_up_step_until[processor][t - 1] + expr_send_up);

            model.AddConstr(M * (1 - comp_phase_ends[t]) + send_down_step_until[processor][t] >=
                            send_down_step_until[processor][t - 1] + expr_send_down);

            model.AddConstr(work_induced[t] >= work_step_until[processor][t] - M * (1 - comp_phase_ends[t]));
            model.AddConstr(comm_induced[t] >= send_down_step_until[processor][t] - M * (1 - comm_phase_ends[t]));
            model.AddConstr(comm_induced[t] >= send_up_step_until[processor][t] - M * (1 - comm_phase_ends[t]));
        }
    }

    // t = 0
    for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

        Expr expr_work;
        Expr expr_send_up;
        Expr expr_send_down;
        for (unsigned node = 0; node < instance.numberOfVertices(); node++) {
            expr_work += instance.getComputationalDag().nodeWorkWeight(node) * compute[node][processor][0];
            expr_send_up += instance.communicationCosts() * send_up[node][processor][0];
            expr_send_down += instance.communicationCosts() * send_down[node][processor][0];
        }

        model.AddConstr(M * (1 - comm_phase_ends[0]) + work_step_until[processor][0] >= expr_work);

        model.AddConstr(M * (1 - comp_phase_ends[0]) + send_up_step_until[processor][0] >= expr_send_up);

        model.AddConstr(M * (1 - comp_phase_ends[0]) + send_down_step_until[processor][0] >= expr_send_down);

        model.AddConstr(work_induced[0] >= work_step_until[processor][0] - M * (1 - comp_phase_ends[0]));
        model.AddConstr(comm_induced[0] >= send_down_step_until[processor][0] - M * (1 - comm_phase_ends[0]));
        model.AddConstr(comm_induced[0] >= send_up_step_until[processor][0] - M * (1 - comm_phase_ends[0]));
    }

    /*
    Objective
*/

    Expr expr;
    for (unsigned t = 0; t < max_time; t++) {
        expr += work_induced[t] + instance.synchronisationCosts() * comm_phase_ends[t] +
                instance.communicationCosts() * comm_induced[t];
    }

    model.SetObjective(expr, COPT_MINIMIZE);
};

void MultiProcessorPebbling::setupSyncObjective(const BspInstance &instance) {

    Expr expr;
    for (unsigned t = 0; t < max_time; t++) {
        expr += comp_phase[t] + instance.communicationCosts() * send_up_phase[t] +
                instance.communicationCosts() * send_down_phase[t];
    }

    model.SetObjective(expr, COPT_MINIMIZE);
}

void MultiProcessorPebbling::setupAsyncVariablesConstraintsObjective(const BspInstance &instance) {

    std::vector<VarArray> finish_times(instance.numberOfProcessors());

    for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
        finish_times[processor] = model.AddVars(max_time, COPT_INTEGER, "finish_times");
    }

    Var makespan = model.AddVar(0, COPT_INFINITY, 1, COPT_INTEGER, "makespan");

    VarArray gets_blue = model.AddVars(instance.numberOfVertices(), COPT_INTEGER, "gets_blue");

    const unsigned M = instance.numberOfProcessors() * instance.numberOfVertices();

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        for (unsigned t = 0; t < max_time; t++) {

            for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

                model.AddConstr(gets_blue[node] >= finish_times[processor][t] - (1 - send_up[node][processor][t]) * M);
                model.AddConstr(gets_blue[node] <=
                                finish_times[processor][t] + (1 - send_down[node][processor][t]) * M +
                                    instance.communicationCosts() *
                                        instance.getComputationalDag().nodeCommunicationWeight(node));
            }
        }
    }

    // makespan constraint
    for (unsigned t = 0; t < max_time; t++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            model.AddConstr(makespan >= finish_times[processor][t]);
        }
    }

    // t = 0
    for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

        Expr expr;
        for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

            expr += instance.getComputationalDag().nodeWorkWeight(node) * compute[node][processor][0] +
                    instance.communicationCosts() * instance.getComputationalDag().nodeCommunicationWeight(node) *
                        (send_up[node][processor][0] + send_down[node][processor][0]);
        }

        model.AddConstr(finish_times[processor][0] >= expr);
    }

    for (unsigned t = 1; t < max_time; t++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            Expr expr;
            for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

                expr += instance.getComputationalDag().nodeWorkWeight(node) * compute[node][processor][t] +
                        instance.communicationCosts() * instance.getComputationalDag().nodeCommunicationWeight(node) *
                            (send_up[node][processor][t] + send_down[node][processor][t]);
            }

            model.AddConstr(finish_times[processor][t] >= finish_times[processor][t - 1] + expr);
        }
    }

    /*
    Objective
      */

    model.SetObjective(makespan, COPT_MINIMIZE);
}

PebblingStrategy MultiProcessorPebbling::constructStrategyFromSolution(const BspInstance &instance) {

    return PebblingStrategy();
}

void MultiProcessorPebbling::WriteSolutionCallback::callback() {

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
