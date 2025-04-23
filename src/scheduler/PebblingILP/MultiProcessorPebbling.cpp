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

#include "scheduler/PebblingILP/MultiProcessorPebbling.hpp"
#include <stdexcept>

void MultiProcessorPebbling::solveILP(const BspInstance &instance) {

    if(!verbose)
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

std::pair<RETURN_STATUS, BspSchedule> MultiProcessorPebbling::computeSchedule(const BspInstance &instance) {

    if(max_time == 0)
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

std::pair<RETURN_STATUS, BspMemSchedule> MultiProcessorPebbling::computeSynchPebbling(const BspInstance &instance) {

    if(max_time == 0)
        max_time = 2 * instance.numberOfVertices();
    
    mergeSteps = false;

    setupBaseVariablesConstraints(instance);
    setupSyncPhaseVariablesConstraints(instance);
    setupSyncObjective(instance);

    solveILP(instance);

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {

        return {SUCCESS, constructMemScheduleFromSolution(instance)};

    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {

        return {ERROR, BspMemSchedule()};

    } else {

        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {

            return {BEST_FOUND, constructMemScheduleFromSolution(instance)};

        } else {
            return {TIMEOUT, BspMemSchedule()};
        }
    }
}

std::pair<RETURN_STATUS, BspMemSchedule> MultiProcessorPebbling::computePebbling(const BspInstance &instance, bool use_async) {

    if(max_time == 0)
        max_time = 2 * instance.numberOfVertices();

    synchronous = !use_async;

    setupBaseVariablesConstraints(instance);
    if(synchronous)
    {
        setupSyncPhaseVariablesConstraints(instance);
        setupBspVariablesConstraintsObjective(instance);
    }
    else
        setupAsyncVariablesConstraintsObjective(instance);

    solveILP(instance);

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {

        BspMemSchedule schedule = constructMemScheduleFromSolution(instance);
        return {schedule.isValid() ? SUCCESS : ERROR, schedule};

    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {

        return {ERROR, BspMemSchedule()};

    } else {

        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {

            BspMemSchedule schedule = constructMemScheduleFromSolution(instance);
            return {schedule.isValid() ? SUCCESS : ERROR, schedule};

        } else {
            return {TIMEOUT, BspMemSchedule()};
        }
    }
}

std::pair<RETURN_STATUS, BspMemSchedule> MultiProcessorPebbling::computePebblingWithInitialSolution(const BspInstance &instance, const BspMemSchedule& initial_solution, bool use_async)
{
    std::vector<std::vector<std::vector<unsigned> > > computeSteps;
    std::vector<std::vector<std::vector<unsigned> > > sendUpSteps;
    std::vector<std::vector<std::vector<unsigned> > > sendDownSteps;
    std::vector<std::vector<std::vector<unsigned> > > nodesEvictedAfterStep;

    synchronous = !use_async;
    
    initial_solution.getDataForMultiprocessorPebbling(computeSteps, sendUpSteps, sendDownSteps, nodesEvictedAfterStep);

    max_time = computeMaxTimeForInitialSolution(instance, computeSteps, sendUpSteps, sendDownSteps, nodesEvictedAfterStep);

    if(verbose)
        std::cout<<"Max time set at "<<max_time<<std::endl;

    setupBaseVariablesConstraints(instance);
    if(synchronous)
    {
        setupSyncPhaseVariablesConstraints(instance);
        setupBspVariablesConstraintsObjective(instance);
    }
    else
        setupAsyncVariablesConstraintsObjective(instance);

    setInitialSolution(instance, computeSteps, sendUpSteps, sendDownSteps, nodesEvictedAfterStep);

    if(verbose)
        std::cout<<"Initial solution set."<<std::endl;

    solveILP(instance);

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {

        BspMemSchedule schedule = constructMemScheduleFromSolution(instance);
        return {schedule.isValid() ? SUCCESS : ERROR, schedule};

    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {

        return {ERROR, BspMemSchedule()};

    } else {

        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {

            BspMemSchedule schedule = constructMemScheduleFromSolution(instance);
            return {schedule.isValid() ? SUCCESS : ERROR, schedule};

        } else {
            return {TIMEOUT, BspMemSchedule()};
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

    compute_exists.resize(instance.numberOfVertices(),
                            std::vector<std::vector<bool>>(instance.numberOfProcessors(), std::vector<bool>(max_time, true)));

    send_up = std::vector<std::vector<VarArray>>(instance.numberOfVertices(),
                                                 std::vector<VarArray>(instance.numberOfProcessors()));

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            send_up[node][processor] = model.AddVars(max_time, COPT_BINARY, "send_up");
        }
    }

    send_up_exists.resize(instance.numberOfVertices(),
                            std::vector<std::vector<bool>>(instance.numberOfProcessors(), std::vector<bool>(max_time, true)));

    send_down = std::vector<std::vector<VarArray>>(instance.numberOfVertices(),
                                                   std::vector<VarArray>(instance.numberOfProcessors()));

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            send_down[node][processor] = model.AddVars(max_time, COPT_BINARY, "send_down");
        }
    }

    send_down_exists.resize(instance.numberOfVertices(),
                            std::vector<std::vector<bool>>(instance.numberOfProcessors(), std::vector<bool>(max_time, true)));

    has_blue = std::vector<VarArray>(instance.numberOfVertices());

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        has_blue[node] = model.AddVars(max_time, COPT_BINARY, "blue_pebble");
    }

    has_blue_exists.resize(instance.numberOfVertices(), std::vector<bool>(max_time, true));

    has_red = std::vector<std::vector<VarArray>>(instance.numberOfVertices(),
                                                 std::vector<VarArray>(instance.numberOfProcessors()));

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            has_red[node][processor] = model.AddVars(max_time, COPT_BINARY, "red_pebble");
        }
    }

    /*
        Invalidate variables based on various factors (node types, input loading, step type restriction)
    */

   for (unsigned node = 0; node < instance.numberOfVertices(); node++)
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++)
            if(!instance.isCompatible(node, processor))
                for (unsigned t = 0; t < max_time; t++)
                {
                    compute_exists[node][processor][t] = false;
                    send_up_exists[node][processor][t] = false;
                }
    
    // restrict source nodes if they need to be loaded
    if(need_to_load_inputs)
        for (unsigned node = 0; node < instance.numberOfVertices(); node++)
            if (instance.getComputationalDag().numberOfParents(node) == 0)
                for (unsigned t = 0; t < max_time; t++)
                {
                    for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++)
                    {
                        compute_exists[node][processor][t] = false;
                        send_up_exists[node][processor][t] = false;
                    }
                    has_blue_exists[node][t] = false;
                }

    // restrict step types for simpler ILP
    if(restrict_step_types)
        for (unsigned t = 0; t < max_time; t++)
        {
            bool this_is_a_comm_step = (t % (compute_steps_per_cycle + 2) == compute_steps_per_cycle + 1);
            if(!need_to_load_inputs && t % (compute_steps_per_cycle + 2) == compute_steps_per_cycle)
                this_is_a_comm_step = true;
            if(need_to_load_inputs && t % (compute_steps_per_cycle + 2) == 0)
                this_is_a_comm_step = true;
            if(this_is_a_comm_step)
                for (unsigned node = 0; node < instance.numberOfVertices(); node++)
                    for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++)
                        compute_exists[node][processor][t] = false;
            else
                for (unsigned node = 0; node < instance.numberOfVertices(); node++)
                    for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++)
                    {
                        send_up_exists[node][processor][t] = false;
                        send_down_exists[node][processor][t] = false;
                    }
        }

    /*
        Constraints
    */

    if(!mergeSteps)
    {
        for (unsigned t = 0; t < max_time; t++) {

            for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

                Expr expr;
                for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

                    if(compute_exists[node][processor][t])
                        expr += compute[node][processor][t];
                    if(send_up_exists[node][processor][t])
                        expr += send_up[node][processor][t];
                    if(send_down_exists[node][processor][t])
                        expr += send_down[node][processor][t];
                }
                model.AddConstr(expr <= 1);
            }
        }
    }
    else
    {
        //extra variables to indicate step types in step merging
        std::vector<VarArray> comp_step_on_proc = std::vector<VarArray>(instance.numberOfProcessors());
        std::vector<VarArray> comm_step_on_proc = std::vector<VarArray>(instance.numberOfProcessors());

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            comp_step_on_proc[processor] = model.AddVars(max_time, COPT_BINARY, "comp_step_on_proc");
            comm_step_on_proc[processor] = model.AddVars(max_time, COPT_BINARY, "comm_step_on_proc");
        }

        const unsigned M = instance.numberOfVertices();

        for (unsigned t = 0; t < max_time; t++)
            for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++)
            {
                Expr expr_comp, expr_comm;
                for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

                    if(compute_exists[node][processor][t])
                        expr_comp += compute[node][processor][t];
                    if(send_up_exists[node][processor][t])
                        expr_comm += send_up[node][processor][t];
                    if(send_down_exists[node][processor][t])
                        expr_comm += send_down[node][processor][t];
                }

                model.AddConstr(M * comp_step_on_proc[processor][t] >= expr_comp);
                model.AddConstr(2 * M * comm_step_on_proc[processor][t] >= expr_comm);

                model.AddConstr(comp_step_on_proc[processor][t] + comm_step_on_proc[processor][t] <= 1);
            }
    }

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        for (unsigned t = 1; t < max_time; t++) {

            if(!has_blue_exists[node][t])
                continue;

            Expr expr;

            for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

                if(send_up_exists[node][processor][t-1])
                    expr += send_up[node][processor][t - 1];
            }
            model.AddConstr(has_blue[node][t] <= has_blue[node][t - 1] + expr);
        }
    }

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            for (unsigned t = 1; t < max_time; t++) {

                Expr expr;

                if(compute_exists[node][processor][t-1])
                    expr += compute[node][processor][t - 1];

                if(send_down_exists[node][processor][t-1])
                    expr += send_down[node][processor][t - 1];

                model.AddConstr(has_red[node][processor][t] <= has_red[node][processor][t - 1] + expr);
            }
        }
    }

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            for (unsigned t = 0; t < max_time; t++) {

                if(!compute_exists[node][processor][t])
                    continue;

                for (const auto &source : instance.getComputationalDag().parents(node)) {

                    if(!mergeSteps || !compute_exists[source][processor][t])
                        model.AddConstr(compute[node][processor][t] <= has_red[source][processor][t]);
                    else
                        model.AddConstr(compute[node][processor][t] <= has_red[source][processor][t] + compute[source][processor][t]);
                }
            }
        }
    }

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            for (unsigned t = 0; t < max_time; t++) {

                if(send_up_exists[node][processor][t])
                    model.AddConstr(send_up[node][processor][t] <= has_red[node][processor][t]);
            }
        }
    }

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            for (unsigned t = 0; t < max_time; t++) {

                if(send_down_exists[node][processor][t] && has_blue_exists[node][t])
                    model.AddConstr(send_down[node][processor][t] <= has_blue[node][t]);
            }
        }
    }

    for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

        for (unsigned t = 0; t < max_time; t++) {
            Expr expr;
            for (unsigned node = 0; node < instance.numberOfVertices(); node++) {
                expr += has_red[node][processor][t] * instance.getComputationalDag().nodeMemoryWeight(node);
                if(!slidingPebbles && compute_exists[node][processor][t])
                    expr += compute[node][processor][t] * instance.getComputationalDag().nodeMemoryWeight(node);
            }

            model.AddConstr(expr <= instance.getArchitecture().memoryBound(processor));
        }
    }

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            if(has_red_in_beginning.empty() || has_red_in_beginning[processor].find(node) == has_red_in_beginning[processor].end())
                model.AddConstr(has_red[node][processor][0] == 0);
            else
                model.AddConstr(has_red[node][processor][0] == 1);
        }
    }

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {
        if(!need_to_load_inputs || instance.getComputationalDag().numberOfParents(node) > 0)
            model.AddConstr(has_blue[node][0] == 0);
    }

    if(needs_blue_at_end.empty()) // default case: blue pebbles required on sinks at the end
    {
        for (unsigned node = 0; node < instance.numberOfVertices(); node++)
            if (instance.getComputationalDag().numberOfChildren(node) == 0 && has_blue_exists[node][max_time-1])
                model.AddConstr(has_blue[node][max_time-1] == 1);
    }
    else // otherwise: specified set of nodes that need blue at the end
    {
        for (unsigned node : needs_blue_at_end)
            if(has_blue_exists[node][max_time-1])
                model.AddConstr(has_blue[node][max_time-1] == 1);
    }
    
    // disable recomputation if needed
    if(!allows_recomputation)
        for (unsigned node = 0; node < instance.numberOfVertices(); node++)
        {
            Expr expr;
            for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++)
                for (unsigned t = 0; t < max_time; t++)
                    if(compute_exists[node][processor][t])
                        expr += compute[node][processor][t];

            model.AddConstr(expr <= 1);
        }
       
};

void MultiProcessorPebbling::setupSyncPhaseVariablesConstraints(const BspInstance &instance) {

    comp_phase = model.AddVars(max_time, COPT_BINARY, "comp_phase");

    if(mergeSteps)
        comm_phase = model.AddVars(max_time, COPT_BINARY, "comm_phase");
    else
    {
        send_up_phase = model.AddVars(max_time, COPT_BINARY, "send_up_phase");
        send_down_phase = model.AddVars(max_time, COPT_BINARY, "send_down_phase");
    }

    const unsigned M = instance.numberOfProcessors() * instance.numberOfVertices();

    for (unsigned t = 0; t < max_time; t++) {

        Expr expr_comp, expr_comm, expr_send_up, expr_send_down;
        for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

            for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
                if(compute_exists[node][processor][t])
                    expr_comp += compute[node][processor][t];
                if(mergeSteps)
                {
                    if(send_up_exists[node][processor][t])
                        expr_comm += send_up[node][processor][t];
                    
                    if(send_down_exists[node][processor][t])
                        expr_comm += send_down[node][processor][t];
                }
                else
                {
                    if(send_up_exists[node][processor][t])
                        expr_send_up += send_up[node][processor][t];

                    if(send_down_exists[node][processor][t])
                        expr_send_down += send_down[node][processor][t];
                }
            }
        }

        model.AddConstr(M * comp_phase[t] >= expr_comp);
        if(mergeSteps)
        {
            model.AddConstr(2 * M * comm_phase[t] >= expr_comm);
            model.AddConstr(comp_phase[t] + comm_phase[t] <= 1);
        }
        else
        {
            model.AddConstr(M * send_up_phase[t] >= expr_send_up);
            model.AddConstr(M * send_down_phase[t] >= expr_send_down);
            model.AddConstr(comp_phase[t] + send_up_phase[t] + send_down_phase[t] <= 1);
        }
    }
};

void MultiProcessorPebbling::setupBspVariablesConstraintsObjective(const BspInstance &instance) {

    comp_phase_ends = model.AddVars(max_time, COPT_BINARY, "comp_phase_ends");

    comm_phase_ends = model.AddVars(max_time, COPT_BINARY, "comm_phase_ends");

    VarArray work_induced = model.AddVars(max_time, COPT_CONTINUOUS, "work_induced");
    VarArray comm_induced = model.AddVars(max_time, COPT_CONTINUOUS, "comm_induced");

    std::vector<VarArray> work_step_until(instance.numberOfProcessors());
    std::vector<VarArray> comm_step_until(instance.numberOfProcessors());
    std::vector<VarArray> send_up_step_until(instance.numberOfProcessors());
    std::vector<VarArray> send_down_step_until(instance.numberOfProcessors());

    VarArray send_up_induced;
    VarArray send_down_induced;
    if(up_and_down_cost_summed)
    {
        send_up_induced = model.AddVars(max_time, COPT_CONTINUOUS, "send_up_induced");
        send_down_induced = model.AddVars(max_time, COPT_CONTINUOUS, "send_down_induced");
    }

    for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
        work_step_until[processor] = model.AddVars(max_time, COPT_CONTINUOUS, "work_step_until");
        send_up_step_until[processor] = model.AddVars(max_time, COPT_CONTINUOUS, "send_up_step_until");
        send_down_step_until[processor] = model.AddVars(max_time, COPT_CONTINUOUS, "send_up_step_until");
    }

    for (unsigned t = 0; t < max_time; t++) {

        model.AddConstr(comp_phase[t] >= comp_phase_ends[t]);
        if(mergeSteps)
            model.AddConstr(comm_phase[t] >= comm_phase_ends[t]);
        else
            model.AddConstr(send_down_phase[t] + send_up_phase[t] >= comm_phase_ends[t]);
    }
    for (unsigned t = 0; t < max_time - 1; t++) {

        model.AddConstr(comp_phase_ends[t] >= comp_phase[t] - comp_phase[t+1]);
        if(mergeSteps)
            model.AddConstr(comm_phase_ends[t] >= comm_phase[t] - comm_phase[t+1]);
        else
            model.AddConstr(comm_phase_ends[t] >= send_down_phase[t] + send_up_phase[t] - send_down_phase[t+1] - send_up_phase[t+1]);
    }

    model.AddConstr(comp_phase_ends[max_time-1] >= comp_phase[max_time-1]);
    if(mergeSteps)
        model.AddConstr(comm_phase_ends[max_time-1] >= comm_phase[max_time-1]);
    else
        model.AddConstr(comm_phase_ends[max_time-1] >= send_down_phase[max_time-1] + send_up_phase[max_time-1]);

    const unsigned M = instance.numberOfProcessors() * (instance.getComputationalDag().sumOfVerticesWorkWeights(
                                                           instance.getComputationalDag().vertices().begin(),
                                                           instance.getComputationalDag().vertices().end()) +
                                                        instance.getComputationalDag().sumOfVerticesCommunicationWeights(
                                                           instance.getComputationalDag().vertices().begin(),
                                                           instance.getComputationalDag().vertices().end()));

    for (unsigned t = 1; t < max_time; t++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            Expr expr_work;
            Expr expr_send_up;
            Expr expr_send_down;
            for (unsigned node = 0; node < instance.numberOfVertices(); node++) {
                if(compute_exists[node][processor][t])
                    expr_work += instance.getComputationalDag().nodeWorkWeight(node) * compute[node][processor][t];
                if(send_up_exists[node][processor][t])
                    expr_send_up += instance.getComputationalDag().nodeCommunicationWeight(node) * send_up[node][processor][t];
                if(send_down_exists[node][processor][t])
                    expr_send_down += instance.getComputationalDag().nodeCommunicationWeight(node) * send_down[node][processor][t];
            }

            model.AddConstr(M * comm_phase_ends[t] + work_step_until[processor][t] >=
                            work_step_until[processor][t - 1] + expr_work);

            model.AddConstr(M * comp_phase_ends[t] + send_up_step_until[processor][t] >=
                            send_up_step_until[processor][t - 1] + expr_send_up);

            model.AddConstr(M * comp_phase_ends[t] + send_down_step_until[processor][t] >=
                            send_down_step_until[processor][t - 1] + expr_send_down);

            model.AddConstr(work_induced[t] >= work_step_until[processor][t] - M * (1 - comp_phase_ends[t]));
            if(up_and_down_cost_summed)
            {
                model.AddConstr(send_up_induced[t] >= send_up_step_until[processor][t] - M * (1 - comm_phase_ends[t]));
                model.AddConstr(send_down_induced[t] >= send_down_step_until[processor][t] - M * (1 - comm_phase_ends[t]));
                model.AddConstr(comm_induced[t] >= send_up_induced[t] + send_down_induced[t]);
             }
            else
            {
                model.AddConstr(comm_induced[t] >= send_down_step_until[processor][t] - M * (1 - comm_phase_ends[t]));
                model.AddConstr(comm_induced[t] >= send_up_step_until[processor][t] - M * (1 - comm_phase_ends[t]));
            }
        }
    }

    // t = 0
    for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

        Expr expr_work;
        Expr expr_send_up;
        Expr expr_send_down;
        for (unsigned node = 0; node < instance.numberOfVertices(); node++) {
            if(compute_exists[node][processor][0])
                expr_work += instance.getComputationalDag().nodeWorkWeight(node) * compute[node][processor][0];
            if(send_up_exists[node][processor][0])
                expr_send_up += instance.getComputationalDag().nodeCommunicationWeight(node) * send_up[node][processor][0];
            if(send_down_exists[node][processor][0])
                expr_send_down += instance.getComputationalDag().nodeCommunicationWeight(node) * send_down[node][processor][0];
        }

        model.AddConstr(M * comm_phase_ends[0] + work_step_until[processor][0] >= expr_work);

        model.AddConstr(M * comp_phase_ends[0] + send_up_step_until[processor][0] >= expr_send_up);

        model.AddConstr(M * comp_phase_ends[0] + send_down_step_until[processor][0] >= expr_send_down);

        model.AddConstr(work_induced[0] >= work_step_until[processor][0] - M * (1 - comp_phase_ends[0]));
        if(up_and_down_cost_summed)
        {
            model.AddConstr(send_up_induced[0] >= send_up_step_until[processor][0] - M * (1 - comm_phase_ends[0]));
            model.AddConstr(send_down_induced[0] >= send_down_step_until[processor][0] - M * (1 - comm_phase_ends[0]));
            model.AddConstr(comm_induced[0] >= send_up_induced[0] + send_down_induced[0]);
        }
        else
        {
            model.AddConstr(comm_induced[0] >= send_down_step_until[processor][0] - M * (1 - comm_phase_ends[0]));
            model.AddConstr(comm_induced[0] >= send_up_step_until[processor][0] - M * (1 - comm_phase_ends[0]));
        }
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
        if(!mergeSteps)
        {
            expr += comp_phase[t] + instance.communicationCosts() * send_up_phase[t] +
                instance.communicationCosts() * send_down_phase[t];
        }
        else
        {
            // this objective+parameter combination is not very meaningful, but still defined here to avoid a segfault otherwise
            expr += comp_phase[t] + instance.communicationCosts() * comm_phase[t];
        }
    }

    model.SetObjective(expr, COPT_MINIMIZE);
}

void MultiProcessorPebbling::setupAsyncVariablesConstraintsObjective(const BspInstance &instance) {

    std::vector<VarArray> finish_times(instance.numberOfProcessors());

    for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
        finish_times[processor] = model.AddVars(max_time, COPT_CONTINUOUS, "finish_times");
    }

    Var makespan = model.AddVar(0, COPT_INFINITY, 1, COPT_CONTINUOUS, "makespan");

    VarArray gets_blue = model.AddVars(instance.numberOfVertices(), COPT_CONTINUOUS, "gets_blue");

    const unsigned M = instance.numberOfProcessors() * (instance.getComputationalDag().sumOfVerticesWorkWeights(
                                                           instance.getComputationalDag().vertices().begin(),
                                                           instance.getComputationalDag().vertices().end()) +
                                                        instance.getComputationalDag().sumOfVerticesCommunicationWeights(
                                                           instance.getComputationalDag().vertices().begin(),
                                                           instance.getComputationalDag().vertices().end()));

    for (unsigned t = 0; t < max_time; t++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            Expr send_down_step_length;
            for (unsigned node = 0; node < instance.numberOfVertices(); node++)
                if(send_down_exists[node][processor][t])
                    send_down_step_length += instance.communicationCosts() *
                        instance.getComputationalDag().nodeCommunicationWeight(node) * send_down[node][processor][t];
            
            for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

                if(send_up_exists[node][processor][t])
                    model.AddConstr(gets_blue[node] >= finish_times[processor][t] - (1 - send_up[node][processor][t]) * M);
                if(send_down_exists[node][processor][t])
                    model.AddConstr(gets_blue[node] <=
                                finish_times[processor][t] + (1 - send_down[node][processor][t]) * M - send_down_step_length);
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

            if(compute_exists[node][processor][0])
                expr += instance.getComputationalDag().nodeWorkWeight(node) * compute[node][processor][0];

            if(send_up_exists[node][processor][0])
                expr += instance.communicationCosts() * instance.getComputationalDag().nodeCommunicationWeight(node) * send_up[node][processor][0];

            if(send_down_exists[node][processor][0])
                expr += instance.communicationCosts() * instance.getComputationalDag().nodeCommunicationWeight(node) * send_down[node][processor][0];
        }

        model.AddConstr(finish_times[processor][0] >= expr);
    }

    for (unsigned t = 1; t < max_time; t++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            Expr expr;
            for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

                if(compute_exists[node][processor][t])
                    expr += instance.getComputationalDag().nodeWorkWeight(node) * compute[node][processor][t];

                if(send_up_exists[node][processor][t])
                    expr += instance.communicationCosts() * instance.getComputationalDag().nodeCommunicationWeight(node) * send_up[node][processor][t];

                if(send_down_exists[node][processor][t])
                    expr += instance.communicationCosts() * instance.getComputationalDag().nodeCommunicationWeight(node) * send_down[node][processor][t];
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

BspMemSchedule MultiProcessorPebbling::constructMemScheduleFromSolution(const BspInstance &instance)
{
    std::vector<std::vector<std::set< std::pair<unsigned, unsigned> > > > nodes_computed(instance.numberOfProcessors(), std::vector<std::set<std::pair<unsigned, unsigned> > >(max_time));
    std::vector<std::vector<std::deque<unsigned> > > nodes_sent_up(instance.numberOfProcessors(), std::vector<std::deque<unsigned> >(max_time));
    std::vector<std::vector<std::deque<unsigned> > > nodes_sent_down(instance.numberOfProcessors(), std::vector<std::deque<unsigned> >(max_time));
    std::vector<std::vector<std::set<unsigned> > > evicted_after(instance.numberOfProcessors(), std::vector<std::set<unsigned> >(max_time));

    // used to remove unneeded steps when a node is sent down and then up (which becomes invalid after reordering the comm phases)
    std::vector<std::vector<bool > > sent_down_already(instance.numberOfVertices(), std::vector<bool>(instance.numberOfProcessors(), false));
    std::vector<std::vector<bool > > ignore_red(instance.numberOfVertices(), std::vector<bool>(instance.numberOfProcessors(), false));

    std::vector<size_t> topOrder = instance.getComputationalDag().GetTopOrder();
    std::vector<unsigned> topOrderPosition(instance.numberOfVertices());
    for(unsigned index = 0; index < instance.numberOfVertices(); ++index)
        topOrderPosition[topOrder[index]] = index;

    std::vector<bool> empty_step(max_time, true);
    std::vector<std::vector<unsigned> > step_type_on_proc(instance.numberOfProcessors(), std::vector<unsigned>(max_time, 0));
    
    for (unsigned step = 0; step < max_time; step++) 
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++)
            for (unsigned node = 0; node < instance.numberOfVertices(); node++)
                if (compute_exists[node][processor][step] && compute[node][processor][step].Get(COPT_DBLINFO_VALUE) >= .99)
                    step_type_on_proc[processor][step] = 1;


    for (unsigned step = 0; step < max_time; step++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

                if (step>0 && has_red[node][processor][step-1].Get(COPT_DBLINFO_VALUE) >= .99 && has_red[node][processor][step].Get(COPT_DBLINFO_VALUE) <= .01 && !ignore_red[node][processor])
                {
                    for(int previous_step = step - 1; previous_step >= 0; --previous_step)
                        if(!nodes_computed[processor][previous_step].empty() || !nodes_sent_up[processor][previous_step].empty() || !nodes_sent_down[processor][previous_step].empty() || previous_step == 0)
                        {
                            evicted_after[processor][previous_step].insert(node);
                            empty_step[previous_step] = false;
                            break;
                        }
                }
                
                if (compute_exists[node][processor][step] && compute[node][processor][step].Get(COPT_DBLINFO_VALUE) >= .99)
                {
                    nodes_computed[processor][step].emplace(topOrderPosition[node], node);
                    empty_step[step] = false;
                    ignore_red[node][processor] = false;

                    //implicit eviction in case of mergesteps - never having "has_red=1"
                    if(step + 1 < max_time && has_red[node][processor][step+1].Get(COPT_DBLINFO_VALUE) <= .01)
                        evicted_after[processor][step].insert(node);
                }

                if (send_down_exists[node][processor][step] && send_down[node][processor][step].Get(COPT_DBLINFO_VALUE) >= .99)
                {
                    bool keep_step = false;

                    for(unsigned next_step = step+1; next_step < max_time && has_red[node][processor][next_step].Get(COPT_DBLINFO_VALUE) >= .99 ; ++next_step)
                        if(step_type_on_proc[processor][next_step] == 1)
                        {
                            keep_step = true;
                            break;
                        }

                    if(keep_step)
                    {
                        nodes_sent_down[processor][step].push_back(node);
                        empty_step[step] = false;
                        step_type_on_proc[processor][step] = 3;
                        ignore_red[node][processor] = false;
                    }
                    else
                        ignore_red[node][processor] = true;

                    sent_down_already[node][processor] = true;
                }

                if (send_up_exists[node][processor][step] && send_up[node][processor][step].Get(COPT_DBLINFO_VALUE) >= .99 && !sent_down_already[node][processor])
                {
                    nodes_sent_up[processor][step].push_back(node);
                    empty_step[step] = false;
                    step_type_on_proc[processor][step] = 2;
                }
            }
        }
    }

    // components of the final BspMemSchedule - the first two dimensions are always processor and superstep
    std::vector<std::vector<std::vector<unsigned> > > compute_steps_per_supstep(instance.numberOfProcessors());
    std::vector<std::vector<std::vector<std::vector<unsigned> > > > nodes_evicted_after_compute(instance.numberOfProcessors());
    std::vector<std::vector<std::vector<unsigned> > > nodes_sent_up_in_supstep(instance.numberOfProcessors());
    std::vector<std::vector<std::vector<unsigned> > > nodes_sent_down_in_supstep(instance.numberOfProcessors());
    std::vector<std::vector<std::vector<unsigned> > > nodes_evicted_in_comm_phase(instance.numberOfProcessors());

    // edge case: check if an extra superstep must be added in the beginning to evict values that are initially in cache
    bool needs_evict_step_in_beginning = false;
    for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++)
        for (unsigned step = 0; step < max_time; step++)
        {
            if(step_type_on_proc[proc][step] == 0 && !evicted_after[proc][step].empty())
            {
                needs_evict_step_in_beginning = true;
                break;
            }
            else if(step_type_on_proc[proc][step]>0)
                break;
        }

    // create the actual BspMemSchedule - iterating over the steps
    int superstepIndex = 0;
    if(synchronous)
    {
        bool in_comm = true;
        superstepIndex = -1;

        if(needs_evict_step_in_beginning)
        {
            // artificially insert comm step in beginning, if it would start with compute otherwise
            bool begins_with_compute = false;
            for (unsigned step = 0; step < max_time; step++)
            {
                bool is_comp = false, is_comm = false;
                for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++)
                {
                    if(step_type_on_proc[proc][step] == 1)
                        is_comp = true;
                    if(step_type_on_proc[proc][step] > 1)
                        is_comm = true;
                }
                if(is_comp)
                    begins_with_compute = true;
                if(is_comp || is_comm)
                    break;
            }
            
            if(begins_with_compute)
            {
                superstepIndex = 0;
                for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++)
                {
                    compute_steps_per_supstep[proc].push_back(std::vector<unsigned>());
                    nodes_evicted_after_compute[proc].push_back(std::vector<std::vector<unsigned> >());
                    nodes_sent_up_in_supstep[proc].push_back(std::vector<unsigned>());
                    nodes_sent_down_in_supstep[proc].push_back(std::vector<unsigned>());
                    nodes_evicted_in_comm_phase[proc].push_back(std::vector<unsigned>());
                }
            }
        }

        // process steps
        for (unsigned step = 0; step < max_time; step++)
        {
            if(empty_step[step])
                continue;

            unsigned step_type = 0;
            for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++)
                step_type = std::max(step_type, step_type_on_proc[proc][step]);

            if (step_type == 1)
            {
                if(in_comm)
                {
                    for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++)
                    {
                        compute_steps_per_supstep[proc].push_back(std::vector<unsigned>());
                        nodes_evicted_after_compute[proc].push_back(std::vector<std::vector<unsigned> >());
                        nodes_sent_up_in_supstep[proc].push_back(std::vector<unsigned>());
                        nodes_sent_down_in_supstep[proc].push_back(std::vector<unsigned>());
                        nodes_evicted_in_comm_phase[proc].push_back(std::vector<unsigned>());
                    }
                    ++superstepIndex;
                    in_comm = false;
                }
                for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++)
                {
                    for(auto index_and_node : nodes_computed[proc][step])
                    {
                        compute_steps_per_supstep[proc][superstepIndex].push_back(index_and_node.second);
                        nodes_evicted_after_compute[proc][superstepIndex].push_back(std::vector<unsigned>());
                    }
                    for(unsigned node : evicted_after[proc][step])
                    {
                        if(!nodes_evicted_after_compute[proc][superstepIndex].empty())
                            nodes_evicted_after_compute[proc][superstepIndex].back().push_back(node);
                        else
                        {
                            // can only happen in special case: eviction in the very beginning
                            nodes_evicted_in_comm_phase[proc][0].push_back(node);
                        }
                    }
                }
            }
            
            if (step_type == 2 || step_type == 3)
            {
                if(superstepIndex < 0)
                {
                    for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++)
                    {
                        compute_steps_per_supstep[proc].push_back(std::vector<unsigned>());
                        nodes_evicted_after_compute[proc].push_back(std::vector<std::vector<unsigned> >());
                        nodes_sent_up_in_supstep[proc].push_back(std::vector<unsigned>());
                        nodes_sent_down_in_supstep[proc].push_back(std::vector<unsigned>());
                        nodes_evicted_in_comm_phase[proc].push_back(std::vector<unsigned>());
                    }
                    ++superstepIndex;
                }

                in_comm = true;
                for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++)
                {
                    for(unsigned node : nodes_sent_up[proc][step])
                        nodes_sent_up_in_supstep[proc][superstepIndex].push_back(node);
                    for(unsigned node : evicted_after[proc][step])
                        nodes_evicted_in_comm_phase[proc][superstepIndex].push_back(node);
                    for(unsigned node : nodes_sent_down[proc][step])
                        nodes_sent_down_in_supstep[proc][superstepIndex].push_back(node);
                }
            }
        }
    }
    else
    {
        std::vector<unsigned> step_idx_on_proc(instance.numberOfProcessors(), 0);

        std::vector<bool> already_has_blue(instance.numberOfVertices(), false);
        if(need_to_load_inputs)
            for (unsigned node = 0; node < instance.numberOfVertices(); node++)
                if(instance.getComputationalDag().numberOfParents(node) == 0)
                    already_has_blue[node] = true;

        std::vector<bool> proc_finished(instance.numberOfProcessors(), false);
        unsigned nr_proc_finished = 0;
        while(nr_proc_finished < instance.numberOfProcessors())
        {
            // preliminary sweep of superstep, to see if we need to wait for other processors
            std::vector<unsigned> idx_limit_on_proc = step_idx_on_proc;

            // first add compute steps
            if(!needs_evict_step_in_beginning || superstepIndex > 0)
            {
                for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++)
                    while(idx_limit_on_proc[proc] < max_time && step_type_on_proc[proc][idx_limit_on_proc[proc]] <= 1)
                        ++idx_limit_on_proc[proc];
            }

            // then add communications step until possible (note - they might not be valid if all put into a single superstep!)
            std::set<unsigned> new_blues;
            bool still_making_progress = true;
            while(still_making_progress)
            {
                still_making_progress = false;
                for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++)
                    while(idx_limit_on_proc[proc] < max_time && step_type_on_proc[proc][idx_limit_on_proc[proc]] != 1)
                    {
                        bool accept_step = true;
                        for(unsigned node : nodes_sent_down[proc][idx_limit_on_proc[proc]])
                            if(!already_has_blue[node] && new_blues.find(node) == new_blues.end())
                                accept_step = false;
                        
                        if(!accept_step)
                            break;

                        for(unsigned node : nodes_sent_up[proc][idx_limit_on_proc[proc]])
                            if(!already_has_blue[node])
                                new_blues.insert(node);
                        
                        still_making_progress = true;
                        ++idx_limit_on_proc[proc];
                    }   
            }

            // actually process the superstep
            for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++)
            {
                compute_steps_per_supstep[proc].push_back(std::vector<unsigned>());
                nodes_evicted_after_compute[proc].push_back(std::vector<std::vector<unsigned> >());
                nodes_sent_up_in_supstep[proc].push_back(std::vector<unsigned>());
                nodes_sent_down_in_supstep[proc].push_back(std::vector<unsigned>());
                nodes_evicted_in_comm_phase[proc].push_back(std::vector<unsigned>());

                while(step_idx_on_proc[proc] < idx_limit_on_proc[proc] && step_type_on_proc[proc][step_idx_on_proc[proc]] <= 1)
                {
                    for(auto index_and_node : nodes_computed[proc][step_idx_on_proc[proc]])
                    {
                        compute_steps_per_supstep[proc][superstepIndex].push_back(index_and_node.second);
                        nodes_evicted_after_compute[proc][superstepIndex].push_back(std::vector<unsigned>());
                    }
                    for(unsigned node : evicted_after[proc][step_idx_on_proc[proc]])
                    {
                        if(!nodes_evicted_after_compute[proc][superstepIndex].empty())
                            nodes_evicted_after_compute[proc][superstepIndex].back().push_back(node);
                        else
                        {
                            // can only happen in special case: eviction in the very beginning
                            nodes_evicted_in_comm_phase[proc][superstepIndex].push_back(node);
                        }
                    }

                    ++step_idx_on_proc[proc];
                }
                while(step_idx_on_proc[proc] < idx_limit_on_proc[proc] && step_type_on_proc[proc][step_idx_on_proc[proc]] != 1)
                {
                    for(unsigned node : nodes_sent_up[proc][step_idx_on_proc[proc]])
                    {
                        nodes_sent_up_in_supstep[proc][superstepIndex].push_back(node);
                        already_has_blue[node] = true;
                    }
                    for(unsigned node : nodes_sent_down[proc][step_idx_on_proc[proc]])
                        nodes_sent_down_in_supstep[proc][superstepIndex].push_back(node);
                    for(unsigned node : evicted_after[proc][step_idx_on_proc[proc]])
                        nodes_evicted_in_comm_phase[proc][superstepIndex].push_back(node);

                    ++step_idx_on_proc[proc];
                }
                if(step_idx_on_proc[proc] == max_time && !proc_finished[proc])
                {
                    proc_finished[proc] = true;
                    ++nr_proc_finished;
                }
            }
            ++superstepIndex;
        }
    }

    std::cout<<"MPP ILP best solution value: "<<model.GetDblAttr(COPT_DBLATTR_BESTOBJ)<<", best lower bound: "<<model.GetDblAttr(COPT_DBLATTR_BESTBND)<<std::endl;

    return BspMemSchedule(instance, compute_steps_per_supstep, nodes_evicted_after_compute,
                            nodes_sent_up_in_supstep, nodes_sent_down_in_supstep, nodes_evicted_in_comm_phase, needs_blue_at_end, has_red_in_beginning, need_to_load_inputs);
}

void MultiProcessorPebbling::setInitialSolution(const BspInstance &instance,
                                                const std::vector<std::vector<std::vector<unsigned> > >& computeSteps,
                                                const std::vector<std::vector<std::vector<unsigned> > >& sendUpSteps,
                                                const std::vector<std::vector<std::vector<unsigned> > >& sendDownSteps,
                                                const std::vector<std::vector<std::vector<unsigned> > >& nodesEvictedAfterStep)
{
    const unsigned N = instance.numberOfVertices();

    std::vector<bool> in_slow_mem(N, false);
    if(need_to_load_inputs)
        for(unsigned node=0; node < N; ++node)
            if(instance.getComputationalDag().numberOfParents(node) == 0)
                in_slow_mem[node] = true;

    std::vector<std::vector<unsigned> > in_fast_mem(N, std::vector<unsigned>(instance.numberOfProcessors(), false));
    if(!has_red_in_beginning.empty())
        for(unsigned proc=0; proc<instance.numberOfProcessors(); ++proc)
            for(unsigned node : has_red_in_beginning[proc])
                in_fast_mem[node][proc] = true;            

    unsigned step = 0, new_step_idx = 0;
    for(; step < computeSteps[0].size(); ++step)
    {
        for(unsigned node=0; node < N; ++node)
        {
            if(has_blue_exists[node][new_step_idx])
                model.SetMipStart(has_blue[node][new_step_idx], (int)in_slow_mem[node]);
            for(unsigned proc=0; proc<instance.numberOfProcessors(); ++proc)
                model.SetMipStart(has_red[node][proc][new_step_idx], (int)in_fast_mem[node][proc]);
        }

        if(restrict_step_types)
        {
            // align step number with step type cycle's phase, if needed
            bool skip_step = true;
            while(skip_step)
            {
                skip_step = false;
                bool is_compute = false, is_send_up = false, is_send_down = false;
                for(unsigned proc=0; proc<instance.numberOfProcessors(); ++proc)
                {
                    if(!computeSteps[proc][step].empty())
                        is_compute = true;
                    if(!sendUpSteps[proc][step].empty())
                        is_send_up = true;
                    if(!sendDownSteps[proc][step].empty())
                        is_send_down = true;
                }
                
                bool send_up_step_idx = (need_to_load_inputs && (new_step_idx % (compute_steps_per_cycle + 2) == compute_steps_per_cycle + 1))
                                        || (!need_to_load_inputs && (new_step_idx % (compute_steps_per_cycle + 2) == compute_steps_per_cycle));
                bool send_down_step_idx = (need_to_load_inputs && (new_step_idx % (compute_steps_per_cycle + 2) == 0))
                                        || (!need_to_load_inputs && (new_step_idx % (compute_steps_per_cycle + 2) == compute_steps_per_cycle + 1));

                if(is_compute && (send_up_step_idx || send_down_step_idx))
                    skip_step = true;
                if(is_send_up && !send_up_step_idx)
                    skip_step = true;
                if(is_send_down && !send_down_step_idx)
                    skip_step = true;
                
                if(skip_step)
                {
                    ++new_step_idx;
                    for(unsigned node=0; node < N; ++node)
                    {
                        if(has_blue_exists[node][new_step_idx])
                            model.SetMipStart(has_blue[node][new_step_idx], (int)in_slow_mem[node]);
                        for(unsigned proc=0; proc<instance.numberOfProcessors(); ++proc)
                            model.SetMipStart(has_red[node][proc][new_step_idx], (int)in_fast_mem[node][proc]);
                    }
                }
            }
        }

        for(unsigned proc=0; proc<instance.numberOfProcessors(); ++proc)
        {
            std::vector<bool> value_of_node(N, false);
            for(unsigned node : computeSteps[proc][step])
            {
                value_of_node[node] = true;
                if(compute_exists[node][proc][new_step_idx])
                    model.SetMipStart(compute[node][proc][new_step_idx], 1);
                in_fast_mem[node][proc] = true;
            }
            for(unsigned node : computeSteps[proc][step])
            {
                if(!value_of_node[node])
                    {
                        if(compute_exists[node][proc][new_step_idx])
                            model.SetMipStart(compute[node][proc][new_step_idx], 0);
                    }
                else
                    value_of_node[node] = false;
            }

            for(unsigned node : sendUpSteps[proc][step])
            {
                value_of_node[node] = true;
                if(send_up_exists[node][proc][new_step_idx])
                    model.SetMipStart(send_up[node][proc][new_step_idx], 1);
                in_slow_mem[node] = true;
            }
            for(unsigned node : sendUpSteps[proc][step])
            {
                if(!value_of_node[node])
                {
                    if(send_up_exists[node][proc][new_step_idx])
                        model.SetMipStart(send_up[node][proc][new_step_idx], 0);
                }
                else
                    value_of_node[node] = false;
            }

            for(unsigned node : sendDownSteps[proc][step])
            {
                value_of_node[node] = true;
                if(send_down_exists[node][proc][new_step_idx])
                    model.SetMipStart(send_down[node][proc][new_step_idx], 1);
                in_fast_mem[node][proc] = true;
            }
            for(unsigned node : sendDownSteps[proc][step])
            {
                if(!value_of_node[node])
                {
                    if(send_down_exists[node][proc][new_step_idx])
                        model.SetMipStart(send_down[node][proc][new_step_idx], 0);
                }
                else
                    value_of_node[node] = false;
            }

            for(unsigned node : nodesEvictedAfterStep[proc][step])
                in_fast_mem[node][proc] = false;
            
        }
        ++new_step_idx;
    }
    for(; new_step_idx < max_time; ++new_step_idx)
    {
        for(unsigned node=0; node < N; ++node)
        {
            if(has_blue_exists[node][new_step_idx])
                model.SetMipStart(has_blue[node][new_step_idx], (int)in_slow_mem[node]);
            for(unsigned proc=0; proc < instance.numberOfProcessors(); ++proc)
            {
                model.SetMipStart(has_red[node][proc][new_step_idx], 0);
                if(compute_exists[node][proc][new_step_idx])
                    model.SetMipStart(compute[node][proc][new_step_idx], 0);
                if(send_up_exists[node][proc][new_step_idx])
                    model.SetMipStart(send_up[node][proc][new_step_idx], 0);
                if(send_down_exists[node][proc][new_step_idx])
                    model.SetMipStart(send_down[node][proc][new_step_idx], 0);
            }
        }
    }
    model.LoadMipStart();
}

unsigned MultiProcessorPebbling::computeMaxTimeForInitialSolution(const BspInstance &instance,
                            const std::vector<std::vector<std::vector<unsigned> > >& computeSteps,
                            const std::vector<std::vector<std::vector<unsigned> > >& sendUpSteps,
                            const std::vector<std::vector<std::vector<unsigned> > >& sendDownSteps,
                            const std::vector<std::vector<std::vector<unsigned> > >& nodesEvictedAfterStep) const
{
    if(!restrict_step_types)
        return computeSteps[0].size() + 3;
    
    unsigned step = 0, new_step_idx = 0;
    for(; step < computeSteps[0].size(); ++step)
    {
        // align step number with step type cycle's phase, if needed
        bool skip_step = true;
        while(skip_step)
        {
            skip_step = false;
            bool is_compute = false, is_send_up = false, is_send_down = false;
            for(unsigned proc=0; proc<instance.numberOfProcessors(); ++proc)
            {
                if(!computeSteps[proc][step].empty())
                    is_compute = true;
                if(!sendUpSteps[proc][step].empty())
                    is_send_up = true;
                if(!sendDownSteps[proc][step].empty())
                    is_send_down = true;
            }

            bool send_up_step_idx = (need_to_load_inputs && (new_step_idx % (compute_steps_per_cycle + 2) == compute_steps_per_cycle + 1))
                                        || (!need_to_load_inputs && (new_step_idx % (compute_steps_per_cycle + 2) == compute_steps_per_cycle));
            bool send_down_step_idx = (need_to_load_inputs && (new_step_idx % (compute_steps_per_cycle + 2) == 0))
                                    || (!need_to_load_inputs && (new_step_idx % (compute_steps_per_cycle + 2) == compute_steps_per_cycle + 1));


            if(is_compute && (send_up_step_idx || send_down_step_idx))
                skip_step = true;
            if(is_send_up && !send_up_step_idx)
                skip_step = true;
            if(is_send_down && !send_down_step_idx)
                skip_step = true;
            
            if(skip_step)
                ++new_step_idx;
        }
            
        ++new_step_idx;
    }

    new_step_idx += compute_steps_per_cycle + 2;
    return new_step_idx;
}

bool MultiProcessorPebbling::hasEmptyStep(const BspInstance &instance)
{
    for (unsigned step = 0; step < max_time; ++step)
    {
        bool empty = true;
        for (unsigned node = 0; node < instance.numberOfVertices(); node++)
            for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++)
            {
                if((compute_exists[node][processor][step] && compute[node][processor][step].Get(COPT_DBLINFO_VALUE) >= .99) || 
                   (send_up_exists[node][processor][step] && send_up[node][processor][step].Get(COPT_DBLINFO_VALUE) >= .99) ||
                    (send_down_exists[node][processor][step] && send_down[node][processor][step].Get(COPT_DBLINFO_VALUE) >= .99 ))
                    empty = false;
            }
        if(empty)
            return true;
    }
    return false;
}
