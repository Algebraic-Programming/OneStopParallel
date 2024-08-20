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

#include "scheduler/IlpSchedulers/TotalCommunicationScheduler.hpp"
#include <stdexcept>

std::pair<RETURN_STATUS, BspSchedule> TotalCommunicationScheduler::computeSchedule(const BspInstance &instance) {

    assert(!ignore_workload_balance || !use_lk_heuristic_callback);

    if (use_initial_schedule &&
        (max_number_supersteps < initial_schedule->numberOfSupersteps() ||
         instance.numberOfProcessors() != initial_schedule->getInstance().numberOfProcessors() ||
         instance.numberOfVertices() != initial_schedule->getInstance().numberOfVertices())) {
        throw std::invalid_argument("Invalid Argument while computeSchedule(instance): instance parameters do not "
                                    "agree with those of the initial schedule's instance!");
    }

    setupVariablesConstraintsObjective(instance);

    if (use_initial_schedule) {
        loadInitialSchedule();
    }

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

    if (write_solutions_found) {

        solution_callback.instance_ptr = &instance;
        model.SetCallback(&solution_callback, COPT_CBCONTEXT_MIPSOL);
    }
    if (use_lk_heuristic_callback) {

        heuristic_callback.instance_ptr = &instance;
        heuristic_callback.num_step = max_number_supersteps;
        model.SetCallback(&heuristic_callback, COPT_CBCONTEXT_MIPSOL);
    }

    model.Solve();

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {

        return {SUCCESS, constructBspScheduleFromSolution(instance, true)};

    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {

        return {ERROR, BspSchedule()};

    } else {

        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {

            return {BEST_FOUND, constructBspScheduleFromSolution(instance, true)};

        } else {
            return {TIMEOUT, BspSchedule()};
        }
    }
};

void TotalCommunicationScheduler::setupVariablesConstraintsObjective(const BspInstance &instance) {

    /*
    Variables
    */

    // variables indicating if superstep is used at all
    superstep_used_var = model.AddVars(max_number_supersteps, COPT_BINARY, "superstep_used");

    node_to_processor_superstep_var = std::vector<std::vector<VarArray>>(
        instance.numberOfVertices(), std::vector<VarArray>(instance.numberOfProcessors()));

    // variables for assigments of nodes to processor and superstep
    for (unsigned int node = 0; node < instance.numberOfVertices(); node++) {

        for (unsigned int processor = 0; processor < instance.numberOfProcessors(); processor++) {

            node_to_processor_superstep_var[node][processor] =
                model.AddVars(max_number_supersteps, COPT_BINARY, "node_to_processor_superstep");
        }
    }

    /*
    Constraints
      */

    /*
    Constraints
      */
    if (use_memory_constraint) {

        for (unsigned int node = 0; node < instance.numberOfVertices(); node++) {

            Expr expr;
            for (unsigned int processor = 0; processor < instance.numberOfProcessors(); processor++) {

                for (unsigned step = 0; step < max_number_supersteps; step++) {

                    expr += node_to_processor_superstep_var[node][processor][step] *
                            instance.getComputationalDag().nodeMemoryWeight(node);
                }
            }

            model.AddConstr(expr <= instance.getArchitecture().memoryBound());
        }
    }

    //  use consecutive supersteps starting from 0
    model.AddConstr(superstep_used_var[0] == 1);

    for (unsigned int step = 0; step < max_number_supersteps - 1; step++) {
        model.AddConstr(superstep_used_var[step] >= superstep_used_var[step + 1]);
    }

    // superstep is used at all
    for (unsigned int step = 0; step < max_number_supersteps; step++) {

        Expr expr;
        for (unsigned int node = 0; node < instance.numberOfVertices(); node++) {

            for (unsigned int processor = 0; processor < instance.numberOfProcessors(); processor++) {
                expr += node_to_processor_superstep_var[node][processor][step];
            }
        }
        model.AddConstr(expr <=
                        instance.numberOfVertices() * instance.numberOfProcessors() * superstep_used_var.GetVar(step));
    }

    // nodes are assigend depending on whether recomputation is allowed or not
    for (unsigned int node = 0; node < instance.numberOfVertices(); node++) {

        Expr expr;
        for (unsigned int processor = 0; processor < instance.numberOfProcessors(); processor++) {

            for (unsigned int step = 0; step < max_number_supersteps; step++) {
                expr += node_to_processor_superstep_var[node][processor].GetVar(step);
            }
        }

        model.AddConstr(expr == 1);
        // model.AddConstr(instance.allowRecomputation() ? expr >= .99 : expr == 1);
    }

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            for (unsigned step = 0; step < max_number_supersteps; step++) {

                for (const auto &source : instance.getComputationalDag().parents(node)) {

                    Expr expr1;

                    for (unsigned p2 = 0; p2 < instance.numberOfProcessors(); p2++) {

                        for (unsigned step_prime = 0; step_prime < step; step_prime++) {

                            expr1 += node_to_processor_superstep_var[source][p2][step_prime];
                        }
                    }

                    expr1 += node_to_processor_superstep_var[source][processor][step];

                    model.AddConstr(node_to_processor_superstep_var[node][processor][step] <= expr1);
                }
            }
        }
    }

    Expr total_edges_cut;

    if (instance.getArchitecture().isNumaArchitecture()) {

        edge_vars = std::vector<std::vector<VarArray>>(instance.numberOfProcessors(),
                                                       std::vector<VarArray>(instance.numberOfProcessors()));

        for (unsigned int p1 = 0; p1 < instance.numberOfProcessors(); p1++) {
            for (unsigned int p2 = 0; p2 < instance.numberOfProcessors(); p2++) {
                if (p1 != p2) {

                    edge_vars[p1][p2] =
                        model.AddVars(instance.getComputationalDag().numberOfEdges(), COPT_BINARY, "edge");

                    unsigned edge_id = 0;
                    for (const auto &ep : instance.getComputationalDag().edges()) {

                        // Var edge_var = model.AddVar(0, 1, 0, COPT_BINARY, "edge");

                        const int &source = instance.getComputationalDag().source(ep);
                        const int &target = instance.getComputationalDag().target(ep);

                        Expr expr1, expr2;
                        for (unsigned step = 0; step < max_number_supersteps; step++) {
                            expr1 += node_to_processor_superstep_var[source][p1][step];
                            expr2 += node_to_processor_superstep_var[target][p2][step];
                        }
                        model.AddConstr(edge_vars[p1][p2][edge_id] >= expr1 + expr2 - 1.001);

                        total_edges_cut += edge_vars[p1][p2][edge_id] *
                                           instance.getComputationalDag().nodeCommunicationWeight(source) *
                                           instance.sendCosts(source, target);

                        edge_id++;
                    }
                }
            }
        }

    } else {

        edge_vars = std::vector<std::vector<VarArray>>(1, std::vector<VarArray>(1));
        edge_vars[0][0] = model.AddVars(instance.getComputationalDag().numberOfEdges(), COPT_BINARY, "edge");

        unsigned edge_id = 0;
        for (const auto &ep : instance.getComputationalDag().edges()) {

            const int &source = instance.getComputationalDag().source(ep);
            const int &target = instance.getComputationalDag().target(ep);

            // Var edge_var = model.AddVar(0, 1, 0, COPT_BINARY, "edge_var");

            for (unsigned p1 = 0; p1 < instance.numberOfProcessors(); p1++) {
                Expr expr1, expr2;
                for (unsigned step = 0; step < max_number_supersteps; step++) {
                    expr1 += node_to_processor_superstep_var[source][p1][step];
                }

                for (unsigned p2 = 0; p2 < instance.numberOfProcessors(); p2++) {
                    if (p1 != p2) {

                        for (unsigned step = 0; step < max_number_supersteps; step++) {
                            expr2 += node_to_processor_superstep_var[target][p2][step];
                        }
                    }
                }
                model.AddConstr(edge_vars[0][0][edge_id] >= expr1 + expr2 - 1.001);
            }

            total_edges_cut +=
                instance.getComputationalDag().nodeCommunicationWeight(source) * edge_vars[0][0][edge_id];

            edge_id++;
        }
    }

    Expr expr;

    if (ignore_workload_balance) {

        for (unsigned step = 0; step < max_number_supersteps; step++) {
            expr += instance.synchronisationCosts() * superstep_used_var[step];
        }

    } else {
        max_work_superstep_var = model.AddVars(max_number_supersteps, COPT_CONTINUOUS, "max_work_superstep");
        // coptModel.AddVars(max_number_supersteps, 0, COPT_INFINITY, 0, COPT_INTEGER, "max_work_superstep");

        for (unsigned int step = 0; step < max_number_supersteps; step++) {
            for (unsigned int processor = 0; processor < instance.numberOfProcessors(); processor++) {

                Expr expr;
                for (unsigned int node = 0; node < instance.numberOfVertices(); node++) {
                    expr += instance.getComputationalDag().nodeWorkWeight(node) *
                            node_to_processor_superstep_var[node][processor][step];
                }

                model.AddConstr(max_work_superstep_var[step] >= expr);
            }
        }

        for (unsigned step = 0; step < max_number_supersteps; step++) {
            expr += max_work_superstep_var[step] + instance.synchronisationCosts() * superstep_used_var[step];
        }
    }

    /*
    Objective function
      */

    double comm_cost = instance.communicationCosts() / instance.numberOfProcessors();
    model.SetObjective(comm_cost * total_edges_cut + expr - instance.synchronisationCosts(), COPT_MINIMIZE);
};

BspSchedule TotalCommunicationScheduler::constructBspScheduleFromSolution(const BspInstance &instance, bool cleanup_) {

    BspSchedule schedule(instance);

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            for (unsigned step = 0; step < max_number_supersteps; step++) {

                if (node_to_processor_superstep_var[node][processor][step].Get(COPT_DBLINFO_VALUE) >= .99) {
                    schedule.setAssignedProcessor(node, processor);
                    schedule.setAssignedSuperstep(node, step);
                }
            }
        }
    }

    if (cleanup_) {
        node_to_processor_superstep_var.clear();
    }

    schedule.setAutoCommunicationSchedule();

    return schedule;
}

void TotalCommunicationScheduler::loadInitialSchedule() {

    for (unsigned step = 0; step < max_number_supersteps; step++) {

        if (step < initial_schedule->numberOfSupersteps()) {
            model.SetMipStart(superstep_used_var[step], 1);

        } else {
            model.SetMipStart(superstep_used_var[step], 0);
        }

        // model.SetMipStart(max_work_superstep_var[step], COPT_INFINITY);
        // model.SetMipStart(max_comm_superstep_var[step], COPT_INFINITY);
    }

    for (unsigned node = 0; node < initial_schedule->getInstance().numberOfVertices(); node++) {

        for (unsigned proc = 0; proc < initial_schedule->getInstance().numberOfProcessors(); proc++) {

            for (unsigned step = 0; step < max_number_supersteps; step++) {

                if (proc == initial_schedule->assignedProcessor(node) &&
                    step == initial_schedule->assignedSuperstep(node)) {
                    model.SetMipStart(node_to_processor_superstep_var[node][proc][step], 1);

                } else {

                    model.SetMipStart(node_to_processor_superstep_var[node][proc][step], 0);
                }
            }
        }
    }

    std::vector<std::vector<unsigned>> work(
        max_number_supersteps, std::vector<unsigned>(initial_schedule->getInstance().numberOfProcessors(), 0));

    for (unsigned int node = 0; node < initial_schedule->getInstance().numberOfVertices(); node++) {
        work[initial_schedule->assignedSuperstep(node)][initial_schedule->assignedProcessor(node)] +=
            initial_schedule->getInstance().getComputationalDag().nodeWorkWeight(node);
    }

    for (unsigned step = 0; step < max_number_supersteps; step++) {
        unsigned max_work = 0;
        for (unsigned i = 0; i < initial_schedule->getInstance().numberOfProcessors(); i++) {
            if (max_work < work[step][i]) {
                max_work = work[step][i];
            }
        }

        model.SetMipStart(max_work_superstep_var[step], max_work);
    }

    model.LoadMipStart();
    model.SetIntParam(COPT_INTPARAM_MIPSTARTMODE, 2);
}

void TotalCommunicationScheduler::WriteSolutionCallback::callback() {

    if (Where() == COPT_CBCONTEXT_MIPSOL && counter < max_number_solution && GetIntInfo(COPT_CBINFO_HASINCUMBENT)) {

        try {

            if (GetDblInfo(COPT_CBINFO_BESTOBJ) < best_obj && 0.0 < GetDblInfo(COPT_CBINFO_BESTBND)) {

                best_obj = GetDblInfo(COPT_CBINFO_BESTOBJ);

                auto sched = constructBspScheduleFromCallback();
                BspScheduleWriter sched_writer(sched);
                sched_writer.write_dot(write_solutions_path_cb + "intmed_sol_" + solution_file_prefix_cb + "_" +
                                       std::to_string(counter) + "_schedule.dot");
                counter++;
            }

        } catch (const std::exception &e) {
        }
    }
};

BspSchedule TotalCommunicationScheduler::WriteSolutionCallback::constructBspScheduleFromCallback() {

    BspSchedule schedule(*instance_ptr);

    for (unsigned node = 0; node < instance_ptr->numberOfVertices(); node++) {

        for (unsigned processor = 0; processor < instance_ptr->numberOfProcessors(); processor++) {

            for (unsigned step = 0; step < (unsigned)(*node_to_processor_superstep_var_ptr)[0][0].Size(); step++) {

                if (GetSolution((*node_to_processor_superstep_var_ptr)[node][processor][step]) >= .99) {
                    schedule.setAssignedProcessor(node, processor);
                    schedule.setAssignedSuperstep(node, step);
                }
            }
        }
    }

    return schedule;
};

void TotalCommunicationScheduler::LKHeuristicCallback::callback() {

    if (Where() == COPT_CBCONTEXT_MIPSOL && GetIntInfo(COPT_CBINFO_HASINCUMBENT)) {

        try {

            if (0.0 < GetDblInfo(COPT_CBINFO_BESTBND) && 1.0 < GetDblInfo(COPT_CBINFO_BESTOBJ) &&
                // GetDblInfo(COPT_CBINFO_BESTOBJ) < best_obj &&
                0.1 < (GetDblInfo(COPT_CBINFO_BESTOBJ) - GetDblInfo(COPT_CBINFO_BESTBND)) /
                          GetDblInfo(COPT_CBINFO_BESTOBJ)) {

                // best_obj = GetDblInfo(COPT_CBINFO_BESTOBJ);

                auto sched = constructBspScheduleFromCallback();

                if (sched.numberOfSupersteps() > 2) {
                    auto status = lk_heuristic.improveSchedule(sched);

                    if (status == SUCCESS) {
                        feedImprovedSchedule(sched);
                    }
                }
            }

        } catch (const std::exception &e) {
        }
    }
};

BspSchedule TotalCommunicationScheduler::LKHeuristicCallback::constructBspScheduleFromCallback() {

    BspSchedule schedule(*instance_ptr);

    for (unsigned node = 0; node < instance_ptr->numberOfVertices(); node++) {

        for (unsigned processor = 0; processor < instance_ptr->numberOfProcessors(); processor++) {

            for (unsigned step = 0; step < (unsigned)(*node_to_processor_superstep_var_ptr)[0][0].Size(); step++) {

                if (GetSolution((*node_to_processor_superstep_var_ptr)[node][processor][step]) >= .99) {
                    schedule.setAssignedProcessor(node, processor);
                    schedule.setAssignedSuperstep(node, step);
                }
            }
        }
    }

    return schedule;
};

void TotalCommunicationScheduler::LKHeuristicCallback::feedImprovedSchedule(const BspSchedule &schedule) {

    for (unsigned step = 0; step < num_step; step++) {

        if (step < schedule.numberOfSupersteps()) {
            SetSolution((*superstep_used_var_ptr)[step], 1.0);
        } else {
            SetSolution((*superstep_used_var_ptr)[step], 0.0);
        }
    }

    for (unsigned node = 0; node < instance_ptr->numberOfVertices(); node++) {

        for (unsigned processor = 0; processor < instance_ptr->numberOfProcessors(); processor++) {

            for (unsigned step = 0; step < (unsigned)(*node_to_processor_superstep_var_ptr)[0][0].Size(); step++) {

                if (schedule.assignedProcessor(node) == processor && schedule.assignedSuperstep(node) == step) {

                    SetSolution((*node_to_processor_superstep_var_ptr)[node][processor][step], 1.0);
                } else {
                    SetSolution((*node_to_processor_superstep_var_ptr)[node][processor][step], 0.0);
                }
            }
        }
    }

    std::vector<std::vector<unsigned>> work(num_step, std::vector<unsigned>(instance_ptr->numberOfProcessors(), 0));

    for (unsigned node = 0; node < instance_ptr->numberOfVertices(); node++) {
        work[schedule.assignedSuperstep(node)][schedule.assignedProcessor(node)] +=
            instance_ptr->getComputationalDag().nodeWorkWeight(node);
    }

    for (unsigned step = 0; step < num_step; step++) {

        unsigned max_work = 0;
        for (unsigned proc = 0; proc < instance_ptr->numberOfProcessors(); proc++) {
            if (max_work < work[step][proc]) {
                max_work = work[step][proc];
            }
        }

        SetSolution((*max_work_superstep_var_ptr)[step], max_work);
    }

    if (instance_ptr->isNumaInstance()) {

        for (unsigned p1 = 0; p1 < instance_ptr->numberOfProcessors(); p1++) {
            for (unsigned p2 = 0; p2 < instance_ptr->numberOfProcessors(); p2++) {
                if (p1 != p2) {

                    unsigned edge_id = 0;
                    for (const auto &ep : instance_ptr->getComputationalDag().edges()) {

                        const unsigned &source = instance_ptr->getComputationalDag().source(ep);
                        const unsigned &target = instance_ptr->getComputationalDag().target(ep);

                        if (schedule.assignedProcessor(source) == p1 && schedule.assignedProcessor(target) == p2) {

                            SetSolution((*edge_vars_ptr)[p1][p2][edge_id], 1.0);
                        } else {
                            SetSolution((*edge_vars_ptr)[p1][p2][edge_id], 0.0);
                        }

                        edge_id++;
                    }
                }
            }
        }

    } else {

        unsigned edge_id = 0;
        for (const auto &ep : instance_ptr->getComputationalDag().edges()) {

            const unsigned &source = instance_ptr->getComputationalDag().source(ep);
            const unsigned &target = instance_ptr->getComputationalDag().target(ep);

            if (schedule.assignedProcessor(source) != schedule.assignedProcessor(target)) {

                SetSolution((*edge_vars_ptr)[0][0][edge_id], 1.0);
            } else {
                SetSolution((*edge_vars_ptr)[0][0][edge_id], 0.0);
            }

            edge_id++;
        }
    }

    LoadSolution();
}
