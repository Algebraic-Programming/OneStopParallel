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

#include "algorithms/IlpSchedulers/CoptFullScheduler.hpp"
#include <stdexcept>

std::pair<RETURN_STATUS, BspSchedule> CoptFullScheduler::computeSchedule(const BspInstance &instance) {

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

std::pair<RETURN_STATUS, BspScheduleRecomp> CoptFullScheduler::computeScheduleRecomp(const BspInstance &instance) {

    allow_recomputation = true;

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
        solution_callback.allow_recomputation_cb = allow_recomputation;
        std::cout << "setting up callback with recomputation: " << allow_recomputation << std::endl;
        model.SetCallback(&solution_callback, COPT_CBCONTEXT_MIPSOL);
    }

    model.Solve();

    allow_recomputation = false;

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {

        return {SUCCESS, constructBspScheduleRecompFromSolution(instance, true)};

    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {

        return {ERROR, BspScheduleRecomp()};

    } else {

        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {

            return {BEST_FOUND, constructBspScheduleRecompFromSolution(instance, true)};

        } else {
            return {TIMEOUT, BspScheduleRecomp()};
        }
    }
};

void CoptFullScheduler::setupVariablesConstraintsObjective(const BspInstance &instance) {

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

        model.AddConstr(allow_recomputation ? expr >= .99 : expr == 1);
    }
    if (allow_recomputation)
        std::cout << "setting up constraints with recomputation: " << allow_recomputation << std::endl;


    comm_processor_to_processor_superstep_node_var = std::vector<std::vector<std::vector<VarArray>>>(
        instance.numberOfProcessors(),
        std::vector<std::vector<VarArray>>(instance.numberOfProcessors(),
                                           std::vector<VarArray>(max_number_supersteps)));

    for (unsigned int p1 = 0; p1 < instance.numberOfProcessors(); p1++) {

        for (unsigned int p2 = 0; p2 < instance.numberOfProcessors(); p2++) {
            for (unsigned int step = 0; step < max_number_supersteps; step++) {

                comm_processor_to_processor_superstep_node_var[p1][p2][step] = model.AddVars(
                    instance.numberOfVertices(), COPT_BINARY, "comm_processor_to_processor_superstep_node");
            }
        }
    }

    // precedence constraint: if task is computed then all of its predecessors must have been present
    for (unsigned int node = 0; node < instance.numberOfVertices(); node++) {

        if (instance.getComputationalDag().numberOfParents(node) > 0) {
            for (unsigned int step = 0; step < max_number_supersteps; step++) {
                for (unsigned int processor = 0; processor < instance.numberOfProcessors(); processor++) {

                    Expr expr;
                    for (const auto &ep : instance.getComputationalDag().in_edges(node)) {
                        expr +=
                            comm_processor_to_processor_superstep_node_var[processor][processor][step]
                                                                          [instance.getComputationalDag().source(ep)];
                    }

                    model.AddConstr(expr >= instance.getComputationalDag().numberOfParents(node) *
                                                node_to_processor_superstep_var[node][processor][step]);
                }
            }
        }
    }

    // combines two constraints: node can only be communicated if it is present; and node is present if it was
    // computed or communicated
    for (unsigned int step = 0; step < max_number_supersteps; step++) {
        for (unsigned int processor = 0; processor < instance.numberOfProcessors(); processor++) {
            for (unsigned int node = 0; node < instance.numberOfVertices(); node++) {

                Expr expr1, expr2;
                if (step > 0) {

                    for (unsigned int p_from = 0; p_from < instance.numberOfProcessors(); p_from++) {
                        expr1 += comm_processor_to_processor_superstep_node_var[p_from][processor][step - 1][node];
                    }
                }

                expr1 += node_to_processor_superstep_var[node][processor][step];

                for (unsigned int p_to = 0; p_to < instance.numberOfProcessors(); p_to++) {
                    expr2 += comm_processor_to_processor_superstep_node_var[processor][p_to][step][node];
                }

                model.AddConstr(instance.numberOfProcessors() * (expr1) >= expr2);
            }
        }
    }

    max_comm_superstep_var = model.AddVars(max_number_supersteps, COPT_INTEGER, "max_comm_superstep");
    // coptModel.AddVars(max_number_supersteps, 0, COPT_INFINITY, 0, COPT_INTEGER, "max_comm_superstep");

    max_work_superstep_var = model.AddVars(max_number_supersteps, COPT_INTEGER, "max_work_superstep");
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

    for (unsigned int step = 0; step < max_number_supersteps; step++) {
        for (unsigned int processor = 0; processor < instance.numberOfProcessors(); processor++) {

            Expr expr;
            for (unsigned int node = 0; node < instance.numberOfVertices(); node++) {
                for (unsigned int p_to = 0; p_to < instance.numberOfProcessors(); p_to++) {
                    if (processor != p_to) {
                        expr += instance.getComputationalDag().nodeCommunicationWeight(node) *
                                instance.sendCosts(processor, p_to) *
                                comm_processor_to_processor_superstep_node_var[processor][p_to][step][node];
                    }
                }
            }

            model.AddConstr(max_comm_superstep_var[step] >= expr);
        }
    }

    for (unsigned int step = 0; step < max_number_supersteps; step++) {
        for (unsigned int processor = 0; processor < instance.numberOfProcessors(); processor++) {

            Expr expr;
            for (unsigned int node = 0; node < instance.numberOfVertices(); node++) {
                for (unsigned int p_from = 0; p_from < instance.numberOfProcessors(); p_from++) {
                    if (processor != p_from) {
                        expr += instance.getComputationalDag().nodeCommunicationWeight(node) *
                                instance.sendCosts(p_from, processor) *
                                comm_processor_to_processor_superstep_node_var[p_from][processor][step][node];
                    }
                }
            }

            model.AddConstr(max_comm_superstep_var[step] >= expr);
        }
    }

    /*
    Objective function
      */
    Expr expr;

    for (unsigned int step = 0; step < max_number_supersteps; step++) {
        expr += max_work_superstep_var[step] + instance.communicationCosts() * max_comm_superstep_var[step] +
                instance.synchronisationCosts() * superstep_used_var[step];
    }

    model.SetObjective(expr - instance.synchronisationCosts(), COPT_MINIMIZE);
};

BspScheduleRecomp CoptFullScheduler::constructBspScheduleRecompFromSolution(const BspInstance &instance,
                                                                            bool cleanup_) {
    unsigned number_of_supersteps = 0;

    for (unsigned step = 0; step < max_number_supersteps; step++) {

        if (superstep_used_var[step].Get(COPT_DBLINFO_VALUE) >= .99) {
            number_of_supersteps++;
        }
    }

    BspScheduleRecomp schedule(instance, number_of_supersteps);

    for (unsigned node = 0; node < instance.numberOfVertices(); node++) {

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

            for (unsigned step = 0; step < max_number_supersteps; step++) {

                if (node_to_processor_superstep_var[node][processor][step].Get(COPT_DBLINFO_VALUE) >= .99) {
                    schedule.assignedProcessors(node).push_back(processor);
                    schedule.assignedSupersteps(node).push_back(step);
                }
            }
        }
    }

    std::map<KeyTriple, unsigned> cs;

    for (unsigned int node = 0; node < instance.numberOfVertices(); node++) {

        for (unsigned int p_from = 0; p_from < instance.numberOfProcessors(); p_from++) {
            for (unsigned int p_to = 0; p_to < instance.numberOfProcessors(); p_to++) {
                if (p_from != p_to) {
                    for (unsigned int step = 0; step < max_number_supersteps; step++) {
                        if (comm_processor_to_processor_superstep_node_var[p_from][p_to][step][node].Get(
                                COPT_DBLINFO_VALUE) >= .99) {
                            schedule.addCommunicationScheduleEntry(node, p_from, p_to, step);
                        }
                    }
                }
            }
        }
    }

    if (cleanup_) {
        node_to_processor_superstep_var.clear();
        comm_processor_to_processor_superstep_node_var.clear();
    }

    return schedule;
}

BspSchedule CoptFullScheduler::constructBspScheduleFromSolution(const BspInstance &instance, bool cleanup_) {

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

    std::map<KeyTriple, unsigned> cs;

    for (unsigned int node = 0; node < instance.numberOfVertices(); node++) {

        for (unsigned int p_from = 0; p_from < instance.numberOfProcessors(); p_from++) {
            for (unsigned int p_to = 0; p_to < instance.numberOfProcessors(); p_to++) {
                if (p_from != p_to) {
                    for (unsigned int step = 0; step < max_number_supersteps; step++) {
                        if (comm_processor_to_processor_superstep_node_var[p_from][p_to][step][node].Get(
                                COPT_DBLINFO_VALUE) >= .99) {
                            schedule.addCommunicationScheduleEntry(node, p_from, p_to, step);
                        }
                    }
                }
            }
        }
    }

    if (cleanup_) {
        node_to_processor_superstep_var.clear();
        comm_processor_to_processor_superstep_node_var.clear();
    }

    return schedule;
}

void CoptFullScheduler::loadInitialSchedule() {

    for (unsigned step = 0; step < max_number_supersteps; step++) {

        if (step < initial_schedule->numberOfSupersteps()) {
            model.SetMipStart(superstep_used_var[step], 1);

        } else {
            model.SetMipStart(superstep_used_var[step], 0);
        }

        // model.SetMipStart(max_work_superstep_var[step], COPT_INFINITY);
        // model.SetMipStart(max_comm_superstep_var[step], COPT_INFINITY);
    }

    const auto &cs = initial_schedule->getCommunicationSchedule();
    for (unsigned node = 0; node < initial_schedule->getInstance().numberOfVertices(); node++) {

        for (unsigned p1 = 0; p1 < initial_schedule->getInstance().numberOfProcessors(); p1++) {

            for (unsigned step = 0; step < max_number_supersteps; step++) {

                for (unsigned p2 = 0; p2 < initial_schedule->getInstance().numberOfProcessors(); p2++) {

                    if (p1 != p2) {

                        const auto &key = std::make_tuple(node, p1, p2);
                        if (cs.find(key) != cs.end()) {

                            if (cs.at(key) == step) {
                                model.SetMipStart(comm_processor_to_processor_superstep_node_var[p1][p2][step][node],
                                                  1);
                            } else {
                                model.SetMipStart(comm_processor_to_processor_superstep_node_var[p1][p2][step][node],
                                                  0);
                            }
                        }
                    } else {
                        // p1 == p2
                        if (p1 == initial_schedule->assignedProcessor(node) &&
                            step == initial_schedule->assignedSuperstep(node)) {
                            model.SetMipStart(comm_processor_to_processor_superstep_node_var[p1][p2][step][node], 1);
                        }
                    }
                }
            }
        }
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

    std::vector<std::vector<unsigned>> send(
        max_number_supersteps, std::vector<unsigned>(initial_schedule->getInstance().numberOfProcessors(), 0));

    std::vector<std::vector<unsigned>> rec(
        max_number_supersteps, std::vector<unsigned>(initial_schedule->getInstance().numberOfProcessors(), 0));

    for (const auto &[key, val] : initial_schedule->getCommunicationSchedule()) {

        send[val][get<1>(key)] +=
            initial_schedule->getInstance().getComputationalDag().nodeCommunicationWeight(get<0>(key)) *
            initial_schedule->getInstance().sendCosts(get<1>(key), get<2>(key));

        rec[val][get<2>(key)] +=
            initial_schedule->getInstance().getComputationalDag().nodeCommunicationWeight(get<0>(key)) *
            initial_schedule->getInstance().sendCosts(get<1>(key), get<2>(key));
    }

    for (unsigned step = 0; step < max_number_supersteps; step++) {
        unsigned max_work = 0;
        for (unsigned i = 0; i < initial_schedule->getInstance().numberOfProcessors(); i++) {
            if (max_work < work[step][i]) {
                max_work = work[step][i];
            }
        }

        unsigned max_comm = 0;
        for (unsigned i = 0; i < initial_schedule->getInstance().numberOfProcessors(); i++) {
            if (max_comm < send[step][i]) {
                max_comm = send[step][i];
            }
            if (max_comm < rec[step][i]) {
                max_comm = rec[step][i];
            }
        }

        model.SetMipStart(max_work_superstep_var[step], max_work);
        model.SetMipStart(max_comm_superstep_var[step], max_comm);
    }

    model.LoadMipStart();
    model.SetIntParam(COPT_INTPARAM_MIPSTARTMODE, 2);
}

void CoptFullScheduler::WriteSolutionCallback::callback() {

    if (Where() == COPT_CBCONTEXT_MIPSOL && counter < max_number_solution && GetIntInfo(COPT_CBINFO_HASINCUMBENT)) {

        try {

            if (GetDblInfo(COPT_CBINFO_BESTOBJ) < best_obj && 0.0 < GetDblInfo(COPT_CBINFO_BESTBND)) {

                best_obj = GetDblInfo(COPT_CBINFO_BESTOBJ);

                if (allow_recomputation_cb) {

                    auto sched = constructBspScheduleRecompFromCallback();
                    BspScheduleRecompWriter sched_writer(sched);
                    sched_writer.write_dot(write_solutions_path_cb + "intmed_sol_" + solution_file_prefix_cb + "_" +
                                           std::to_string(counter) + "_schedule.dot");
                

                } else {

                    auto sched = constructBspScheduleFromCallback();
                    BspScheduleWriter sched_writer(sched);
                    sched_writer.write_dot(write_solutions_path_cb + "intmed_sol_" + solution_file_prefix_cb + "_" +
                                           std::to_string(counter) + "_schedule.dot");
                }
                counter++;
            }

        } catch (const std::exception &e) {
        }
    }
};

BspSchedule CoptFullScheduler::WriteSolutionCallback::constructBspScheduleFromCallback() {

    VectorSchedule schedule(*instance_ptr);

    int supersteps = 1;

    for (unsigned int node = 0; node < instance_ptr->numberOfVertices(); node++) {

        for (unsigned int processor = 0; processor < instance_ptr->numberOfProcessors(); processor++) {

            for (int step = 0; step < (*node_to_processor_superstep_var_ptr)[0][0].Size(); step++) {

                if (GetSolution((*node_to_processor_superstep_var_ptr)[node][processor][step]) >= .99) {
                    schedule.node_to_processor_assignment[node] = processor;
                    schedule.node_to_superstep_assignment[node] = step;

                    if (step >= supersteps) {
                        supersteps = step + 1;
                    }
                }
            }
        }
    }

    schedule.number_of_supersteps = supersteps;

    std::map<KeyTriple, unsigned> cs;
    for (unsigned int node = 0; node < instance_ptr->numberOfVertices(); node++) {

        for (unsigned int p_from = 0; p_from < instance_ptr->numberOfProcessors(); p_from++) {
            for (unsigned int p_to = 0; p_to < instance_ptr->numberOfProcessors(); p_to++) {
                if (p_from != p_to) {
                    for (int step = 0; step < (*node_to_processor_superstep_var_ptr)[0][0].Size(); step++) {
                        if (GetSolution(
                                (*comm_processor_to_processor_superstep_node_var_ptr)[p_from][p_to][step][node]) >=
                            .99) {
                            cs[std::make_tuple(node, p_from, p_to)] = step;
                        }
                    }
                }
            }
        }
    }

    return schedule.buildBspSchedule(cs);
};

BspScheduleRecomp CoptFullScheduler::WriteSolutionCallback::constructBspScheduleRecompFromCallback() {

    unsigned number_of_supersteps = 1;
    BspScheduleRecomp schedule(*instance_ptr, number_of_supersteps);

    for (unsigned int node = 0; node < instance_ptr->numberOfVertices(); node++) {

        for (unsigned int processor = 0; processor < instance_ptr->numberOfProcessors(); processor++) {

            for (int step = 0; step < (*node_to_processor_superstep_var_ptr)[0][0].Size(); step++) {

                if (GetSolution((*node_to_processor_superstep_var_ptr)[node][processor][step]) >= .99) {
                    schedule.assignedProcessors(node).push_back(processor);
                    schedule.assignedSupersteps(node).push_back(step);

                    if (step >= number_of_supersteps) {
                        number_of_supersteps = step + 1;
                    }
                }
            }
        }
    }

    schedule.setNumberOfSupersteps(number_of_supersteps);

    for (unsigned int node = 0; node < instance_ptr->numberOfVertices(); node++) {

        for (unsigned int p_from = 0; p_from < instance_ptr->numberOfProcessors(); p_from++) {
            for (unsigned int p_to = 0; p_to < instance_ptr->numberOfProcessors(); p_to++) {
                if (p_from != p_to) {
                    for (int step = 0; step < (*node_to_processor_superstep_var_ptr)[0][0].Size(); step++) {
                        if (GetSolution(
                                (*comm_processor_to_processor_superstep_node_var_ptr)[p_from][p_to][step][node]) >=
                            .99) {

                            schedule.addCommunicationScheduleEntry(node, p_from, p_to, step);
                        }
                    }
                }
            }
        }
    }

    return schedule;
};