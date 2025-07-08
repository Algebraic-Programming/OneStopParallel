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

#include "bsp/scheduler/LocalSearch/KernighanLin/kl_total_comm.hpp"
#include "bsp/scheduler/Scheduler.hpp"
#include "graph_algorithms/directed_graph_edge_view.hpp"
#include "auxiliary/io/DotFileWriter.hpp"

namespace osp {

template<typename Graph_t>
class TotalCommunicationScheduler : public Scheduler<Graph_t> {

  private:
    Envr env;
    Model model;

    bool use_memory_constraint;
    bool ignore_workload_balance;

    bool use_initial_schedule;
    const BspSchedule<Graph_t> *initial_schedule;

    bool write_solutions_found;
    bool use_lk_heuristic_callback;

    class WriteSolutionCallback : public CallbackBase {

      private:
        unsigned counter;
        unsigned max_number_solution;

        double best_obj;

      public:
        WriteSolutionCallback()
            : counter(0), max_number_solution(100), best_obj(COPT_INFINITY), write_solutions_path_cb(""),
              solution_file_prefix_cb(""), instance_ptr(0), node_to_processor_superstep_var_ptr() {}

        std::string write_solutions_path_cb;
        std::string solution_file_prefix_cb;
        const BspInstance<Graph_t> *instance_ptr;

        std::vector<std::vector<VarArray>> *node_to_processor_superstep_var_ptr;

        void callback() override {

            if (Where() == COPT_CBCONTEXT_MIPSOL && counter < max_number_solution &&
                GetIntInfo(COPT_CBINFO_HASINCUMBENT)) {

                try {

                    if (GetDblInfo(COPT_CBINFO_BESTOBJ) < best_obj && 0.0 < GetDblInfo(COPT_CBINFO_BESTBND)) {

                        best_obj = GetDblInfo(COPT_CBINFO_BESTOBJ);

                        auto sched = constructBspScheduleFromCallback();
                        DotFileWriter sched_writer;
                        sched_writer.write_dot(write_solutions_path_cb + "intmed_sol_" + solution_file_prefix_cb + "_" +
                                                   std::to_string(counter) + "_schedule.dot",
                                               sched);
                        counter++;
                    }

                } catch (const std::exception &e) {
                }
            }
        }

        BspSchedule<Graph_t> constructBspScheduleFromCallback() {

            BspSchedule<Graph_t> schedule(*instance_ptr);

            for (const auto &node : instance_ptr->vertices()) {

                for (unsigned processor = 0; processor < instance_ptr->numberOfProcessors(); processor++) {

                    for (unsigned step = 0; step < (unsigned)(*node_to_processor_superstep_var_ptr)[0][0].Size();
                         step++) {

                        assert(size < std::numeric_limits<int>::max());
                        if (GetSolution(
                                (*node_to_processor_superstep_var_ptr)[node][processor][static_cast<int>(step)]) >=
                            .99) {
                            schedule.setAssignedProcessor(node, processor);
                            schedule.setAssignedSuperstep(node, step);
                        }
                    }
                }
            }

            return schedule;
        }
    };

    class LKHeuristicCallback : public CallbackBase {

      private:
        kl_total_comm<Graph_t> lk_heuristic;

        double best_obj;

      public:
        LKHeuristicCallback()
            : lk_heuristic(), best_obj(COPT_INFINITY), num_step(0), instance_ptr(0), max_work_superstep_var_ptr(0),
              superstep_used_var_ptr(0), node_to_processor_superstep_var_ptr(0), edge_vars_ptr(0) {}

        unsigned num_step;
        const BspInstance<Graph_t> *instance_ptr;

        VarArray *max_work_superstep_var_ptr;
        VarArray *superstep_used_var_ptr;
        std::vector<std::vector<VarArray>> *node_to_processor_superstep_var_ptr;
        std::vector<std::vector<VarArray>> *edge_vars_ptr;

        void callback() override {

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

                            if (status == RETURN_STATUS::OSP_SUCCESS) {
                                feedImprovedSchedule(sched);
                            }
                        }
                    }

                } catch (const std::exception &e) {
                }
            }
        }

        BspSchedule<Graph_t> constructBspScheduleFromCallback() {

            BspSchedule schedule(*instance_ptr);

            for (const auto &node : instance_ptr->vertices()) {

                for (unsigned processor = 0; processor < instance_ptr->numberOfProcessors(); processor++) {

                    for (unsigned step = 0; step < (unsigned)(*node_to_processor_superstep_var_ptr)[0][0].Size();
                         step++) {
                        assert(step <= std::numeric_limits<int>::max());
                        if (GetSolution(
                                (*node_to_processor_superstep_var_ptr)[node][processor][static_cast<int>(step)]) >=
                            .99) {
                            schedule.setAssignedProcessor(node, processor);
                            schedule.setAssignedSuperstep(node, step);
                        }
                    }
                }
            }

            return schedule;
        };

        void feedImprovedSchedule(const BspSchedule<Graph_t> &schedule) {

            for (unsigned step = 0; step < num_step; step++) {

                if (step < schedule.numberOfSupersteps()) {
                    assert(step <= std::numeric_limits<int>::max());
                    SetSolution((*superstep_used_var_ptr)[static_cast<int>(step)], 1.0);
                } else {
                    assert(step <= std::numeric_limits<int>::max());
                    SetSolution((*superstep_used_var_ptr)[static_cast<int>(step)], 0.0);
                }
            }

            for (const auto &node : instance_ptr->vertices()) {

                for (unsigned processor = 0; processor < instance_ptr->numberOfProcessors(); processor++) {

                    for (unsigned step = 0; step < (unsigned)(*node_to_processor_superstep_var_ptr)[0][0].Size();
                         step++) {

                        if (schedule.assignedProcessor(node) == processor && schedule.assignedSuperstep(node) == step) {
                            assert(step <= std::numeric_limits<int>::max());
                            SetSolution((*node_to_processor_superstep_var_ptr)[node][processor][static_cast<int>(step)],
                                        1.0);
                        } else {
                            assert(step <= std::numeric_limits<int>::max());
                            SetSolution((*node_to_processor_superstep_var_ptr)[node][processor][static_cast<int>(step)],
                                        0.0);
                        }
                    }
                }
            }

            std::vector<std::vector<v_workw_t<Graph_t>>> work(
                num_step, std::vector<v_workw_t<Graph_t>>(instance_ptr->numberOfProcessors(), 0));

            for (const auto &node : instance_ptr->vertices()) {
                work[schedule.assignedSuperstep(node)][schedule.assignedProcessor(node)] +=
                    instance_ptr->getComputationalDag().vertex_work_weight(node);
            }

            for (unsigned step = 0; step < num_step; step++) {

                v_workw_t<Graph_t> max_work = 0;
                for (unsigned proc = 0; proc < instance_ptr->numberOfProcessors(); proc++) {
                    if (max_work < work[step][proc]) {
                        max_work = work[step][proc];
                    }
                }

                assert(step <= std::numeric_limits<int>::max());
                SetSolution((*max_work_superstep_var_ptr)[static_cast<int>(step)], max_work);
            }

            if (instance_ptr->isNumaInstance()) {

                for (unsigned p1 = 0; p1 < instance_ptr->numberOfProcessors(); p1++) {
                    for (unsigned p2 = 0; p2 < instance_ptr->numberOfProcessors(); p2++) {
                        if (p1 != p2) {

                            int edge_id = 0;
                            for (const auto &ep : edge_view(instance_ptr->getComputationalDag())) {

                                if (schedule.assignedProcessor(ep.source) == p1 &&
                                    schedule.assignedProcessor(ep.target) == p2) {

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

                int edge_id = 0;
                for (const auto &ep : edge_view(instance_ptr->getComputationalDag())) {

                    if (schedule.assignedProcessor(ep.source) != schedule.assignedProcessor(ep.target)) {

                        SetSolution((*edge_vars_ptr)[0][0][edge_id], 1.0);
                    } else {
                        SetSolution((*edge_vars_ptr)[0][0][edge_id], 0.0);
                    }

                    edge_id++;
                }
            }

            LoadSolution();
        }
    };

    WriteSolutionCallback solution_callback;
    LKHeuristicCallback heuristic_callback;

  protected:
    unsigned int max_number_supersteps;

    VarArray superstep_used_var;
    std::vector<std::vector<VarArray>> node_to_processor_superstep_var;
    std::vector<std::vector<VarArray>> edge_vars;
    VarArray max_work_superstep_var;

    void constructBspScheduleFromSolution(BspSchedule<Graph_t> &schedule, bool cleanup_ = false) {

        const auto &instance = schedule.getInstance();

        for (const auto &node : instance.vertices()) {

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
    }

    void loadInitialSchedule() {

        for (unsigned step = 0; step < max_number_supersteps; step++) {

            if (step < initial_schedule->numberOfSupersteps()) {
                assert(step <= std::numeric_limits<int>::max());
                model.SetMipStart(superstep_used_var[static_cast<int>(step)], 1);

            } else {
                assert(step <= std::numeric_limits<int>::max());
                model.SetMipStart(superstep_used_var[static_cast<int>(step)], 0);
            }
        }

        for (const auto &node : initial_schedule->getInstance().vertices()) {

            for (unsigned proc = 0; proc < initial_schedule->getInstance().numberOfProcessors(); proc++) {

                for (unsigned step = 0; step < max_number_supersteps; step++) {

                    if (proc == initial_schedule->assignedProcessor(node) &&
                        step == initial_schedule->assignedSuperstep(node)) {

                        assert(step <= std::numeric_limits<int>::max());
                        model.SetMipStart(node_to_processor_superstep_var[node][proc][static_cast<int>(step)], 1);

                    } else {

                        assert(step <= std::numeric_limits<int>::max());
                        model.SetMipStart(node_to_processor_superstep_var[node][proc][static_cast<int>(step)], 0);
                    }
                }
            }
        }

        std::vector<std::vector<v_workw_t<Graph_t>>> work(
            max_number_supersteps,
            std::vector<v_workw_t<Graph_t>>(initial_schedule->getInstance().numberOfProcessors(), 0));

        for (const auto &node : initial_schedule->getInstance().vertices()) {
            work[initial_schedule->assignedSuperstep(node)][initial_schedule->assignedProcessor(node)] +=
                initial_schedule->getInstance().getComputationalDag().vertex_work_weight(node);
        }

        for (unsigned step = 0; step < max_number_supersteps; step++) {
            v_workw_t<Graph_t> max_work = 0;
            for (unsigned i = 0; i < initial_schedule->getInstance().numberOfProcessors(); i++) {
                if (max_work < work[step][i]) {
                    max_work = work[step][i];
                }
            }

            assert(step <= std::numeric_limits<int>::max());
            model.SetMipStart(max_work_superstep_var[static_cast<int>(step)], max_work);
        }

        model.LoadMipStart();
        model.SetIntParam(COPT_INTPARAM_MIPSTARTMODE, 2);
    }

    void setupVariablesConstraintsObjective(const BspInstance<Graph_t> &instance) {

        /*
        Variables
        */

        // variables indicating if superstep is used at all
        superstep_used_var = model.AddVars(static_cast<int>(max_number_supersteps), COPT_BINARY, "superstep_used");

        node_to_processor_superstep_var = std::vector<std::vector<VarArray>>(
            instance.numberOfVertices(), std::vector<VarArray>(instance.numberOfProcessors()));
        assert(max_number_supersteps <= std::numeric_limits<int>::max());
        // variables for assigments of nodes to processor and superstep
        for (const auto &node : instance.vertices()) {

            for (unsigned int processor = 0; processor < instance.numberOfProcessors(); processor++) {

                node_to_processor_superstep_var[node][processor] =
                    model.AddVars(static_cast<int>(max_number_supersteps), COPT_BINARY, "node_to_processor_superstep");
            }
        }

        /*
        Constraints
          */

        /*
        Constraints
          */
        if (use_memory_constraint) {

            for (unsigned int processor = 0; processor < instance.numberOfProcessors(); processor++) {

                for (unsigned step = 0; step < max_number_supersteps; step++) {
                    Expr expr;
                    for (unsigned int node = 0; node < instance.numberOfVertices(); node++) {
                        expr += node_to_processor_superstep_var[node][processor][static_cast<int>(step)] *
                                instance.getComputationalDag().vertex_mem_weight(node);
                    }
                    model.AddConstr(expr <= instance.getArchitecture().memoryBound(processor));
                }
            }
        }

        //  use consecutive supersteps starting from 0
        model.AddConstr(superstep_used_var[0] == 1);

        for (unsigned int step = 0; step < max_number_supersteps - 1; step++) {
            model.AddConstr(superstep_used_var[static_cast<int>(step)] >= superstep_used_var[static_cast<int>(step + 1)]);
        }

        // superstep is used at all
        for (unsigned int step = 0; step < max_number_supersteps; step++) {

            Expr expr;
            for (const auto &node : instance.vertices()) {

                for (unsigned int processor = 0; processor < instance.numberOfProcessors(); processor++) {
                    expr += node_to_processor_superstep_var[node][processor][static_cast<int>(step)];
                }
            }
            model.AddConstr(expr <= (double)(instance.numberOfVertices() * instance.numberOfProcessors()) *
                                        superstep_used_var.GetVar(static_cast<int>(step)));
        }

        // nodes are assigend depending on whether recomputation is allowed or not
        for (const auto &node : instance.vertices()) {

            Expr expr;
            for (unsigned int processor = 0; processor < instance.numberOfProcessors(); processor++) {

                assert(max_number_supersteps <= std::numeric_limits<int>::max());
                for (unsigned int step = 0; step < max_number_supersteps; step++) {
                    expr += node_to_processor_superstep_var[node][processor].GetVar(static_cast<int>(step));
                }
            }

            model.AddConstr(expr == 1);
            // model.AddConstr(instance.allowRecomputation() ? expr >= .99 : expr == 1);
        }

        for (const auto &node : instance.vertices()) {

            for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

                assert(max_number_supersteps <= std::numeric_limits<int>::max());
                for (unsigned step = 0; step < max_number_supersteps; step++) {

                    for (const auto &source : instance.getComputationalDag().parents(node)) {

                        Expr expr1;

                        for (unsigned p2 = 0; p2 < instance.numberOfProcessors(); p2++) {

                            for (unsigned step_prime = 0; step_prime < step; step_prime++) {

                                expr1 += node_to_processor_superstep_var[source][p2][static_cast<int>(step_prime)];
                            }
                        }

                        expr1 += node_to_processor_superstep_var[source][processor][static_cast<int>(step)];

                        model.AddConstr(node_to_processor_superstep_var[node][processor][static_cast<int>(step)] <=
                                        expr1);
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

                        assert(instance.getComputationalDag().num_edges() <= std::numeric_limits<int>::max());
                        edge_vars[p1][p2] = model.AddVars(static_cast<int>(instance.getComputationalDag().num_edges()),
                                                          COPT_BINARY, "edge");

                        int edge_id = 0;
                        for (const auto &ep : edge_view(instance.getComputationalDag())) {

                            Expr expr1, expr2;
                            assert(max_number_supersteps <= std::numeric_limits<int>::max());
                            for (unsigned step = 0; step < max_number_supersteps; step++) {
                                expr1 += node_to_processor_superstep_var[ep.source][p1][static_cast<int>(step)];
                                expr2 += node_to_processor_superstep_var[ep.target][p2][static_cast<int>(step)];
                            }
                            model.AddConstr(edge_vars[p1][p2][edge_id] >= expr1 + expr2 - 1.001);

                            total_edges_cut += edge_vars[p1][p2][edge_id] *
                                               instance.getComputationalDag().vertex_comm_weight(ep.source) *
                                               instance.sendCosts(p1, p2);

                            edge_id++;
                        }
                    }
                }
            }

        } else {

            edge_vars = std::vector<std::vector<VarArray>>(1, std::vector<VarArray>(1));
            assert(instance.getComputationalDag().num_edges() <= std::numeric_limits<int>::max());
            edge_vars[0][0] =
                model.AddVars(static_cast<int>(instance.getComputationalDag().num_edges()), COPT_BINARY, "edge");

            int edge_id = 0;
            for (const auto &ep : edge_view(instance.getComputationalDag())) {

                for (unsigned p1 = 0; p1 < instance.numberOfProcessors(); p1++) {
                    Expr expr1, expr2;
                    for (unsigned step = 0; step < max_number_supersteps; step++) {
                        expr1 += node_to_processor_superstep_var[ep.source][p1][static_cast<int>(step)];
                    }

                    for (unsigned p2 = 0; p2 < instance.numberOfProcessors(); p2++) {
                        if (p1 != p2) {

                            for (unsigned step = 0; step < max_number_supersteps; step++) {

                                expr2 += node_to_processor_superstep_var[ep.target][p2][static_cast<int>(step)];
                            }
                        }
                    }
                    model.AddConstr(edge_vars[0][0][edge_id] >= expr1 + expr2 - 1.001);
                }

                total_edges_cut +=
                    instance.getComputationalDag().vertex_comm_weight(ep.source) * edge_vars[0][0][edge_id];

                edge_id++;
            }
        }

        Expr expr;

        if (ignore_workload_balance) {

            for (unsigned step = 0; step < max_number_supersteps; step++) {
                assert(step <= std::numeric_limits<int>::max());
                expr += instance.synchronisationCosts() * superstep_used_var[static_cast<int>(step)];
            }

        } else {
            assert(max_number_supersteps <= std::numeric_limits<int>::max());
            max_work_superstep_var =
                model.AddVars(static_cast<int>(max_number_supersteps), COPT_CONTINUOUS, "max_work_superstep");
            // coptModel.AddVars(max_number_supersteps, 0, COPT_INFINITY, 0, COPT_INTEGER, "max_work_superstep");

            for (unsigned int step = 0; step < max_number_supersteps; step++) {
                assert(step <= std::numeric_limits<int>::max());
                for (unsigned int processor = 0; processor < instance.numberOfProcessors(); processor++) {

                    Expr expr;
                    for (const auto &node : instance.vertices()) {
                        expr += instance.getComputationalDag().vertex_work_weight(node) *
                                node_to_processor_superstep_var[node][processor][static_cast<int>(step)];
                    }

                    model.AddConstr(max_work_superstep_var[static_cast<int>(step)] >= expr);
                }
            }

            for (unsigned step = 0; step < max_number_supersteps; step++) {
                assert(step <= std::numeric_limits<int>::max());
                expr += max_work_superstep_var[static_cast<int>(step)] +
                        instance.synchronisationCosts() * superstep_used_var[static_cast<int>(step)];
            }
        }

        /*
        Objective function
          */

        double comm_cost = (double)instance.communicationCosts() / instance.numberOfProcessors();
        model.SetObjective(comm_cost * total_edges_cut + expr - instance.synchronisationCosts(), COPT_MINIMIZE);
    }

  public:
    TotalCommunicationScheduler(unsigned steps = 5)
        : Scheduler<Graph_t>(), env(), model(env.CreateModel("TotalCommScheduler")), use_memory_constraint(false),
          ignore_workload_balance(false), use_initial_schedule(false), initial_schedule(0),
          write_solutions_found(false), use_lk_heuristic_callback(true), solution_callback(), heuristic_callback(),
          max_number_supersteps(steps) {

        heuristic_callback.max_work_superstep_var_ptr = &max_work_superstep_var;
        heuristic_callback.superstep_used_var_ptr = &superstep_used_var;
        heuristic_callback.node_to_processor_superstep_var_ptr = &node_to_processor_superstep_var;
        heuristic_callback.edge_vars_ptr = &edge_vars;

        solution_callback.node_to_processor_superstep_var_ptr = &node_to_processor_superstep_var;
    }

    TotalCommunicationScheduler(const BspSchedule<Graph_t> &schedule)
        : Scheduler<Graph_t>(), env(), model(env.CreateModel("TotalCommScheduler")), use_memory_constraint(false),
          ignore_workload_balance(false), use_initial_schedule(true), initial_schedule(&schedule),
          write_solutions_found(false), use_lk_heuristic_callback(true), solution_callback(), heuristic_callback(),
          max_number_supersteps(schedule.numberOfSupersteps()) {

        heuristic_callback.max_work_superstep_var_ptr = &max_work_superstep_var;
        heuristic_callback.superstep_used_var_ptr = &superstep_used_var;
        heuristic_callback.node_to_processor_superstep_var_ptr = &node_to_processor_superstep_var;
        heuristic_callback.edge_vars_ptr = &edge_vars;

        solution_callback.node_to_processor_superstep_var_ptr = &node_to_processor_superstep_var;
    }

    virtual ~TotalCommunicationScheduler() = default;

    /**
     * @brief Compute the schedule for the given BspInstance using the COPT solver.
     *
     * @param instance the BspInstance for which to compute the schedule
     *
     * @return a pair containing the return status and the computed BspSchedule
     *
     * @throws std::invalid_argument if the instance parameters do not
     *         agree with those of the initial schedule's instance
     */
    virtual RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &schedule) override {

        auto &instance = schedule.getInstance();

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

        model.SetDblParam(COPT_DBLPARAM_TIMELIMIT, Scheduler<Graph_t>::timeLimitSeconds);
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

            return RETURN_STATUS::OSP_SUCCESS; //, constructBspScheduleFromSolution(instance, true)};

        } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {

            return RETURN_STATUS::ERROR;

        } else {

            if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {

                return RETURN_STATUS::BEST_FOUND; //, constructBspScheduleFromSolution(instance, true)};

            } else {
                return RETURN_STATUS::TIMEOUT;
            }
        }
    };

    /**
     * @brief Sets the provided schedule as the initial solution for the ILP.
     *
     * This function sets the provided schedule as the initial solution for the ILP.
     * The maximum number of allowed supersteps is set to the number of supersteps
     * in the provided schedule.
     *
     * @param schedule The provided schedule.
     */
    inline void setInitialSolutionFromBspSchedule(const BspSchedule<Graph_t> &schedule) {

        initial_schedule = &schedule;

        max_number_supersteps = schedule.numberOfSupersteps();

        use_initial_schedule = true;
    }

    /**
     * @brief Sets the maximum number of supersteps allowed.
     *
     * This function sets the maximum number of supersteps allowed
     * for the computation of the BSP schedule. If an initial
     * solution is used, the maximum number of supersteps must be
     * greater or equal to the number of supersteps in the initial
     * solution.
     *
     * @param max The maximum number of supersteps allowed.
     *
     * @throws std::invalid_argument If the maximum number of
     *         supersteps is less than the number of supersteps in
     *         the initial solution.
     */
    void setMaxNumberOfSupersteps(unsigned max) {
        if (use_initial_schedule && max < initial_schedule->numberOfSupersteps()) {

            throw std::invalid_argument("Invalid Argument while setting "
                                        "max number of supersteps to a value "
                                        "which is less than the number of "
                                        "supersteps of the initial schedule!");
        }
        max_number_supersteps = max;
    }

    /**
     * @brief Enables writing intermediate solutions.
     *
     * This function enables the writing of intermediate solutions. The
     * `path` parameter specifies the path where the solutions will be
     * written, and the `file_prefix` parameter specifies the prefix
     * that will be used for the solution files.
     *
     * @param path The path where the solutions will be written.
     * @param file_prefix The prefix that will be used for the solution files.
     */
    inline void enableWriteIntermediateSol(std::string path, std::string file_prefix) {
        write_solutions_found = true;
        solution_callback.write_solutions_path_cb = path;
        solution_callback.solution_file_prefix_cb = file_prefix;
    }

    /**
     * @brief Set the use of memory constraint.
     *
     * This function sets the use of memory constraint. If the memory
     * constraint is enabled, the solver will use a memory constraint
     * to limit total memory of nodes assigend to a processor in a superstep.
     *
     * @param use True if the memory constraint should be used, false otherwise.
     */
    inline void setUseMemoryConstraint(bool use) { use_memory_constraint = use; }

    /**
     * @brief Set the use of workload balance constraint.
     *
     * This function sets the use of workload balance constraint. If the
     * workload balance constraint is enabled, the solver will use a workload
     * balance constraint to limit the difference of total work of nodes
     * assigned to a processor in a superstep.
     *
     * @param use True if the workload balance constraint should be used, false otherwise.
     */
    inline void setIgnoreWorkloadBalance(bool use) { ignore_workload_balance = use; }

    /**
     * @brief Set the use of LK heuristic callback.
     *
     * This function sets the use of LK heuristic callback. If the LK heuristic
     * callback is enabled, the solver will use the LK heuristic on any feasible solution
     * that is found to improve it.
     *
     *
     * @param use True if the LK heuristic callback should be used, false otherwise.
     */
    inline void setUseLkHeuristicCallback(bool use) { use_lk_heuristic_callback = use; }

    /**
     * Disables writing intermediate solutions.
     *
     * This function disables the writing of intermediate solutions. After
     * calling this function, the `enableWriteIntermediateSol` function needs
     * to be called again in order to enable writing of intermediate solutions.
     */
    inline void disableWriteIntermediateSol() { write_solutions_found = false; }

    /**
     * @brief Get the maximum number of supersteps.
     *
     * @return The maximum number of supersteps.
     */
    inline unsigned getMaxNumberOfSupersteps() const { return max_number_supersteps; }

    /**
     * @brief Get the best gap found by the solver.
     *
     * @return The best gap found by the solver.
     */
    inline double bestGap() { return model.GetDblAttr(COPT_DBLATTR_BESTGAP); }

    /**
     * @brief Get the best objective value found by the solver.
     *
     * @return The best objective value found by the solver.
     */
    inline double bestObjective() { return model.GetDblAttr(COPT_DBLATTR_BESTOBJ); }

    /**
     * @brief Get the best bound found by the solver.
     *
     * @return The best bound found by the solver.
     */
    inline double bestBound() { return model.GetDblAttr(COPT_DBLATTR_BESTBND); }

    /**
     * @brief Get the name of the schedule.
     *
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const override { return "TotalCommIlp"; }
};

} // namespace osp