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

#include "bsp/model/BspSchedule.hpp"
#include "bsp/model/BspScheduleCS.hpp"
#include "bsp/model/BspScheduleRecomp.hpp"
#include "bsp/model/VectorSchedule.hpp"
#include "bsp/scheduler/Scheduler.hpp"
#include "io/DotFileWriter.hpp"

namespace osp {

/**
 * @class CoptFullScheduler
 * @brief A class that represents a scheduler using the COPT solver for computing the schedule of a BSP instance.
 *
 * The `CoptFullScheduler` class is a subclass of the `Scheduler` class and provides an implementation of the
 * `computeSchedule` method using the COPT solver. It uses an ILP (Integer Linear Programming) formulation to find an
 * optimal schedule for a given BSP instance. The scheduler supports various options such as setting an initial
 * solution, setting the maximum number of supersteps, enabling/disabling writing intermediate solutions, and setting
 * communication constraints.
 *
 * The COPT solver is used to solve the ILP formulation and find the optimal schedule. It provides methods to set up the
 * ILP model, define variables and constraints, and solve the ILP problem. The scheduler constructs a `Model` object
 * from the COPT library to represent the ILP model and uses various callbacks to define the objective function,
 * constraints, and solution handling.
 *
 * To compute the schedule, the `computeSchedule` method is called with a `BspInstance` object representing the BSP
 * instance for which the schedule needs to be computed. The method returns a pair containing the return status and the
 * computed `BspSchedule`.
 *
 * The `CoptFullScheduler` class also provides methods to set the initial solution, set the maximum number of
 * supersteps, enable/disable writing intermediate solutions, and get information about the best gap, objective value,
 * and bound found by the solver.
 */
template<typename Graph_t>
class CoptFullScheduler : public Scheduler<Graph_t> {

    static_assert(is_computational_dag_v<Graph_t>, "CoptFullScheduler can only be used with computational DAGs.");

  private:
    bool allow_recomputation;
    bool use_memory_constraint;
    bool use_initial_schedule = false;
    const BspScheduleCS<Graph_t> *initial_schedule;

    bool use_initial_schedule_recomp = false;
    const BspScheduleRecomp<Graph_t> *initial_schedule_recomp;

    bool write_solutions_found;
    std::string write_solutions_path;
    std::string solution_file_prefix;

    class WriteSolutionCallback : public CallbackBase {

      private:
        unsigned counter;
        unsigned max_number_solution;

        double best_obj;

      public:
        WriteSolutionCallback()
            : counter(0), max_number_solution(500), best_obj(COPT_INFINITY), allow_recomputation_cb(false),
              write_solutions_path_cb(""), solution_file_prefix_cb(""), instance_ptr(),
              node_to_processor_superstep_var_ptr(), comm_processor_to_processor_superstep_node_var_ptr() {}

        bool allow_recomputation_cb;
        std::string write_solutions_path_cb;
        std::string solution_file_prefix_cb;
        const BspInstance<Graph_t> *instance_ptr;

        std::vector<std::vector<VarArray>> *node_to_processor_superstep_var_ptr;
        std::vector<std::vector<std::vector<VarArray>>> *comm_processor_to_processor_superstep_node_var_ptr;

        void callback() override {

            if (Where() == COPT_CBCONTEXT_MIPSOL && counter < max_number_solution &&
                GetIntInfo(COPT_CBINFO_HASINCUMBENT)) {

                try {

                    if (GetDblInfo(COPT_CBINFO_BESTOBJ) < best_obj && 0.0 < GetDblInfo(COPT_CBINFO_BESTBND)) {

                        best_obj = GetDblInfo(COPT_CBINFO_BESTOBJ);

                        if (allow_recomputation_cb) {

                        auto sched = constructBspScheduleRecompFromCallback();
                        DotFileWriter sched_writer;
                        sched_writer.write_dot(write_solutions_path_cb + "intmed_sol_" + solution_file_prefix_cb + "_" +
                                            std::to_string(counter) + "_schedule.dot", sched.getInstance().getComputationalDag());
                                            // TODO replace with recomp schedule file writing once it's available

                        } else {

                        BspSchedule<Graph_t> sched = constructBspScheduleFromCallback();
                        DotFileWriter sched_writer;
                        sched_writer.write_dot(write_solutions_path_cb + "intmed_sol_" + solution_file_prefix_cb + "_" +
                                                   std::to_string(counter) + "_schedule.dot", sched);
                        }
                        counter++;
                    }

                } catch (const std::exception &e) {
                }
            }
        }

        BspScheduleCS<Graph_t> constructBspScheduleFromCallback() {

            BspScheduleCS<Graph_t> schedule(*instance_ptr);

            for (const auto &node : instance_ptr->vertices()) {

                for (unsigned int processor = 0; processor < instance_ptr->numberOfProcessors(); processor++) {

                    for (unsigned step = 0;
                         step < static_cast<unsigned>((*node_to_processor_superstep_var_ptr)[0][0].Size()); step++) {

                        if (GetSolution(
                                (*node_to_processor_superstep_var_ptr)[node][processor][static_cast<int>(step)]) >=
                            .99) {
                            schedule.setAssignedProcessor(node, processor);
                            schedule.setAssignedSuperstep(node, step);
                        }
                    }
                }
            }

            for (const auto &node : instance_ptr->vertices()) {

                for (unsigned int p_from = 0; p_from < instance_ptr->numberOfProcessors(); p_from++) {
                    for (unsigned int p_to = 0; p_to < instance_ptr->numberOfProcessors(); p_to++) {
                        if (p_from != p_to) {
                            for (int step = 0; step < (*node_to_processor_superstep_var_ptr)[0][0].Size(); step++) {
                                if (GetSolution(
                                        (*comm_processor_to_processor_superstep_node_var_ptr)[p_from][p_to][static_cast<
                                            unsigned>(step)][static_cast<int>(node)]) >= .99) {
                                    schedule.addCommunicationScheduleEntry(node, p_from, p_to,
                                                                           static_cast<unsigned>(step));
                                }
                            }
                        }
                    }
                }
            }

            return schedule;
        }
        
        BspScheduleRecomp<Graph_t> constructBspScheduleRecompFromCallback() {

            unsigned number_of_supersteps = 0;
            BspScheduleRecomp<Graph_t> schedule(*instance_ptr);

            for (unsigned int node = 0; node < instance_ptr->numberOfVertices(); node++) {

                for (unsigned int processor = 0; processor < instance_ptr->numberOfProcessors(); processor++) {

                    for (unsigned step = 0; step < static_cast<unsigned>((*node_to_processor_superstep_var_ptr)[0][0].Size()); step++) {

                        if (GetSolution((*node_to_processor_superstep_var_ptr)[node][processor][static_cast<int>(step)]) >= .99) {
                            schedule.assignments(node).emplace_back(processor, step);

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
                            for (unsigned step = 0; step < static_cast<unsigned>((*node_to_processor_superstep_var_ptr)[0][0].Size()); step++) {
                                if (GetSolution(
                                        (*comm_processor_to_processor_superstep_node_var_ptr)[p_from][p_to][step][static_cast<int>(node)]) >=
                                    .99) {

                                    schedule.addCommunicationScheduleEntry(node, p_from, p_to, step);
                                }
                            }
                        }
                    }
                }
            }

            return schedule;
        }
    };

    // WriteSolutionCallback solution_callback;

  protected:
    unsigned int max_number_supersteps;

    VarArray superstep_used_var;
    std::vector<std::vector<VarArray>> node_to_processor_superstep_var;
    std::vector<std::vector<std::vector<VarArray>>> comm_processor_to_processor_superstep_node_var;

    VarArray max_comm_superstep_var;
    VarArray max_work_superstep_var;

    void constructBspScheduleFromSolution(BspScheduleCS<Graph_t> &schedule, bool cleanup_ = false) {

        const auto &instance = schedule.getInstance();

        for (const auto &node : instance.vertices()) {

            for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {

                for (unsigned step = 0; step < max_number_supersteps; step++) {

                    if (node_to_processor_superstep_var[node][processor][static_cast<int>(step)].Get(
                            COPT_DBLINFO_VALUE) >= .99) {
                        schedule.setAssignedProcessor(node, processor);
                        schedule.setAssignedSuperstep(node, step);
                    }
                }
            }
        }

        for (const auto &node : instance.vertices()) {

            for (unsigned int p_from = 0; p_from < instance.numberOfProcessors(); p_from++) {
                for (unsigned int p_to = 0; p_to < instance.numberOfProcessors(); p_to++) {
                    if (p_from != p_to) {
                        for (unsigned int step = 0; step < max_number_supersteps; step++) {
                            if (comm_processor_to_processor_superstep_node_var[p_from][p_to][step]
                                                                              [static_cast<int>(node)]
                                                                                  .Get(COPT_DBLINFO_VALUE) >= .99) {
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
    }

    void constructBspScheduleRecompFromSolution(BspScheduleRecomp<Graph_t> &schedule, bool cleanup_) {
        unsigned number_of_supersteps = 0;

        for (unsigned step = 0; step < max_number_supersteps; step++) {

            if (superstep_used_var[static_cast<int>(step)].Get(COPT_DBLINFO_VALUE) >= .99) {
                number_of_supersteps++;
            }
        }

        schedule.setNumberOfSupersteps(number_of_supersteps);

        for (unsigned node = 0; node < schedule.getInstance().numberOfVertices(); node++) {

            for (unsigned processor = 0; processor < schedule.getInstance().numberOfProcessors(); processor++) {

                for (unsigned step = 0; step < max_number_supersteps; step++) {

                    if (node_to_processor_superstep_var[node][processor][static_cast<int>(step)].Get(COPT_DBLINFO_VALUE) >= .99) {
                        schedule.assignments(node).emplace_back(processor, step);
                    }
                }
            }
        }

        for (unsigned int node = 0; node < schedule.getInstance().numberOfVertices(); node++) {

            for (unsigned int p_from = 0; p_from < schedule.getInstance().numberOfProcessors(); p_from++) {
                for (unsigned int p_to = 0; p_to < schedule.getInstance().numberOfProcessors(); p_to++) {
                    if (p_from != p_to) {
                        for (unsigned int step = 0; step < max_number_supersteps; step++) {
                            if (comm_processor_to_processor_superstep_node_var[p_from][p_to][step][static_cast<int>(node)].Get(
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
    }


    void loadInitialSchedule(Model &model) {

        const auto& DAG = use_initial_schedule_recomp ?
                        initial_schedule_recomp->getInstance().getComputationalDag() :
                        initial_schedule->getInstance().getComputationalDag();

        const auto& arch = use_initial_schedule_recomp ?
                        initial_schedule_recomp->getInstance().getArchitecture() :
                        initial_schedule->getInstance().getArchitecture();

        const unsigned& num_processors = use_initial_schedule_recomp ?
                        initial_schedule_recomp->getInstance().numberOfProcessors() :
                        initial_schedule->getInstance().numberOfProcessors();

        const unsigned& num_supersteps = use_initial_schedule_recomp ?
                        initial_schedule_recomp->numberOfSupersteps() :
                        initial_schedule->numberOfSupersteps();

        const auto &cs = use_initial_schedule_recomp ?
                        initial_schedule_recomp->getCommunicationSchedule() :
                        initial_schedule->getCommunicationSchedule();

        assert(max_number_supersteps <= std::numeric_limits<int>::max());
        for (unsigned step = 0; step < max_number_supersteps; step++) {

            if (step < num_supersteps) {
                model.SetMipStart(superstep_used_var[static_cast<int>(step)], 1);

            } else {
                model.SetMipStart(superstep_used_var[static_cast<int>(step)], 0);
            }

            // model.SetMipStart(max_work_superstep_var[step], COPT_INFINITY);
            // model.SetMipStart(max_comm_superstep_var[step], COPT_INFINITY);
        }

        std::vector<std::set<std::pair<unsigned, unsigned> > > computed(DAG.num_vertices());
        for (const auto &node : DAG.vertices())
        {
            if(use_initial_schedule_recomp)
                for (const std::pair<unsigned, unsigned>& assignment : initial_schedule_recomp->assignments(node))
                    computed[node].emplace(assignment);
            else
                computed[node].emplace(initial_schedule->assignedProcessor(node),initial_schedule->assignedSuperstep(node));
        }

        for (const auto &node : DAG.vertices()) {

            for (unsigned p1 = 0; p1 < num_processors; p1++) {

                for (unsigned step = 0; step < max_number_supersteps; step++) {

                    for (unsigned p2 = 0; p2 < num_processors; p2++) {

                        if (p1 != p2) {

                            const auto &key = std::make_tuple(node, p1, p2);
                            if (cs.find(key) != cs.end()) {

                                if (cs.at(key) == step) {
                                    model.SetMipStart(
                                        comm_processor_to_processor_superstep_node_var[p1][p2][step]
                                                                                      [static_cast<int>(node)],
                                        1);
                                } else {
                                    model.SetMipStart(
                                        comm_processor_to_processor_superstep_node_var[p1][p2][step]
                                                                                      [static_cast<int>(node)],
                                        0);
                                }
                            }
                        }
                    }
                }
            }

            for(const std::pair<unsigned, unsigned>& proc_step : computed[node]){
                for(unsigned step = proc_step.second; step < max_number_supersteps; step++){
                    model.SetMipStart(comm_processor_to_processor_superstep_node_var[proc_step.first][proc_step.first][step]
                                                                                    [static_cast<int>(node)], 1);
                }
            }
        }

        for (const auto &node : DAG.vertices()) {            

            for (unsigned proc = 0; proc < num_processors; proc++) {

                for (unsigned step = 0; step < max_number_supersteps; step++) {

                    if (computed[node].find(std::make_pair(proc, step)) != computed[node].end()) {
                        model.SetMipStart(node_to_processor_superstep_var[node][proc][static_cast<int>(step)], 1);

                    } else {

                        model.SetMipStart(node_to_processor_superstep_var[node][proc][static_cast<int>(step)], 0);
                    }
                }
            }
        }

        std::vector<std::vector<v_workw_t<Graph_t>>> work(
            max_number_supersteps,
            std::vector<v_workw_t<Graph_t>>(num_processors, 0));

        if(use_initial_schedule_recomp)
        {
            for (const auto &node : initial_schedule_recomp->getInstance().vertices()) {
                for (const std::pair<unsigned, unsigned>& assignment : initial_schedule_recomp->assignments(node)) {
                    work[assignment.second][assignment.first] +=
                    DAG.vertex_work_weight(node);
                }
            }
        }
        else
        {
            for (const auto &node : initial_schedule->getInstance().vertices())
                work[initial_schedule->assignedSuperstep(node)][initial_schedule->assignedProcessor(node)] +=
                    DAG.vertex_work_weight(node);
        }

        std::vector<std::vector<v_commw_t<Graph_t>>> send(
            max_number_supersteps,
            std::vector<v_commw_t<Graph_t>>(num_processors, 0));

        std::vector<std::vector<v_commw_t<Graph_t>>> rec(
            max_number_supersteps,
            std::vector<v_commw_t<Graph_t>>(num_processors, 0));

        for (const auto &[key, val] : cs) {

            send[val][std::get<1>(key)] +=
                DAG.vertex_comm_weight(std::get<0>(key)) *
                arch.sendCosts(std::get<1>(key), std::get<2>(key));

            rec[val][std::get<2>(key)] +=
                DAG.vertex_comm_weight(std::get<0>(key)) *
                arch.sendCosts(std::get<1>(key), std::get<2>(key));
        }

        for (unsigned step = 0; step < max_number_supersteps; step++) {
            v_workw_t<Graph_t> max_work = 0;
            for (unsigned i = 0; i < num_processors; i++) {
                if (max_work < work[step][i]) {
                    max_work = work[step][i];
                }
            }

            v_commw_t<Graph_t> max_comm = 0;
            for (unsigned i = 0; i < num_processors; i++) {
                if (max_comm < send[step][i]) {
                    max_comm = send[step][i];
                }
                if (max_comm < rec[step][i]) {
                    max_comm = rec[step][i];
                }
            }

            model.SetMipStart(max_work_superstep_var[static_cast<int>(step)], max_work);
            model.SetMipStart(max_comm_superstep_var[static_cast<int>(step)], max_comm);
        }

        model.LoadMipStart();
        model.SetIntParam(COPT_INTPARAM_MIPSTARTMODE, 2);
    }

    void setupVariablesConstraintsObjective(const BspInstance<Graph_t> &instance, Model &model) {

        /*
       Variables
       */

        assert(max_number_supersteps <= std::numeric_limits<int>::max());
        assert(instance.numberOfProcessors() <= std::numeric_limits<int>::max());

        // variables indicating if superstep is used at all
        superstep_used_var = model.AddVars(static_cast<int>(max_number_supersteps), COPT_BINARY, "superstep_used");

        node_to_processor_superstep_var = std::vector<std::vector<VarArray>>(
            instance.numberOfVertices(), std::vector<VarArray>(instance.numberOfProcessors()));

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
        if (use_memory_constraint) {

            for (unsigned int processor = 0; processor < instance.numberOfProcessors(); processor++) {
                for (unsigned step = 0; step < max_number_supersteps; step++) {

                    Expr expr;
                    for (const auto &node : instance.vertices()) {
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

                for (unsigned int step = 0; step < max_number_supersteps; step++) {
                    expr += node_to_processor_superstep_var[node][processor].GetVar(static_cast<int>(step));
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

                    comm_processor_to_processor_superstep_node_var[p1][p2][step] =
                        model.AddVars(static_cast<int>(instance.numberOfVertices()), COPT_BINARY,
                                      "comm_processor_to_processor_superstep_node");
                }
            }
        }

        // precedence constraint: if task is computed then all of its predecessors must have been present
        for (const auto &node : instance.vertices()) {

            if (instance.getComputationalDag().in_degree(node) > 0) {
                for (unsigned int step = 0; step < max_number_supersteps; step++) {
                    for (unsigned int processor = 0; processor < instance.numberOfProcessors(); processor++) {

                        Expr expr;
                        for (const auto &parent : instance.getComputationalDag().parents(node)) {
                            expr += comm_processor_to_processor_superstep_node_var[processor][processor][step]
                                                                                  [static_cast<int>(parent)];
                        }

                        model.AddConstr(expr >=
                                        (double)instance.getComputationalDag().in_degree(node) *
                                            node_to_processor_superstep_var[node][processor][static_cast<int>(step)]);
                    }
                }
            }
        }

        // combines two constraints: node can only be communicated if it is present; and node is present if it was
        // computed or communicated
        for (unsigned int step = 0; step < max_number_supersteps; step++) {
            for (unsigned int processor = 0; processor < instance.numberOfProcessors(); processor++) {
                for (const auto &node : instance.vertices()) {

                    Expr expr1, expr2;
                    if (step > 0) {

                        for (unsigned int p_from = 0; p_from < instance.numberOfProcessors(); p_from++) {
                            expr1 += comm_processor_to_processor_superstep_node_var[p_from][processor][step - 1]
                                                                                   [static_cast<int>(node)];
                        }
                    }

                    expr1 += node_to_processor_superstep_var[node][processor][static_cast<int>(step)];

                    for (unsigned int p_to = 0; p_to < instance.numberOfProcessors(); p_to++) {
                        expr2 += comm_processor_to_processor_superstep_node_var[processor][p_to][step]
                                                                               [static_cast<int>(node)];
                    }

                    model.AddConstr(instance.numberOfProcessors() * (expr1) >= expr2);
                }
            }
        }

        max_comm_superstep_var =
            model.AddVars(static_cast<int>(max_number_supersteps), COPT_INTEGER, "max_comm_superstep");
        // coptModel.AddVars(max_number_supersteps, 0, COPT_INFINITY, 0, COPT_INTEGER, "max_comm_superstep");

        max_work_superstep_var =
            model.AddVars(static_cast<int>(max_number_supersteps), COPT_INTEGER, "max_work_superstep");
        // coptModel.AddVars(max_number_supersteps, 0, COPT_INFINITY, 0, COPT_INTEGER, "max_work_superstep");

        for (unsigned int step = 0; step < max_number_supersteps; step++) {
            for (unsigned int processor = 0; processor < instance.numberOfProcessors(); processor++) {

                Expr expr;
                for (unsigned int node = 0; node < instance.numberOfVertices(); node++) {
                    expr += instance.getComputationalDag().vertex_work_weight(node) *
                            node_to_processor_superstep_var[node][processor][static_cast<int>(step)];
                }

                model.AddConstr(max_work_superstep_var[static_cast<int>(step)] >= expr);
            }
        }

        for (unsigned int step = 0; step < max_number_supersteps; step++) {
            for (unsigned int processor = 0; processor < instance.numberOfProcessors(); processor++) {

                Expr expr;
                for (const auto &node : instance.vertices()) {
                    for (unsigned int p_to = 0; p_to < instance.numberOfProcessors(); p_to++) {
                        if (processor != p_to) {
                            expr += instance.getComputationalDag().vertex_comm_weight(node) *
                                    instance.sendCosts(processor, p_to) *
                                    comm_processor_to_processor_superstep_node_var[processor][p_to][step]
                                                                                  [static_cast<int>(node)];
                        }
                    }
                }

                model.AddConstr(max_comm_superstep_var[static_cast<int>(step)] >= expr);
            }
        }

        for (unsigned int step = 0; step < max_number_supersteps; step++) {
            for (unsigned int processor = 0; processor < instance.numberOfProcessors(); processor++) {

                Expr expr;
                for (const auto &node : instance.vertices()) {
                    for (unsigned int p_from = 0; p_from < instance.numberOfProcessors(); p_from++) {
                        if (processor != p_from) {
                            expr += instance.getComputationalDag().vertex_comm_weight(node) *
                                    instance.sendCosts(p_from, processor) *
                                    comm_processor_to_processor_superstep_node_var[p_from][processor][step]
                                                                                  [static_cast<int>(node)];
                        }
                    }
                }

                model.AddConstr(max_comm_superstep_var[static_cast<int>(step)] >= expr);
            }
        }

        // vertex type restrictions
        for (const vertex_idx_t<Graph_t> &node : instance.vertices()) {
            for (unsigned int processor = 0; processor < instance.numberOfProcessors(); processor++) {
                if(!instance.isCompatible(node, processor)) {
                    for (unsigned int step = 0; step < max_number_supersteps; step++) {
                        model.AddConstr(node_to_processor_superstep_var[node][processor][static_cast<int>(step)] == 0);
                    }
                }
            }
        }

        /*
        Objective function
          */
        Expr expr;

        for (unsigned int step = 0; step < max_number_supersteps; step++) {
            expr += max_work_superstep_var[static_cast<int>(step)] +
                    instance.communicationCosts() * max_comm_superstep_var[static_cast<int>(step)] +
                    instance.synchronisationCosts() * superstep_used_var[static_cast<int>(step)];
        }

        model.SetObjective(expr - instance.synchronisationCosts(), COPT_MINIMIZE);
    }

  public:
    CoptFullScheduler(unsigned steps = 5)
        : allow_recomputation(false), use_memory_constraint(false), use_initial_schedule(false), initial_schedule(0),
          write_solutions_found(false), max_number_supersteps(steps) {

        // solution_callback.comm_processor_to_processor_superstep_node_var_ptr =
        //     &comm_processor_to_processor_superstep_node_var;
        // solution_callback.node_to_processor_superstep_var_ptr = &node_to_processor_superstep_var;
    }

    CoptFullScheduler(const BspScheduleCS<Graph_t> &schedule)
        : allow_recomputation(false), use_memory_constraint(false), use_initial_schedule(true),
          initial_schedule(&schedule), write_solutions_found(false),
          max_number_supersteps(schedule.numberOfSupersteps()) {

        // solution_callback.comm_processor_to_processor_superstep_node_var_ptr =
        //     &comm_processor_to_processor_superstep_node_var;
        // solution_callback.node_to_processor_superstep_var_ptr = &node_to_processor_superstep_var;
    }

        CoptFullScheduler(const BspScheduleRecomp<Graph_t> &schedule)
        : allow_recomputation(true), use_memory_constraint(false), use_initial_schedule_recomp(true),
          initial_schedule_recomp(&schedule), write_solutions_found(false),
          max_number_supersteps(schedule.numberOfSupersteps()) {
    }

    virtual ~CoptFullScheduler() = default;

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

        BspScheduleCS<Graph_t> schedule_cs(schedule.getInstance());
        RETURN_STATUS status = computeScheduleCS(schedule_cs);
        if (status == SUCCESS || status == BEST_FOUND) {
            schedule = std::move(schedule_cs);
            return status;
        } else {
            return status;
        }
    }
    virtual RETURN_STATUS computeScheduleCS(BspScheduleCS<Graph_t> &schedule) override {

        auto &instance = schedule.getInstance();

        if (use_initial_schedule &&
            (max_number_supersteps < initial_schedule->numberOfSupersteps() ||
             instance.numberOfProcessors() != initial_schedule->getInstance().numberOfProcessors() ||
             instance.numberOfVertices() != initial_schedule->getInstance().numberOfVertices())) {
            throw std::invalid_argument("Invalid Argument while computeSchedule(instance): instance parameters do not "
                                        "agree with those of the initial schedule's instance!");
        }

        Envr env;
        Model model = env.CreateModel("bsp_schedule_cs");

        setupVariablesConstraintsObjective(instance, model);

        if (use_initial_schedule) {
            loadInitialSchedule(model);
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

            WriteSolutionCallback solution_callback;
            solution_callback.comm_processor_to_processor_superstep_node_var_ptr =
                &comm_processor_to_processor_superstep_node_var;
            solution_callback.node_to_processor_superstep_var_ptr = &node_to_processor_superstep_var;
            solution_callback.solution_file_prefix_cb = solution_file_prefix;
            solution_callback.write_solutions_path_cb = write_solutions_path;
            solution_callback.instance_ptr = &instance;

            model.SetCallback(&solution_callback, COPT_CBCONTEXT_MIPSOL);
        }

        model.Solve();

        if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {

            constructBspScheduleFromSolution(schedule, true);
            return SUCCESS;

        } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {

            return ERROR;

        } else {

            if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {

                constructBspScheduleFromSolution(schedule, true);
                return BEST_FOUND;

            } else {
                return TIMEOUT;
            }
        }
    }

    virtual RETURN_STATUS computeScheduleRecomp(BspScheduleRecomp<Graph_t> &schedule) {

        allow_recomputation = true;

        if (use_initial_schedule &&
            (max_number_supersteps < initial_schedule->numberOfSupersteps() ||
            schedule.getInstance().numberOfProcessors() != initial_schedule->getInstance().numberOfProcessors() ||
            schedule.getInstance().numberOfVertices() != initial_schedule->getInstance().numberOfVertices())) {
            throw std::invalid_argument("Invalid Argument while computeScheduleRecomp: instance parameters do not "
                                        "agree with those of the initial schedule's instance!");
        }

        if (use_initial_schedule_recomp &&
            (max_number_supersteps < initial_schedule_recomp->numberOfSupersteps() ||
            schedule.getInstance().numberOfProcessors() != initial_schedule_recomp->getInstance().numberOfProcessors() ||
            schedule.getInstance().numberOfVertices() != initial_schedule_recomp->getInstance().numberOfVertices())) {
            throw std::invalid_argument("Invalid Argument while computeScheduleRecomp: instance parameters do not "
                                        "agree with those of the initial schedule's instance!");
        }

        Envr env;
        Model model = env.CreateModel("bsp_schedule");

        setupVariablesConstraintsObjective(schedule.getInstance(), model);

        if (use_initial_schedule || use_initial_schedule_recomp) {
            loadInitialSchedule(model);
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

            WriteSolutionCallback solution_callback;
            solution_callback.instance_ptr = &schedule.getInstance();
            solution_callback.comm_processor_to_processor_superstep_node_var_ptr =
                &comm_processor_to_processor_superstep_node_var;
            solution_callback.node_to_processor_superstep_var_ptr = &node_to_processor_superstep_var;
            solution_callback.solution_file_prefix_cb = solution_file_prefix;
            solution_callback.write_solutions_path_cb = write_solutions_path;
            solution_callback.allow_recomputation_cb = allow_recomputation;
            std::cout << "setting up callback with recomputation: " << allow_recomputation << std::endl;
            model.SetCallback(&solution_callback, COPT_CBCONTEXT_MIPSOL);
        }

        model.Solve();

        allow_recomputation = false;

        if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {

            constructBspScheduleRecompFromSolution(schedule, true);
            return SUCCESS;

        } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {

            return ERROR;

        } else {

            if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {

                constructBspScheduleRecompFromSolution(schedule, true);
                return BEST_FOUND;

            } else {
                return TIMEOUT;
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
    inline void setInitialSolutionFromBspSchedule(const BspScheduleCS<Graph_t> &schedule) {

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
        write_solutions_path = path;
        solution_file_prefix = file_prefix;
    }

    /**
     * Disables writing intermediate solutions.
     *
     * This function disables the writing of intermediate solutions. After
     * calling this function, the `enableWriteIntermediateSol` function needs
     * to be called again in order to enable writing of intermediate solutions.
     */
    inline void disableWriteIntermediateSol() { write_solutions_found = false; }

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
     * @brief Get the maximum number of supersteps.
     *
     * @return The maximum number of supersteps.
     */
    inline unsigned getMaxNumberOfSupersteps() const { return max_number_supersteps; }

    /**
     * @brief Get the name of the schedule.
     *
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const override { return "FullIlp"; }
};

} // namespace osp