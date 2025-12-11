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

#include "callbackbase.h"
#include "coptcpp_pch.h"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/computational_dag_util.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"
#include "osp/pebbling/PebblingSchedule.hpp"
#include "osp/pebbling/pebblers/pebblingILP/COPTEnv.hpp"

namespace osp {

template <typename Graph_t>
class MultiProcessorPebbling : public Scheduler<Graph_t> {
    static_assert(is_computational_dag_v<Graph_t>, "PebblingSchedule can only be used with computational DAGs.");

  private:
    using vertex_idx = vertex_idx_t<Graph_t>;
    using workweight_type = v_workw_t<Graph_t>;
    using commweight_type = v_commw_t<Graph_t>;
    using memweight_type = v_memw_t<Graph_t>;

    Model model;

    bool write_solutions_found;

    class WriteSolutionCallback : public CallbackBase {
      private:
        unsigned counter;
        unsigned max_number_solution;

        double best_obj;

      public:
        WriteSolutionCallback()
            : counter(0),
              max_number_solution(500),
              best_obj(COPT_INFINITY),
              write_solutions_path_cb(""),
              solution_file_prefix_cb("") {}

        std::string write_solutions_path_cb;
        std::string solution_file_prefix_cb;

        void callback() override;
    };

    WriteSolutionCallback solution_callback;

  protected:
    std::vector<std::vector<VarArray>> compute;
    std::vector<std::vector<VarArray>> send_up;
    std::vector<std::vector<VarArray>> send_down;
    std::vector<std::vector<VarArray>> has_red;
    std::vector<VarArray> has_blue;

    std::vector<std::vector<std::vector<bool>>> compute_exists;
    std::vector<std::vector<std::vector<bool>>> send_up_exists;
    std::vector<std::vector<std::vector<bool>>> send_down_exists;
    std::vector<std::vector<bool>> has_blue_exists;

    VarArray comp_phase;
    VarArray comm_phase;
    VarArray send_up_phase;
    VarArray send_down_phase;

    VarArray comm_phase_ends;
    VarArray comp_phase_ends;

    unsigned max_time = 0;
    unsigned time_limit_seconds;

    // problem settings
    bool slidingPebbles = false;
    bool mergeSteps = true;
    bool synchronous = true;
    bool up_and_down_cost_summed = true;
    bool allows_recomputation = true;
    bool restrict_step_types = false;
    unsigned compute_steps_per_cycle = 3;
    bool need_to_load_inputs = true;
    std::set<vertex_idx> needs_blue_at_end;
    std::vector<std::set<vertex_idx>> has_red_in_beginning;
    bool verbose = false;

    void constructPebblingScheduleFromSolution(PebblingSchedule<Graph_t> &schedule);

    void setInitialSolution(const BspInstance<Graph_t> &instance,
                            const std::vector<std::vector<std::vector<vertex_idx>>> &computeSteps,
                            const std::vector<std::vector<std::vector<vertex_idx>>> &sendUpSteps,
                            const std::vector<std::vector<std::vector<vertex_idx>>> &sendDownSteps,
                            const std::vector<std::vector<std::vector<vertex_idx>>> &nodesEvictedAfterStep);

    unsigned computeMaxTimeForInitialSolution(const BspInstance<Graph_t> &instance,
                                              const std::vector<std::vector<std::vector<vertex_idx>>> &computeSteps,
                                              const std::vector<std::vector<std::vector<vertex_idx>>> &sendUpSteps,
                                              const std::vector<std::vector<std::vector<vertex_idx>>> &sendDownSteps) const;

    void setupBaseVariablesConstraints(const BspInstance<Graph_t> &instance);

    void setupSyncPhaseVariablesConstraints(const BspInstance<Graph_t> &instance);
    void setupSyncObjective(const BspInstance<Graph_t> &instance);

    void setupAsyncVariablesConstraintsObjective(const BspInstance<Graph_t> &instance);
    void setupBspVariablesConstraintsObjective(const BspInstance<Graph_t> &instance);

    void solveILP();

  public:
    MultiProcessorPebbling()
        : Scheduler<Graph_t>(), model(COPTEnv::getInstance().CreateModel("MPP")), write_solutions_found(false), max_time(0) {}

    virtual ~MultiProcessorPebbling() = default;

    virtual RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &schedule) override;
    virtual RETURN_STATUS computeSynchPebbling(PebblingSchedule<Graph_t> &schedule);

    virtual RETURN_STATUS computePebbling(PebblingSchedule<Graph_t> &schedule, bool use_async = false);

    virtual RETURN_STATUS computePebblingWithInitialSolution(const PebblingSchedule<Graph_t> &initial_solution,
                                                             PebblingSchedule<Graph_t> &out_schedule,
                                                             bool use_async = false);

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
     * Disables writing intermediate solutions.
     *
     * This function disables the writing of intermediate solutions. After
     * calling this function, the `enableWriteIntermediateSol` function needs
     * to be called again in order to enable writing of intermediate solutions.
     */
    inline void disableWriteIntermediateSol() { write_solutions_found = false; }

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
    virtual std::string getScheduleName() const override { return "MultiProcessorPebbling"; }

    // getters and setters for problem parameters
    inline bool allowsSlidingPebbles() const { return slidingPebbles; }

    inline bool allowsMergingSteps() const { return mergeSteps; }

    inline bool isUpAndDownCostSummed() const { return up_and_down_cost_summed; }

    inline bool allowsRecomputation() const { return allows_recomputation; }

    inline bool hasRestrictedStepTypes() const { return restrict_step_types; }

    inline bool needsToLoadInputs() const { return need_to_load_inputs; }

    inline unsigned getComputeStepsPerCycle() const { return compute_steps_per_cycle; }

    inline unsigned getMaxTime() const { return max_time; }

    inline void setSlidingPebbles(const bool slidingPebbles_) { slidingPebbles = slidingPebbles_; }

    inline void setMergingSteps(const bool mergeSteps_) { mergeSteps = mergeSteps_; }

    inline void setUpAndDownCostSummed(const bool is_summed_) { up_and_down_cost_summed = is_summed_; }

    inline void setRecomputation(const bool allow_recompute_) { allows_recomputation = allow_recompute_; }

    inline void setRestrictStepTypes(const bool restrict_) {
        restrict_step_types = restrict_;
        if (restrict_) { mergeSteps = true; }
    }

    inline void setNeedToLoadInputs(const bool load_inputs_) { need_to_load_inputs = load_inputs_; }

    inline void setComputeStepsPerCycle(const unsigned steps_per_cycle_) { compute_steps_per_cycle = steps_per_cycle_; }

    inline void setMaxTime(const unsigned max_time_) { max_time = max_time_; }

    inline void setNeedsBlueAtEnd(const std::set<vertex_idx> &needs_blue_) { needs_blue_at_end = needs_blue_; }

    inline void setHasRedInBeginning(const std::vector<std::set<vertex_idx>> &has_red_) { has_red_in_beginning = has_red_; }

    inline void setVerbose(const bool verbose_) { verbose = verbose_; }

    inline void setTimeLimitSeconds(unsigned time_limit_seconds_) { time_limit_seconds = time_limit_seconds_; }

    bool hasEmptyStep(const BspInstance<Graph_t> &instance);
};

// implementation

template <typename Graph_t>
void MultiProcessorPebbling<Graph_t>::solveILP() {
    if (!verbose) { model.SetIntParam(COPT_INTPARAM_LOGTOCONSOLE, 0); }

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

template <typename Graph_t>
RETURN_STATUS MultiProcessorPebbling<Graph_t>::computeSchedule(BspSchedule<Graph_t> &schedule) {
    if (max_time == 0) { max_time = 2 * static_cast<unsigned>(schedule.getInstance().numberOfVertices()); }

    setupBaseVariablesConstraints(schedule.getInstance());
    setupSyncPhaseVariablesConstraints(schedule.getInstance());
    setupBspVariablesConstraintsObjective(schedule.getInstance());

    solveILP();

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {
        return RETURN_STATUS::OSP_SUCCESS;

    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {
        return RETURN_STATUS::ERROR;

    } else {
        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
            return RETURN_STATUS::BEST_FOUND;

        } else {
            return RETURN_STATUS::TIMEOUT;
        }
    }
};

template <typename Graph_t>
RETURN_STATUS MultiProcessorPebbling<Graph_t>::computeSynchPebbling(PebblingSchedule<Graph_t> &schedule) {
    const BspInstance<Graph_t> &instance = schedule.getInstance();

    if (max_time == 0) { max_time = 2 * static_cast<unsigned>(instance.numberOfVertices()); }

    mergeSteps = false;

    setupBaseVariablesConstraints(instance);
    setupSyncPhaseVariablesConstraints(instance);
    setupSyncObjective(instance);

    solveILP();

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {
        constructPebblingScheduleFromSolution(schedule);
        return RETURN_STATUS::OSP_SUCCESS;

    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {
        return RETURN_STATUS::ERROR;

    } else {
        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
            constructPebblingScheduleFromSolution(schedule);
            return RETURN_STATUS::OSP_SUCCESS;

        } else {
            return RETURN_STATUS::TIMEOUT;
        }
    }
}

template <typename Graph_t>
RETURN_STATUS MultiProcessorPebbling<Graph_t>::computePebbling(PebblingSchedule<Graph_t> &schedule, bool use_async) {
    const BspInstance<Graph_t> &instance = schedule.getInstance();

    if (max_time == 0) { max_time = 2 * static_cast<unsigned>(instance.numberOfVertices()); }

    synchronous = !use_async;

    setupBaseVariablesConstraints(instance);
    if (synchronous) {
        setupSyncPhaseVariablesConstraints(instance);
        setupBspVariablesConstraintsObjective(instance);
    } else {
        setupAsyncVariablesConstraintsObjective(instance);
    }

    solveILP();

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {
        constructPebblingScheduleFromSolution(schedule);
        return schedule.isValid() ? RETURN_STATUS::OSP_SUCCESS : RETURN_STATUS::ERROR;

    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {
        return RETURN_STATUS::ERROR;

    } else {
        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
            constructPebblingScheduleFromSolution(schedule);
            return schedule.isValid() ? RETURN_STATUS::OSP_SUCCESS : RETURN_STATUS::ERROR;

        } else {
            return RETURN_STATUS::TIMEOUT;
        }
    }
}

template <typename Graph_t>
RETURN_STATUS MultiProcessorPebbling<Graph_t>::computePebblingWithInitialSolution(const PebblingSchedule<Graph_t> &initial_solution,
                                                                                  PebblingSchedule<Graph_t> &out_schedule,
                                                                                  bool use_async) {
    const BspInstance<Graph_t> &instance = initial_solution.getInstance();

    std::vector<std::vector<std::vector<vertex_idx>>> computeSteps;
    std::vector<std::vector<std::vector<vertex_idx>>> sendUpSteps;
    std::vector<std::vector<std::vector<vertex_idx>>> sendDownSteps;
    std::vector<std::vector<std::vector<vertex_idx>>> nodesEvictedAfterStep;

    synchronous = !use_async;

    initial_solution.getDataForMultiprocessorPebbling(computeSteps, sendUpSteps, sendDownSteps, nodesEvictedAfterStep);

    max_time = computeMaxTimeForInitialSolution(instance, computeSteps, sendUpSteps, sendDownSteps);

    if (verbose) { std::cout << "Max time set at " << max_time << std::endl; }

    setupBaseVariablesConstraints(instance);
    if (synchronous) {
        setupSyncPhaseVariablesConstraints(instance);
        setupBspVariablesConstraintsObjective(instance);
    } else {
        setupAsyncVariablesConstraintsObjective(instance);
    }

    setInitialSolution(instance, computeSteps, sendUpSteps, sendDownSteps, nodesEvictedAfterStep);

    if (verbose) { std::cout << "Initial solution set." << std::endl; }

    solveILP();

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {
        constructPebblingScheduleFromSolution(out_schedule);
        return out_schedule.isValid() ? RETURN_STATUS::OSP_SUCCESS : RETURN_STATUS::ERROR;

    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {
        return RETURN_STATUS::ERROR;

    } else {
        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
            constructPebblingScheduleFromSolution(out_schedule);
            return out_schedule.isValid() ? RETURN_STATUS::OSP_SUCCESS : RETURN_STATUS::ERROR;

        } else {
            return RETURN_STATUS::TIMEOUT;
        }
    }
}

template <typename Graph_t>
void MultiProcessorPebbling<Graph_t>::setupBaseVariablesConstraints(const BspInstance<Graph_t> &instance) {
    /*
        Variables
    */
    compute = std::vector<std::vector<VarArray>>(instance.numberOfVertices(), std::vector<VarArray>(instance.numberOfProcessors()));

    for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            compute[node][processor] = model.AddVars(static_cast<int>(max_time), COPT_BINARY, "node_processor_time");
        }
    }

    compute_exists.resize(instance.numberOfVertices(),
                          std::vector<std::vector<bool>>(instance.numberOfProcessors(), std::vector<bool>(max_time, true)));

    send_up = std::vector<std::vector<VarArray>>(instance.numberOfVertices(), std::vector<VarArray>(instance.numberOfProcessors()));

    for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            send_up[node][processor] = model.AddVars(static_cast<int>(max_time), COPT_BINARY, "send_up");
        }
    }

    send_up_exists.resize(instance.numberOfVertices(),
                          std::vector<std::vector<bool>>(instance.numberOfProcessors(), std::vector<bool>(max_time, true)));

    send_down
        = std::vector<std::vector<VarArray>>(instance.numberOfVertices(), std::vector<VarArray>(instance.numberOfProcessors()));

    for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            send_down[node][processor] = model.AddVars(static_cast<int>(max_time), COPT_BINARY, "send_down");
        }
    }

    send_down_exists.resize(instance.numberOfVertices(),
                            std::vector<std::vector<bool>>(instance.numberOfProcessors(), std::vector<bool>(max_time, true)));

    has_blue = std::vector<VarArray>(instance.numberOfVertices());

    for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
        has_blue[node] = model.AddVars(static_cast<int>(max_time), COPT_BINARY, "blue_pebble");
    }

    has_blue_exists.resize(instance.numberOfVertices(), std::vector<bool>(max_time, true));

    has_red = std::vector<std::vector<VarArray>>(instance.numberOfVertices(), std::vector<VarArray>(instance.numberOfProcessors()));

    for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            has_red[node][processor] = model.AddVars(static_cast<int>(max_time), COPT_BINARY, "red_pebble");
        }
    }

    /*
        Invalidate variables based on various factors (node types, input loading, step type restriction)
    */

    for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            if (!instance.isCompatible(node, processor)) {
                for (unsigned t = 0; t < max_time; t++) {
                    compute_exists[node][processor][t] = false;
                    send_up_exists[node][processor][t] = false;
                }
            }
        }
    }

    // restrict source nodes if they need to be loaded
    if (need_to_load_inputs) {
        for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
            if (instance.getComputationalDag().in_degree(node) == 0) {
                for (unsigned t = 0; t < max_time; t++) {
                    for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
                        compute_exists[node][processor][t] = false;
                        send_up_exists[node][processor][t] = false;
                    }
                    has_blue_exists[node][t] = false;
                }
            }
        }
    }

    // restrict step types for simpler ILP
    if (restrict_step_types) {
        for (unsigned t = 0; t < max_time; t++) {
            bool this_is_a_comm_step = (t % (compute_steps_per_cycle + 2) == compute_steps_per_cycle + 1);
            if (!need_to_load_inputs && t % (compute_steps_per_cycle + 2) == compute_steps_per_cycle) {
                this_is_a_comm_step = true;
            }
            if (need_to_load_inputs && t % (compute_steps_per_cycle + 2) == 0) { this_is_a_comm_step = true; }
            if (this_is_a_comm_step) {
                for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
                    for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
                        compute_exists[node][processor][t] = false;
                    }
                }
            } else {
                for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
                    for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
                        send_up_exists[node][processor][t] = false;
                        send_down_exists[node][processor][t] = false;
                    }
                }
            }
        }
    }

    /*
        Constraints
    */

    if (!mergeSteps) {
        for (unsigned t = 0; t < max_time; t++) {
            for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
                Expr expr;
                for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
                    if (compute_exists[node][processor][t]) { expr += compute[node][processor][static_cast<int>(t)]; }
                    if (send_up_exists[node][processor][t]) { expr += send_up[node][processor][static_cast<int>(t)]; }
                    if (send_down_exists[node][processor][t]) { expr += send_down[node][processor][static_cast<int>(t)]; }
                }
                model.AddConstr(expr <= 1);
            }
        }
    } else {
        // extra variables to indicate step types in step merging
        std::vector<VarArray> comp_step_on_proc = std::vector<VarArray>(instance.numberOfProcessors());
        std::vector<VarArray> comm_step_on_proc = std::vector<VarArray>(instance.numberOfProcessors());

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            comp_step_on_proc[processor] = model.AddVars(static_cast<int>(max_time), COPT_BINARY, "comp_step_on_proc");
            comm_step_on_proc[processor] = model.AddVars(static_cast<int>(max_time), COPT_BINARY, "comm_step_on_proc");
        }

        const unsigned M = static_cast<unsigned>(instance.numberOfVertices());

        for (unsigned t = 0; t < max_time; t++) {
            for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
                Expr expr_comp, expr_comm;
                for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
                    if (compute_exists[node][processor][t]) { expr_comp += compute[node][processor][static_cast<int>(t)]; }
                    if (send_up_exists[node][processor][t]) { expr_comm += send_up[node][processor][static_cast<int>(t)]; }
                    if (send_down_exists[node][processor][t]) { expr_comm += send_down[node][processor][static_cast<int>(t)]; }
                }

                model.AddConstr(M * comp_step_on_proc[processor][static_cast<int>(t)] >= expr_comp);
                model.AddConstr(2 * M * comm_step_on_proc[processor][static_cast<int>(t)] >= expr_comm);

                model.AddConstr(
                    comp_step_on_proc[processor][static_cast<int>(t)] + comm_step_on_proc[processor][static_cast<int>(t)] <= 1);
            }
        }
    }

    for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
        for (unsigned t = 1; t < max_time; t++) {
            if (!has_blue_exists[node][t]) { continue; }

            Expr expr;

            for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
                if (send_up_exists[node][processor][t - 1]) { expr += send_up[node][processor][static_cast<int>(t) - 1]; }
            }
            model.AddConstr(has_blue[node][static_cast<int>(t)] <= has_blue[node][static_cast<int>(t) - 1] + expr);
        }
    }

    for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            for (unsigned t = 1; t < max_time; t++) {
                Expr expr;

                if (compute_exists[node][processor][t - 1]) { expr += compute[node][processor][static_cast<int>(t) - 1]; }

                if (send_down_exists[node][processor][t - 1]) { expr += send_down[node][processor][static_cast<int>(t) - 1]; }

                model.AddConstr(has_red[node][processor][static_cast<int>(t)]
                                <= has_red[node][processor][static_cast<int>(t) - 1] + expr);
            }
        }
    }

    for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            for (unsigned t = 0; t < max_time; t++) {
                if (!compute_exists[node][processor][t]) { continue; }

                for (const auto &source : instance.getComputationalDag().parents(node)) {
                    if (!mergeSteps || !compute_exists[source][processor][t]) {
                        model.AddConstr(compute[node][processor][static_cast<int>(t)]
                                        <= has_red[source][processor][static_cast<int>(t)]);
                    } else {
                        model.AddConstr(compute[node][processor][static_cast<int>(t)]
                                        <= has_red[source][processor][static_cast<int>(t)]
                                               + compute[source][processor][static_cast<int>(t)]);
                    }
                }
            }
        }
    }

    for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            for (unsigned t = 0; t < max_time; t++) {
                if (send_up_exists[node][processor][t]) {
                    model.AddConstr(send_up[node][processor][static_cast<int>(t)] <= has_red[node][processor][static_cast<int>(t)]);
                }
            }
        }
    }

    for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            for (unsigned t = 0; t < max_time; t++) {
                if (send_down_exists[node][processor][t] && has_blue_exists[node][t]) {
                    model.AddConstr(send_down[node][processor][static_cast<int>(t)] <= has_blue[node][static_cast<int>(t)]);
                }
            }
        }
    }

    for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
        for (unsigned t = 0; t < max_time; t++) {
            Expr expr;
            for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
                expr += has_red[node][processor][static_cast<int>(t)] * instance.getComputationalDag().vertex_mem_weight(node);
                if (!slidingPebbles && compute_exists[node][processor][t]) {
                    expr += compute[node][processor][static_cast<int>(t)] * instance.getComputationalDag().vertex_mem_weight(node);
                }
            }

            model.AddConstr(expr <= instance.getArchitecture().memoryBound(processor));
        }
    }

    for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            if (has_red_in_beginning.empty()
                || has_red_in_beginning[processor].find(node) == has_red_in_beginning[processor].end()) {
                model.AddConstr(has_red[node][processor][0] == 0);
            } else {
                model.AddConstr(has_red[node][processor][0] == 1);
            }
        }
    }

    for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
        if (!need_to_load_inputs || instance.getComputationalDag().in_degree(node) > 0) {
            model.AddConstr(has_blue[node][0] == 0);
        }
    }

    if (needs_blue_at_end.empty())    // default case: blue pebbles required on sinks at the end
    {
        for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
            if (instance.getComputationalDag().out_degree(node) == 0 && has_blue_exists[node][max_time - 1]) {
                model.AddConstr(has_blue[node][static_cast<int>(max_time) - 1] == 1);
            }
        }
    } else    // otherwise: specified set of nodes that need blue at the end
    {
        for (vertex_idx node : needs_blue_at_end) {
            if (has_blue_exists[node][max_time - 1]) { model.AddConstr(has_blue[node][static_cast<int>(max_time) - 1] == 1); }
        }
    }

    // disable recomputation if needed
    if (!allows_recomputation) {
        for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
            Expr expr;
            for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
                for (unsigned t = 0; t < max_time; t++) {
                    if (compute_exists[node][processor][t]) { expr += compute[node][processor][static_cast<int>(t)]; }
                }
            }

            model.AddConstr(expr <= 1);
        }
    }
};

template <typename Graph_t>
void MultiProcessorPebbling<Graph_t>::setupSyncPhaseVariablesConstraints(const BspInstance<Graph_t> &instance) {
    comp_phase = model.AddVars(static_cast<int>(max_time), COPT_BINARY, "comp_phase");

    if (mergeSteps) {
        comm_phase = model.AddVars(static_cast<int>(max_time), COPT_BINARY, "comm_phase");
    } else {
        send_up_phase = model.AddVars(static_cast<int>(max_time), COPT_BINARY, "send_up_phase");
        send_down_phase = model.AddVars(static_cast<int>(max_time), COPT_BINARY, "send_down_phase");
    }

    const unsigned M = static_cast<unsigned>(instance.numberOfProcessors() * instance.numberOfVertices());

    for (unsigned t = 0; t < max_time; t++) {
        Expr expr_comp, expr_comm, expr_send_up, expr_send_down;
        for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
            for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
                if (compute_exists[node][processor][t]) { expr_comp += compute[node][processor][static_cast<int>(t)]; }
                if (mergeSteps) {
                    if (send_up_exists[node][processor][t]) { expr_comm += send_up[node][processor][static_cast<int>(t)]; }

                    if (send_down_exists[node][processor][t]) { expr_comm += send_down[node][processor][static_cast<int>(t)]; }
                } else {
                    if (send_up_exists[node][processor][t]) { expr_send_up += send_up[node][processor][static_cast<int>(t)]; }

                    if (send_down_exists[node][processor][t]) {
                        expr_send_down += send_down[node][processor][static_cast<int>(t)];
                    }
                }
            }
        }

        model.AddConstr(M * comp_phase[static_cast<int>(t)] >= expr_comp);
        if (mergeSteps) {
            model.AddConstr(2 * M * comm_phase[static_cast<int>(t)] >= expr_comm);
            model.AddConstr(comp_phase[static_cast<int>(t)] + comm_phase[static_cast<int>(t)] <= 1);
        } else {
            model.AddConstr(M * send_up_phase[static_cast<int>(t)] >= expr_send_up);
            model.AddConstr(M * send_down_phase[static_cast<int>(t)] >= expr_send_down);
            model.AddConstr(
                comp_phase[static_cast<int>(t)] + send_up_phase[static_cast<int>(t)] + send_down_phase[static_cast<int>(t)] <= 1);
        }
    }
};

template <typename Graph_t>
void MultiProcessorPebbling<Graph_t>::setupBspVariablesConstraintsObjective(const BspInstance<Graph_t> &instance) {
    comp_phase_ends = model.AddVars(static_cast<int>(max_time), COPT_BINARY, "comp_phase_ends");

    comm_phase_ends = model.AddVars(static_cast<int>(max_time), COPT_BINARY, "comm_phase_ends");

    VarArray work_induced = model.AddVars(static_cast<int>(max_time), COPT_CONTINUOUS, "work_induced");
    VarArray comm_induced = model.AddVars(static_cast<int>(max_time), COPT_CONTINUOUS, "comm_induced");

    std::vector<VarArray> work_step_until(instance.numberOfProcessors());
    std::vector<VarArray> comm_step_until(instance.numberOfProcessors());
    std::vector<VarArray> send_up_step_until(instance.numberOfProcessors());
    std::vector<VarArray> send_down_step_until(instance.numberOfProcessors());

    VarArray send_up_induced;
    VarArray send_down_induced;
    if (up_and_down_cost_summed) {
        send_up_induced = model.AddVars(static_cast<int>(max_time), COPT_CONTINUOUS, "send_up_induced");
        send_down_induced = model.AddVars(static_cast<int>(max_time), COPT_CONTINUOUS, "send_down_induced");
    }

    for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
        work_step_until[processor] = model.AddVars(static_cast<int>(max_time), COPT_CONTINUOUS, "work_step_until");
        send_up_step_until[processor] = model.AddVars(static_cast<int>(max_time), COPT_CONTINUOUS, "send_up_step_until");
        send_down_step_until[processor] = model.AddVars(static_cast<int>(max_time), COPT_CONTINUOUS, "send_up_step_until");
    }

    for (unsigned t = 0; t < max_time; t++) {
        model.AddConstr(comp_phase[static_cast<int>(t)] >= comp_phase_ends[static_cast<int>(t)]);
        if (mergeSteps) {
            model.AddConstr(comm_phase[static_cast<int>(t)] >= comm_phase_ends[static_cast<int>(t)]);
        } else {
            model.AddConstr(send_down_phase[static_cast<int>(t)] + send_up_phase[static_cast<int>(t)]
                            >= comm_phase_ends[static_cast<int>(t)]);
        }
    }
    for (unsigned t = 0; t < max_time - 1; t++) {
        model.AddConstr(comp_phase_ends[static_cast<int>(t)]
                        >= comp_phase[static_cast<int>(t)] - comp_phase[static_cast<int>(t) + 1]);
        if (mergeSteps) {
            model.AddConstr(comm_phase_ends[static_cast<int>(t)]
                            >= comm_phase[static_cast<int>(t)] - comm_phase[static_cast<int>(t) + 1]);
        } else {
            model.AddConstr(comm_phase_ends[static_cast<int>(t)]
                            >= send_down_phase[static_cast<int>(t)] + send_up_phase[static_cast<int>(t)]
                                   - send_down_phase[static_cast<int>(t) + 1] - send_up_phase[static_cast<int>(t) + 1]);
        }
    }

    model.AddConstr(comp_phase_ends[static_cast<int>(max_time) - 1] >= comp_phase[static_cast<int>(max_time) - 1]);
    if (mergeSteps) {
        model.AddConstr(comm_phase_ends[static_cast<int>(max_time) - 1] >= comm_phase[static_cast<int>(max_time) - 1]);
    } else {
        model.AddConstr(comm_phase_ends[static_cast<int>(max_time) - 1]
                        >= send_down_phase[static_cast<int>(max_time) - 1] + send_up_phase[static_cast<int>(max_time) - 1]);
    }

    const unsigned M = static_cast<unsigned>(instance.numberOfProcessors()
                                             * (sumOfVerticesWorkWeights(instance.getComputationalDag())
                                                + sumOfVerticesCommunicationWeights(instance.getComputationalDag())));

    for (unsigned t = 1; t < max_time; t++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            Expr expr_work;
            Expr expr_send_up;
            Expr expr_send_down;
            for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
                if (compute_exists[node][processor][t]) {
                    expr_work += instance.getComputationalDag().vertex_work_weight(node)
                                 * compute[node][processor][static_cast<int>(t)];
                }
                if (send_up_exists[node][processor][t]) {
                    expr_send_up += instance.getComputationalDag().vertex_comm_weight(node)
                                    * send_up[node][processor][static_cast<int>(t)];
                }
                if (send_down_exists[node][processor][t]) {
                    expr_send_down += instance.getComputationalDag().vertex_comm_weight(node)
                                      * send_down[node][processor][static_cast<int>(t)];
                }
            }

            model.AddConstr(M * comm_phase_ends[static_cast<int>(t)] + work_step_until[processor][static_cast<int>(t)]
                            >= work_step_until[processor][static_cast<int>(t) - 1] + expr_work);

            model.AddConstr(M * comp_phase_ends[static_cast<int>(t)] + send_up_step_until[processor][static_cast<int>(t)]
                            >= send_up_step_until[processor][static_cast<int>(t) - 1] + expr_send_up);

            model.AddConstr(M * comp_phase_ends[static_cast<int>(t)] + send_down_step_until[processor][static_cast<int>(t)]
                            >= send_down_step_until[processor][static_cast<int>(t) - 1] + expr_send_down);

            model.AddConstr(work_induced[static_cast<int>(t)]
                            >= work_step_until[processor][static_cast<int>(t)] - M * (1 - comp_phase_ends[static_cast<int>(t)]));
            if (up_and_down_cost_summed) {
                model.AddConstr(send_up_induced[static_cast<int>(t)] >= send_up_step_until[processor][static_cast<int>(t)]
                                                                            - M * (1 - comm_phase_ends[static_cast<int>(t)]));
                model.AddConstr(send_down_induced[static_cast<int>(t)] >= send_down_step_until[processor][static_cast<int>(t)]
                                                                              - M * (1 - comm_phase_ends[static_cast<int>(t)]));
                model.AddConstr(comm_induced[static_cast<int>(t)]
                                >= send_up_induced[static_cast<int>(t)] + send_down_induced[static_cast<int>(t)]);
            } else {
                model.AddConstr(comm_induced[static_cast<int>(t)] >= send_down_step_until[processor][static_cast<int>(t)]
                                                                         - M * (1 - comm_phase_ends[static_cast<int>(t)]));
                model.AddConstr(comm_induced[static_cast<int>(t)] >= send_up_step_until[processor][static_cast<int>(t)]
                                                                         - M * (1 - comm_phase_ends[static_cast<int>(t)]));
            }
        }
    }

    // t = 0
    for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
        Expr expr_work;
        Expr expr_send_up;
        Expr expr_send_down;
        for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
            if (compute_exists[node][processor][0]) {
                expr_work += instance.getComputationalDag().vertex_work_weight(node) * compute[node][processor][0];
            }
            if (send_up_exists[node][processor][0]) {
                expr_send_up += instance.getComputationalDag().vertex_comm_weight(node) * send_up[node][processor][0];
            }
            if (send_down_exists[node][processor][0]) {
                expr_send_down += instance.getComputationalDag().vertex_comm_weight(node) * send_down[node][processor][0];
            }
        }

        model.AddConstr(M * comm_phase_ends[0] + work_step_until[processor][0] >= expr_work);

        model.AddConstr(M * comp_phase_ends[0] + send_up_step_until[processor][0] >= expr_send_up);

        model.AddConstr(M * comp_phase_ends[0] + send_down_step_until[processor][0] >= expr_send_down);

        model.AddConstr(work_induced[0] >= work_step_until[processor][0] - M * (1 - comp_phase_ends[0]));
        if (up_and_down_cost_summed) {
            model.AddConstr(send_up_induced[0] >= send_up_step_until[processor][0] - M * (1 - comm_phase_ends[0]));
            model.AddConstr(send_down_induced[0] >= send_down_step_until[processor][0] - M * (1 - comm_phase_ends[0]));
            model.AddConstr(comm_induced[0] >= send_up_induced[0] + send_down_induced[0]);
        } else {
            model.AddConstr(comm_induced[0] >= send_down_step_until[processor][0] - M * (1 - comm_phase_ends[0]));
            model.AddConstr(comm_induced[0] >= send_up_step_until[processor][0] - M * (1 - comm_phase_ends[0]));
        }
    }

    /*
    Objective
*/

    Expr expr;
    for (unsigned t = 0; t < max_time; t++) {
        expr += work_induced[static_cast<int>(t)] + instance.synchronisationCosts() * comm_phase_ends[static_cast<int>(t)]
                + instance.communicationCosts() * comm_induced[static_cast<int>(t)];
    }

    model.SetObjective(expr, COPT_MINIMIZE);
};

template <typename Graph_t>
void MultiProcessorPebbling<Graph_t>::setupSyncObjective(const BspInstance<Graph_t> &instance) {
    Expr expr;
    for (unsigned t = 0; t < max_time; t++) {
        if (!mergeSteps) {
            expr += comp_phase[static_cast<int>(t)] + instance.communicationCosts() * send_up_phase[static_cast<int>(t)]
                    + instance.communicationCosts() * send_down_phase[static_cast<int>(t)];
        } else {
            // this objective+parameter combination is not very meaningful, but still defined here to avoid a segfault otherwise
            expr += comp_phase[static_cast<int>(t)] + instance.communicationCosts() * comm_phase[static_cast<int>(t)];
        }
    }

    model.SetObjective(expr, COPT_MINIMIZE);
}

template <typename Graph_t>
void MultiProcessorPebbling<Graph_t>::setupAsyncVariablesConstraintsObjective(const BspInstance<Graph_t> &instance) {
    std::vector<VarArray> finish_times(instance.numberOfProcessors());

    for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
        finish_times[processor] = model.AddVars(static_cast<int>(max_time), COPT_CONTINUOUS, "finish_times");
    }

    Var makespan = model.AddVar(0, COPT_INFINITY, 1, COPT_CONTINUOUS, "makespan");

    VarArray gets_blue = model.AddVars(static_cast<int>(instance.numberOfVertices()), COPT_CONTINUOUS, "gets_blue");

    const unsigned M = static_cast<unsigned>(instance.numberOfProcessors()
                                             * (sumOfVerticesWorkWeights(instance.getComputationalDag())
                                                + sumOfVerticesCommunicationWeights(instance.getComputationalDag())));

    for (unsigned t = 0; t < max_time; t++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            Expr send_down_step_length;
            for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
                if (send_down_exists[node][processor][t]) {
                    send_down_step_length += instance.communicationCosts()
                                             * instance.getComputationalDag().vertex_comm_weight(node)
                                             * send_down[node][processor][static_cast<int>(t)];
                }
            }

            for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
                if (send_up_exists[node][processor][t]) {
                    model.AddConstr(gets_blue[static_cast<int>(node)]
                                    >= finish_times[processor][static_cast<int>(t)]
                                           - (1 - send_up[node][processor][static_cast<int>(t)]) * M);
                }
                if (send_down_exists[node][processor][t]) {
                    model.AddConstr(gets_blue[static_cast<int>(node)]
                                    <= finish_times[processor][static_cast<int>(t)]
                                           + (1 - send_down[node][processor][static_cast<int>(t)]) * M - send_down_step_length);
                }
            }
        }
    }

    // makespan constraint
    for (unsigned t = 0; t < max_time; t++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            model.AddConstr(makespan >= finish_times[processor][static_cast<int>(t)]);
        }
    }

    // t = 0
    for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
        Expr expr;
        for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
            if (compute_exists[node][processor][0]) {
                expr += instance.getComputationalDag().vertex_work_weight(node) * compute[node][processor][0];
            }

            if (send_up_exists[node][processor][0]) {
                expr += instance.communicationCosts() * instance.getComputationalDag().vertex_comm_weight(node)
                        * send_up[node][processor][0];
            }

            if (send_down_exists[node][processor][0]) {
                expr += instance.communicationCosts() * instance.getComputationalDag().vertex_comm_weight(node)
                        * send_down[node][processor][0];
            }
        }

        model.AddConstr(finish_times[processor][0] >= expr);
    }

    for (unsigned t = 1; t < max_time; t++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            Expr expr;
            for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
                if (compute_exists[node][processor][t]) {
                    expr += instance.getComputationalDag().vertex_work_weight(node) * compute[node][processor][static_cast<int>(t)];
                }

                if (send_up_exists[node][processor][t]) {
                    expr += instance.communicationCosts() * instance.getComputationalDag().vertex_comm_weight(node)
                            * send_up[node][processor][static_cast<int>(t)];
                }

                if (send_down_exists[node][processor][t]) {
                    expr += instance.communicationCosts() * instance.getComputationalDag().vertex_comm_weight(node)
                            * send_down[node][processor][static_cast<int>(t)];
                }
            }

            model.AddConstr(finish_times[processor][static_cast<int>(t)]
                            >= finish_times[processor][static_cast<int>(t) - 1] + expr);
        }
    }

    /*
    Objective
      */

    model.SetObjective(makespan, COPT_MINIMIZE);
}

template <typename Graph_t>
void MultiProcessorPebbling<Graph_t>::WriteSolutionCallback::callback() {
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

        } catch (const std::exception &e) {}
    }
};

template <typename Graph_t>
void MultiProcessorPebbling<Graph_t>::constructPebblingScheduleFromSolution(PebblingSchedule<Graph_t> &schedule) {
    const BspInstance<Graph_t> &instance = schedule.getInstance();

    std::vector<std::vector<std::set<std::pair<unsigned, vertex_idx>>>> nodes_computed(
        instance.numberOfProcessors(), std::vector<std::set<std::pair<unsigned, vertex_idx>>>(max_time));
    std::vector<std::vector<std::deque<vertex_idx>>> nodes_sent_up(instance.numberOfProcessors(),
                                                                   std::vector<std::deque<vertex_idx>>(max_time));
    std::vector<std::vector<std::deque<vertex_idx>>> nodes_sent_down(instance.numberOfProcessors(),
                                                                     std::vector<std::deque<vertex_idx>>(max_time));
    std::vector<std::vector<std::set<vertex_idx>>> evicted_after(instance.numberOfProcessors(),
                                                                 std::vector<std::set<vertex_idx>>(max_time));

    // used to remove unneeded steps when a node is sent down and then up (which becomes invalid after reordering the comm phases)
    std::vector<std::vector<bool>> sent_down_already(instance.numberOfVertices(),
                                                     std::vector<bool>(instance.numberOfProcessors(), false));
    std::vector<std::vector<bool>> ignore_red(instance.numberOfVertices(), std::vector<bool>(instance.numberOfProcessors(), false));

    std::vector<vertex_idx> topOrder = GetTopOrder(instance.getComputationalDag());
    std::vector<unsigned> topOrderPosition(instance.numberOfVertices());
    for (unsigned index = 0; index < instance.numberOfVertices(); ++index) { topOrderPosition[topOrder[index]] = index; }

    std::vector<bool> empty_step(max_time, true);
    std::vector<std::vector<unsigned>> step_type_on_proc(instance.numberOfProcessors(), std::vector<unsigned>(max_time, 0));

    for (unsigned step = 0; step < max_time; step++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
                if (compute_exists[node][processor][step]
                    && compute[node][processor][static_cast<int>(step)].Get(COPT_DBLINFO_VALUE) >= .99) {
                    step_type_on_proc[processor][step] = 1;
                }
            }
        }
    }

    for (unsigned step = 0; step < max_time; step++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
                if (step > 0 && has_red[node][processor][static_cast<int>(step) - 1].Get(COPT_DBLINFO_VALUE) >= .99
                    && has_red[node][processor][static_cast<int>(step)].Get(COPT_DBLINFO_VALUE) <= .01
                    && !ignore_red[node][processor]) {
                    for (size_t previous_step = step - 1; previous_step < step; --previous_step) {
                        if (!nodes_computed[processor][previous_step].empty() || !nodes_sent_up[processor][previous_step].empty()
                            || !nodes_sent_down[processor][previous_step].empty() || previous_step == 0) {
                            evicted_after[processor][previous_step].insert(node);
                            empty_step[previous_step] = false;
                            break;
                        }
                    }
                }

                if (compute_exists[node][processor][step]
                    && compute[node][processor][static_cast<int>(step)].Get(COPT_DBLINFO_VALUE) >= .99) {
                    nodes_computed[processor][step].emplace(topOrderPosition[node], node);
                    empty_step[step] = false;
                    ignore_red[node][processor] = false;

                    // implicit eviction in case of mergesteps - never having "has_red=1"
                    if (step + 1 < max_time && has_red[node][processor][static_cast<int>(step) + 1].Get(COPT_DBLINFO_VALUE) <= .01) {
                        evicted_after[processor][step].insert(node);
                    }
                }

                if (send_down_exists[node][processor][step]
                    && send_down[node][processor][static_cast<int>(step)].Get(COPT_DBLINFO_VALUE) >= .99) {
                    bool keep_step = false;

                    for (unsigned next_step = step + 1;
                         next_step < max_time
                         && has_red[node][processor][static_cast<int>(next_step)].Get(COPT_DBLINFO_VALUE) >= .99;
                         ++next_step) {
                        if (step_type_on_proc[processor][next_step] == 1) {
                            keep_step = true;
                            break;
                        }
                    }

                    if (keep_step) {
                        nodes_sent_down[processor][step].push_back(node);
                        empty_step[step] = false;
                        step_type_on_proc[processor][step] = 3;
                        ignore_red[node][processor] = false;
                    } else {
                        ignore_red[node][processor] = true;
                    }

                    sent_down_already[node][processor] = true;
                }

                if (send_up_exists[node][processor][step]
                    && send_up[node][processor][static_cast<int>(step)].Get(COPT_DBLINFO_VALUE) >= .99
                    && !sent_down_already[node][processor]) {
                    nodes_sent_up[processor][step].push_back(node);
                    empty_step[step] = false;
                    step_type_on_proc[processor][step] = 2;
                }
            }
        }
    }

    // components of the final PebblingSchedule - the first two dimensions are always processor and superstep
    std::vector<std::vector<std::vector<vertex_idx>>> compute_steps_per_supstep(instance.numberOfProcessors());
    std::vector<std::vector<std::vector<std::vector<vertex_idx>>>> nodes_evicted_after_compute(instance.numberOfProcessors());
    std::vector<std::vector<std::vector<vertex_idx>>> nodes_sent_up_in_supstep(instance.numberOfProcessors());
    std::vector<std::vector<std::vector<vertex_idx>>> nodes_sent_down_in_supstep(instance.numberOfProcessors());
    std::vector<std::vector<std::vector<vertex_idx>>> nodes_evicted_in_comm_phase(instance.numberOfProcessors());

    // edge case: check if an extra superstep must be added in the beginning to evict values that are initially in cache
    bool needs_evict_step_in_beginning = false;
    for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
        for (unsigned step = 0; step < max_time; step++) {
            if (step_type_on_proc[proc][step] == 0 && !evicted_after[proc][step].empty()) {
                needs_evict_step_in_beginning = true;
                break;
            } else if (step_type_on_proc[proc][step] > 0) {
                break;
            }
        }
    }

    // create the actual PebblingSchedule - iterating over the steps
    unsigned superstepIndex = 0;
    if (synchronous) {
        bool in_comm = true;
        superstepIndex = UINT_MAX;

        if (needs_evict_step_in_beginning) {
            // artificially insert comm step in beginning, if it would start with compute otherwise
            bool begins_with_compute = false;
            for (unsigned step = 0; step < max_time; step++) {
                bool is_comp = false, is_comm = false;
                for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
                    if (step_type_on_proc[proc][step] == 1) { is_comp = true; }
                    if (step_type_on_proc[proc][step] > 1) { is_comm = true; }
                }
                if (is_comp) { begins_with_compute = true; }
                if (is_comp || is_comm) { break; }
            }

            if (begins_with_compute) {
                superstepIndex = 0;
                for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
                    compute_steps_per_supstep[proc].push_back(std::vector<vertex_idx>());
                    nodes_evicted_after_compute[proc].push_back(std::vector<std::vector<vertex_idx>>());
                    nodes_sent_up_in_supstep[proc].push_back(std::vector<vertex_idx>());
                    nodes_sent_down_in_supstep[proc].push_back(std::vector<vertex_idx>());
                    nodes_evicted_in_comm_phase[proc].push_back(std::vector<vertex_idx>());
                }
            }
        }

        // process steps
        for (unsigned step = 0; step < max_time; step++) {
            if (empty_step[step]) { continue; }

            unsigned step_type = 0;
            for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
                step_type = std::max(step_type, step_type_on_proc[proc][step]);
            }

            if (step_type == 1) {
                if (in_comm) {
                    for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
                        compute_steps_per_supstep[proc].push_back(std::vector<vertex_idx>());
                        nodes_evicted_after_compute[proc].push_back(std::vector<std::vector<vertex_idx>>());
                        nodes_sent_up_in_supstep[proc].push_back(std::vector<vertex_idx>());
                        nodes_sent_down_in_supstep[proc].push_back(std::vector<vertex_idx>());
                        nodes_evicted_in_comm_phase[proc].push_back(std::vector<vertex_idx>());
                    }
                    ++superstepIndex;
                    in_comm = false;
                }
                for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
                    for (auto index_and_node : nodes_computed[proc][step]) {
                        compute_steps_per_supstep[proc][superstepIndex].push_back(index_and_node.second);
                        nodes_evicted_after_compute[proc][superstepIndex].push_back(std::vector<vertex_idx>());
                    }
                    for (vertex_idx node : evicted_after[proc][step]) {
                        if (!nodes_evicted_after_compute[proc][superstepIndex].empty()) {
                            nodes_evicted_after_compute[proc][superstepIndex].back().push_back(node);
                        } else {
                            // can only happen in special case: eviction in the very beginning
                            nodes_evicted_in_comm_phase[proc][0].push_back(node);
                        }
                    }
                }
            }

            if (step_type == 2 || step_type == 3) {
                if (superstepIndex == UINT_MAX) {
                    for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
                        compute_steps_per_supstep[proc].push_back(std::vector<vertex_idx>());
                        nodes_evicted_after_compute[proc].push_back(std::vector<std::vector<vertex_idx>>());
                        nodes_sent_up_in_supstep[proc].push_back(std::vector<vertex_idx>());
                        nodes_sent_down_in_supstep[proc].push_back(std::vector<vertex_idx>());
                        nodes_evicted_in_comm_phase[proc].push_back(std::vector<vertex_idx>());
                    }
                    ++superstepIndex;
                }

                in_comm = true;
                for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
                    for (vertex_idx node : nodes_sent_up[proc][step]) {
                        nodes_sent_up_in_supstep[proc][superstepIndex].push_back(node);
                    }
                    for (vertex_idx node : evicted_after[proc][step]) {
                        nodes_evicted_in_comm_phase[proc][superstepIndex].push_back(node);
                    }
                    for (vertex_idx node : nodes_sent_down[proc][step]) {
                        nodes_sent_down_in_supstep[proc][superstepIndex].push_back(node);
                    }
                }
            }
        }
    } else {
        std::vector<unsigned> step_idx_on_proc(instance.numberOfProcessors(), 0);

        std::vector<bool> already_has_blue(instance.numberOfVertices(), false);
        if (need_to_load_inputs) {
            for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
                if (instance.getComputationalDag().in_degree(node) == 0) { already_has_blue[node] = true; }
            }
        }

        std::vector<bool> proc_finished(instance.numberOfProcessors(), false);
        unsigned nr_proc_finished = 0;
        while (nr_proc_finished < instance.numberOfProcessors()) {
            // preliminary sweep of superstep, to see if we need to wait for other processors
            std::vector<unsigned> idx_limit_on_proc = step_idx_on_proc;

            // first add compute steps
            if (!needs_evict_step_in_beginning || superstepIndex > 0) {
                for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
                    while (idx_limit_on_proc[proc] < max_time && step_type_on_proc[proc][idx_limit_on_proc[proc]] <= 1) {
                        ++idx_limit_on_proc[proc];
                    }
                }
            }

            // then add communications step until possible (note - they might not be valid if all put into a single superstep!)
            std::set<vertex_idx> new_blues;
            bool still_making_progress = true;
            while (still_making_progress) {
                still_making_progress = false;
                for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
                    while (idx_limit_on_proc[proc] < max_time && step_type_on_proc[proc][idx_limit_on_proc[proc]] != 1) {
                        bool accept_step = true;
                        for (vertex_idx node : nodes_sent_down[proc][idx_limit_on_proc[proc]]) {
                            if (!already_has_blue[node] && new_blues.find(node) == new_blues.end()) { accept_step = false; }
                        }

                        if (!accept_step) { break; }

                        for (vertex_idx node : nodes_sent_up[proc][idx_limit_on_proc[proc]]) {
                            if (!already_has_blue[node]) { new_blues.insert(node); }
                        }

                        still_making_progress = true;
                        ++idx_limit_on_proc[proc];
                    }
                }
            }

            // actually process the superstep
            for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
                compute_steps_per_supstep[proc].push_back(std::vector<vertex_idx>());
                nodes_evicted_after_compute[proc].push_back(std::vector<std::vector<vertex_idx>>());
                nodes_sent_up_in_supstep[proc].push_back(std::vector<vertex_idx>());
                nodes_sent_down_in_supstep[proc].push_back(std::vector<vertex_idx>());
                nodes_evicted_in_comm_phase[proc].push_back(std::vector<vertex_idx>());

                while (step_idx_on_proc[proc] < idx_limit_on_proc[proc] && step_type_on_proc[proc][step_idx_on_proc[proc]] <= 1) {
                    for (auto index_and_node : nodes_computed[proc][step_idx_on_proc[proc]]) {
                        compute_steps_per_supstep[proc][superstepIndex].push_back(index_and_node.second);
                        nodes_evicted_after_compute[proc][superstepIndex].push_back(std::vector<vertex_idx>());
                    }
                    for (vertex_idx node : evicted_after[proc][step_idx_on_proc[proc]]) {
                        if (!nodes_evicted_after_compute[proc][superstepIndex].empty()) {
                            nodes_evicted_after_compute[proc][superstepIndex].back().push_back(node);
                        } else {
                            // can only happen in special case: eviction in the very beginning
                            nodes_evicted_in_comm_phase[proc][superstepIndex].push_back(node);
                        }
                    }

                    ++step_idx_on_proc[proc];
                }
                while (step_idx_on_proc[proc] < idx_limit_on_proc[proc] && step_type_on_proc[proc][step_idx_on_proc[proc]] != 1) {
                    for (vertex_idx node : nodes_sent_up[proc][step_idx_on_proc[proc]]) {
                        nodes_sent_up_in_supstep[proc][superstepIndex].push_back(node);
                        already_has_blue[node] = true;
                    }
                    for (vertex_idx node : nodes_sent_down[proc][step_idx_on_proc[proc]]) {
                        nodes_sent_down_in_supstep[proc][superstepIndex].push_back(node);
                    }
                    for (vertex_idx node : evicted_after[proc][step_idx_on_proc[proc]]) {
                        nodes_evicted_in_comm_phase[proc][superstepIndex].push_back(node);
                    }

                    ++step_idx_on_proc[proc];
                }
                if (step_idx_on_proc[proc] == max_time && !proc_finished[proc]) {
                    proc_finished[proc] = true;
                    ++nr_proc_finished;
                }
            }
            ++superstepIndex;
        }
    }

    std::cout << "MPP ILP best solution value: " << model.GetDblAttr(COPT_DBLATTR_BESTOBJ)
              << ", best lower bound: " << model.GetDblAttr(COPT_DBLATTR_BESTBND) << std::endl;

    schedule = PebblingSchedule<Graph_t>(instance,
                                         compute_steps_per_supstep,
                                         nodes_evicted_after_compute,
                                         nodes_sent_up_in_supstep,
                                         nodes_sent_down_in_supstep,
                                         nodes_evicted_in_comm_phase,
                                         needs_blue_at_end,
                                         has_red_in_beginning,
                                         need_to_load_inputs);
}

template <typename Graph_t>
void MultiProcessorPebbling<Graph_t>::setInitialSolution(
    const BspInstance<Graph_t> &instance,
    const std::vector<std::vector<std::vector<vertex_idx>>> &computeSteps,
    const std::vector<std::vector<std::vector<vertex_idx>>> &sendUpSteps,
    const std::vector<std::vector<std::vector<vertex_idx>>> &sendDownSteps,
    const std::vector<std::vector<std::vector<vertex_idx>>> &nodesEvictedAfterStep) {
    const unsigned N = static_cast<unsigned>(instance.numberOfVertices());

    std::vector<bool> in_slow_mem(N, false);
    if (need_to_load_inputs) {
        for (vertex_idx node = 0; node < N; ++node) {
            if (instance.getComputationalDag().in_degree(node) == 0) { in_slow_mem[node] = true; }
        }
    }

    std::vector<std::vector<unsigned>> in_fast_mem(N, std::vector<unsigned>(instance.numberOfProcessors(), false));
    if (!has_red_in_beginning.empty()) {
        for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {
            for (vertex_idx node : has_red_in_beginning[proc]) { in_fast_mem[node][proc] = true; }
        }
    }

    unsigned step = 0, new_step_idx = 0;
    for (; step < computeSteps[0].size(); ++step) {
        for (vertex_idx node = 0; node < N; ++node) {
            if (has_blue_exists[node][new_step_idx]) {
                model.SetMipStart(has_blue[node][static_cast<int>(new_step_idx)], static_cast<int>(in_slow_mem[node]));
            }
            for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {
                model.SetMipStart(has_red[node][proc][static_cast<int>(new_step_idx)], static_cast<int>(in_fast_mem[node][proc]));
            }
        }

        if (restrict_step_types) {
            // align step number with step type cycle's phase, if needed
            bool skip_step = true;
            while (skip_step) {
                skip_step = false;
                bool is_compute = false, is_send_up = false, is_send_down = false;
                for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {
                    if (!computeSteps[proc][step].empty()) { is_compute = true; }
                    if (!sendUpSteps[proc][step].empty()) { is_send_up = true; }
                    if (!sendDownSteps[proc][step].empty()) { is_send_down = true; }
                }

                bool send_up_step_idx
                    = (need_to_load_inputs && (new_step_idx % (compute_steps_per_cycle + 2) == compute_steps_per_cycle + 1))
                      || (!need_to_load_inputs && (new_step_idx % (compute_steps_per_cycle + 2) == compute_steps_per_cycle));
                bool send_down_step_idx
                    = (need_to_load_inputs && (new_step_idx % (compute_steps_per_cycle + 2) == 0))
                      || (!need_to_load_inputs && (new_step_idx % (compute_steps_per_cycle + 2) == compute_steps_per_cycle + 1));

                if (is_compute && (send_up_step_idx || send_down_step_idx)) { skip_step = true; }
                if (is_send_up && !send_up_step_idx) { skip_step = true; }
                if (is_send_down && !send_down_step_idx) { skip_step = true; }

                if (skip_step) {
                    ++new_step_idx;
                    for (vertex_idx node = 0; node < N; ++node) {
                        if (has_blue_exists[node][new_step_idx]) {
                            model.SetMipStart(has_blue[node][static_cast<int>(new_step_idx)], static_cast<int>(in_slow_mem[node]));
                        }
                        for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {
                            model.SetMipStart(has_red[node][proc][static_cast<int>(new_step_idx)],
                                              static_cast<int>(in_fast_mem[node][proc]));
                        }
                    }
                }
            }
        }

        for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {
            std::vector<bool> value_of_node(N, false);
            for (vertex_idx node : computeSteps[proc][step]) {
                value_of_node[node] = true;
                if (compute_exists[node][proc][new_step_idx]) {
                    model.SetMipStart(compute[node][proc][static_cast<int>(new_step_idx)], 1);
                }
                in_fast_mem[node][proc] = true;
            }
            for (vertex_idx node : computeSteps[proc][step]) {
                if (!value_of_node[node]) {
                    if (compute_exists[node][proc][new_step_idx]) {
                        model.SetMipStart(compute[node][proc][static_cast<int>(new_step_idx)], 0);
                    }
                } else {
                    value_of_node[node] = false;
                }
            }

            for (vertex_idx node : sendUpSteps[proc][step]) {
                value_of_node[node] = true;
                if (send_up_exists[node][proc][new_step_idx]) {
                    model.SetMipStart(send_up[node][proc][static_cast<int>(new_step_idx)], 1);
                }
                in_slow_mem[node] = true;
            }
            for (vertex_idx node : sendUpSteps[proc][step]) {
                if (!value_of_node[node]) {
                    if (send_up_exists[node][proc][new_step_idx]) {
                        model.SetMipStart(send_up[node][proc][static_cast<int>(new_step_idx)], 0);
                    }
                } else {
                    value_of_node[node] = false;
                }
            }

            for (vertex_idx node : sendDownSteps[proc][step]) {
                value_of_node[node] = true;
                if (send_down_exists[node][proc][new_step_idx]) {
                    model.SetMipStart(send_down[node][proc][static_cast<int>(new_step_idx)], 1);
                }
                in_fast_mem[node][proc] = true;
            }
            for (vertex_idx node : sendDownSteps[proc][step]) {
                if (!value_of_node[node]) {
                    if (send_down_exists[node][proc][new_step_idx]) {
                        model.SetMipStart(send_down[node][proc][static_cast<int>(new_step_idx)], 0);
                    }
                } else {
                    value_of_node[node] = false;
                }
            }

            for (vertex_idx node : nodesEvictedAfterStep[proc][step]) { in_fast_mem[node][proc] = false; }
        }
        ++new_step_idx;
    }
    for (; new_step_idx < max_time; ++new_step_idx) {
        for (vertex_idx node = 0; node < N; ++node) {
            if (has_blue_exists[node][new_step_idx]) {
                model.SetMipStart(has_blue[node][static_cast<int>(new_step_idx)], static_cast<int>(in_slow_mem[node]));
            }
            for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {
                model.SetMipStart(has_red[node][proc][static_cast<int>(new_step_idx)], 0);
                if (compute_exists[node][proc][new_step_idx]) {
                    model.SetMipStart(compute[node][proc][static_cast<int>(new_step_idx)], 0);
                }
                if (send_up_exists[node][proc][new_step_idx]) {
                    model.SetMipStart(send_up[node][proc][static_cast<int>(new_step_idx)], 0);
                }
                if (send_down_exists[node][proc][new_step_idx]) {
                    model.SetMipStart(send_down[node][proc][static_cast<int>(new_step_idx)], 0);
                }
            }
        }
    }
    model.LoadMipStart();
}

template <typename Graph_t>
unsigned MultiProcessorPebbling<Graph_t>::computeMaxTimeForInitialSolution(
    const BspInstance<Graph_t> &instance,
    const std::vector<std::vector<std::vector<vertex_idx>>> &computeSteps,
    const std::vector<std::vector<std::vector<vertex_idx>>> &sendUpSteps,
    const std::vector<std::vector<std::vector<vertex_idx>>> &sendDownSteps) const {
    if (!restrict_step_types) { return static_cast<unsigned>(computeSteps[0].size()) + 3; }

    unsigned step = 0, new_step_idx = 0;
    for (; step < computeSteps[0].size(); ++step) {
        // align step number with step type cycle's phase, if needed
        bool skip_step = true;
        while (skip_step) {
            skip_step = false;
            bool is_compute = false, is_send_up = false, is_send_down = false;
            for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {
                if (!computeSteps[proc][step].empty()) { is_compute = true; }
                if (!sendUpSteps[proc][step].empty()) { is_send_up = true; }
                if (!sendDownSteps[proc][step].empty()) { is_send_down = true; }
            }

            bool send_up_step_idx
                = (need_to_load_inputs && (new_step_idx % (compute_steps_per_cycle + 2) == compute_steps_per_cycle + 1))
                  || (!need_to_load_inputs && (new_step_idx % (compute_steps_per_cycle + 2) == compute_steps_per_cycle));
            bool send_down_step_idx
                = (need_to_load_inputs && (new_step_idx % (compute_steps_per_cycle + 2) == 0))
                  || (!need_to_load_inputs && (new_step_idx % (compute_steps_per_cycle + 2) == compute_steps_per_cycle + 1));

            if (is_compute && (send_up_step_idx || send_down_step_idx)) { skip_step = true; }
            if (is_send_up && !send_up_step_idx) { skip_step = true; }
            if (is_send_down && !send_down_step_idx) { skip_step = true; }

            if (skip_step) { ++new_step_idx; }
        }

        ++new_step_idx;
    }

    new_step_idx += compute_steps_per_cycle + 2;
    return new_step_idx;
}

template <typename Graph_t>
bool MultiProcessorPebbling<Graph_t>::hasEmptyStep(const BspInstance<Graph_t> &instance) {
    for (unsigned step = 0; step < max_time; ++step) {
        bool empty = true;
        for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
            for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
                if ((compute_exists[node][processor][step] && compute[node][processor][step].Get(COPT_DBLINFO_VALUE) >= .99)
                    || (send_up_exists[node][processor][step] && send_up[node][processor][step].Get(COPT_DBLINFO_VALUE) >= .99)
                    || (send_down_exists[node][processor][step]
                        && send_down[node][processor][step].Get(COPT_DBLINFO_VALUE) >= .99)) {
                    empty = false;
                }
            }
        }
        if (empty) { return true; }
    }
    return false;
}

}    // namespace osp
