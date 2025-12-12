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

template <typename GraphT>
class MultiProcessorPebbling : public Scheduler<GraphT> {
    static_assert(IsComputationalDagV<Graph_t>, "PebblingSchedule can only be used with computational DAGs.");

  private:
    using vertex_idx = vertex_idx_t<Graph_t>;
    using workweight_type = v_workw_t<Graph_t>;
    using commweight_type = v_commw_t<Graph_t>;
    using memweight_type = v_memw_t<Graph_t>;

    Model model_;

    bool writeSolutionsFound_;

    class WriteSolutionCallback : public CallbackBase {
      private:
        unsigned counter_;
        unsigned maxNumberSolution_;

        double bestObj_;

      public:
        WriteSolutionCallback()
            : counter_(0), maxNumberSolution_(500), best_obj(COPT_INFINITY), writeSolutionsPathCb_(""), solutionFilePrefixCb_("") {}

        std::string writeSolutionsPathCb_;
        std::string solutionFilePrefixCb_;

        void Callback() override;
    };

    WriteSolutionCallback solutionCallback_;

  protected:
    std::vector<std::vector<VarArray>> compute_;
    std::vector<std::vector<VarArray>> sendUp_;
    std::vector<std::vector<VarArray>> sendDown_;
    std::vector<std::vector<VarArray>> hasRed_;
    std::vector<VarArray> hasBlue_;

    std::vector<std::vector<std::vector<bool>>> computeExists_;
    std::vector<std::vector<std::vector<bool>>> sendUpExists_;
    std::vector<std::vector<std::vector<bool>>> sendDownExists_;
    std::vector<std::vector<bool>> hasBlueExists_;

    VarArray compPhase_;
    VarArray commPhase_;
    VarArray sendUpPhase_;
    VarArray sendDownPhase_;

    VarArray commPhaseEnds_;
    VarArray compPhaseEnds_;

    unsigned maxTime_ = 0;
    unsigned timeLimitSeconds_;

    // problem settings
    bool slidingPebbles_ = false;
    bool mergeSteps_ = true;
    bool synchronous_ = true;
    bool upAndDownCostSummed_ = true;
    bool allowsRecomputation_ = true;
    bool restrictStepTypes_ = false;
    unsigned computeStepsPerCycle_ = 3;
    bool needToLoadInputs_ = true;
    std::set<vertex_idx> needsBlueAtEnd_;
    std::vector<std::set<vertex_idx>> hasRedInBeginning_;
    bool verbose_ = false;

    void ConstructPebblingScheduleFromSolution(PebblingSchedule<GraphT> &schedule);

    void SetInitialSolution(const BspInstance<GraphT> &instance,
                            const std::vector<std::vector<std::vector<vertex_idx>>> &computeSteps,
                            const std::vector<std::vector<std::vector<vertex_idx>>> &sendUpSteps,
                            const std::vector<std::vector<std::vector<vertex_idx>>> &sendDownSteps,
                            const std::vector<std::vector<std::vector<vertex_idx>>> &nodesEvictedAfterStep);

    unsigned ComputeMaxTimeForInitialSolution(const BspInstance<GraphT> &instance,
                                              const std::vector<std::vector<std::vector<vertex_idx>>> &computeSteps,
                                              const std::vector<std::vector<std::vector<vertex_idx>>> &sendUpSteps,
                                              const std::vector<std::vector<std::vector<vertex_idx>>> &sendDownSteps) const;

    void SetupBaseVariablesConstraints(const BspInstance<GraphT> &instance);

    void SetupSyncPhaseVariablesConstraints(const BspInstance<GraphT> &instance);
    void SetupSyncObjective(const BspInstance<GraphT> &instance);

    void SetupAsyncVariablesConstraintsObjective(const BspInstance<GraphT> &instance);
    void SetupBspVariablesConstraintsObjective(const BspInstance<GraphT> &instance);

    void SolveIlp();

  public:
    MultiProcessorPebbling()
        : Scheduler<GraphT>(), model(COPTEnv::getInstance().CreateModel("MPP")), writeSolutionsFound_(false), maxTime_(0) {}

    virtual ~MultiProcessorPebbling() = default;

    virtual RETURN_STATUS computeSchedule(BspSchedule<GraphT> &schedule) override;
    virtual RETURN_STATUS ComputeSynchPebbling(PebblingSchedule<GraphT> &schedule);

    virtual RETURN_STATUS ComputePebbling(PebblingSchedule<GraphT> &schedule, bool useAsync = false);

    virtual RETURN_STATUS ComputePebblingWithInitialSolution(const PebblingSchedule<GraphT> &initialSolution,
                                                             PebblingSchedule<GraphT> &outSchedule,
                                                             bool useAsync = false);

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
    inline void EnableWriteIntermediateSol(std::string path, std::string filePrefix) {
        writeSolutionsFound_ = true;
        solutionCallback_.writeSolutionsPathCb_ = path;
        solutionCallback_.solutionFilePrefixCb_ = filePrefix;
    }

    /**
     * Disables writing intermediate solutions.
     *
     * This function disables the writing of intermediate solutions. After
     * calling this function, the `enableWriteIntermediateSol` function needs
     * to be called again in order to enable writing of intermediate solutions.
     */
    inline void DisableWriteIntermediateSol() { writeSolutionsFound_ = false; }

    /**
     * @brief Get the best gap found by the solver.
     *
     * @return The best gap found by the solver.
     */
    inline double BestGap() { return model.GetDblAttr(COPT_DBLATTR_BESTGAP); }

    /**
     * @brief Get the best objective value found by the solver.
     *
     * @return The best objective value found by the solver.
     */
    inline double BestObjective() { return model.GetDblAttr(COPT_DBLATTR_BESTOBJ); }

    /**
     * @brief Get the best bound found by the solver.
     *
     * @return The best bound found by the solver.
     */
    inline double BestBound() { return model.GetDblAttr(COPT_DBLATTR_BESTBND); }

    /**
     * @brief Get the name of the schedule.
     *
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const override { return "MultiProcessorPebbling"; }

    // getters and setters for problem parameters
    inline bool AllowsSlidingPebbles() const { return slidingPebbles_; }

    inline bool AllowsMergingSteps() const { return mergeSteps_; }

    inline bool IsUpAndDownCostSummed() const { return upAndDownCostSummed_; }

    inline bool AllowsRecomputation() const { return allowsRecomputation_; }

    inline bool HasRestrictedStepTypes() const { return restrictStepTypes_; }

    inline bool NeedsToLoadInputs() const { return needToLoadInputs_; }

    inline unsigned GetComputeStepsPerCycle() const { return computeStepsPerCycle_; }

    inline unsigned GetMaxTime() const { return maxTime_; }

    inline void SetSlidingPebbles(const bool slidingPebbles) { slidingPebbles_ = slidingPebbles; }

    inline void SetMergingSteps(const bool mergeSteps) { mergeSteps_ = mergeSteps; }

    inline void SetUpAndDownCostSummed(const bool isSummed) { upAndDownCostSummed_ = isSummed; }

    inline void SetRecomputation(const bool allowRecompute) { allowsRecomputation_ = allowRecompute; }

    inline void SetRestrictStepTypes(const bool restrict) {
        restrictStepTypes_ = restrict;
        if (restrict) {
            mergeSteps_ = true;
        }
    }

    inline void SetNeedToLoadInputs(const bool loadInputs) { needToLoadInputs_ = loadInputs; }

    inline void SetComputeStepsPerCycle(const unsigned stepsPerCycle) { computeStepsPerCycle_ = stepsPerCycle; }

    inline void SetMaxTime(const unsigned maxTime) { maxTime_ = maxTime; }

    inline void SetNeedsBlueAtEnd(const std::set<vertex_idx> &needsBlue) { needs_blue_at_end = needs_blue_; }

    inline void SetHasRedInBeginning(const std::vector<std::set<vertex_idx>> &hasRed) { has_red_in_beginning = has_red_; }

    inline void SetVerbose(const bool verbose) { verbose_ = verbose; }

    inline void SetTimeLimitSeconds(unsigned timeLimitSeconds) { timeLimitSeconds_ = timeLimitSeconds; }

    bool HasEmptyStep(const BspInstance<GraphT> &instance);
};

// implementation

template <typename GraphT>
void MultiProcessorPebbling<GraphT>::SolveIlp() {
    if (!verbose_) {
        model.SetIntParam(COPT_INTPARAM_LOGTOCONSOLE, 0);
    }

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

template <typename GraphT>
RETURN_STATUS MultiProcessorPebbling<GraphT>::ComputeSchedule(BspSchedule<GraphT> &schedule) {
    if (maxTime_ == 0) {
        maxTime_ = 2 * static_cast<unsigned>(schedule.getInstance().numberOfVertices());
    }

    SetupBaseVariablesConstraints(schedule.getInstance());
    SetupSyncPhaseVariablesConstraints(schedule.getInstance());
    SetupBspVariablesConstraintsObjective(schedule.getInstance());

    SolveIlp();

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

template <typename GraphT>
RETURN_STATUS MultiProcessorPebbling<GraphT>::ComputeSynchPebbling(PebblingSchedule<GraphT> &schedule) {
    const BspInstance<GraphT> &instance = schedule.getInstance();

    if (maxTime_ == 0) {
        maxTime_ = 2 * static_cast<unsigned>(instance.numberOfVertices());
    }

    mergeSteps_ = false;

    SetupBaseVariablesConstraints(instance);
    SetupSyncPhaseVariablesConstraints(instance);
    SetupSyncObjective(instance);

    SolveIlp();

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {
        ConstructPebblingScheduleFromSolution(schedule);
        return RETURN_STATUS::OSP_SUCCESS;

    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {
        return RETURN_STATUS::ERROR;

    } else {
        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
            ConstructPebblingScheduleFromSolution(schedule);
            return RETURN_STATUS::OSP_SUCCESS;

        } else {
            return RETURN_STATUS::TIMEOUT;
        }
    }
}

template <typename GraphT>
RETURN_STATUS MultiProcessorPebbling<GraphT>::ComputePebbling(PebblingSchedule<GraphT> &schedule, bool useAsync) {
    const BspInstance<GraphT> &instance = schedule.getInstance();

    if (maxTime_ == 0) {
        maxTime_ = 2 * static_cast<unsigned>(instance.numberOfVertices());
    }

    synchronous_ = !useAsync;

    SetupBaseVariablesConstraints(instance);
    if (synchronous_) {
        SetupSyncPhaseVariablesConstraints(instance);
        SetupBspVariablesConstraintsObjective(instance);
    } else {
        SetupAsyncVariablesConstraintsObjective(instance);
    }

    SolveIlp();

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {
        ConstructPebblingScheduleFromSolution(schedule);
        return schedule.isValid() ? RETURN_STATUS::OSP_SUCCESS : RETURN_STATUS::ERROR;

    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {
        return RETURN_STATUS::ERROR;

    } else {
        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
            ConstructPebblingScheduleFromSolution(schedule);
            return schedule.isValid() ? RETURN_STATUS::OSP_SUCCESS : RETURN_STATUS::ERROR;

        } else {
            return RETURN_STATUS::TIMEOUT;
        }
    }
}

template <typename GraphT>
RETURN_STATUS MultiProcessorPebbling<GraphT>::ComputePebblingWithInitialSolution(const PebblingSchedule<GraphT> &initialSolution,
                                                                                 PebblingSchedule<GraphT> &outSchedule,
                                                                                 bool useAsync) {
    const BspInstance<GraphT> &instance = initialSolution.getInstance();

    std::vector<std::vector<std::vector<vertex_idx>>> computeSteps;
    std::vector<std::vector<std::vector<vertex_idx>>> sendUpSteps;
    std::vector<std::vector<std::vector<vertex_idx>>> sendDownSteps;
    std::vector<std::vector<std::vector<vertex_idx>>> nodesEvictedAfterStep;

    synchronous_ = !useAsync;

    initialSolution.getDataForMultiprocessorPebbling(computeSteps, sendUpSteps, sendDownSteps, nodesEvictedAfterStep);

    max_time = computeMaxTimeForInitialSolution(instance, computeSteps, sendUpSteps, sendDownSteps);

    if (verbose_) {
        std::cout << "Max time set at " << maxTime_ << std::endl;
    }

    SetupBaseVariablesConstraints(instance);
    if (synchronous_) {
        SetupSyncPhaseVariablesConstraints(instance);
        SetupBspVariablesConstraintsObjective(instance);
    } else {
        SetupAsyncVariablesConstraintsObjective(instance);
    }

    setInitialSolution(instance, computeSteps, sendUpSteps, sendDownSteps, nodesEvictedAfterStep);

    if (verbose_) {
        std::cout << "Initial solution set." << std::endl;
    }

    SolveIlp();

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {
        ConstructPebblingScheduleFromSolution(outSchedule);
        return out_schedule.isValid() ? RETURN_STATUS::OSP_SUCCESS : RETURN_STATUS::ERROR;

    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {
        return RETURN_STATUS::ERROR;

    } else {
        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
            ConstructPebblingScheduleFromSolution(outSchedule);
            return out_schedule.isValid() ? RETURN_STATUS::OSP_SUCCESS : RETURN_STATUS::ERROR;

        } else {
            return RETURN_STATUS::TIMEOUT;
        }
    }
}

template <typename GraphT>
void MultiProcessorPebbling<GraphT>::SetupBaseVariablesConstraints(const BspInstance<GraphT> &instance) {
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
                for (unsigned t = 0; t < maxTime_; t++) {
                    compute_exists[node][processor][t] = false;
                    send_up_exists[node][processor][t] = false;
                }
            }
        }
    }

    // restrict source nodes if they need to be loaded
    if (needToLoadInputs_) {
        for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
            if (instance.getComputationalDag().in_degree(node) == 0) {
                for (unsigned t = 0; t < maxTime_; t++) {
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
    if (restrictStepTypes_) {
        for (unsigned t = 0; t < maxTime_; t++) {
            bool thisIsACommStep = (t % (computeStepsPerCycle_ + 2) == computeStepsPerCycle_ + 1);
            if (!needToLoadInputs_ && t % (computeStepsPerCycle_ + 2) == computeStepsPerCycle_) {
                thisIsACommStep = true;
            }
            if (needToLoadInputs_ && t % (computeStepsPerCycle_ + 2) == 0) {
                thisIsACommStep = true;
            }
            if (thisIsACommStep) {
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

    if (!mergeSteps_) {
        for (unsigned t = 0; t < maxTime_; t++) {
            for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
                Expr expr;
                for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
                    if (compute_exists[node][processor][t]) {
                        expr += compute[node][processor][static_cast<int>(t)];
                    }
                    if (send_up_exists[node][processor][t]) {
                        expr += send_up[node][processor][static_cast<int>(t)];
                    }
                    if (send_down_exists[node][processor][t]) {
                        expr += send_down[node][processor][static_cast<int>(t)];
                    }
                }
                model.AddConstr(expr <= 1);
            }
        }
    } else {
        // extra variables to indicate step types in step merging
        std::vector<VarArray> compStepOnProc = std::vector<VarArray>(instance.numberOfProcessors());
        std::vector<VarArray> commStepOnProc = std::vector<VarArray>(instance.numberOfProcessors());

        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            comp_step_on_proc[processor] = model.AddVars(static_cast<int>(max_time), COPT_BINARY, "comp_step_on_proc");
            comm_step_on_proc[processor] = model.AddVars(static_cast<int>(max_time), COPT_BINARY, "comm_step_on_proc");
        }

        const unsigned m = static_cast<unsigned>(instance.numberOfVertices());

        for (unsigned t = 0; t < maxTime_; t++) {
            for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
                Expr exprComp, expr_comm;
                for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
                    if (compute_exists[node][processor][t]) {
                        expr_comp += compute[node][processor][static_cast<int>(t)];
                    }
                    if (send_up_exists[node][processor][t]) {
                        expr_comm += send_up[node][processor][static_cast<int>(t)];
                    }
                    if (send_down_exists[node][processor][t]) {
                        expr_comm += send_down[node][processor][static_cast<int>(t)];
                    }
                }

                model.AddConstr(M * comp_step_on_proc[processor][static_cast<int>(t)] >= expr_comp);
                model.AddConstr(2 * M * comm_step_on_proc[processor][static_cast<int>(t)] >= expr_comm);

                model.AddConstr(
                    comp_step_on_proc[processor][static_cast<int>(t)] + comm_step_on_proc[processor][static_cast<int>(t)] <= 1);
            }
        }
    }

    for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
        for (unsigned t = 1; t < maxTime_; t++) {
            if (!has_blue_exists[node][t]) {
                continue;
            }

            Expr expr;

            for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
                if (send_up_exists[node][processor][t - 1]) {
                    expr += send_up[node][processor][static_cast<int>(t) - 1];
                }
            }
            model.AddConstr(has_blue[node][static_cast<int>(t)] <= has_blue[node][static_cast<int>(t) - 1] + expr);
        }
    }

    for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            for (unsigned t = 1; t < maxTime_; t++) {
                Expr expr;

                if (compute_exists[node][processor][t - 1]) {
                    expr += compute[node][processor][static_cast<int>(t) - 1];
                }

                if (send_down_exists[node][processor][t - 1]) {
                    expr += send_down[node][processor][static_cast<int>(t) - 1];
                }

                model.AddConstr(has_red[node][processor][static_cast<int>(t)]
                                <= has_red[node][processor][static_cast<int>(t) - 1] + expr);
            }
        }
    }

    for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            for (unsigned t = 0; t < maxTime_; t++) {
                if (!compute_exists[node][processor][t]) {
                    continue;
                }

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
            for (unsigned t = 0; t < maxTime_; t++) {
                if (send_up_exists[node][processor][t]) {
                    model.AddConstr(send_up[node][processor][static_cast<int>(t)] <= has_red[node][processor][static_cast<int>(t)]);
                }
            }
        }
    }

    for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            for (unsigned t = 0; t < maxTime_; t++) {
                if (send_down_exists[node][processor][t] && has_blue_exists[node][t]) {
                    model.AddConstr(send_down[node][processor][static_cast<int>(t)] <= has_blue[node][static_cast<int>(t)]);
                }
            }
        }
    }

    for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
        for (unsigned t = 0; t < maxTime_; t++) {
            Expr expr;
            for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
                expr += has_red[node][processor][static_cast<int>(t)] * instance.getComputationalDag().VertexMemWeight(node);
                if (!slidingPebbles && compute_exists[node][processor][t]) {
                    expr += compute[node][processor][static_cast<int>(t)] * instance.getComputationalDag().VertexMemWeight(node);
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
        if (!needToLoadInputs_ || instance.getComputationalDag().in_degree(node) > 0) {
            model.AddConstr(has_blue[node][0] == 0);
        }
    }

    if (needs_blue_at_end.empty())    // default case: blue pebbles required on sinks at the end
    {
        for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
            if (instance.getComputationalDag().OutDegree(node) == 0 && has_blue_exists[node][max_time - 1]) {
                model.AddConstr(has_blue[node][static_cast<int>(max_time) - 1] == 1);
            }
        }
    } else    // otherwise: specified set of nodes that need blue at the end
    {
        for (vertex_idx node : needs_blue_at_end) {
            if (has_blue_exists[node][max_time - 1]) {
                model.AddConstr(has_blue[node][static_cast<int>(max_time) - 1] == 1);
            }
        }
    }

    // disable recomputation if needed
    if (!allowsRecomputation_) {
        for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
            Expr expr;
            for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
                for (unsigned t = 0; t < maxTime_; t++) {
                    if (compute_exists[node][processor][t]) {
                        expr += compute[node][processor][static_cast<int>(t)];
                    }
                }
            }

            model.AddConstr(expr <= 1);
        }
    }
};

template <typename GraphT>
void MultiProcessorPebbling<GraphT>::SetupSyncPhaseVariablesConstraints(const BspInstance<GraphT> &instance) {
    comp_phase = model.AddVars(static_cast<int>(max_time), COPT_BINARY, "comp_phase");

    if (mergeSteps_) {
        comm_phase = model.AddVars(static_cast<int>(max_time), COPT_BINARY, "comm_phase");
    } else {
        send_up_phase = model.AddVars(static_cast<int>(max_time), COPT_BINARY, "send_up_phase");
        send_down_phase = model.AddVars(static_cast<int>(max_time), COPT_BINARY, "send_down_phase");
    }

    const unsigned m = static_cast<unsigned>(instance.numberOfProcessors() * instance.numberOfVertices());

    for (unsigned t = 0; t < maxTime_; t++) {
        Expr exprComp, expr_comm, expr_send_up, expr_send_down;
        for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
            for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
                if (compute_exists[node][processor][t]) {
                    expr_comp += compute[node][processor][static_cast<int>(t)];
                }
                if (mergeSteps_) {
                    if (send_up_exists[node][processor][t]) {
                        expr_comm += send_up[node][processor][static_cast<int>(t)];
                    }

                    if (send_down_exists[node][processor][t]) {
                        expr_comm += send_down[node][processor][static_cast<int>(t)];
                    }
                } else {
                    if (send_up_exists[node][processor][t]) {
                        expr_send_up += send_up[node][processor][static_cast<int>(t)];
                    }

                    if (send_down_exists[node][processor][t]) {
                        expr_send_down += send_down[node][processor][static_cast<int>(t)];
                    }
                }
            }
        }

        model.AddConstr(M * comp_phase[static_cast<int>(t)] >= expr_comp);
        if (mergeSteps_) {
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

template <typename GraphT>
void MultiProcessorPebbling<GraphT>::SetupBspVariablesConstraintsObjective(const BspInstance<GraphT> &instance) {
    comp_phase_ends = model.AddVars(static_cast<int>(max_time), COPT_BINARY, "comp_phase_ends");

    comm_phase_ends = model.AddVars(static_cast<int>(max_time), COPT_BINARY, "comm_phase_ends");

    VarArray workInduced = model.AddVars(static_cast<int>(max_time), COPT_CONTINUOUS, "work_induced");
    VarArray commInduced = model.AddVars(static_cast<int>(max_time), COPT_CONTINUOUS, "comm_induced");

    std::vector<VarArray> workStepUntil(instance.numberOfProcessors());
    std::vector<VarArray> commStepUntil(instance.numberOfProcessors());
    std::vector<VarArray> sendUpStepUntil(instance.numberOfProcessors());
    std::vector<VarArray> sendDownStepUntil(instance.numberOfProcessors());

    VarArray sendUpInduced;
    VarArray sendDownInduced;
    if (upAndDownCostSummed_) {
        send_up_induced = model.AddVars(static_cast<int>(max_time), COPT_CONTINUOUS, "send_up_induced");
        send_down_induced = model.AddVars(static_cast<int>(max_time), COPT_CONTINUOUS, "send_down_induced");
    }

    for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
        work_step_until[processor] = model.AddVars(static_cast<int>(max_time), COPT_CONTINUOUS, "work_step_until");
        send_up_step_until[processor] = model.AddVars(static_cast<int>(max_time), COPT_CONTINUOUS, "send_up_step_until");
        send_down_step_until[processor] = model.AddVars(static_cast<int>(max_time), COPT_CONTINUOUS, "send_up_step_until");
    }

    for (unsigned t = 0; t < maxTime_; t++) {
        model.AddConstr(comp_phase[static_cast<int>(t)] >= comp_phase_ends[static_cast<int>(t)]);
        if (mergeSteps_) {
            model.AddConstr(comm_phase[static_cast<int>(t)] >= comm_phase_ends[static_cast<int>(t)]);
        } else {
            model.AddConstr(send_down_phase[static_cast<int>(t)] + send_up_phase[static_cast<int>(t)]
                            >= comm_phase_ends[static_cast<int>(t)]);
        }
    }
    for (unsigned t = 0; t < maxTime_ - 1; t++) {
        model.AddConstr(comp_phase_ends[static_cast<int>(t)]
                        >= comp_phase[static_cast<int>(t)] - comp_phase[static_cast<int>(t) + 1]);
        if (mergeSteps_) {
            model.AddConstr(comm_phase_ends[static_cast<int>(t)]
                            >= comm_phase[static_cast<int>(t)] - comm_phase[static_cast<int>(t) + 1]);
        } else {
            model.AddConstr(comm_phase_ends[static_cast<int>(t)]
                            >= send_down_phase[static_cast<int>(t)] + send_up_phase[static_cast<int>(t)]
                                   - send_down_phase[static_cast<int>(t) + 1] - send_up_phase[static_cast<int>(t) + 1]);
        }
    }

    model.AddConstr(comp_phase_ends[static_cast<int>(max_time) - 1] >= comp_phase[static_cast<int>(max_time) - 1]);
    if (mergeSteps_) {
        model.AddConstr(comm_phase_ends[static_cast<int>(max_time) - 1] >= comm_phase[static_cast<int>(max_time) - 1]);
    } else {
        model.AddConstr(comm_phase_ends[static_cast<int>(max_time) - 1]
                        >= send_down_phase[static_cast<int>(max_time) - 1] + send_up_phase[static_cast<int>(max_time) - 1]);
    }

    const unsigned m = static_cast<unsigned>(instance.numberOfProcessors()
                                             * (sumOfVerticesWorkWeights(instance.getComputationalDag())
                                                + sumOfVerticesCommunicationWeights(instance.getComputationalDag())));

    for (unsigned t = 1; t < maxTime_; t++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            Expr exprWork;
            Expr exprSendUp;
            Expr exprSendDown;
            for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
                if (compute_exists[node][processor][t]) {
                    expr_work
                        += instance.getComputationalDag().VertexWorkWeight(node) * compute[node][processor][static_cast<int>(t)];
                }
                if (send_up_exists[node][processor][t]) {
                    expr_send_up
                        += instance.getComputationalDag().VertexCommWeight(node) * send_up[node][processor][static_cast<int>(t)];
                }
                if (send_down_exists[node][processor][t]) {
                    expr_send_down += instance.getComputationalDag().VertexCommWeight(node)
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
            if (upAndDownCostSummed_) {
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
        Expr exprWork;
        Expr exprSendUp;
        Expr exprSendDown;
        for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
            if (compute_exists[node][processor][0]) {
                expr_work += instance.getComputationalDag().VertexWorkWeight(node) * compute[node][processor][0];
            }
            if (send_up_exists[node][processor][0]) {
                expr_send_up += instance.getComputationalDag().VertexCommWeight(node) * send_up[node][processor][0];
            }
            if (send_down_exists[node][processor][0]) {
                expr_send_down += instance.getComputationalDag().VertexCommWeight(node) * send_down[node][processor][0];
            }
        }

        model.AddConstr(M * comm_phase_ends[0] + work_step_until[processor][0] >= expr_work);

        model.AddConstr(M * comp_phase_ends[0] + send_up_step_until[processor][0] >= expr_send_up);

        model.AddConstr(M * comp_phase_ends[0] + send_down_step_until[processor][0] >= expr_send_down);

        model.AddConstr(work_induced[0] >= work_step_until[processor][0] - M * (1 - comp_phase_ends[0]));
        if (upAndDownCostSummed_) {
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
    for (unsigned t = 0; t < maxTime_; t++) {
        expr += work_induced[static_cast<int>(t)] + instance.synchronisationCosts() * comm_phase_ends[static_cast<int>(t)]
                + instance.communicationCosts() * comm_induced[static_cast<int>(t)];
    }

    model.SetObjective(expr, COPT_MINIMIZE);
};

template <typename GraphT>
void MultiProcessorPebbling<GraphT>::SetupSyncObjective(const BspInstance<GraphT> &instance) {
    Expr expr;
    for (unsigned t = 0; t < maxTime_; t++) {
        if (!mergeSteps_) {
            expr += comp_phase[static_cast<int>(t)] + instance.communicationCosts() * send_up_phase[static_cast<int>(t)]
                    + instance.communicationCosts() * send_down_phase[static_cast<int>(t)];
        } else {
            // this objective+parameter combination is not very meaningful, but still defined here to avoid a segfault otherwise
            expr += comp_phase[static_cast<int>(t)] + instance.communicationCosts() * comm_phase[static_cast<int>(t)];
        }
    }

    model.SetObjective(expr, COPT_MINIMIZE);
}

template <typename GraphT>
void MultiProcessorPebbling<GraphT>::SetupAsyncVariablesConstraintsObjective(const BspInstance<GraphT> &instance) {
    std::vector<VarArray> finishTimes(instance.numberOfProcessors());

    for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
        finish_times[processor] = model.AddVars(static_cast<int>(max_time), COPT_CONTINUOUS, "finish_times");
    }

    Var makespan = model.AddVar(0, COPT_INFINITY, 1, COPT_CONTINUOUS, "makespan");

    VarArray getsBlue = model.AddVars(static_cast<int>(instance.numberOfVertices()), COPT_CONTINUOUS, "gets_blue");

    const unsigned m = static_cast<unsigned>(instance.numberOfProcessors()
                                             * (sumOfVerticesWorkWeights(instance.getComputationalDag())
                                                + sumOfVerticesCommunicationWeights(instance.getComputationalDag())));

    for (unsigned t = 0; t < maxTime_; t++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            Expr sendDownStepLength;
            for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
                if (send_down_exists[node][processor][t]) {
                    send_down_step_length += instance.communicationCosts() * instance.getComputationalDag().VertexCommWeight(node)
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
    for (unsigned t = 0; t < maxTime_; t++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            model.AddConstr(makespan >= finish_times[processor][static_cast<int>(t)]);
        }
    }

    // t = 0
    for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
        Expr expr;
        for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
            if (compute_exists[node][processor][0]) {
                expr += instance.getComputationalDag().VertexWorkWeight(node) * compute[node][processor][0];
            }

            if (send_up_exists[node][processor][0]) {
                expr += instance.communicationCosts() * instance.getComputationalDag().VertexCommWeight(node)
                        * send_up[node][processor][0];
            }

            if (send_down_exists[node][processor][0]) {
                expr += instance.communicationCosts() * instance.getComputationalDag().VertexCommWeight(node)
                        * send_down[node][processor][0];
            }
        }

        model.AddConstr(finish_times[processor][0] >= expr);
    }

    for (unsigned t = 1; t < maxTime_; t++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            Expr expr;
            for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
                if (compute_exists[node][processor][t]) {
                    expr += instance.getComputationalDag().VertexWorkWeight(node) * compute[node][processor][static_cast<int>(t)];
                }

                if (send_up_exists[node][processor][t]) {
                    expr += instance.communicationCosts() * instance.getComputationalDag().VertexCommWeight(node)
                            * send_up[node][processor][static_cast<int>(t)];
                }

                if (send_down_exists[node][processor][t]) {
                    expr += instance.communicationCosts() * instance.getComputationalDag().VertexCommWeight(node)
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

template <typename GraphT>
void MultiProcessorPebbling<GraphT>::WriteSolutionCallback::Callback() {
    if (Where() == COPT_CBCONTEXT_MIPSOL && counter < max_number_solution && GetIntInfo(COPT_CBINFO_HASINCUMBENT)) {
        try {
            if (GetDblInfo(COPT_CBINFO_BESTOBJ) < best_obj && 0.0 < GetDblInfo(COPT_CBINFO_BESTBND)) {
                best_obj = GetDblInfo(COPT_CBINFO_BESTOBJ);

                //    auto sched = constructBspScheduleFromCallback();
                //    BspScheduleWriter sched_writer(sched);
                //    sched_writer.write_dot(write_solutions_path_cb + "intmed_sol_" + solution_file_prefix_cb + "_"
                //    +
                //                           std::to_string(counter) + "_schedule.dot");
                counter_++;
            }

        } catch (const std::exception &e) {}
    }
};

template <typename GraphT>
void MultiProcessorPebbling<GraphT>::ConstructPebblingScheduleFromSolution(PebblingSchedule<GraphT> &schedule) {
    const BspInstance<GraphT> &instance = schedule.getInstance();

    std::vector<std::vector<std::set<std::pair<unsigned, vertex_idx>>>> nodesComputed(
        instance.numberOfProcessors(), std::vector<std::set<std::pair<unsigned, vertex_idx>>>(max_time));
    std::vector<std::vector<std::deque<vertex_idx>>> nodesSentUp(instance.numberOfProcessors(),
                                                                 std::vector<std::deque<vertex_idx>>(max_time));
    std::vector<std::vector<std::deque<vertex_idx>>> nodesSentDown(instance.numberOfProcessors(),
                                                                   std::vector<std::deque<vertex_idx>>(max_time));
    std::vector<std::vector<std::set<vertex_idx>>> evictedAfter(instance.numberOfProcessors(),
                                                                std::vector<std::set<vertex_idx>>(max_time));

    // used to remove unneeded steps when a node is sent down and then up (which becomes invalid after reordering the comm phases)
    std::vector<std::vector<bool>> sentDownAlready(instance.numberOfVertices(),
                                                   std::vector<bool>(instance.numberOfProcessors(), false));
    std::vector<std::vector<bool>> ignoreRed(instance.numberOfVertices(), std::vector<bool>(instance.numberOfProcessors(), false));

    std::vector<vertex_idx> topOrder = GetTopOrder(instance.getComputationalDag());
    std::vector<unsigned> topOrderPosition(instance.numberOfVertices());
    for (unsigned index = 0; index < instance.numberOfVertices(); ++index) {
        topOrderPosition[topOrder[index]] = index;
    }

    std::vector<bool> emptyStep(maxTime_, true);
    std::vector<std::vector<unsigned>> stepTypeOnProc(instance.numberOfProcessors(), std::vector<unsigned>(max_time, 0));

    for (unsigned step = 0; step < maxTime_; step++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
                if (compute_exists[node][processor][step]
                    && compute[node][processor][static_cast<int>(step)].Get(COPT_DBLINFO_VALUE) >= .99) {
                    stepTypeOnProc[processor][step] = 1;
                }
            }
        }
    }

    for (unsigned step = 0; step < maxTime_; step++) {
        for (unsigned processor = 0; processor < instance.numberOfProcessors(); processor++) {
            for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
                if (step > 0 && has_red[node][processor][static_cast<int>(step) - 1].Get(COPT_DBLINFO_VALUE) >= .99
                    && has_red[node][processor][static_cast<int>(step)].Get(COPT_DBLINFO_VALUE) <= .01
                    && !ignore_red[node][processor]) {
                    for (size_t previousStep = step - 1; previousStep < step; --previousStep) {
                        if (!nodes_computed[processor][previousStep].empty() || !nodes_sent_up[processor][previousStep].empty()
                            || !nodes_sent_down[processor][previousStep].empty() || previousStep == 0) {
                            evictedAfter[processor][previousStep].insert(node);
                            emptyStep[previousStep] = false;
                            break;
                        }
                    }
                }

                if (compute_exists[node][processor][step]
                    && compute[node][processor][static_cast<int>(step)].Get(COPT_DBLINFO_VALUE) >= .99) {
                    nodesComputed[processor][step].emplace(topOrderPosition[node], node);
                    emptyStep[step] = false;
                    ignoreRed[node][processor] = false;

                    // implicit eviction in case of mergesteps - never having "has_red=1"
                    if (step + 1 < max_time && has_red[node][processor][static_cast<int>(step) + 1].Get(COPT_DBLINFO_VALUE) <= .01) {
                        evictedAfter[processor][step].insert(node);
                    }
                }

                if (send_down_exists[node][processor][step]
                    && send_down[node][processor][static_cast<int>(step)].Get(COPT_DBLINFO_VALUE) >= .99) {
                    bool keepStep = false;

                    for (unsigned nextStep = step + 1;
                         next_step < max_time
                         && has_red[node][processor][static_cast<int>(next_step)].Get(COPT_DBLINFO_VALUE) >= .99;
                         ++nextStep) {
                        if (stepTypeOnProc[processor][nextStep] == 1) {
                            keepStep = true;
                            break;
                        }
                    }

                    if (keepStep) {
                        nodesSentDown[processor][step].push_back(node);
                        emptyStep[step] = false;
                        stepTypeOnProc[processor][step] = 3;
                        ignoreRed[node][processor] = false;
                    } else {
                        ignoreRed[node][processor] = true;
                    }

                    sentDownAlready[node][processor] = true;
                }

                if (send_up_exists[node][processor][step]
                    && send_up[node][processor][static_cast<int>(step)].Get(COPT_DBLINFO_VALUE) >= .99
                    && !sent_down_already[node][processor]) {
                    nodesSentUp[processor][step].push_back(node);
                    emptyStep[step] = false;
                    stepTypeOnProc[processor][step] = 2;
                }
            }
        }
    }

    // components of the final PebblingSchedule - the first two dimensions are always processor and superstep
    std::vector<std::vector<std::vector<vertex_idx>>> computeStepsPerSupstep(instance.numberOfProcessors());
    std::vector<std::vector<std::vector<std::vector<vertex_idx>>>> nodesEvictedAfterCompute(instance.numberOfProcessors());
    std::vector<std::vector<std::vector<vertex_idx>>> nodesSentUpInSupstep(instance.numberOfProcessors());
    std::vector<std::vector<std::vector<vertex_idx>>> nodesSentDownInSupstep(instance.numberOfProcessors());
    std::vector<std::vector<std::vector<vertex_idx>>> nodesEvictedInCommPhase(instance.numberOfProcessors());

    // edge case: check if an extra superstep must be added in the beginning to evict values that are initially in cache
    bool needsEvictStepInBeginning = false;
    for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
        for (unsigned step = 0; step < maxTime_; step++) {
            if (stepTypeOnProc[proc][step] == 0 && !evicted_after[proc][step].empty()) {
                needsEvictStepInBeginning = true;
                break;
            } else if (stepTypeOnProc[proc][step] > 0) {
                break;
            }
        }
    }

    // create the actual PebblingSchedule - iterating over the steps
    unsigned superstepIndex = 0;
    if (synchronous_) {
        bool inComm = true;
        superstepIndex = UINT_MAX;

        if (needsEvictStepInBeginning) {
            // artificially insert comm step in beginning, if it would start with compute otherwise
            bool beginsWithCompute = false;
            for (unsigned step = 0; step < maxTime_; step++) {
                bool isComp = false, isComm = false;
                for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
                    if (stepTypeOnProc[proc][step] == 1) {
                        isComp = true;
                    }
                    if (stepTypeOnProc[proc][step] > 1) {
                        isComm = true;
                    }
                }
                if (isComp) {
                    beginsWithCompute = true;
                }
                if (isComp || isComm) {
                    break;
                }
            }

            if (beginsWithCompute) {
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
        for (unsigned step = 0; step < maxTime_; step++) {
            if (emptyStep[step]) {
                continue;
            }

            unsigned stepType = 0;
            for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
                stepType = std::max(stepType, step_type_on_proc[proc][step]);
            }

            if (stepType == 1) {
                if (inComm) {
                    for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
                        compute_steps_per_supstep[proc].push_back(std::vector<vertex_idx>());
                        nodes_evicted_after_compute[proc].push_back(std::vector<std::vector<vertex_idx>>());
                        nodes_sent_up_in_supstep[proc].push_back(std::vector<vertex_idx>());
                        nodes_sent_down_in_supstep[proc].push_back(std::vector<vertex_idx>());
                        nodes_evicted_in_comm_phase[proc].push_back(std::vector<vertex_idx>());
                    }
                    ++superstepIndex;
                    inComm = false;
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

            if (stepType == 2 || stepType == 3) {
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

                inComm = true;
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
        std::vector<unsigned> stepIdxOnProc(instance.numberOfProcessors(), 0);

        std::vector<bool> alreadyHasBlue(instance.numberOfVertices(), false);
        if (needToLoadInputs_) {
            for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {
                if (instance.getComputationalDag().in_degree(node) == 0) {
                    alreadyHasBlue[node] = true;
                }
            }
        }

        std::vector<bool> procFinished(instance.numberOfProcessors(), false);
        unsigned nrProcFinished = 0;
        while (nrProcFinished < instance.numberOfProcessors()) {
            // preliminary sweep of superstep, to see if we need to wait for other processors
            std::vector<unsigned> idxLimitOnProc = step_idx_on_proc;

            // first add compute steps
            if (!needsEvictStepInBeginning || superstepIndex > 0) {
                for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
                    while (idxLimitOnProc[proc] < maxTime_ && step_type_on_proc[proc][idx_limit_on_proc[proc]] <= 1) {
                        ++idx_limit_on_proc[proc];
                    }
                }
            }

            // then add communications step until possible (note - they might not be valid if all put into a single superstep!)
            std::set<vertex_idx> newBlues;
            bool stillMakingProgress = true;
            while (stillMakingProgress) {
                stillMakingProgress = false;
                for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
                    while (idxLimitOnProc[proc] < maxTime_ && step_type_on_proc[proc][idx_limit_on_proc[proc]] != 1) {
                        bool acceptStep = true;
                        for (vertex_idx node : nodes_sent_down[proc][idx_limit_on_proc[proc]]) {
                            if (!already_has_blue[node] && new_blues.find(node) == new_blues.end()) {
                                accept_step = false;
                            }
                        }

                        if (!acceptStep) {
                            break;
                        }

                        for (vertex_idx node : nodes_sent_up[proc][idx_limit_on_proc[proc]]) {
                            if (!already_has_blue[node]) {
                                new_blues.insert(node);
                            }
                        }

                        stillMakingProgress = true;
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

                while (stepIdxOnProc[proc] < idx_limit_on_proc[proc] && step_type_on_proc[proc][step_idx_on_proc[proc]] <= 1) {
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
                while (stepIdxOnProc[proc] < idx_limit_on_proc[proc] && step_type_on_proc[proc][step_idx_on_proc[proc]] != 1) {
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
                if (stepIdxOnProc[proc] == maxTime_ && !proc_finished[proc]) {
                    procFinished[proc] = true;
                    ++nrProcFinished;
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

template <typename GraphT>
void MultiProcessorPebbling<GraphT>::SetInitialSolution(
    const BspInstance<GraphT> &instance,
    const std::vector<std::vector<std::vector<vertex_idx>>> &computeSteps,
    const std::vector<std::vector<std::vector<vertex_idx>>> &sendUpSteps,
    const std::vector<std::vector<std::vector<vertex_idx>>> &sendDownSteps,
    const std::vector<std::vector<std::vector<vertex_idx>>> &nodesEvictedAfterStep) {
    const unsigned n = static_cast<unsigned>(instance.numberOfVertices());

    std::vector<bool> inSlowMem(n, false);
    if (needToLoadInputs_) {
        for (vertex_idx node = 0; node < n; ++node) {
            if (instance.getComputationalDag().in_degree(node) == 0) {
                inSlowMem[node] = true;
            }
        }
    }

    std::vector<std::vector<unsigned>> inFastMem(N, std::vector<unsigned>(instance.numberOfProcessors(), false));
    if (!has_red_in_beginning.empty()) {
        for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {
            for (vertex_idx node : has_red_in_beginning[proc]) {
                in_fast_mem[node][proc] = true;
            }
        }
    }

    unsigned step = 0, newStepIdx = 0;
    for (; step < computeSteps[0].size(); ++step) {
        for (vertex_idx node = 0; node < n; ++node) {
            if (has_blue_exists[node][new_step_idx]) {
                model.SetMipStart(has_blue[node][static_cast<int>(new_step_idx)], static_cast<int>(in_slow_mem[node]));
            }
            for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {
                model.SetMipStart(has_red[node][proc][static_cast<int>(new_step_idx)], static_cast<int>(in_fast_mem[node][proc]));
            }
        }

        if (restrictStepTypes_) {
            // align step number with step type cycle's phase, if needed
            bool skipStep = true;
            while (skipStep) {
                skipStep = false;
                bool isCompute = false, isSendUp = false, isSendDown = false;
                for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {
                    if (!computeSteps[proc][step].empty()) {
                        isCompute = true;
                    }
                    if (!sendUpSteps[proc][step].empty()) {
                        isSendUp = true;
                    }
                    if (!sendDownSteps[proc][step].empty()) {
                        isSendDown = true;
                    }
                }

                bool sendUpStepIdx
                    = (needToLoadInputs_ && (newStepIdx % (computeStepsPerCycle_ + 2) == computeStepsPerCycle_ + 1))
                      || (!needToLoadInputs_ && (newStepIdx % (computeStepsPerCycle_ + 2) == computeStepsPerCycle_));
                bool sendDownStepIdx
                    = (needToLoadInputs_ && (newStepIdx % (computeStepsPerCycle_ + 2) == 0))
                      || (!needToLoadInputs_ && (newStepIdx % (computeStepsPerCycle_ + 2) == computeStepsPerCycle_ + 1));

                if (isCompute && (sendUpStepIdx || sendDownStepIdx)) {
                    skipStep = true;
                }
                if (isSendUp && !sendUpStepIdx) {
                    skipStep = true;
                }
                if (isSendDown && !sendDownStepIdx) {
                    skipStep = true;
                }

                if (skipStep) {
                    ++newStepIdx;
                    for (vertex_idx node = 0; node < n; ++node) {
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
            std::vector<bool> valueOfNode(n, false);
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

            for (vertex_idx node : nodesEvictedAfterStep[proc][step]) {
                in_fast_mem[node][proc] = false;
            }
        }
        ++newStepIdx;
    }
    for (; newStepIdx < maxTime_; ++newStepIdx) {
        for (vertex_idx node = 0; node < n; ++node) {
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

template <typename GraphT>
unsigned MultiProcessorPebbling<GraphT>::ComputeMaxTimeForInitialSolution(
    const BspInstance<GraphT> &instance,
    const std::vector<std::vector<std::vector<vertex_idx>>> &computeSteps,
    const std::vector<std::vector<std::vector<vertex_idx>>> &sendUpSteps,
    const std::vector<std::vector<std::vector<vertex_idx>>> &sendDownSteps) const {
    if (!restrictStepTypes_) {
        return static_cast<unsigned>(computeSteps[0].size()) + 3;
    }

    unsigned step = 0, newStepIdx = 0;
    for (; step < computeSteps[0].size(); ++step) {
        // align step number with step type cycle's phase, if needed
        bool skipStep = true;
        while (skipStep) {
            skipStep = false;
            bool isCompute = false, isSendUp = false, isSendDown = false;
            for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {
                if (!computeSteps[proc][step].empty()) {
                    isCompute = true;
                }
                if (!sendUpSteps[proc][step].empty()) {
                    isSendUp = true;
                }
                if (!sendDownSteps[proc][step].empty()) {
                    isSendDown = true;
                }
            }

            bool sendUpStepIdx = (needToLoadInputs_ && (newStepIdx % (computeStepsPerCycle_ + 2) == computeStepsPerCycle_ + 1))
                                 || (!needToLoadInputs_ && (newStepIdx % (computeStepsPerCycle_ + 2) == computeStepsPerCycle_));
            bool sendDownStepIdx
                = (needToLoadInputs_ && (newStepIdx % (computeStepsPerCycle_ + 2) == 0))
                  || (!needToLoadInputs_ && (newStepIdx % (computeStepsPerCycle_ + 2) == computeStepsPerCycle_ + 1));

            if (isCompute && (sendUpStepIdx || sendDownStepIdx)) {
                skipStep = true;
            }
            if (isSendUp && !sendUpStepIdx) {
                skipStep = true;
            }
            if (isSendDown && !sendDownStepIdx) {
                skipStep = true;
            }

            if (skipStep) {
                ++newStepIdx;
            }
        }

        ++newStepIdx;
    }

    newStepIdx += computeStepsPerCycle_ + 2;
    return newStepIdx;
}

template <typename GraphT>
bool MultiProcessorPebbling<GraphT>::HasEmptyStep(const BspInstance<GraphT> &instance) {
    for (unsigned step = 0; step < maxTime_; ++step) {
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
        if (empty) {
            return true;
        }
    }
    return false;
}

}    // namespace osp
