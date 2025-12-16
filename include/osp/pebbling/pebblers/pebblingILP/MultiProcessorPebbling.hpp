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
    static_assert(isComputationalDagV<GraphT>, "PebblingSchedule can only be used with computational DAGs.");

  private:
    using VertexIdx = VertexIdxT<GraphT>;

    Model model_;

    bool writeSolutionsFound_;

    class WriteSolutionCallback : public CallbackBase {
      private:
        unsigned counter_;
        unsigned maxNumberSolution_;

        double bestObj_;

      public:
        WriteSolutionCallback()
            : counter_(0), maxNumberSolution_(500), bestObj_(COPT_INFINITY), writeSolutionsPathCb_(""), solutionFilePrefixCb_("") {}

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
    std::set<VertexIdx> needsBlueAtEnd_;
    std::vector<std::set<VertexIdx>> hasRedInBeginning_;
    bool verbose_ = false;

    void ConstructPebblingScheduleFromSolution(PebblingSchedule<GraphT> &schedule);

    void SetInitialSolution(const BspInstance<GraphT> &instance,
                            const std::vector<std::vector<std::vector<VertexIdx>>> &computeSteps,
                            const std::vector<std::vector<std::vector<VertexIdx>>> &sendUpSteps,
                            const std::vector<std::vector<std::vector<VertexIdx>>> &sendDownSteps,
                            const std::vector<std::vector<std::vector<VertexIdx>>> &nodesEvictedAfterStep);

    unsigned ComputeMaxTimeForInitialSolution(const BspInstance<GraphT> &instance,
                                              const std::vector<std::vector<std::vector<VertexIdx>>> &computeSteps,
                                              const std::vector<std::vector<std::vector<VertexIdx>>> &sendUpSteps,
                                              const std::vector<std::vector<std::vector<VertexIdx>>> &sendDownSteps) const;

    void SetupBaseVariablesConstraints(const BspInstance<GraphT> &instance);

    void SetupSyncPhaseVariablesConstraints(const BspInstance<GraphT> &instance);
    void SetupSyncObjective(const BspInstance<GraphT> &instance);

    void SetupAsyncVariablesConstraintsObjective(const BspInstance<GraphT> &instance);
    void SetupBspVariablesConstraintsObjective(const BspInstance<GraphT> &instance);

    void SolveIlp();

  public:
    MultiProcessorPebbling()
        : Scheduler<GraphT>(), model_(COPTEnv::GetInstance().CreateModel("MPP")), writeSolutionsFound_(false), maxTime_(0) {}

    virtual ~MultiProcessorPebbling() = default;

    virtual ReturnStatus ComputeSchedule(BspSchedule<GraphT> &schedule) override;
    virtual ReturnStatus ComputeSynchPebbling(PebblingSchedule<GraphT> &schedule);

    virtual ReturnStatus ComputePebbling(PebblingSchedule<GraphT> &schedule, bool useAsync = false);

    virtual ReturnStatus ComputePebblingWithInitialSolution(const PebblingSchedule<GraphT> &initialSolution,
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
    inline double BestGap() { return model_.GetDblAttr(COPT_DBLATTR_BESTGAP); }

    /**
     * @brief Get the best objective value found by the solver.
     *
     * @return The best objective value found by the solver.
     */
    inline double BestObjective() { return model_.GetDblAttr(COPT_DBLATTR_BESTOBJ); }

    /**
     * @brief Get the best bound found by the solver.
     *
     * @return The best bound found by the solver.
     */
    inline double BestBound() { return model_.GetDblAttr(COPT_DBLATTR_BESTBND); }

    /**
     * @brief Get the name of the schedule.
     *
     * @return The name of the schedule.
     */
    virtual std::string GetScheduleName() const override { return "MultiProcessorPebbling"; }

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

    inline void SetNeedsBlueAtEnd(const std::set<VertexIdx> &needsBlue) { needsBlueAtEnd_ = needsBlue; }

    inline void SetHasRedInBeginning(const std::vector<std::set<VertexIdx>> &hasRed) { hasRedInBeginning_ = hasRed; }

    inline void SetVerbose(const bool verbose) { verbose_ = verbose; }

    inline void SetTimeLimitSeconds(unsigned timeLimitSeconds) { timeLimitSeconds_ = timeLimitSeconds; }

    bool HasEmptyStep(const BspInstance<GraphT> &instance);
};

// implementation

template <typename GraphT>
void MultiProcessorPebbling<GraphT>::SolveIlp() {
    if (!verbose_) {
        model_.SetIntParam(COPT_INTPARAM_LOGTOCONSOLE, 0);
    }

    model_.SetDblParam(COPT_DBLPARAM_TIMELIMIT, timeLimitSeconds_);
    model_.SetIntParam(COPT_INTPARAM_THREADS, 128);

    model_.SetIntParam(COPT_INTPARAM_STRONGBRANCHING, 1);
    model_.SetIntParam(COPT_INTPARAM_LPMETHOD, 1);
    model_.SetIntParam(COPT_INTPARAM_ROUNDINGHEURLEVEL, 1);

    model_.SetIntParam(COPT_INTPARAM_SUBMIPHEURLEVEL, 1);
    // model.SetIntParam(COPT_INTPARAM_PRESOLVE, 1);
    // model.SetIntParam(COPT_INTPARAM_CUTLEVEL, 0);
    model_.SetIntParam(COPT_INTPARAM_TREECUTLEVEL, 2);
    // model.SetIntParam(COPT_INTPARAM_DIVINGHEURLEVEL, 2);

    model_.Solve();
}

template <typename GraphT>
ReturnStatus MultiProcessorPebbling<GraphT>::ComputeSchedule(BspSchedule<GraphT> &schedule) {
    if (maxTime_ == 0) {
        maxTime_ = 2 * static_cast<unsigned>(schedule.GetInstance().NumberOfVertices());
    }

    SetupBaseVariablesConstraints(schedule.GetInstance());
    SetupSyncPhaseVariablesConstraints(schedule.GetInstance());
    SetupBspVariablesConstraintsObjective(schedule.GetInstance());

    SolveIlp();

    if (model_.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {
        return ReturnStatus::OSP_SUCCESS;

    } else if (model_.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {
        return ReturnStatus::ERROR;

    } else {
        if (model_.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
            return ReturnStatus::BEST_FOUND;

        } else {
            return ReturnStatus::TIMEOUT;
        }
    }
};

template <typename GraphT>
ReturnStatus MultiProcessorPebbling<GraphT>::ComputeSynchPebbling(PebblingSchedule<GraphT> &schedule) {
    const BspInstance<GraphT> &instance = schedule.GetInstance();

    if (maxTime_ == 0) {
        maxTime_ = 2 * static_cast<unsigned>(instance.NumberOfVertices());
    }

    mergeSteps_ = false;

    SetupBaseVariablesConstraints(instance);
    SetupSyncPhaseVariablesConstraints(instance);
    SetupSyncObjective(instance);

    SolveIlp();

    if (model_.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {
        ConstructPebblingScheduleFromSolution(schedule);
        return ReturnStatus::OSP_SUCCESS;

    } else if (model_.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {
        return ReturnStatus::ERROR;

    } else {
        if (model_.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
            ConstructPebblingScheduleFromSolution(schedule);
            return ReturnStatus::OSP_SUCCESS;

        } else {
            return ReturnStatus::TIMEOUT;
        }
    }
}

template <typename GraphT>
ReturnStatus MultiProcessorPebbling<GraphT>::ComputePebbling(PebblingSchedule<GraphT> &schedule, bool useAsync) {
    const BspInstance<GraphT> &instance = schedule.GetInstance();

    if (maxTime_ == 0) {
        maxTime_ = 2 * static_cast<unsigned>(instance.NumberOfVertices());
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

    if (model_.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {
        ConstructPebblingScheduleFromSolution(schedule);
        return schedule.IsValid() ? ReturnStatus::OSP_SUCCESS : ReturnStatus::ERROR;

    } else if (model_.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {
        return ReturnStatus::ERROR;

    } else {
        if (model_.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
            ConstructPebblingScheduleFromSolution(schedule);
            return schedule.IsValid() ? ReturnStatus::OSP_SUCCESS : ReturnStatus::ERROR;

        } else {
            return ReturnStatus::TIMEOUT;
        }
    }
}

template <typename GraphT>
ReturnStatus MultiProcessorPebbling<GraphT>::ComputePebblingWithInitialSolution(const PebblingSchedule<GraphT> &initialSolution,
                                                                                PebblingSchedule<GraphT> &outSchedule,
                                                                                bool useAsync) {
    const BspInstance<GraphT> &instance = initialSolution.GetInstance();

    std::vector<std::vector<std::vector<VertexIdx>>> computeSteps;
    std::vector<std::vector<std::vector<VertexIdx>>> sendUpSteps;
    std::vector<std::vector<std::vector<VertexIdx>>> sendDownSteps;
    std::vector<std::vector<std::vector<VertexIdx>>> nodesEvictedAfterStep;

    synchronous_ = !useAsync;

    initialSolution.GetDataForMultiprocessorPebbling(computeSteps, sendUpSteps, sendDownSteps, nodesEvictedAfterStep);

    maxTime_ = ComputeMaxTimeForInitialSolution(instance, computeSteps, sendUpSteps, sendDownSteps);

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

    if (model_.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {
        ConstructPebblingScheduleFromSolution(outSchedule);
        return outSchedule.IsValid() ? ReturnStatus::OSP_SUCCESS : ReturnStatus::ERROR;

    } else if (model_.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {
        return ReturnStatus::ERROR;

    } else {
        if (model_.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
            ConstructPebblingScheduleFromSolution(outSchedule);
            return outSchedule.IsValid() ? ReturnStatus::OSP_SUCCESS : ReturnStatus::ERROR;

        } else {
            return ReturnStatus::TIMEOUT;
        }
    }
}

template <typename GraphT>
void MultiProcessorPebbling<GraphT>::SetupBaseVariablesConstraints(const BspInstance<GraphT> &instance) {
    /*
        Variables
    */
    compute_
        = std::vector<std::vector<VarArray>>(instance.NumberOfVertices(), std::vector<VarArray>(instance.NumberOfProcessors()));

    for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
            compute_[node][processor] = model_.AddVars(static_cast<int>(maxTime_), COPT_BINARY, "node_processor_time");
        }
    }

    computeExists_.resize(instance.NumberOfVertices(),
                          std::vector<std::vector<bool>>(instance.NumberOfProcessors(), std::vector<bool>(maxTime_, true)));

    sendUp_ = std::vector<std::vector<VarArray>>(instance.NumberOfVertices(), std::vector<VarArray>(instance.NumberOfProcessors()));

    for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
            sendUp_[node][processor] = model_.AddVars(static_cast<int>(maxTime_), COPT_BINARY, "sendUp");
        }
    }

    sendUpExists_.resize(instance.NumberOfVertices(),
                         std::vector<std::vector<bool>>(instance.NumberOfProcessors(), std::vector<bool>(maxTime_, true)));

    sendDown_
        = std::vector<std::vector<VarArray>>(instance.NumberOfVertices(), std::vector<VarArray>(instance.NumberOfProcessors()));

    for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
            sendDown_[node][processor] = model_.AddVars(static_cast<int>(maxTime_), COPT_BINARY, "sendDown");
        }
    }

    sendDownExists_.resize(instance.NumberOfVertices(),
                           std::vector<std::vector<bool>>(instance.NumberOfProcessors(), std::vector<bool>(maxTime_, true)));

    hasBlue_ = std::vector<VarArray>(instance.NumberOfVertices());

    for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
        hasBlue_[node] = model_.AddVars(static_cast<int>(maxTime_), COPT_BINARY, "blue_pebble");
    }

    hasBlueExists_.resize(instance.NumberOfVertices(), std::vector<bool>(maxTime_, true));

    hasRed_ = std::vector<std::vector<VarArray>>(instance.NumberOfVertices(), std::vector<VarArray>(instance.NumberOfProcessors()));

    for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
            hasRed_[node][processor] = model_.AddVars(static_cast<int>(maxTime_), COPT_BINARY, "red_pebble");
        }
    }

    /*
        Invalidate variables based on various factors (node types, input loading, step type restriction)
    */

    for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
            if (!instance.IsCompatible(node, processor)) {
                for (unsigned t = 0; t < maxTime_; t++) {
                    computeExists_[node][processor][t] = false;
                    sendUpExists_[node][processor][t] = false;
                }
            }
        }
    }

    // restrict source nodes if they need to be loaded
    if (needToLoadInputs_) {
        for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
            if (instance.GetComputationalDag().InDegree(node) == 0) {
                for (unsigned t = 0; t < maxTime_; t++) {
                    for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                        computeExists_[node][processor][t] = false;
                        sendUpExists_[node][processor][t] = false;
                    }
                    hasBlueExists_[node][t] = false;
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
                for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
                    for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                        computeExists_[node][processor][t] = false;
                    }
                }
            } else {
                for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
                    for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                        sendUpExists_[node][processor][t] = false;
                        sendDownExists_[node][processor][t] = false;
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
            for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                Expr expr;
                for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
                    if (computeExists_[node][processor][t]) {
                        expr += compute_[node][processor][static_cast<int>(t)];
                    }
                    if (sendUpExists_[node][processor][t]) {
                        expr += sendUp_[node][processor][static_cast<int>(t)];
                    }
                    if (sendDownExists_[node][processor][t]) {
                        expr += sendDown_[node][processor][static_cast<int>(t)];
                    }
                }
                model_.AddConstr(expr <= 1);
            }
        }
    } else {
        // extra variables to indicate step types in step merging
        std::vector<VarArray> compStepOnProc = std::vector<VarArray>(instance.NumberOfProcessors());
        std::vector<VarArray> commStepOnProc = std::vector<VarArray>(instance.NumberOfProcessors());

        for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
            compStepOnProc[processor] = model_.AddVars(static_cast<int>(maxTime_), COPT_BINARY, "comp_step_on_proc");
            commStepOnProc[processor] = model_.AddVars(static_cast<int>(maxTime_), COPT_BINARY, "comm_step_on_proc");
        }

        const unsigned m = static_cast<unsigned>(instance.NumberOfVertices());

        for (unsigned t = 0; t < maxTime_; t++) {
            for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                Expr exprComp, exprComm;
                for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
                    if (computeExists_[node][processor][t]) {
                        exprComp += compute_[node][processor][static_cast<int>(t)];
                    }
                    if (sendUpExists_[node][processor][t]) {
                        exprComm += sendUp_[node][processor][static_cast<int>(t)];
                    }
                    if (sendDownExists_[node][processor][t]) {
                        exprComm += sendDown_[node][processor][static_cast<int>(t)];
                    }
                }

                model_.AddConstr(m * compStepOnProc[processor][static_cast<int>(t)] >= exprComp);
                model_.AddConstr(2 * m * commStepOnProc[processor][static_cast<int>(t)] >= exprComm);

                model_.AddConstr(compStepOnProc[processor][static_cast<int>(t)] + commStepOnProc[processor][static_cast<int>(t)]
                                 <= 1);
            }
        }
    }

    for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
        for (unsigned t = 1; t < maxTime_; t++) {
            if (!hasBlueExists_[node][t]) {
                continue;
            }

            Expr expr;

            for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                if (sendUpExists_[node][processor][t - 1]) {
                    expr += sendUp_[node][processor][static_cast<int>(t) - 1];
                }
            }
            model_.AddConstr(hasBlue_[node][static_cast<int>(t)] <= hasBlue_[node][static_cast<int>(t) - 1] + expr);
        }
    }

    for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
            for (unsigned t = 1; t < maxTime_; t++) {
                Expr expr;

                if (computeExists_[node][processor][t - 1]) {
                    expr += compute_[node][processor][static_cast<int>(t) - 1];
                }

                if (sendDownExists_[node][processor][t - 1]) {
                    expr += sendDown_[node][processor][static_cast<int>(t) - 1];
                }

                model_.AddConstr(hasRed_[node][processor][static_cast<int>(t)]
                                 <= hasRed_[node][processor][static_cast<int>(t) - 1] + expr);
            }
        }
    }

    for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
            for (unsigned t = 0; t < maxTime_; t++) {
                if (!computeExists_[node][processor][t]) {
                    continue;
                }

                for (const auto &source : instance.GetComputationalDag().Parents(node)) {
                    if (!mergeSteps_ || !computeExists_[source][processor][t]) {
                        model_.AddConstr(compute_[node][processor][static_cast<int>(t)]
                                         <= hasRed_[source][processor][static_cast<int>(t)]);
                    } else {
                        model_.AddConstr(compute_[node][processor][static_cast<int>(t)]
                                         <= hasRed_[source][processor][static_cast<int>(t)]
                                                + compute_[source][processor][static_cast<int>(t)]);
                    }
                }
            }
        }
    }

    for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
            for (unsigned t = 0; t < maxTime_; t++) {
                if (sendUpExists_[node][processor][t]) {
                    model_.AddConstr(sendUp_[node][processor][static_cast<int>(t)]
                                     <= hasRed_[node][processor][static_cast<int>(t)]);
                }
            }
        }
    }

    for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
            for (unsigned t = 0; t < maxTime_; t++) {
                if (sendDownExists_[node][processor][t] && hasBlueExists_[node][t]) {
                    model_.AddConstr(sendDown_[node][processor][static_cast<int>(t)] <= hasBlue_[node][static_cast<int>(t)]);
                }
            }
        }
    }

    for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
        for (unsigned t = 0; t < maxTime_; t++) {
            Expr expr;
            for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
                expr += hasRed_[node][processor][static_cast<int>(t)] * instance.GetComputationalDag().VertexMemWeight(node);
                if (!slidingPebbles_ && computeExists_[node][processor][t]) {
                    expr += compute_[node][processor][static_cast<int>(t)] * instance.GetComputationalDag().VertexMemWeight(node);
                }
            }

            model_.AddConstr(expr <= instance.GetArchitecture().MemoryBound(processor));
        }
    }

    for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
        for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
            if (hasRedInBeginning_.empty() || hasRedInBeginning_[processor].find(node) == hasRedInBeginning_[processor].end()) {
                model_.AddConstr(hasRed_[node][processor][0] == 0);
            } else {
                model_.AddConstr(hasRed_[node][processor][0] == 1);
            }
        }
    }

    for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
        if (!needToLoadInputs_ || instance.GetComputationalDag().InDegree(node) > 0) {
            model_.AddConstr(hasBlue_[node][0] == 0);
        }
    }

    if (needsBlueAtEnd_.empty())    // default case: blue pebbles required on sinks at the end
    {
        for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
            if (instance.GetComputationalDag().OutDegree(node) == 0 && hasBlueExists_[node][maxTime_ - 1]) {
                model_.AddConstr(hasBlue_[node][static_cast<int>(maxTime_) - 1] == 1);
            }
        }
    } else    // otherwise: specified set of nodes that need blue at the end
    {
        for (VertexIdx node : needsBlueAtEnd_) {
            if (hasBlueExists_[node][maxTime_ - 1]) {
                model_.AddConstr(hasBlue_[node][static_cast<int>(maxTime_) - 1] == 1);
            }
        }
    }

    // disable recomputation if needed
    if (!allowsRecomputation_) {
        for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
            Expr expr;
            for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                for (unsigned t = 0; t < maxTime_; t++) {
                    if (computeExists_[node][processor][t]) {
                        expr += compute_[node][processor][static_cast<int>(t)];
                    }
                }
            }

            model_.AddConstr(expr <= 1);
        }
    }
};

template <typename GraphT>
void MultiProcessorPebbling<GraphT>::SetupSyncPhaseVariablesConstraints(const BspInstance<GraphT> &instance) {
    compPhase_ = model_.AddVars(static_cast<int>(maxTime_), COPT_BINARY, "comp_phase");

    if (mergeSteps_) {
        commPhase_ = model_.AddVars(static_cast<int>(maxTime_), COPT_BINARY, "comm_phase");
    } else {
        sendUpPhase_ = model_.AddVars(static_cast<int>(maxTime_), COPT_BINARY, "sendUp_phase");
        sendDownPhase_ = model_.AddVars(static_cast<int>(maxTime_), COPT_BINARY, "sendDown_phase");
    }

    const unsigned m = static_cast<unsigned>(instance.NumberOfProcessors() * instance.NumberOfVertices());

    for (unsigned t = 0; t < maxTime_; t++) {
        Expr exprComp, exprComm, exprSendUp, exprSendDown;
        for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
            for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                if (computeExists_[node][processor][t]) {
                    exprComp += compute_[node][processor][static_cast<int>(t)];
                }
                if (mergeSteps_) {
                    if (sendUpExists_[node][processor][t]) {
                        exprComm += sendUp_[node][processor][static_cast<int>(t)];
                    }

                    if (sendDownExists_[node][processor][t]) {
                        exprComm += sendDown_[node][processor][static_cast<int>(t)];
                    }
                } else {
                    if (sendUpExists_[node][processor][t]) {
                        exprSendUp += sendUp_[node][processor][static_cast<int>(t)];
                    }

                    if (sendDownExists_[node][processor][t]) {
                        exprSendDown += sendDown_[node][processor][static_cast<int>(t)];
                    }
                }
            }
        }

        model_.AddConstr(m * compPhase_[static_cast<int>(t)] >= exprComp);
        if (mergeSteps_) {
            model_.AddConstr(2 * m * commPhase_[static_cast<int>(t)] >= exprComm);
            model_.AddConstr(compPhase_[static_cast<int>(t)] + commPhase_[static_cast<int>(t)] <= 1);
        } else {
            model_.AddConstr(m * sendUpPhase_[static_cast<int>(t)] >= exprSendUp);
            model_.AddConstr(m * sendDownPhase_[static_cast<int>(t)] >= exprSendDown);
            model_.AddConstr(
                compPhase_[static_cast<int>(t)] + sendUpPhase_[static_cast<int>(t)] + sendDownPhase_[static_cast<int>(t)] <= 1);
        }
    }
};

template <typename GraphT>
void MultiProcessorPebbling<GraphT>::SetupBspVariablesConstraintsObjective(const BspInstance<GraphT> &instance) {
    compPhaseEnds_ = model_.AddVars(static_cast<int>(maxTime_), COPT_BINARY, "comp_phase_ends");

    commPhaseEnds_ = model_.AddVars(static_cast<int>(maxTime_), COPT_BINARY, "comm_phase_ends");

    VarArray workInduced_ = model_.AddVars(static_cast<int>(maxTime_), COPT_CONTINUOUS, "work_induced");
    VarArray commInduced_ = model_.AddVars(static_cast<int>(maxTime_), COPT_CONTINUOUS, "comm_induced");

    std::vector<VarArray> workStepUntil(instance.NumberOfProcessors());
    std::vector<VarArray> commStepUntil(instance.NumberOfProcessors());
    std::vector<VarArray> sendUpStepUntil(instance.NumberOfProcessors());
    std::vector<VarArray> sendDownStepUntil(instance.NumberOfProcessors());

    VarArray sendUpInduced;
    VarArray sendDownInduced;
    if (upAndDownCostSummed_) {
        sendUpInduced = model_.AddVars(static_cast<int>(maxTime_), COPT_CONTINUOUS, "sendUp_induced");
        sendDownInduced = model_.AddVars(static_cast<int>(maxTime_), COPT_CONTINUOUS, "sendDown_induced");
    }

    for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
        workStepUntil[processor] = model_.AddVars(static_cast<int>(maxTime_), COPT_CONTINUOUS, "work_step_until");
        sendUpStepUntil[processor] = model_.AddVars(static_cast<int>(maxTime_), COPT_CONTINUOUS, "sendUp_step_until");
        sendDownStepUntil[processor] = model_.AddVars(static_cast<int>(maxTime_), COPT_CONTINUOUS, "sendUp_step_until");
    }

    for (unsigned t = 0; t < maxTime_; t++) {
        model_.AddConstr(compPhase_[static_cast<int>(t)] >= compPhaseEnds_[static_cast<int>(t)]);
        if (mergeSteps_) {
            model_.AddConstr(commPhase_[static_cast<int>(t)] >= commPhaseEnds_[static_cast<int>(t)]);
        } else {
            model_.AddConstr(sendDownPhase_[static_cast<int>(t)] + sendUpPhase_[static_cast<int>(t)]
                             >= commPhaseEnds_[static_cast<int>(t)]);
        }
    }
    for (unsigned t = 0; t < maxTime_ - 1; t++) {
        model_.AddConstr(compPhaseEnds_[static_cast<int>(t)]
                         >= compPhase_[static_cast<int>(t)] - compPhase_[static_cast<int>(t) + 1]);
        if (mergeSteps_) {
            model_.AddConstr(commPhaseEnds_[static_cast<int>(t)]
                             >= commPhase_[static_cast<int>(t)] - commPhase_[static_cast<int>(t) + 1]);
        } else {
            model_.AddConstr(commPhaseEnds_[static_cast<int>(t)]
                             >= sendDownPhase_[static_cast<int>(t)] + sendUpPhase_[static_cast<int>(t)]
                                    - sendDownPhase_[static_cast<int>(t) + 1] - sendUpPhase_[static_cast<int>(t) + 1]);
        }
    }

    model_.AddConstr(compPhaseEnds_[static_cast<int>(maxTime_) - 1] >= compPhase_[static_cast<int>(maxTime_) - 1]);
    if (mergeSteps_) {
        model_.AddConstr(commPhaseEnds_[static_cast<int>(maxTime_) - 1] >= commPhase_[static_cast<int>(maxTime_) - 1]);
    } else {
        model_.AddConstr(commPhaseEnds_[static_cast<int>(maxTime_) - 1]
                         >= sendDownPhase_[static_cast<int>(maxTime_) - 1] + sendUpPhase_[static_cast<int>(maxTime_) - 1]);
    }

    const unsigned m = static_cast<unsigned>(instance.NumberOfProcessors()
                                             * (sumOfVerticesWorkWeights(instance.GetComputationalDag())
                                                + sumOfVerticesCommunicationWeights(instance.GetComputationalDag())));

    for (unsigned t = 1; t < maxTime_; t++) {
        for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
            Expr exprWork;
            Expr exprSendUp;
            Expr exprSendDown;
            for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
                if (computeExists_[node][processor][t]) {
                    exprWork
                        += instance.GetComputationalDag().VertexWorkWeight(node) * compute_[node][processor][static_cast<int>(t)];
                }
                if (sendUpExists_[node][processor][t]) {
                    exprSendUp
                        += instance.GetComputationalDag().VertexCommWeight(node) * sendUp_[node][processor][static_cast<int>(t)];
                }
                if (sendDownExists_[node][processor][t]) {
                    exprSendDown += instance.GetComputationalDag().VertexCommWeight(node)
                                    * sendDown_[node][processor][static_cast<int>(t)];
                }
            }

            model_.AddConstr(m * commPhaseEnds_[static_cast<int>(t)] + workStepUntil[processor][static_cast<int>(t)]
                             >= workStepUntil[processor][static_cast<int>(t) - 1] + exprWork);

            model_.AddConstr(m * compPhaseEnds_[static_cast<int>(t)] + sendUpStepUntil[processor][static_cast<int>(t)]
                             >= sendUpStepUntil[processor][static_cast<int>(t) - 1] + exprSendUp);

            model_.AddConstr(m * compPhaseEnds_[static_cast<int>(t)] + sendDownStepUntil[processor][static_cast<int>(t)]
                             >= sendDownStepUntil[processor][static_cast<int>(t) - 1] + exprSendDown);

            model_.AddConstr(workInduced_[static_cast<int>(t)]
                             >= workStepUntil[processor][static_cast<int>(t)] - m * (1 - compPhaseEnds_[static_cast<int>(t)]));
            if (upAndDownCostSummed_) {
                model_.AddConstr(sendUpInduced[static_cast<int>(t)] >= sendUpStepUntil[processor][static_cast<int>(t)]
                                                                           - m * (1 - commPhaseEnds_[static_cast<int>(t)]));
                model_.AddConstr(sendDownInduced[static_cast<int>(t)] >= sendDownStepUntil[processor][static_cast<int>(t)]
                                                                             - m * (1 - commPhaseEnds_[static_cast<int>(t)]));
                model_.AddConstr(commInduced_[static_cast<int>(t)]
                                 >= sendUpInduced[static_cast<int>(t)] + sendDownInduced[static_cast<int>(t)]);
            } else {
                model_.AddConstr(commInduced_[static_cast<int>(t)] >= sendDownStepUntil[processor][static_cast<int>(t)]
                                                                          - m * (1 - commPhaseEnds_[static_cast<int>(t)]));
                model_.AddConstr(commInduced_[static_cast<int>(t)] >= sendUpStepUntil[processor][static_cast<int>(t)]
                                                                          - m * (1 - commPhaseEnds_[static_cast<int>(t)]));
            }
        }
    }

    // t = 0
    for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
        Expr exprWork;
        Expr exprSendUp;
        Expr exprSendDown;
        for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
            if (computeExists_[node][processor][0]) {
                exprWork += instance.GetComputationalDag().VertexWorkWeight(node) * compute_[node][processor][0];
            }
            if (sendUpExists_[node][processor][0]) {
                exprSendUp += instance.GetComputationalDag().VertexCommWeight(node) * sendUp_[node][processor][0];
            }
            if (sendDownExists_[node][processor][0]) {
                exprSendDown += instance.GetComputationalDag().VertexCommWeight(node) * sendDown_[node][processor][0];
            }
        }

        model_.AddConstr(m * commPhaseEnds_[0] + workStepUntil[processor][0] >= exprWork);

        model_.AddConstr(m * compPhaseEnds_[0] + sendUpStepUntil[processor][0] >= exprSendUp);

        model_.AddConstr(m * compPhaseEnds_[0] + sendDownStepUntil[processor][0] >= exprSendDown);

        model_.AddConstr(workInduced_[0] >= workStepUntil[processor][0] - m * (1 - compPhaseEnds_[0]));
        if (upAndDownCostSummed_) {
            model_.AddConstr(sendUpInduced[0] >= sendUpStepUntil[processor][0] - m * (1 - commPhaseEnds_[0]));
            model_.AddConstr(sendDownInduced[0] >= sendDownStepUntil[processor][0] - m * (1 - commPhaseEnds_[0]));
            model_.AddConstr(commInduced_[0] >= sendUpInduced[0] + sendDownInduced[0]);
        } else {
            model_.AddConstr(commInduced_[0] >= sendDownStepUntil[processor][0] - m * (1 - commPhaseEnds_[0]));
            model_.AddConstr(commInduced_[0] >= sendUpStepUntil[processor][0] - m * (1 - commPhaseEnds_[0]));
        }
    }

    /*
    Objective
*/

    Expr expr;
    for (unsigned t = 0; t < maxTime_; t++) {
        expr += workInduced_[static_cast<int>(t)] + instance.SynchronisationCosts() * commPhaseEnds_[static_cast<int>(t)]
                + instance.CommunicationCosts() * commInduced_[static_cast<int>(t)];
    }

    model_.SetObjective(expr, COPT_MINIMIZE);
};

template <typename GraphT>
void MultiProcessorPebbling<GraphT>::SetupSyncObjective(const BspInstance<GraphT> &instance) {
    Expr expr;
    for (unsigned t = 0; t < maxTime_; t++) {
        if (!mergeSteps_) {
            expr += compPhase_[static_cast<int>(t)] + instance.CommunicationCosts() * sendUpPhase_[static_cast<int>(t)]
                    + instance.CommunicationCosts() * sendDownPhase_[static_cast<int>(t)];
        } else {
            // this objective+parameter combination is not very meaningful, but still defined here to avoid a segfault otherwise
            expr += compPhase_[static_cast<int>(t)] + instance.CommunicationCosts() * commPhase_[static_cast<int>(t)];
        }
    }

    model_.SetObjective(expr, COPT_MINIMIZE);
}

template <typename GraphT>
void MultiProcessorPebbling<GraphT>::SetupAsyncVariablesConstraintsObjective(const BspInstance<GraphT> &instance) {
    std::vector<VarArray> finishTimes(instance.NumberOfProcessors());

    for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
        finishTimes[processor] = model_.AddVars(static_cast<int>(maxTime_), COPT_CONTINUOUS, "finish_times");
    }

    Var makespan = model_.AddVar(0, COPT_INFINITY, 1, COPT_CONTINUOUS, "makespan");

    VarArray getsBlue = model_.AddVars(static_cast<int>(instance.NumberOfVertices()), COPT_CONTINUOUS, "gets_blue");

    const unsigned m = static_cast<unsigned>(instance.NumberOfProcessors()
                                             * (sumOfVerticesWorkWeights(instance.GetComputationalDag())
                                                + sumOfVerticesCommunicationWeights(instance.GetComputationalDag())));

    for (unsigned t = 0; t < maxTime_; t++) {
        for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
            Expr sendDownStepLength;
            for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
                if (sendDownExists_[node][processor][t]) {
                    sendDownStepLength += instance.CommunicationCosts() * instance.GetComputationalDag().VertexCommWeight(node)
                                          * sendDown_[node][processor][static_cast<int>(t)];
                }
            }

            for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
                if (sendUpExists_[node][processor][t]) {
                    model_.AddConstr(getsBlue[static_cast<int>(node)]
                                     >= finishTimes[processor][static_cast<int>(t)]
                                            - (1 - sendUp_[node][processor][static_cast<int>(t)]) * m);
                }
                if (sendDownExists_[node][processor][t]) {
                    model_.AddConstr(getsBlue[static_cast<int>(node)]
                                     <= finishTimes[processor][static_cast<int>(t)]
                                            + (1 - sendDown_[node][processor][static_cast<int>(t)]) * m - sendDownStepLength);
                }
            }
        }
    }

    // makespan constraint
    for (unsigned t = 0; t < maxTime_; t++) {
        for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
            model_.AddConstr(makespan >= finishTimes[processor][static_cast<int>(t)]);
        }
    }

    // t = 0
    for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
        Expr expr;
        for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
            if (computeExists_[node][processor][0]) {
                expr += instance.GetComputationalDag().VertexWorkWeight(node) * compute_[node][processor][0];
            }

            if (sendUpExists_[node][processor][0]) {
                expr += instance.CommunicationCosts() * instance.GetComputationalDag().VertexCommWeight(node)
                        * sendUp_[node][processor][0];
            }

            if (sendDownExists_[node][processor][0]) {
                expr += instance.CommunicationCosts() * instance.GetComputationalDag().VertexCommWeight(node)
                        * sendDown_[node][processor][0];
            }
        }

        model_.AddConstr(finishTimes[processor][0] >= expr);
    }

    for (unsigned t = 1; t < maxTime_; t++) {
        for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
            Expr expr;
            for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
                if (computeExists_[node][processor][t]) {
                    expr += instance.GetComputationalDag().VertexWorkWeight(node) * compute_[node][processor][static_cast<int>(t)];
                }

                if (sendUpExists_[node][processor][t]) {
                    expr += instance.CommunicationCosts() * instance.GetComputationalDag().VertexCommWeight(node)
                            * sendUp_[node][processor][static_cast<int>(t)];
                }

                if (sendDownExists_[node][processor][t]) {
                    expr += instance.CommunicationCosts() * instance.GetComputationalDag().VertexCommWeight(node)
                            * sendDown_[node][processor][static_cast<int>(t)];
                }
            }

            model_.AddConstr(finishTimes[processor][static_cast<int>(t)] >= finishTimes[processor][static_cast<int>(t) - 1] + expr);
        }
    }

    /*
    Objective
      */

    model_.SetObjective(makespan, COPT_MINIMIZE);
}

template <typename GraphT>
void MultiProcessorPebbling<GraphT>::WriteSolutionCallback::Callback() {
    if (Where() == COPT_CBCONTEXT_MIPSOL && counter_ < maxNumberSolution_ && GetIntInfo(COPT_CBINFO_HASINCUMBENT)) {
        try {
            if (GetDblInfo(COPT_CBINFO_BESTOBJ) < bestObj_ && 0.0 < GetDblInfo(COPT_CBINFO_BESTBND)) {
                bestObj_ = GetDblInfo(COPT_CBINFO_BESTOBJ);

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
    const BspInstance<GraphT> &instance = schedule.GetInstance();

    std::vector<std::vector<std::set<std::pair<unsigned, VertexIdx>>>> nodesComputed(
        instance.NumberOfProcessors(), std::vector<std::set<std::pair<unsigned, VertexIdx>>>(maxTime_));
    std::vector<std::vector<std::deque<VertexIdx>>> nodesSentUp(instance.NumberOfProcessors(),
                                                                std::vector<std::deque<VertexIdx>>(maxTime_));
    std::vector<std::vector<std::deque<VertexIdx>>> nodesSentDown(instance.NumberOfProcessors(),
                                                                  std::vector<std::deque<VertexIdx>>(maxTime_));
    std::vector<std::vector<std::set<VertexIdx>>> evictedAfter(instance.NumberOfProcessors(),
                                                               std::vector<std::set<VertexIdx>>(maxTime_));

    // used to remove unneeded steps when a node is sent down and then up (which becomes invalid after reordering the comm phases)
    std::vector<std::vector<bool>> sentDownAlready(instance.NumberOfVertices(),
                                                   std::vector<bool>(instance.NumberOfProcessors(), false));
    std::vector<std::vector<bool>> ignoreRed(instance.NumberOfVertices(), std::vector<bool>(instance.NumberOfProcessors(), false));

    std::vector<VertexIdx> topOrder = GetTopOrder(instance.GetComputationalDag());
    std::vector<unsigned> topOrderPosition(instance.NumberOfVertices());
    for (unsigned index = 0; index < instance.NumberOfVertices(); ++index) {
        topOrderPosition[topOrder[index]] = index;
    }

    std::vector<bool> emptyStep(maxTime_, true);
    std::vector<std::vector<unsigned>> stepTypeOnProc(instance.NumberOfProcessors(), std::vector<unsigned>(maxTime_, 0));

    for (unsigned step = 0; step < maxTime_; step++) {
        for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
            for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
                if (computeExists_[node][processor][step]
                    && compute_[node][processor][static_cast<int>(step)].Get(COPT_DBLINFO_VALUE) >= .99) {
                    stepTypeOnProc[processor][step] = 1;
                }
            }
        }
    }

    for (unsigned step = 0; step < maxTime_; step++) {
        for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
            for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
                if (step > 0 && hasRed_[node][processor][static_cast<int>(step) - 1].Get(COPT_DBLINFO_VALUE) >= .99
                    && hasRed_[node][processor][static_cast<int>(step)].Get(COPT_DBLINFO_VALUE) <= .01
                    && !ignoreRed[node][processor]) {
                    for (size_t previousStep = step - 1; previousStep < step; --previousStep) {
                        if (!nodesComputed_[processor][previousStep].empty() || !nodesSentUp_[processor][previousStep].empty()
                            || !nodesSentDown_[processor][previousStep].empty() || previousStep == 0) {
                            evictedAfter[processor][previousStep].insert(node);
                            emptyStep[previousStep] = false;
                            break;
                        }
                    }
                }

                if (computeExists_[node][processor][step]
                    && compute_[node][processor][static_cast<int>(step)].Get(COPT_DBLINFO_VALUE) >= .99) {
                    nodesComputed_[processor][step].emplace(topOrderPosition[node], node);
                    emptyStep[step] = false;
                    ignoreRed_[node][processor] = false;

                    // implicit eviction in case of mergesteps - never having "hasRed=1"
                    if (step + 1 < max_time && hasRed_[node][processor][static_cast<int>(step) + 1].Get(COPT_DBLINFO_VALUE) <= .01) {
                        evictedAfter[processor][step].insert(node);
                    }
                }

                if (sendDownExists_[node][processor][step]
                    && sendDown_[node][processor][static_cast<int>(step)].Get(COPT_DBLINFO_VALUE) >= .99) {
                    bool keepStep = false;

                    for (unsigned nextStep = step + 1;
                         nextStep < maxTime_ && hasRed_[node][processor][static_cast<int>(nextStep)].Get(COPT_DBLINFO_VALUE) >= .99;
                         ++nextStep) {
                        if (stepTypeOnProc[processor][nextStep] == 1) {
                            keepStep = true;
                            break;
                        }
                    }

                    if (keepStep) {
                        nodesSentDown_[processor][step].push_back(node);
                        emptyStep[step] = false;
                        stepTypeOnProc[processor][step] = 3;
                        ignoreRed_[node][processor] = false;
                    } else {
                        ignoreRed_[node][processor] = true;
                    }

                    sentDownAlready[node][processor] = true;
                }

                if (sendUpExists_[node][processor][step]
                    && sendUp_[node][processor][static_cast<int>(step)].Get(COPT_DBLINFO_VALUE) >= .99
                    && !sent_down_already[node][processor]) {
                    nodesSentUp[processor][step].push_back(node);
                    emptyStep[step] = false;
                    stepTypeOnProc[processor][step] = 2;
                }
            }
        }
    }

    // components of the final PebblingSchedule - the first two dimensions are always processor and superstep
    std::vector<std::vector<std::vector<VertexIdx>>> computeStepsPerSupstep(instance.NumberOfProcessors());
    std::vector<std::vector<std::vector<std::vector<VertexIdx>>>> nodesEvictedAfterCompute(instance.NumberOfProcessors());
    std::vector<std::vector<std::vector<VertexIdx>>> nodesSentUpInSupstep(instance.NumberOfProcessors());
    std::vector<std::vector<std::vector<VertexIdx>>> nodesSentDownInSupstep(instance.NumberOfProcessors());
    std::vector<std::vector<std::vector<VertexIdx>>> nodesEvictedInCommPhase(instance.NumberOfProcessors());

    // edge case: check if an extra superstep must be added in the beginning to evict values that are initially in cache
    bool needsEvictStepInBeginning = false;
    for (unsigned proc = 0; proc < instance.NumberOfProcessors(); proc++) {
        for (unsigned step = 0; step < maxTime_; step++) {
            if (stepTypeOnProc[proc][step] == 0 && !evictedAfter[proc][step].empty()) {
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
                for (unsigned proc = 0; proc < instance.NumberOfProcessors(); proc++) {
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
                for (unsigned proc = 0; proc < instance.NumberOfProcessors(); proc++) {
                    computeStepsPerSupstep[proc].push_back(std::vector<VertexIdx>());
                    nodesEvictedAfterCompute[proc].push_back(std::vector<std::vector<VertexIdx>>());
                    nodesSentUpInSupstep[proc].push_back(std::vector<VertexIdx>());
                    nodesSentDownInSupstep[proc].push_back(std::vector<VertexIdx>());
                    nodesEvictedInCommPhase[proc].push_back(std::vector<VertexIdx>());
                }
            }
        }

        // process steps
        for (unsigned step = 0; step < maxTime_; step++) {
            if (emptyStep[step]) {
                continue;
            }

            unsigned stepType = 0;
            for (unsigned proc = 0; proc < instance.NumberOfProcessors(); proc++) {
                stepType = std::max(stepType, stepTypeOnProc[proc][step]);
            }

            if (stepType == 1) {
                if (inComm) {
                    for (unsigned proc = 0; proc < instance.NumberOfProcessors(); proc++) {
                        computeStepsPerSupstep[proc].push_back(std::vector<VertexIdx>());
                        nodesEvictedAfterCompute[proc].push_back(std::vector<std::vector<VertexIdx>>());
                        nodesSentUpInSupstep[proc].push_back(std::vector<VertexIdx>());
                        nodesSentDownInSupstep[proc].push_back(std::vector<VertexIdx>());
                        nodesEvictedInCommPhase[proc].push_back(std::vector<VertexIdx>());
                    }
                    ++superstepIndex;
                    inComm = false;
                }
                for (unsigned proc = 0; proc < instance.NumberOfProcessors(); proc++) {
                    for (auto indexAndNode : nodesComputed[proc][step]) {
                        computeStepsPerSupstep[proc][superstepIndex].push_back(indexAndNode.second);
                        nodesEvictedAfterCompute[proc][superstepIndex].push_back(std::vector<VertexIdx>());
                    }
                    for (VertexIdx node : evictedAfter[proc][step]) {
                        if (!nodesEvictedAfterCompute[proc][superstepIndex].empty()) {
                            nodesEvictedAfterCompute[proc][superstepIndex].back().push_back(node);
                        } else {
                            // can only happen in special case: eviction in the very beginning
                            nodesEvictedInCommPhase[proc][0].push_back(node);
                        }
                    }
                }
            }

            if (stepType == 2 || stepType == 3) {
                if (superstepIndex == UINT_MAX) {
                    for (unsigned proc = 0; proc < instance.NumberOfProcessors(); proc++) {
                        computeStepsPerSupstep[proc].push_back(std::vector<VertexIdx>());
                        nodesEvictedAfterCompute[proc].push_back(std::vector<std::vector<VertexIdx>>());
                        nodesSentUpInSupstep[proc].push_back(std::vector<VertexIdx>());
                        nodesSentDownInSupstep[proc].push_back(std::vector<VertexIdx>());
                        nodesEvictedInCommPhase[proc].push_back(std::vector<VertexIdx>());
                    }
                    ++superstepIndex;
                }

                inComm = true;
                for (unsigned proc = 0; proc < instance.NumberOfProcessors(); proc++) {
                    for (VertexIdx node : nodesSentUp[proc][step]) {
                        nodesSentUpInSupstep[proc][superstepIndex].push_back(node);
                    }
                    for (VertexIdx node : evictedAfter[proc][step]) {
                        nodesEvictedInCommPhase[proc][superstepIndex].push_back(node);
                    }
                    for (VertexIdx node : nodesSentDown[proc][step]) {
                        nodesSentDownInSupstep[proc][superstepIndex].push_back(node);
                    }
                }
            }
        }
    } else {
        std::vector<unsigned> stepIdxOnProc(instance.NumberOfProcessors(), 0);

        std::vector<bool> alreadyHasBlue(instance.NumberOfVertices(), false);
        if (needToLoadInputs_) {
            for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
                if (instance.GetComputationalDag().InDegree(node) == 0) {
                    alreadyHasBlue[node] = true;
                }
            }
        }

        std::vector<bool> procFinished(instance.NumberOfProcessors(), false);
        unsigned nrProcFinished = 0;
        while (nrProcFinished < instance.NumberOfProcessors()) {
            // preliminary sweep of superstep, to see if we need to wait for other processors
            std::vector<unsigned> idxLimitOnProc = stepIdxOnProc;

            // first add compute steps
            if (!needsEvictStepInBeginning || superstepIndex > 0) {
                for (unsigned proc = 0; proc < instance.NumberOfProcessors(); proc++) {
                    while (idxLimitOnProc[proc] < maxTime_ && stepTypeOnProc[proc][idxLimitOnProc[proc]] <= 1) {
                        ++idxLimitOnProc[proc];
                    }
                }
            }

            // then add communications step until possible (note - they might not be valid if all put into a single superstep!)
            std::set<VertexIdx> newBlues;
            bool stillMakingProgress = true;
            while (stillMakingProgress) {
                stillMakingProgress = false;
                for (unsigned proc = 0; proc < instance.NumberOfProcessors(); proc++) {
                    while (idxLimitOnProc[proc] < maxTime_ && stepTypeOnProc[proc][idxLimitOnProc[proc]] != 1) {
                        bool acceptStep = true;
                        for (VertexIdx node : nodesSentDown[proc][idxLimitOnProc[proc]]) {
                            if (!alreadyHasBlue[node] && newBlues.find(node) == newBlues.end()) {
                                acceptStep = false;
                            }
                        }

                        if (!acceptStep) {
                            break;
                        }

                        for (VertexIdx node : nodesSentUp[proc][idxLimitOnProc[proc]]) {
                            if (!alreadyHasBlue[node]) {
                                newBlues.insert(node);
                            }
                        }

                        stillMakingProgress = true;
                        ++idxLimitOnProc[proc];
                    }
                }
            }

            // actually process the superstep
            for (unsigned proc = 0; proc < instance.NumberOfProcessors(); proc++) {
                computeStepsPerSupstep_[proc].push_back(std::vector<VertexIdx>());
                nodesEvictedAfterCompute_[proc].push_back(std::vector<std::vector<VertexIdx>>());
                nodesSentUpInSupstep_[proc].push_back(std::vector<VertexIdx>());
                nodesSentDownInSupstep_[proc].push_back(std::vector<VertexIdx>());
                nodesEvictedInCommPhase_[proc].push_back(std::vector<VertexIdx>());

                while (stepIdxOnProc[proc] < idxLimitOnProc[proc] && stepTypeOnProc[proc][stepIdxOnProc[proc]] <= 1) {
                    for (auto indexAndNode : computeSteps[proc][stepIdxOnProc[proc]]) {
                        computeStepsPerSupstep_[proc][superstepIndex].push_back(indexAndNode.second);
                        nodesEvictedAfterCompute_[proc][superstepIndex].push_back(std::vector<VertexIdx>());
                    }
                    for (VertexIdx node : nodesEvictedAfterStep[proc][stepIdxOnProc[proc]]) {
                        if (!nodesEvictedAfterCompute_[proc][superstepIndex].empty()) {
                            nodesEvictedAfterCompute_[proc][superstepIndex].back().push_back(node);
                        } else {
                            // can only happen in special case: eviction in the very beginning
                            nodesEvictedInCommPhase_[proc][superstepIndex].push_back(node);
                        }
                    }

                    ++stepIdxOnProc[proc];
                }
                while (stepIdxOnProc[proc] < idxLimitOnProc[proc] && stepTypeOnProc[proc][stepIdxOnProc[proc]] != 1) {
                    for (VertexIdx node : nodesSentUp[proc][stepIdxOnProc[proc]]) {
                        nodesSentUpInSupstep_[proc][superstepIndex].push_back(node);
                        alreadyHasBlue[node] = true;
                    }
                    for (VertexIdx node : nodesSentDown[proc][stepIdxOnProc[proc]]) {
                        nodesSentDownInSupstep_[proc][superstepIndex].push_back(node);
                    }
                    for (VertexIdx node : evictedAfter[proc][stepIdxOnProc[proc]]) {
                        nodesEvictedInCommPhase_[proc][superstepIndex].push_back(node);
                    }

                    ++stepIdxOnProc[proc];
                }
                if (stepIdxOnProc[proc] == maxTime_ && !procFinished[proc]) {
                    procFinished[proc] = true;
                    ++nrProcFinished;
                }
            }
            ++superstepIndex;
        }
    }

    std::cout << "MPP ILP best solution value: " << model_.GetDblAttr(COPT_DBLATTR_BESTOBJ)
              << ", best lower bound: " << model_.GetDblAttr(COPT_DBLATTR_BESTBND) << std::endl;

    schedule = PebblingSchedule<GraphT>(instance,
                                        computeStepsPerSupstep_,
                                        nodesEvictedAfterCompute_,
                                        nodesSentUpInSupstep_,
                                        nodesSentDownInSupstep_,
                                        nodesEvictedInCommPhase_,
                                        needsBlueAtEnd_,
                                        hasRedInBeginning_,
                                        needToLoadInputs_);
}

template <typename GraphT>
void MultiProcessorPebbling<GraphT>::SetInitialSolution(
    const BspInstance<GraphT> &instance,
    const std::vector<std::vector<std::vector<VertexIdx>>> &computeSteps,
    const std::vector<std::vector<std::vector<VertexIdx>>> &sendUpSteps,
    const std::vector<std::vector<std::vector<VertexIdx>>> &sendDownSteps,
    const std::vector<std::vector<std::vector<VertexIdx>>> &nodesEvictedAfterStep) {
    const unsigned n = static_cast<unsigned>(instance.NumberOfVertices());

    std::vector<bool> inSlowMem(n, false);
    if (needToLoadInputs_) {
        for (VertexIdx node = 0; node < n; ++node) {
            if (instance.GetComputationalDag().InDegree(node) == 0) {
                inSlowMem[node] = true;
            }
        }
    }

    std::vector<std::vector<bool>> inFastMem(n, std::vector<bool>(instance.NumberOfProcessors(), false));
    if (!hasRedInBeginning_.empty()) {
        for (unsigned proc = 0; proc < instance.NumberOfProcessors(); ++proc) {
            for (VertexIdx node : hasRedInBeginning_[proc]) {
                inFastMem[node][proc] = true;
            }
        }
    }

    unsigned step = 0, newStepIdx = 0;
    for (; step < computeSteps[0].size(); ++step) {
        for (VertexIdx node = 0; node < n; ++node) {
            if (hasBlueExists[node][newStepIdx]) {
                model_.SetMipStart(hasBlue[node][static_cast<int>(newStepIdx)], static_cast<int>(inSlowMem[node]));
            }
            for (unsigned proc = 0; proc < instance.NumberOfProcessors(); ++proc) {
                model_.SetMipStart(hasRed[node][proc][static_cast<int>(newStepIdx)], static_cast<int>(inFastMem[node][proc]));
            }
        }

        if (restrictStepTypes_) {
            // align step number with step type cycle's phase, if needed
            bool skipStep = true;
            while (skipStep) {
                skipStep = false;
                bool isCompute = false, isSendUp = false, isSendDown = false;
                for (unsigned proc = 0; proc < instance.NumberOfProcessors(); ++proc) {
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
                    for (VertexIdx node = 0; node < n; ++node) {
                        if (hasBlueExists[node][newStepIdx]) {
                            model_.SetMipStart(hasBlue[node][static_cast<int>(newStepIdx)], static_cast<int>(in_slow_mem[node]));
                        }
                        for (unsigned proc = 0; proc < instance.NumberOfProcessors(); ++proc) {
                            model_.SetMipStart(hasRed[node][proc][static_cast<int>(newStepIdx)],
                                               static_cast<int>(in_fast_mem[node][proc]));
                        }
                    }
                }
            }
        }

        for (unsigned proc = 0; proc < instance.NumberOfProcessors(); ++proc) {
            std::vector<bool> valueOfNode(n, false);
            for (VertexIdx node : computeSteps[proc][step]) {
                value_of_node[node] = true;
                if (computeExists[node][proc][newStepIdx]) {
                    model_.SetMipStart(compute[node][proc][static_cast<int>(newStepIdx)], 1);
                }
                in_fast_mem[node][proc] = true;
            }
            for (VertexIdx node : computeSteps[proc][step]) {
                if (!value_of_node[node]) {
                    if (computeExists[node][proc][newStepIdx]) {
                        model_.SetMipStart(compute[node][proc][static_cast<int>(newStepIdx)], 0);
                    }
                } else {
                    value_of_node[node] = false;
                }
            }

            for (VertexIdx node : sendUpSteps[proc][step]) {
                value_of_node[node] = true;
                if (sendUpExists[node][proc][newStepIdx]) {
                    model_.SetMipStart(sendUp[node][proc][static_cast<int>(newStepIdx)], 1);
                }
                in_slow_mem[node] = true;
            }
            for (VertexIdx node : sendUpSteps[proc][step]) {
                if (!value_of_node[node]) {
                    if (sendUpExists[node][proc][newStepIdx]) {
                        model_.SetMipStart(sendUp[node][proc][static_cast<int>(newStepIdx)], 0);
                    }
                } else {
                    value_of_node[node] = false;
                }
            }

            for (VertexIdx node : sendDownSteps[proc][step]) {
                value_of_node[node] = true;
                if (sendDownExists[node][proc][newStepIdx]) {
                    model_.SetMipStart(sendDown[node][proc][static_cast<int>(newStepIdx)], 1);
                }
                in_fast_mem[node][proc] = true;
            }
            for (VertexIdx node : sendDownSteps[proc][step]) {
                if (!value_of_node[node]) {
                    if (sendDownExists[node][proc][newStepIdx]) {
                        model_.SetMipStart(sendDown[node][proc][static_cast<int>(newStepIdx)], 0);
                    }
                } else {
                    value_of_node[node] = false;
                }
            }

            for (VertexIdx node : nodesEvictedAfterStep[proc][step]) {
                in_fast_mem[node][proc] = false;
            }
        }
        ++newStepIdx;
    }
    for (; newStepIdx < maxTime_; ++newStepIdx) {
        for (VertexIdx node = 0; node < n; ++node) {
            if (hasBlueExists[node][newStepIdx]) {
                model_.SetMipStart(hasBlue[node][static_cast<int>(newStepIdx)], static_cast<int>(inSlowMem[node]));
            }
            for (unsigned proc = 0; proc < instance.NumberOfProcessors(); ++proc) {
                model_.SetMipStart(hasRed[node][proc][static_cast<int>(newStepIdx)], 0);
                if (computeExists[node][proc][newStepIdx]) {
                    model_.SetMipStart(compute[node][proc][static_cast<int>(newStepIdx)], 0);
                }
                if (sendUpExists[node][proc][newStepIdx]) {
                    model_.SetMipStart(sendUp[node][proc][static_cast<int>(newStepIdx)], 0);
                }
                if (sendDownExists[node][proc][newStepIdx]) {
                    model_.SetMipStart(sendDown[node][proc][static_cast<int>(newStepIdx)], 0);
                }
            }
        }
    }
    model_.LoadMipStart();
}

template <typename GraphT>
unsigned MultiProcessorPebbling<GraphT>::ComputeMaxTimeForInitialSolution(
    const BspInstance<GraphT> &instance,
    const std::vector<std::vector<std::vector<VertexIdx>>> &computeSteps,
    const std::vector<std::vector<std::vector<VertexIdx>>> &sendUpSteps,
    const std::vector<std::vector<std::vector<VertexIdx>>> &sendDownSteps) const {
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
            for (unsigned proc = 0; proc < instance.NumberOfProcessors(); ++proc) {
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
        for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
            for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                if ((computeExists[node][processor][step] && compute[node][processor][step].Get(COPT_DBLINFO_VALUE) >= .99)
                    || (sendUpExists[node][processor][step] && sendUp[node][processor][step].Get(COPT_DBLINFO_VALUE) >= .99)
                    || (sendDownExists[node][processor][step] && sendDown[node][processor][step].Get(COPT_DBLINFO_VALUE) >= .99)) {
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
