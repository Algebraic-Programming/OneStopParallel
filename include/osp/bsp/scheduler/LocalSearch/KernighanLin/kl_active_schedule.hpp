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

#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/model/IBspSchedule.hpp"
#include "osp/bsp/model/util/SetSchedule.hpp"
#include "osp/bsp/model/util/VectorSchedule.hpp"
#include "osp/bsp/scheduler/ImprovementScheduler.hpp"
#include "osp/bsp/scheduler/LocalSearch/LocalSearchMemoryConstraintModules.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"

namespace osp {

template <typename CostT, typename VertexIdxT>
struct KlMoveStruct {
    VertexIdxT node_;
    CostT gain_;

    unsigned fromProc_;
    unsigned fromStep_;

    unsigned toProc_;
    unsigned toStep_;

    KlMoveStruct() : node_(0), gain_(0), fromProc_(0), fromStep_(0), toProc_(0), toStep_(0) {}

    KlMoveStruct(VertexIdxT node, CostT gain, unsigned fromProc, unsigned fromStep, unsigned toProc, unsigned toStep)
        : node_(node), gain_(gain), fromProc_(fromProc), fromStep_(fromStep), toProc_(toProc), toStep_(toStep) {}

    bool operator<(KlMoveStruct<CostT, VertexIdxT> const &rhs) const {
        return (gain_ < rhs.gain_) or (gain_ == rhs.gain_ and node_ > rhs.node_);
    }

    bool operator>(KlMoveStruct<CostT, VertexIdxT> const &rhs) const {
        return (gain_ > rhs.gain_) or (gain_ >= rhs.gain_ and node_ < rhs.node_);
    }

    KlMoveStruct<CostT, VertexIdxT> ReverseMove() const {
        return KlMoveStruct(node_, -gain_, toProc_, toStep_, fromProc_, fromStep_);
    }
};

template <typename WorkWeightT>
struct PreMoveWorkData {
    WorkWeightT fromStepMaxWork_;
    WorkWeightT fromStepSecondMaxWork_;
    unsigned fromStepMaxWorkProcessorCount_;

    WorkWeightT toStepMaxWork_;
    WorkWeightT toStepSecondMaxWork_;
    unsigned toStepMaxWorkProcessorCount_;

    PreMoveWorkData() {}

    PreMoveWorkData(WorkWeightT fromStepMaxWork,
                    WorkWeightT fromStepSecondMaxWork,
                    unsigned fromStepMaxWorkProcessorCount,
                    WorkWeightT toStepMaxWork,
                    WorkWeightT toStepSecondMaxWork,
                    unsigned toStepMaxWorkProcessorCount)
        : fromStepMaxWork_(fromStepMaxWork),
          fromStepSecondMaxWork_(fromStepSecondMaxWork),
          fromStepMaxWorkProcessorCount_(fromStepMaxWorkProcessorCount),
          toStepMaxWork_(toStepMaxWork),
          toStepSecondMaxWork_(toStepSecondMaxWork),
          toStepMaxWorkProcessorCount_(toStepMaxWorkProcessorCount) {}
};

template <typename GraphT>
struct KlActiveScheduleWorkDatastructures {
    using WorkWeightT = VWorkwT<GraphT>;

    const BspInstance<GraphT> *instance_;
    const SetSchedule<GraphT> *setSchedule_;

    struct WeightProc {
        WorkWeightT work_;
        unsigned proc_;

        WeightProc() : work_(0), proc_(0) {}

        WeightProc(WorkWeightT work, unsigned proc) : work_(work), proc_(proc) {}

        bool operator<(WeightProc const &rhs) const { return (work_ > rhs.work_) or (work_ == rhs.work_ and proc_ < rhs.proc_); }
    };

    std::vector<std::vector<WeightProc>> stepProcessorWork_;
    std::vector<std::vector<unsigned>> stepProcessorPosition_;
    std::vector<unsigned> stepMaxWorkProcessorCount_;
    WorkWeightT maxWorkWeight_;
    WorkWeightT totalWorkWeight_;

    inline WorkWeightT StepMaxWork(unsigned step) const { return stepProcessorWork_[step][0].work_; }

    inline WorkWeightT StepSecondMaxWork(unsigned step) const {
        return stepProcessorWork_[step][stepMaxWorkProcessorCount_[step]].work_;
    }

    inline WorkWeightT StepProcWork(unsigned step, unsigned proc) const {
        return stepProcessorWork_[step][stepProcessorPosition_[step][proc]].work_;
    }

    inline WorkWeightT &StepProcWork(unsigned step, unsigned proc) {
        return stepProcessorWork_[step][stepProcessorPosition_[step][proc]].work_;
    }

    template <typename CostT, typename VertexIdxT>
    inline PreMoveWorkData<WorkWeightT> GetPreMoveWorkData(KlMoveStruct<CostT, VertexIdxT> move) {
        return PreMoveWorkData<WorkWeightT>(StepMaxWork(move.fromStep_),
                                            StepSecondMaxWork(move.fromStep_),
                                            stepMaxWorkProcessorCount_[move.fromStep_],
                                            StepMaxWork(move.toStep_),
                                            StepSecondMaxWork(move.toStep_),
                                            stepMaxWorkProcessorCount_[move.toStep_]);
    }

    inline void Initialize(const SetSchedule<GraphT> &sched, const BspInstance<GraphT> &inst, unsigned numSteps) {
        instance_ = &inst;
        setSchedule_ = &sched;
        maxWorkWeight_ = 0;
        totalWorkWeight_ = 0;
        stepProcessorWork_.assign(numSteps, std::vector<WeightProc>(instance_->NumberOfProcessors()));
        stepProcessorPosition_.assign(numSteps, std::vector<unsigned>(instance_->NumberOfProcessors(), 0));
        stepMaxWorkProcessorCount_.assign(numSteps, 0);
    }

    inline void Clear() {
        stepProcessorWork_.clear();
        stepProcessorPosition_.clear();
        stepMaxWorkProcessorCount_.clear();
    }

    inline void ArrangeSuperstepData(const unsigned step) {
        std::sort(stepProcessorWork_[step].begin(), stepProcessorWork_[step].end());
        unsigned pos = 0;
        const WorkWeightT maxWorkTo = stepProcessorWork_[step][0].work_;

        for (const auto &wp : stepProcessorWork_[step]) {
            stepProcessorPosition_[step][wp.proc_] = pos++;

            if (wp.work_ == maxWorkTo && pos < instance_->NumberOfProcessors()) {
                stepMaxWorkProcessorCount_[step] = pos;
            }
        }
    }

    template <typename CostT, typename VertexIdxT>
    void ApplyMove(KlMoveStruct<CostT, VertexIdxT> move, WorkWeightT workWeight) {
        if (workWeight == 0) {
            return;
        }

        if (move.toStep_ != move.fromStep_) {
            StepProcWork(move.toStep_, move.toProc_) += workWeight;
            StepProcWork(move.fromStep_, move.fromProc_) -= workWeight;

            ArrangeSuperstepData(move.toStep_);
            ArrangeSuperstepData(move.fromStep_);

        } else {
            StepProcWork(move.toStep_, move.toProc_) += workWeight;
            StepProcWork(move.fromStep_, move.fromProc_) -= workWeight;
            ArrangeSuperstepData(move.toStep_);
        }
    }

    void SwapSteps(const unsigned step1, const unsigned step2) {
        std::swap(stepProcessorWork_[step1], stepProcessorWork_[step2]);
        std::swap(stepProcessorPosition_[step1], stepProcessorPosition_[step2]);
        std::swap(stepMaxWorkProcessorCount_[step1], stepMaxWorkProcessorCount_[step2]);
    }

    void OverrideNextSuperstep(unsigned step) {
        const unsigned nextStep = step + 1;
        for (unsigned i = 0; i < instance_->NumberOfProcessors(); i++) {
            stepProcessorWork_[nextStep][i] = stepProcessorWork_[step][i];
            stepProcessorPosition_[nextStep][i] = stepProcessorPosition_[step][i];
        }
        stepMaxWorkProcessorCount_[nextStep] = stepMaxWorkProcessorCount_[step];
    }

    void ResetSuperstep(unsigned step) {
        for (unsigned i = 0; i < instance_->NumberOfProcessors(); i++) {
            stepProcessorWork_[step][i] = {0, i};
            stepProcessorPosition_[step][i] = i;
        }
        stepMaxWorkProcessorCount_[step] = instance_->NumberOfProcessors() - 1;
    }

    void ComputeWorkDatastructures(unsigned startStep, unsigned endStep) {
        for (unsigned step = startStep; step <= endStep; step++) {
            stepMaxWorkProcessorCount_[step] = 0;
            WorkWeightT maxWork = 0;

            for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
                stepProcessorWork_[step][proc].work_ = 0;
                stepProcessorWork_[step][proc].proc_ = proc;

                for (const auto &node : setSchedule_->GetProcessorStepVertices()[step][proc]) {
                    const WorkWeightT vertexWorkWeight = instance_->GetComputationalDag().VertexWorkWeight(node);
                    totalWorkWeight_ += vertexWorkWeight;
                    maxWorkWeight_ = std::max(vertexWorkWeight, maxWorkWeight_);
                    stepProcessorWork_[step][proc].work_ += vertexWorkWeight;
                }

                if (stepProcessorWork_[step][proc].work_ > maxWork) {
                    maxWork = stepProcessorWork_[step][proc].work_;
                    stepMaxWorkProcessorCount_[step] = 1;
                } else if (stepProcessorWork_[step][proc].work_ == maxWork
                           && stepMaxWorkProcessorCount_[step] < (instance_->NumberOfProcessors() - 1)) {
                    stepMaxWorkProcessorCount_[step]++;
                }
            }

            std::sort(stepProcessorWork_[step].begin(), stepProcessorWork_[step].end());
            unsigned pos = 0;
            for (const auto &wp : stepProcessorWork_[step]) {
                stepProcessorPosition_[step][wp.proc_] = pos++;
            }
        }
    }
};

template <typename GraphT, typename CostT>
struct ThreadLocalActiveScheduleData {
    using VertexType = VertexIdxT<GraphT>;
    using EdgeType = EdgeDescT<GraphT>;

    using KlMove = KlMoveStruct<CostT, VertexType>;

    std::unordered_set<EdgeType> currentViolations_;
    std::vector<KlMove> appliedMoves_;

    CostT cost_ = 0;
    CostT initialCost_ = 0;
    bool feasible_ = true;

    CostT bestCost_ = 0;
    unsigned bestScheduleIdx_ = 0;

    std::unordered_map<VertexType, EdgeType> newViolations_;
    std::unordered_set<EdgeType> resolvedViolations_;

    inline void InitializeCost(CostT cost) {
        initialCost_ = cost;
        cost_ = cost;
        bestCost_ = cost;
        feasible_ = true;
    }

    inline void UpdateCost(CostT changeInCost) {
        cost_ += changeInCost;

        if (cost_ <= bestCost_ && feasible_) {
            bestCost_ = cost_;
            bestScheduleIdx_ = static_cast<unsigned>(appliedMoves_.size());
        }
    }
};

template <typename GraphT, typename CostT, typename MemoryConstraintT>
class KlActiveSchedule {
  private:
    using VertexType = VertexIdxT<GraphT>;
    using EdgeType = EdgeDescT<GraphT>;
    using KlMove = KlMoveStruct<CostT, VertexType>;
    using ThreadDataT = ThreadLocalActiveScheduleData<GraphT, CostT>;

    const BspInstance<GraphT> *instance_;

    VectorSchedule<GraphT> vectorSchedule_;
    SetSchedule<GraphT> setSchedule_;

    CostT cost_ = 0;
    bool feasible_ = true;

    unsigned staleness_ = 1;

  public:
    virtual ~KlActiveSchedule() = default;

    inline const BspInstance<GraphT> &GetInstance() const { return *instance_; }

    inline const VectorSchedule<GraphT> &GetVectorSchedule() const { return vectorSchedule_; }

    inline VectorSchedule<GraphT> &GetVectorSchedule() { return vectorSchedule_; }

    inline const SetSchedule<GraphT> &GetSetSchedule() const { return setSchedule_; }

    unsigned GetStaleness() const { return staleness_; }

    inline CostT GetCost() { return cost_; }

    inline bool IsFeasible() { return feasible_; }

    inline unsigned NumSteps() const { return vectorSchedule_.NumberOfSupersteps(); }

    inline unsigned AssignedProcessor(VertexType node) const { return vectorSchedule_.AssignedProcessor(node); }

    inline unsigned AssignedSuperstep(VertexType node) const { return vectorSchedule_.AssignedSuperstep(node); }

    inline VWorkwT<GraphT> GetStepMaxWork(unsigned step) const { return workDatastructures_.StepMaxWork(step); }

    inline VWorkwT<GraphT> GetStepSecondMaxWork(unsigned step) const { return workDatastructures_.StepSecondMaxWork(step); }

    inline std::vector<unsigned> &GetStepMaxWorkProcessorCount() { return workDatastructures_.stepMaxWorkProcessorCount_; }

    inline VWorkwT<GraphT> GetStepProcessorWork(unsigned step, unsigned proc) const {
        return workDatastructures_.StepProcWork(step, proc);
    }

    inline PreMoveWorkData<VWorkwT<GraphT>> GetPreMoveWorkData(KlMove move) {
        return workDatastructures_.GetPreMoveWorkData(move);
    }

    inline VWorkwT<GraphT> GetMaxWorkWeight() { return workDatastructures_.maxWorkWeight_; }

    inline VWorkwT<GraphT> GetTotalWorkWeight() { return workDatastructures_.totalWorkWeight_; }

    inline void SetCost(CostT cost) { cost_ = cost; }

    constexpr static bool useMemoryConstraint_ = isLocalSearchMemoryConstraintV<MemoryConstraintT>;

    MemoryConstraintT memoryConstraint_;

    KlActiveScheduleWorkDatastructures<GraphT> workDatastructures_;

    inline VWorkwT<GraphT> GetStepTotalWork(unsigned step) const {
        VWorkwT<GraphT> totalWork = 0;
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            totalWork += workDatastructures_.StepProcWork(step, proc);
        }
        return totalWork;
    }

    void ApplyMove(KlMove move, ThreadDataT &threadData) {
        vectorSchedule_.SetAssignedProcessor(move.node_, move.toProc_);
        vectorSchedule_.SetAssignedSuperstep(move.node_, move.toStep_);

        setSchedule_.GetProcessorStepVertices()[move.fromStep_][move.fromProc_].erase(move.node_);
        setSchedule_.GetProcessorStepVertices()[move.toStep_][move.toProc_].insert(move.node_);

        UpdateViolations(move.node_, threadData);
        threadData.appliedMoves_.push_back(move);

        workDatastructures_.ApplyMove(move, instance_->GetComputationalDag().VertexWorkWeight(move.node_));
        if constexpr (useMemoryConstraint_) {
            memoryConstraint_.ApplyMove(move.node_, move.fromProc_, move.fromStep_, move.toProc_, move.toStep_);
        }
    }

    template <typename CommDatastructuresT>
    void RevertToBestSchedule(unsigned startMove,
                              unsigned insertStep,
                              bool stepWasRemoved,
                              CommDatastructuresT &commDatastructures,
                              ThreadDataT &threadData,
                              unsigned startStep,
                              unsigned &endStep) {
        const unsigned bound = std::max(startMove, threadData.bestScheduleIdx_);
        RevertMoves(bound, commDatastructures, threadData, startStep, endStep);

        // Re-insert the removed step when the best schedule predates the
        // removal.  bestScheduleIdx_ <= startMove (== localSearchStartStep_)
        // means the best was saved during scatter or is the initial state,
        // both of which are pre-removal.  bestScheduleIdx_ > startMove means
        // the inner loop (or resolve) found a better state post-removal, so
        // the step stays removed.
        //
        // stepWasRemoved guards against the case where startMove == 0 because
        // the removed step was already empty (zero scatter moves).  Without
        // the flag, startMove == 0 would look identical to "no step removed."
        if (stepWasRemoved && startMove >= threadData.bestScheduleIdx_) {
            SwapEmptyStepBwd(++endStep, insertStep);
        }

        RevertMoves(threadData.bestScheduleIdx_, commDatastructures, threadData, startStep, endStep);

#ifdef KL_DEBUG
        if (not threadData.feasible_) {
            std::cout << "Reverted to best schedule with cost: " << threadData.bestCost_ << " and "
                      << vectorSchedule_.NumberOfSupersteps() << " supersteps" << std::endl;
        }
#endif

        threadData.appliedMoves_.clear();
        threadData.bestScheduleIdx_ = 0;
        threadData.currentViolations_.clear();
        threadData.feasible_ = true;
        threadData.cost_ = threadData.bestCost_;
    }

    template <typename CommDatastructuresT>
    void RevertScheduleToBound(const size_t bound,
                               const CostT newCost,
                               const bool isFeasible,
                               CommDatastructuresT &commDatastructures,
                               ThreadDataT &threadData,
                               unsigned startStep,
                               unsigned endStep) {
        RevertMoves(bound, commDatastructures, threadData, startStep, endStep);

        threadData.currentViolations_.clear();
        threadData.feasible_ = isFeasible;
        threadData.cost_ = newCost;
    }

    void ComputeViolations(ThreadDataT &threadData);
    void UpdateViolationsAfterStepRemoval(unsigned removedStep, ThreadDataT &threadData);
    void ComputeWorkMemoryDatastructures(unsigned startStep, unsigned endStep);
    void WriteSchedule(BspSchedule<GraphT> &schedule);
    inline void Initialize(const IBspSchedule<GraphT> &schedule);
    inline void Clear();
    void RemoveEmptyStep(unsigned step);
    void InsertEmptyStep(unsigned step);
    void SwapEmptyStepFwd(const unsigned step, const unsigned toStep);
    void SwapEmptyStepBwd(const unsigned toStep, const unsigned emptyStep);
    void SwapSteps(const unsigned step1, const unsigned step2);

  private:
    template <typename CommDatastructuresT>
    void RevertMoves(const size_t bound,
                     CommDatastructuresT &commDatastructures,
                     ThreadDataT &threadData,
                     unsigned startStep,
                     unsigned endStep) {
        while (threadData.appliedMoves_.size() > bound) {
            const auto move = threadData.appliedMoves_.back().ReverseMove();
            threadData.appliedMoves_.pop_back();

            vectorSchedule_.SetAssignedProcessor(move.node_, move.toProc_);
            vectorSchedule_.SetAssignedSuperstep(move.node_, move.toStep_);

            setSchedule_.GetProcessorStepVertices()[move.fromStep_][move.fromProc_].erase(move.node_);
            setSchedule_.GetProcessorStepVertices()[move.toStep_][move.toProc_].insert(move.node_);
            workDatastructures_.ApplyMove(move, instance_->GetComputationalDag().VertexWorkWeight(move.node_));
            commDatastructures.UpdateDatastructureAfterMove(move, startStep, endStep);
            if constexpr (useMemoryConstraint_) {
                memoryConstraint_.ApplyMove(move.node_, move.fromProc_, move.fromStep_, move.toProc_, move.toStep_);
            }
        }
    }

    void UpdateViolations(VertexType node, ThreadDataT &threadData) {
        threadData.newViolations_.clear();
        threadData.resolvedViolations_.clear();

        const unsigned nodeStep = vectorSchedule_.AssignedSuperstep(node);
        const unsigned nodeProc = vectorSchedule_.AssignedProcessor(node);

        for (const auto &edge : OutEdges(node, instance_->GetComputationalDag())) {
            const auto &child = Target(edge, instance_->GetComputationalDag());

            if (threadData.currentViolations_.find(edge) == threadData.currentViolations_.end()) {
                const unsigned differentProcessors = (nodeProc == vectorSchedule_.AssignedProcessor(child)) ? 0 : staleness_;
                if (nodeStep + differentProcessors > vectorSchedule_.AssignedSuperstep(child)) {
                    threadData.currentViolations_.insert(edge);
                    threadData.newViolations_[child] = edge;
                }
            } else {
                const unsigned differentProcessors = (nodeProc == vectorSchedule_.AssignedProcessor(child)) ? 0 : staleness_;
                if (nodeStep + differentProcessors <= vectorSchedule_.AssignedSuperstep(child)) {
                    threadData.currentViolations_.erase(edge);
                    threadData.resolvedViolations_.insert(edge);
                }
            }
        }

        for (const auto &edge : InEdges(node, instance_->GetComputationalDag())) {
            const auto &parent = Source(edge, instance_->GetComputationalDag());

            if (threadData.currentViolations_.find(edge) == threadData.currentViolations_.end()) {
                const unsigned differentProcessors = (nodeProc == vectorSchedule_.AssignedProcessor(parent)) ? 0 : staleness_;
                if (vectorSchedule_.AssignedSuperstep(parent) + differentProcessors > nodeStep) {
                    threadData.currentViolations_.insert(edge);
                    threadData.newViolations_[parent] = edge;
                }
            } else {
                const unsigned differentProcessors = (nodeProc == vectorSchedule_.AssignedProcessor(parent)) ? 0 : staleness_;
                if (vectorSchedule_.AssignedSuperstep(parent) + differentProcessors <= nodeStep) {
                    threadData.currentViolations_.erase(edge);
                    threadData.resolvedViolations_.insert(edge);
                }
            }
        }

#ifdef KL_DEBUG

        if (threadData.newViolations_.size() > 0) {
            std::cout << "New violations: " << std::endl;
            for (const auto &edge : threadData.newViolations_) {
                std::cout << "Edge: " << Source(edge.second, instance_->GetComputationalDag()) << " -> "
                          << Target(edge.second, instance_->GetComputationalDag()) << std::endl;
            }
        }

        if (threadData.resolvedViolations_.size() > 0) {
            std::cout << "Resolved violations: " << std::endl;
            for (const auto &edge : threadData.resolvedViolations_) {
                std::cout << "Edge: " << Source(edge, instance_->GetComputationalDag()) << " -> "
                          << Target(edge, instance_->GetComputationalDag()) << std::endl;
            }
        }

#endif

        if (threadData.currentViolations_.size() > 0) {
            threadData.feasible_ = false;
        } else {
            threadData.feasible_ = true;
        }
    }
};

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::Clear() {
    workDatastructures_.Clear();
    vectorSchedule_.Clear();
    setSchedule_.Clear();
    if constexpr (useMemoryConstraint_) {
        memoryConstraint_.Clear();
    }
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::ComputeViolations(ThreadDataT &threadData) {
    threadData.currentViolations_.clear();
    threadData.feasible_ = true;

    for (const auto &edge : Edges(instance_->GetComputationalDag())) {
        const auto &sourceV = Source(edge, instance_->GetComputationalDag());
        const auto &targetV = Target(edge, instance_->GetComputationalDag());

        const unsigned sourceProc = AssignedProcessor(sourceV);
        const unsigned targetProc = AssignedProcessor(targetV);
        const unsigned sourceStep = AssignedSuperstep(sourceV);
        const unsigned targetStep = AssignedSuperstep(targetV);

        const unsigned differentProcessors = (sourceProc == targetProc) ? 0 : staleness_;

        if (sourceStep + differentProcessors > targetStep) {
            threadData.currentViolations_.insert(edge);
            threadData.feasible_ = false;
        }
    }
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::UpdateViolationsAfterStepRemoval(unsigned removedStep,
                                                                                          ThreadDataT &threadData) {
    // After SwapEmptyStepFwd(removedStep, ...) bubbles the empty step forward,
    // all nodes formerly at steps removedStep+1.. shift down by 1.  Nodes at
    // steps 0..removedStep-1 are untouched.
    //
    // Only cross-processor edges that cross from the unshifted region into the
    // shifted region lose 1 from their gap.  For staleness <= 2 the only
    // boundary that can drop below the staleness threshold is:
    //
    //   parent at step removedStep-1  -->  child now at step removedStep
    //                                      (formerly at removedStep+1)
    //   gap went from 2 to 1 -- violates staleness == 2
    //
    // We iterate only the nodes in the affected superstep (removedStep after
    // the swap) and check their incoming edges from the step above.
    //
    // TODO: for staleness > 2, parents at steps removedStep-2 down to
    //       removedStep-(staleness-1) could also create new violations.
    //       Extend the check to cover those additional steps.

    if (staleness_ <= 1 || removedStep == 0) {
        return;
    }

    const auto &dag = instance_->GetComputationalDag();

    for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
        for (const auto &node : setSchedule_.GetProcessorStepVertices()[removedStep][proc]) {
            for (const auto &edge : InEdges(node, dag)) {
                const auto &parent = Source(edge, dag);
                const unsigned parentStep = vectorSchedule_.AssignedSuperstep(parent);
                const unsigned parentProc = vectorSchedule_.AssignedProcessor(parent);

                if (parentProc != proc && parentStep + staleness_ > removedStep) {
                    threadData.currentViolations_.insert(edge);
                }
            }
        }
    }

    threadData.feasible_ = threadData.currentViolations_.empty();
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::Initialize(const IBspSchedule<GraphT> &schedule) {
    instance_ = &schedule.GetInstance();
    vectorSchedule_ = VectorSchedule(schedule);
    setSchedule_ = SetSchedule(schedule);
    workDatastructures_.Initialize(setSchedule_, *instance_, NumSteps());

    staleness_ = schedule.GetStaleness();

    cost_ = 0;
    feasible_ = true;

    if constexpr (useMemoryConstraint_) {
        memoryConstraint_.Initialize(setSchedule_, vectorSchedule_);
    }

    ComputeWorkMemoryDatastructures(0, NumSteps() - 1);
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::ComputeWorkMemoryDatastructures(unsigned startStep, unsigned endStep) {
    if constexpr (useMemoryConstraint_) {
        memoryConstraint_.ComputeMemoryDatastructure(startStep, endStep);
    }
    workDatastructures_.ComputeWorkDatastructures(startStep, endStep);
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::WriteSchedule(BspSchedule<GraphT> &schedule) {
    for (const auto v : instance_->Vertices()) {
        schedule.SetAssignedProcessor(v, vectorSchedule_.AssignedProcessor(v));
        schedule.SetAssignedSuperstep(v, vectorSchedule_.AssignedSuperstep(v));
    }
    schedule.UpdateNumberOfSupersteps();
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::RemoveEmptyStep(unsigned step) {
    for (unsigned i = step; i < NumSteps() - 1; i++) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            for (const auto node : setSchedule_.GetProcessorStepVertices()[i + 1][proc]) {
                vectorSchedule_.SetAssignedSuperstep(node, i);
            }
        }
        std::swap(setSchedule_.GetProcessorStepVertices()[i], setSchedule_.GetProcessorStepVertices()[i + 1]);
        workDatastructures_.SwapSteps(i, i + 1);
        if constexpr (useMemoryConstraint_) {
            memoryConstraint_.SwapSteps(i, i + 1);
        }
    }
    vectorSchedule_.numberOfSupersteps_--;
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::SwapEmptyStepFwd(const unsigned step, const unsigned toStep) {
    for (unsigned i = step; i < toStep; i++) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            for (const auto node : setSchedule_.GetProcessorStepVertices()[i + 1][proc]) {
                vectorSchedule_.SetAssignedSuperstep(node, i);
            }
        }
        std::swap(setSchedule_.GetProcessorStepVertices()[i], setSchedule_.GetProcessorStepVertices()[i + 1]);
        workDatastructures_.SwapSteps(i, i + 1);
        if constexpr (useMemoryConstraint_) {
            memoryConstraint_.SwapSteps(i, i + 1);
        }
    }
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::InsertEmptyStep(unsigned step) {
    unsigned i = vectorSchedule_.IncrementNumberOfSupersteps();

    for (; i > step; i--) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            for (const auto node : setSchedule_.GetProcessorStepVertices()[i - 1][proc]) {
                vectorSchedule_.SetAssignedSuperstep(node, i);
            }
        }
        std::swap(setSchedule_.GetProcessorStepVertices()[i], setSchedule_.GetProcessorStepVertices()[i - 1]);
        workDatastructures_.SwapSteps(i - 1, i);
        if constexpr (useMemoryConstraint_) {
            memoryConstraint_.SwapSteps(i - 1, i);
        }
    }
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::SwapEmptyStepBwd(const unsigned toStep, const unsigned emptyStep) {
    unsigned i = toStep;

    for (; i > emptyStep; i--) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            for (const auto node : setSchedule_.GetProcessorStepVertices()[i - 1][proc]) {
                vectorSchedule_.SetAssignedSuperstep(node, i);
            }
        }
        std::swap(setSchedule_.GetProcessorStepVertices()[i], setSchedule_.GetProcessorStepVertices()[i - 1]);
        workDatastructures_.SwapSteps(i - 1, i);
        if constexpr (useMemoryConstraint_) {
            memoryConstraint_.SwapSteps(i - 1, i);
        }
    }
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::SwapSteps(const unsigned step1, const unsigned step2) {
    if (step1 == step2) {
        return;
    }

    for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
        for (const auto node : setSchedule_.GetProcessorStepVertices()[step1][proc]) {
            vectorSchedule_.SetAssignedSuperstep(node, step2);
        }
        for (const auto node : setSchedule_.GetProcessorStepVertices()[step2][proc]) {
            vectorSchedule_.SetAssignedSuperstep(node, step1);
        }
    }
    std::swap(setSchedule_.GetProcessorStepVertices()[step1], setSchedule_.GetProcessorStepVertices()[step2]);
    workDatastructures_.SwapSteps(step1, step2);
    if constexpr (useMemoryConstraint_) {
        memoryConstraint_.SwapSteps(step1, step2);
    }
}

}    // namespace osp
