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

    inline WorkWeightT StepMaxWork(unsigned step) const { return stepProcessorWork_[step][0].work; }

    inline WorkWeightT StepSecondMaxWork(unsigned step) const {
        return stepProcessorWork_[step][stepMaxWorkProcessorCount_[step]].work;
    }

    inline WorkWeightT StepProcWork(unsigned step, unsigned proc) const {
        return stepProcessorWork_[step][stepProcessorPosition_[step][proc]].work;
    }

    inline WorkWeightT &StepProcWork(unsigned step, unsigned proc) {
        return stepProcessorWork_[step][stepProcessorPosition_[step][proc]].work;
    }

    template <typename CostT, typename VertexIdxT>
    inline PreMoveWorkData GetPreMoveWorkData(KlMoveStruct<CostT, VertexIdxT> move) {
        return PreMoveWorkData(step_max_work(move.from_step),
                               step_second_max_work(move.from_step),
                               step_max_work_processor_count[move.from_step],
                               step_max_work(move.to_step),
                               step_second_max_work(move.to_step),
                               step_max_work_processor_count[move.to_step]);
    }

    inline void Initialize(const SetSchedule<GraphT> &sched, const BspInstance<GraphT> &inst, unsigned numSteps) {
        instance_ = &inst;
        setSchedule_ = &sched;
        max_work_weight = 0;
        total_work_weight = 0;
        stepProcessorWork_
            = std::vector<std::vector<WeightProc>>(numSteps, std::vector<WeightProc>(instance_->NumberOfProcessors()));
        stepProcessorPosition_
            = std::vector<std::vector<unsigned>>(numSteps, std::vector<unsigned>(instance_->NumberOfProcessors(), 0));
        stepMaxWorkProcessorCount_ = std::vector<unsigned>(numSteps, 0);
    }

    inline void Clear() {
        stepProcessorWork_.clear();
        stepProcessorPosition_.clear();
        stepMaxWorkProcessorCount_.clear();
    }

    inline void ArrangeSuperstepData(const unsigned step) {
        std::sort(stepProcessorWork_[step].begin(), stepProcessorWork_[step].end());
        unsigned pos = 0;
        const WorkWeightT maxWorkTo = stepProcessorWork_[step][0].work;

        for (const auto &wp : stepProcessorWork_[step]) {
            stepProcessorPosition_[step][wp.proc] = pos++;

            if (wp.work == maxWorkTo && pos < instance_->NumberOfProcessors()) {
                stepMaxWorkProcessorCount_[step] = pos;
            }
        }
    }

    template <typename CostT, typename VertexIdxT>
    void ApplyMove(KlMoveStruct<CostT, VertexIdxT> move, WorkWeightT workWeight) {
        if (workWeight == 0) {
            return;
        }

        if (move.toStep != move.fromStep) {
            StepProcWork(move.toStep, move.toProc) += workWeight;
            StepProcWork(move.fromStep, move.fromProc) -= workWeight;

            ArrangeSuperstepData(move.toStep);
            ArrangeSuperstepData(move.fromStep);

            // const work_weight_t prev_max_work_to = step_max_work(move.to_step);
            // const work_weight_t new_weight_to = step_proc_work(move.to_step, move.to_proc) += work_weight;

            // if (prev_max_work_to < new_weight_to) {
            //     step_max_work_processor_count[move.to_step] = 1;
            // } else if (prev_max_work_to == new_weight_to) {
            //     step_max_work_processor_count[move.to_step]++;
            // }

            // unsigned to_proc_pos = step_processor_position[move.to_step][move.to_proc];

            // while (to_proc_pos > 0 && step_processor_work_[move.to_step][to_proc_pos - 1].work < new_weight_to) {
            //     std::swap(step_processor_work_[move.to_step][to_proc_pos], step_processor_work_[move.to_step][to_proc_pos -
            //     1]); std::swap(step_processor_position[move.to_step][step_processor_work_[move.to_step][to_proc_pos].proc],
            //     step_processor_position[move.to_step][step_processor_work_[move.to_step][to_proc_pos - 1].proc]);
            //     to_proc_pos--;
            // }

            // const work_weight_t prev_max_work_from = step_max_work(move.from_step);
            // const work_weight_t prev_weight_from = step_proc_work(move.from_step, move.from_proc);
            // const work_weight_t new_weight_from = step_proc_work(move.from_step, move.from_proc) -= work_weight;

            // unsigned from_proc_pos = step_processor_position[move.from_step][move.from_proc];

            // while (from_proc_pos < instance->NumberOfProcessors() - 1 && step_processor_work_[move.from_step][from_proc_pos +
            // 1].work > new_weight_from) {
            //     std::swap(step_processor_work_[move.from_step][from_proc_pos],
            //     step_processor_work_[move.from_step][from_proc_pos + 1]);
            //     std::swap(step_processor_position[move.from_step][step_processor_work_[move.from_step][from_proc_pos].proc],
            //     step_processor_position[move.from_step][step_processor_work_[move.from_step][from_proc_pos + 1].proc]);
            //     from_proc_pos++;
            // }

            // if (prev_max_work_from == prev_weight_from) {
            //     step_max_work_processor_count[move.from_step]--;
            //     if (step_max_work_processor_count[move.from_step] == 0) {
            //         step_max_work_processor_count[move.from_step] = from_proc_pos;
            //     }
            // }

        } else {
            StepProcWork(move.toStep, move.toProc) += workWeight;
            StepProcWork(move.fromStep, move.fromProc) -= workWeight;
            ArrangeSuperstepData(move.toStep);
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
            work_weight_t maxWork = 0;

            for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
                stepProcessorWork_[step][proc].work = 0;
                stepProcessorWork_[step][proc].proc = proc;

                for (const auto &node : setSchedule_->step_processor_vertices[step][proc]) {
                    const work_weight_t vertexWorkWeight = instance_->GetComputationalDag().VertexWorkWeight(node);
                    totalWorkWeight_ += vertexWorkWeight;
                    maxWorkWeight_ = std::max(vertexWorkWeight, maxWorkWeight_);
                    stepProcessorWork_[step][proc].work += vertexWorkWeight;
                }

                if (stepProcessorWork_[step][proc].work > maxWork) {
                    maxWork = stepProcessorWork_[step][proc].work;
                    stepMaxWorkProcessorCount_[step] = 1;
                } else if (stepProcessorWork_[step][proc].work == max_work
                           && stepMaxWorkProcessorCount_[step] < (instance_->NumberOfProcessors() - 1)) {
                    stepMaxWorkProcessorCount_[step]++;
                }
            }

            std::sort(stepProcessorWork_[step].begin(), stepProcessorWork_[step].end());
            unsigned pos = 0;
            for (const auto &wp : stepProcessorWork_[step]) {
                stepProcessorPosition_[step][wp.proc] = pos++;
            }
        }
    }
};

template <typename GraphT, typename CostT>
struct ThreadLocalActiveScheduleData {
    using VertexType = VertexIdxT<GraphT>;
    using EdgeType = EdgeDescT<GraphT>;

    using kl_move = kl_move_struct<cost_t, VertexType>;

    std::unordered_set<EdgeType> currentViolations_;
    std::vector<kl_move> appliedMoves_;

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
            best_schedule_idx = static_cast<unsigned>(applied_moves.size());
        }
    }
};

template <typename GraphT, typename CostT, typename MemoryConstraintT>
class KlActiveSchedule {
  private:
    using VertexType = VertexIdxT<GraphT>;
    using EdgeType = EdgeDescT<GraphT>;
    using kl_move = kl_move_struct<cost_t, VertexType>;
    using ThreadDataT = ThreadLocalActiveScheduleData<GraphT, CostT>;

    const BspInstance<GraphT> *instance_;

    VectorSchedule<GraphT> vectorSchedule_;
    SetSchedule<GraphT> setSchedule_;

    CostT cost_ = 0;
    bool feasible_ = true;

  public:
    virtual ~KlActiveSchedule() = default;

    inline const BspInstance<GraphT> &GetInstance() const { return *instance_; }

    inline const VectorSchedule<GraphT> &GetVectorSchedule() const { return vectorSchedule_; }

    inline VectorSchedule<GraphT> &GetVectorSchedule() { return vectorSchedule_; }

    inline const SetSchedule<GraphT> &GetSetSchedule() const { return setSchedule_; }

    inline CostT GetCost() { return cost_; }

    inline bool IsFeasible() { return feasible_; }

    inline unsigned NumSteps() const { return vectorSchedule_.NumberOfSupersteps(); }

    inline unsigned AssignedProcessor(VertexType node) const { return vectorSchedule_.assignedProcessor(node); }

    inline unsigned AssignedSuperstep(VertexType node) const { return vectorSchedule_.AssignedSuperstep(node); }

    inline VWorkwT<GraphT> GetStepMaxWork(unsigned step) const { return workDatastructures_.step_max_work(step); }

    inline VWorkwT<GraphT> GetStepSecondMaxWork(unsigned step) const { return workDatastructures_.StepSecondMaxWork(step); }

    inline std::vector<unsigned> &GetStepMaxWorkProcessorCount() { return workDatastructures_.step_max_work_processor_count; }

    inline VWorkwT<GraphT> GetStepProcessorWork(unsigned step, unsigned proc) const {
        return workDatastructures_.StepProcWork(step, proc);
    }

    inline pre_move_work_data<VWorkwT<GraphT>> GetPreMoveWorkData(kl_move move) {
        return workDatastructures_.GetPreMoveWorkData(move);
    }

    inline VWorkwT<GraphT> GetMaxWorkWeight() { return workDatastructures_.max_work_weight; }

    inline VWorkwT<GraphT> GetTotalWorkWeight() { return workDatastructures_.total_work_weight; }

    inline void SetCost(CostT cost) { cost_ = cost; }

    constexpr static bool useMemoryConstraint_ = is_local_search_memory_constraint_v<MemoryConstraintT>;

    MemoryConstraintT memoryConstraint_;

    KlActiveScheduleWorkDatastructures<GraphT> workDatastructures_;

    inline VWorkwT<GraphT> GetStepTotalWork(unsigned step) const {
        VWorkwT<GraphT> totalWork = 0;
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            totalWork += StepProcWork(step, proc);
        }
        return totalWork;
    }

    void ApplyMove(kl_move move, ThreadDataT &threadData) {
        vectorSchedule_.SetAssignedProcessor(move.node, move.to_proc);
        vectorSchedule_.SetAssignedSuperstep(move.node, move.to_step);

        setSchedule_.stepProcessorVertices_[move.fromStep][move.fromProc].erase(move.node);
        setSchedule_.stepProcessorVertices_[move.toStep][move.toProc].insert(move.node);

        UpdateViolations(move.node, threadData);
        threadData.appliedMoves_.push_back(move);

        workDatastructures_.ApplyMove(move, instance_->GetComputationalDag().VertexWorkWeight(move.node));
        if constexpr (useMemoryConstraint_) {
            memoryConstraint_.ApplyMove(move.node, move.fromProc, move.fromStep, move.toProc, move.toStep);
        }
    }

    template <typename CommDatastructuresT>
    void RevertToBestSchedule(unsigned startMove,
                              unsigned insertStep,
                              CommDatastructuresT &commDatastructures,
                              ThreadDataT &threadData,
                              unsigned startStep,
                              unsigned &endStep) {
        const unsigned bound = std::max(startMove, threadData.bestScheduleIdx_);
        RevertMoves(bound, commDatastructures, threadData, startStep, endStep);

        if (startMove > threadData.bestScheduleIdx_) {
            SwapEmptyStepBwd(++endStep, insertStep);
        }

        RevertMoves(threadData.bestScheduleIdx_, commDatastructures, threadData, startStep, endStep);

#ifdef KL_DEBUG
        if (not threadData.feasible) {
            std::cout << "Reverted to best schedule with cost: " << threadData.bestCost << " and "
                      << vectorSchedule.NumberOfSupersteps() << " supersteps" << std::endl;
        }
#endif

        threadData.appliedMoves_.clear();
        threadData.bestScheduleIdx_ = 0;
        threadData.currentViolations_.clear();
        threadData.feasible = true;
        threadData.cost = threadData.bestCost;
    }

    template <typename CommDatastructuresT>
    void RevertScheduleToBound(const size_t bound,
                               const CostT newCost,
                               const bool isFeasible,
                               CommDatastructuresT &commDatastructures,
                               ThreadDataT &threadData,
                               unsigned startStep,
                               unsigned endStep) {
        revert_moves(bound, commDatastructures, threadData, startStep, endStep);

        threadData.current_violations.clear();
        threadData.feasible = isFeasible;
        threadData.cost = newCost;
    }

    void ComputeViolations(ThreadDataT &threadData);
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
            const auto move = threadData.appliedMoves_.back().reverseMove();
            threadData.appliedMoves_.pop_back();

            vectorSchedule_.SetAssignedProcessor(move.node, move.toProc);
            vectorSchedule_.SetAssignedSuperstep(move.node, move.toStep);

            setSchedule_.stepProcessorVertices_[move.fromStep][move.fromProc].erase(move.node);
            setSchedule_.stepProcessorVertices_[move.toStep][move.toProc].insert(move.node);
            workDatastructures_.ApplyMove(move, instance_->GetComputationalDag().VertexWorkWeight(move.node));
            commDatastructures.UpdateDatastructureAfterMove(move, startStep, endStep);
            if constexpr (useMemoryConstraint_) {
                memoryConstraint_.ApplyMove(move.node, move.fromProc, move.fromStep, move.toProc, move.toStep);
            }
        }
    }

    void UpdateViolations(VertexType node, ThreadDataT &threadData) {
        threadData.new_violations.clear();
        threadData.resolved_violations.clear();

        const unsigned nodeStep = vectorSchedule_.AssignedSuperstep(node);
        const unsigned nodeProc = vectorSchedule_.AssignedProcessor(node);

        for (const auto &edge : OutEdges(node, instance->GetComputationalDag())) {
            const auto &child = Traget(edge, instance->GetComputationalDag());

            if (threadData.currentViolations_.find(edge) == threadData.currentViolations_.end()) {
                if ((nodeStep > vectorSchedule_.AssignedSuperstep(child))
                    || (nodeStep == vectorSchedule_.AssignedSuperstep(child)
                        && nodeProc != vectorSchedule_.AssignedProcessor(child))) {
                    threadData.currentViolations_.insert(edge);
                    threadData.newViolations_[child] = edge;
                }
            } else {
                if ((nodeStep < vectorSchedule_.AssignedSuperstep(child))
                    || (nodeStep == vectorSchedule_.AssignedSuperstep(child)
                        && nodeProc == vectorSchedule_.AssignedProcessor(child))) {
                    threadData.currentViolations_.erase(edge);
                    threadData.resolvedViolations_.insert(edge);
                }
            }
        }

        for (const auto &edge : InEdges(node, instance->GetComputationalDag())) {
            const auto &parent = Source(edge, instance->GetComputationalDag());

            if (threadData.currentViolations_.find(edge) == threadData.currentViolations_.end()) {
                if ((nodeStep < vectorSchedule_.AssignedSuperstep(parent))
                    || (nodeStep == vectorSchedule_.AssignedSuperstep(parent)
                        && nodeProc != vectorSchedule_.AssignedProcessor(parent))) {
                    threadData.currentViolations_.insert(edge);
                    threadData.newViolations_[parent] = edge;
                }
            } else {
                if ((nodeStep > vectorSchedule_.AssignedSuperstep(parent))
                    || (nodeStep == vectorSchedule_.AssignedSuperstep(parent)
                        && nodeProc == vectorSchedule_.AssignedProcessor(parent))) {
                    threadData.currentViolations_.erase(edge);
                    threadData.resolvedViolations_.insert(edge);
                }
            }
        }

#ifdef KL_DEBUG

        if (threadData.newViolations_.size() > 0) {
            std::cout << "New violations: " << std::endl;
            for (const auto &edge : threadData.newViolations_) {
                std::cout << "Edge: " << Source(edge.second, instance->GetComputationalDag()) << " -> "
                          << Traget(edge.second, instance->GetComputationalDag()) << std::endl;
            }
        }

        if (thread_data.resolved_violations.size() > 0) {
            std::cout << "Resolved violations: " << std::endl;
            for (const auto &edge : thread_data.resolved_violations) {
                std::cout << "Edge: " << Source(edge, instance->GetComputationalDag()) << " -> "
                          << Traget(edge, instance->GetComputationalDag()) << std::endl;
            }
        }

#endif

        if (threadData.current_violations.size() > 0) {
            threadData.feasible = false;
        } else {
            threadData.feasible = true;
        }
    }
};

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::Clear() {
    workDatastructures_.clear();
    vectorSchedule_.clear();
    setSchedule_.clear();
    if constexpr (useMemoryConstraint_) {
        memoryConstraint_.clear();
    }
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::ComputeViolations(ThreadDataT &threadData) {
    threadData.current_violations.clear();
    threadData.feasible = true;

    for (const auto &edge : Edges(instance_->GetComputationalDag())) {
        const auto &sourceV = Source(edge, instance_->GetComputationalDag());
        const auto &targetV = Traget(edge, instance_->GetComputationalDag());

        const unsigned sourceProc = AssignedProcessor(sourceV);
        const unsigned targetProc = AssignedProcessor(targetV);
        const unsigned sourceStep = AssignedSuperstep(sourceV);
        const unsigned targetStep = AssignedSuperstep(targetV);

        if (sourceStep > targetStep || (sourceStep == targetStep && sourceProc != targetProc)) {
            threadData.current_violations.insert(edge);
            threadData.feasible = false;
        }
    }
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::Initialize(const IBspSchedule<GraphT> &schedule) {
    instance_ = &schedule.GetInstance();
    vectorSchedule_ = VectorSchedule(schedule);
    setSchedule_ = SetSchedule(schedule);
    workDatastructures_.Initialize(setSchedule_, *instance_, NumSteps());

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
    for (const auto v : instance_->vertices()) {
        schedule.SetAssignedProcessor(v, vectorSchedule_.AssignedProcessor(v));
        schedule.SetAssignedSuperstep(v, vectorSchedule_.AssignedSuperstep(v));
    }
    schedule.UpdateNumberOfSupersteps();
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::RemoveEmptyStep(unsigned step) {
    for (unsigned i = step; i < NumSteps() - 1; i++) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            for (const auto node : setSchedule_.step_processor_vertices[i + 1][proc]) {
                vectorSchedule_.SetAssignedSuperstep(node, i);
            }
        }
        std::swap(setSchedule_.step_processor_vertices[i], setSchedule_.step_processor_vertices[i + 1]);
        workDatastructures_.SwapSteps(i, i + 1);
        if constexpr (useMemoryConstraint_) {
            memoryConstraint_.SwapSteps(i, i + 1);
        }
    }
    vectorSchedule_.NumberOfSupersteps--;
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::SwapEmptyStepFwd(const unsigned step, const unsigned toStep) {
    for (unsigned i = step; i < toStep; i++) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            for (const auto node : setSchedule_.step_processor_vertices[i + 1][proc]) {
                vectorSchedule_.SetAssignedSuperstep(node, i);
            }
        }
        std::swap(setSchedule_.step_processor_vertices[i], setSchedule_.step_processor_vertices[i + 1]);
        workDatastructures_.SwapSteps(i, i + 1);
        if constexpr (useMemoryConstraint_) {
            memoryConstraint_.SwapSteps(i, i + 1);
        }
    }
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::InsertEmptyStep(unsigned step) {
    unsigned i = vectorSchedule_.NumberOfSupersteps++;

    for (; i > step; i--) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            for (const auto node : setSchedule_.step_processor_vertices[i - 1][proc]) {
                vectorSchedule_.SetAssignedSuperstep(node, i);
            }
        }
        std::swap(setSchedule_.step_processor_vertices[i], setSchedule_.step_processor_vertices[i - 1]);
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
            for (const auto node : setSchedule_.step_processor_vertices[i - 1][proc]) {
                vectorSchedule_.SetAssignedSuperstep(node, i);
            }
        }
        std::swap(setSchedule_.step_processor_vertices[i], setSchedule_.step_processor_vertices[i - 1]);
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
        for (const auto node : setSchedule_.step_processor_vertices[step1][proc]) {
            vectorSchedule_.SetAssignedSuperstep(node, step2);
        }
        for (const auto node : setSchedule_.step_processor_vertices[step2][proc]) {
            vectorSchedule_.SetAssignedSuperstep(node, step1);
        }
    }
    std::swap(setSchedule_.step_processor_vertices[step1], setSchedule_.step_processor_vertices[step2]);
    workDatastructures_.SwapSteps(step1, step2);
    if constexpr (useMemoryConstraint_) {
        memoryConstraint_.SwapSteps(step1, step2);
    }
}

}    // namespace osp
