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
    VertexIdxT node;
    CostT gain;

    unsigned fromProc;
    unsigned fromStep;

    unsigned toProc;
    unsigned toStep;

    KlMoveStruct() : node(0), gain(0), fromProc(0), fromStep(0), toProc(0), toStep(0) {}

    KlMoveStruct(VertexIdxT node_, CostT gain_, unsigned fromProc_, unsigned fromStep_, unsigned toProc_, unsigned toStep_)
        : node(node_), gain(gain_), fromProc(fromProc_), fromStep(fromStep_), toProc(toProc_), toStep(toStep_) {}

    bool operator<(KlMoveStruct<CostT, VertexIdxT> const &rhs) const {
        return (gain < rhs.gain) or (gain == rhs.gain and node > rhs.node);
    }

    bool operator>(KlMoveStruct<CostT, VertexIdxT> const &rhs) const {
        return (gain > rhs.gain) or (gain >= rhs.gain and node < rhs.node);
    }

    KlMoveStruct<CostT, VertexIdxT> ReverseMove() const { return KlMoveStruct(node, -gain, toProc, toStep, fromProc, fromStep); }
};

template <typename WorkWeightT>
struct PreMoveWorkData {
    WorkWeightT fromStepMaxWork;
    WorkWeightT fromStepSecondMaxWork;
    unsigned fromStepMaxWorkProcessorCount;

    WorkWeightT toStepMaxWork;
    WorkWeightT toStepSecondMaxWork;
    unsigned toStepMaxWorkProcessorCount;

    PreMoveWorkData() {}

    PreMoveWorkData(WorkWeightT fromStepMaxWork_,
                    WorkWeightT fromStepSecondMaxWork_,
                    unsigned fromStepMaxWorkProcessorCount_,
                    WorkWeightT toStepMaxWork_,
                    WorkWeightT toStepSecondMaxWork_,
                    unsigned toStepMaxWorkProcessorCount_)
        : fromStepMaxWork(fromStepMaxWork_),
          fromStepSecondMaxWork(fromStepSecondMaxWork_),
          fromStepMaxWorkProcessorCount(fromStepMaxWorkProcessorCount_),
          toStepMaxWork(toStepMaxWork_),
          toStepSecondMaxWork(toStepSecondMaxWork_),
          toStepMaxWorkProcessorCount(toStepMaxWorkProcessorCount_) {}
};

template <typename GraphT>
struct KlActiveScheduleWorkDatastructures {
    using WorkWeightT = VWorkwT<GraphT>;

    const BspInstance<GraphT> *instance;
    const SetSchedule<GraphT> *setSchedule;

    struct WeightProc {
        WorkWeightT work;
        unsigned proc;

        WeightProc() : work(0), proc(0) {}

        WeightProc(WorkWeightT work_, unsigned proc_) : work(work_), proc(proc_) {}

        bool operator<(WeightProc const &rhs) const { return (work > rhs.work) or (work == rhs.work and proc < rhs.proc); }
    };

    std::vector<std::vector<WeightProc>> stepProcessorWork;
    std::vector<std::vector<unsigned>> stepProcessorPosition;
    std::vector<unsigned> stepMaxWorkProcessorCount;
    WorkWeightT maxWorkWeight;
    WorkWeightT totalWorkWeight;

    inline WorkWeightT StepMaxWork(unsigned step) const { return stepProcessorWork[step][0].work; }

    inline WorkWeightT StepSecondMaxWork(unsigned step) const {
        return stepProcessorWork[step][stepMaxWorkProcessorCount[step]].work;
    }

    inline WorkWeightT StepProcWork(unsigned step, unsigned proc) const {
        return stepProcessorWork[step][stepProcessorPosition[step][proc]].work;
    }

    inline WorkWeightT &StepProcWork(unsigned step, unsigned proc) {
        return stepProcessorWork[step][stepProcessorPosition[step][proc]].work;
    }

    template <typename CostT, typename VertexIdxT>
    inline PreMoveWorkData<WorkWeightT> GetPreMoveWorkData(KlMoveStruct<CostT, VertexIdxT> move) {
        return PreMoveWorkData<WorkWeightT>(StepMaxWork(move.fromStep),
                                            StepSecondMaxWork(move.fromStep),
                                            stepMaxWorkProcessorCount[move.fromStep],
                                            StepMaxWork(move.toStep),
                                            StepSecondMaxWork(move.toStep),
                                            stepMaxWorkProcessorCount[move.toStep]);
    }

    inline void Initialize(const SetSchedule<GraphT> &sched, const BspInstance<GraphT> &inst, unsigned numSteps) {
        instance = &inst;
        setSchedule = &sched;
        maxWorkWeight = 0;
        totalWorkWeight = 0;
        stepProcessorWork = std::vector<std::vector<WeightProc>>(numSteps, std::vector<WeightProc>(instance->NumberOfProcessors()));
        stepProcessorPosition
            = std::vector<std::vector<unsigned>>(numSteps, std::vector<unsigned>(instance->NumberOfProcessors(), 0));
        stepMaxWorkProcessorCount = std::vector<unsigned>(numSteps, 0);
    }

    inline void Clear() {
        stepProcessorWork.clear();
        stepProcessorPosition.clear();
        stepMaxWorkProcessorCount.clear();
    }

    inline void ArrangeSuperstepData(const unsigned step) {
        std::sort(stepProcessorWork[step].begin(), stepProcessorWork[step].end());
        unsigned pos = 0;
        const WorkWeightT maxWorkTo = stepProcessorWork[step][0].work;

        for (const auto &wp : stepProcessorWork[step]) {
            stepProcessorPosition[step][wp.proc] = pos++;

            if (wp.work == maxWorkTo && pos < instance->NumberOfProcessors()) {
                stepMaxWorkProcessorCount[step] = pos;
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

            // while (from_proc_pos < instance->numberOfProcessors() - 1 && step_processor_work_[move.from_step][from_proc_pos +
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
        std::swap(stepProcessorWork[step1], stepProcessorWork[step2]);
        std::swap(stepProcessorPosition[step1], stepProcessorPosition[step2]);
        std::swap(stepMaxWorkProcessorCount[step1], stepMaxWorkProcessorCount[step2]);
    }

    void OverrideNextSuperstep(unsigned step) {
        const unsigned nextStep = step + 1;
        for (unsigned i = 0; i < instance->numberOfProcessors(); i++) {
            stepProcessorWork[nextStep][i] = stepProcessorWork[step][i];
            stepProcessorPosition[nextStep][i] = stepProcessorPosition[step][i];
        }
        stepMaxWorkProcessorCount[nextStep] = stepMaxWorkProcessorCount[step];
    }

    void ResetSuperstep(unsigned step) {
        for (unsigned i = 0; i < instance->numberOfProcessors(); i++) {
            stepProcessorWork[step][i] = {0, i};
            stepProcessorPosition[step][i] = i;
        }
        stepMaxWorkProcessorCount[step] = instance->numberOfProcessors() - 1;
    }

    void ComputeWorkDatastructures(unsigned startStep, unsigned endStep) {
        for (unsigned step = startStep; step <= endStep; step++) {
            stepMaxWorkProcessorCount[step] = 0;
            WorkWeightT maxWork = 0;

            for (unsigned proc = 0; proc < instance->NumberOfProcessors(); proc++) {
                stepProcessorWork[step][proc].work = 0;
                stepProcessorWork[step][proc].proc = proc;

                for (const auto &node : setSchedule->stepProcessorVertices[step][proc]) {
                    const WorkWeightT vertexWorkWeight = instance->GetComputationalDag().VertexWorkWeight(node);
                    totalWorkWeight += vertexWorkWeight;
                    maxWorkWeight = std::max(vertexWorkWeight, maxWorkWeight);
                    stepProcessorWork[step][proc].work += vertexWorkWeight;
                }

                if (stepProcessorWork[step][proc].work > maxWork) {
                    maxWork = stepProcessorWork[step][proc].work;
                    stepMaxWorkProcessorCount[step] = 1;
                } else if (stepProcessorWork[step][proc].work == maxWork
                           && stepMaxWorkProcessorCount[step] < (instance->NumberOfProcessors() - 1)) {
                    stepMaxWorkProcessorCount[step]++;
                }
            }

            std::sort(stepProcessorWork[step].begin(), stepProcessorWork[step].end());
            unsigned pos = 0;
            for (const auto &wp : stepProcessorWork[step]) {
                stepProcessorPosition[step][wp.proc] = pos++;
            }
        }
    }
};

template <typename GraphT, typename CostT>
struct ThreadLocalActiveScheduleData {
    using VertexType = VertexIdxT<GraphT>;
    using EdgeType = EdgeDescT<GraphT>;

    using KlMove = KlMoveStruct<CostT, VertexType>;

    std::unordered_set<EdgeType> currentViolations;
    std::vector<KlMove> appliedMoves;

    CostT cost = 0;
    CostT initialCost = 0;
    bool feasible = true;

    CostT bestCost = 0;
    unsigned bestScheduleIdx = 0;

    std::unordered_map<VertexType, EdgeType> newViolations;
    std::unordered_set<EdgeType> resolvedViolations;

    inline void InitializeCost(CostT cost_) {
        initialCost = cost_;
        cost = cost_;
        bestCost = cost_;
        feasible = true;
    }

    inline void UpdateCost(CostT changeInCost) {
        cost += changeInCost;

        if (cost <= bestCost && feasible) {
            bestCost = cost;
            bestScheduleIdx = static_cast<unsigned>(appliedMoves.size());
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

  public:
    virtual ~KlActiveSchedule() = default;

    inline const BspInstance<GraphT> &GetInstance() const { return *instance_; }

    inline const VectorSchedule<GraphT> &GetVectorSchedule() const { return vectorSchedule_; }

    inline VectorSchedule<GraphT> &GetVectorSchedule() { return vectorSchedule_; }

    inline const SetSchedule<GraphT> &GetSetSchedule() const { return setSchedule_; }

    inline CostT GetCost() { return cost_; }

    inline bool IsFeasible() { return feasible_; }

    inline unsigned NumSteps() const { return vectorSchedule_.NumberOfSupersteps(); }

    inline unsigned AssignedProcessor(VertexType node) const { return vectorSchedule_.AssignedProcessor(node); }

    inline unsigned AssignedSuperstep(VertexType node) const { return vectorSchedule_.AssignedSuperstep(node); }

    inline VWorkwT<GraphT> GetStepMaxWork(unsigned step) const { return workDatastructures.StepMaxWork(step); }

    inline VWorkwT<GraphT> GetStepSecondMaxWork(unsigned step) const { return workDatastructures.StepSecondMaxWork(step); }

    inline std::vector<unsigned> &GetStepMaxWorkProcessorCount() { return workDatastructures.stepMaxWorkProcessorCount; }

    inline VWorkwT<GraphT> GetStepProcessorWork(unsigned step, unsigned proc) const {
        return workDatastructures.StepProcWork(step, proc);
    }

    inline PreMoveWorkData<VWorkwT<GraphT>> GetPreMoveWorkData(KlMove move) {
        return workDatastructures.GetPreMoveWorkData(move);
    }

    inline VWorkwT<GraphT> GetMaxWorkWeight() { return workDatastructures.maxWorkWeight; }

    inline VWorkwT<GraphT> GetTotalWorkWeight() { return workDatastructures.total_work_weight; }

    inline void SetCost(CostT cost) { cost_ = cost; }

    constexpr static bool useMemoryConstraint = isLocalSearchMemoryConstraintV<MemoryConstraintT>;

    MemoryConstraintT memoryConstraint;

    KlActiveScheduleWorkDatastructures<GraphT> workDatastructures;

    inline VWorkwT<GraphT> GetStepTotalWork(unsigned step) const {
        VWorkwT<GraphT> totalWork = 0;
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            totalWork += GetStepProcessorWork(step, proc);
        }
        return totalWork;
    }

    void ApplyMove(KlMove move, ThreadDataT &threadData) {
        vectorSchedule_.SetAssignedProcessor(move.node, move.toProc);
        vectorSchedule_.SetAssignedSuperstep(move.node, move.toStep);

        setSchedule_.stepProcessorVertices[move.fromStep][move.fromProc].erase(move.node);
        setSchedule_.stepProcessorVertices[move.toStep][move.toProc].insert(move.node);

        UpdateViolations(move.node, threadData);
        threadData.appliedMoves.push_back(move);

        workDatastructures.ApplyMove(move, instance_->GetComputationalDag().VertexWorkWeight(move.node));
        if constexpr (useMemoryConstraint) {
            memoryConstraint.ApplyMove(move.node, move.fromProc, move.fromStep, move.toProc, move.toStep);
        }
    }

    template <typename CommDatastructuresT>
    void RevertToBestSchedule(unsigned startMove,
                              unsigned insertStep,
                              CommDatastructuresT &commDatastructures,
                              ThreadDataT &threadData,
                              unsigned startStep,
                              unsigned &endStep) {
        const unsigned bound = std::max(startMove, threadData.bestScheduleIdx);
        RevertMoves(bound, commDatastructures, threadData, startStep, endStep);

        if (startMove > threadData.bestScheduleIdx) {
            SwapEmptyStepBwd(++endStep, insertStep);
        }

        RevertMoves(threadData.bestScheduleIdx, commDatastructures, threadData, startStep, endStep);

#ifdef KL_DEBUG
        if (not thread_data.feasible) {
            std::cout << "Reverted to best schedule with cost: " << thread_data.best_cost << " and "
                      << vector_schedule.number_of_supersteps << " supersteps" << std::endl;
        }
#endif

        threadData.appliedMoves.clear();
        threadData.bestScheduleIdx = 0;
        threadData.currentViolations.clear();
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
        RevertMoves(bound, commDatastructures, threadData, startStep, endStep);

        threadData.currentViolations.clear();
        threadData.feasible = isFeasible;
        threadData.cost = newCost;
    }

    void ComputeViolations(ThreadDataT &threadData);
    void ComputeWorkMemoryDatastructures(unsigned start_step, unsigned end_step);
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
        while (threadData.appliedMoves.size() > bound) {
            const auto move = threadData.appliedMoves.back().ReverseMove();
            threadData.appliedMoves.pop_back();

            vectorSchedule_.SetAssignedProcessor(move.node, move.toProc);
            vectorSchedule_.SetAssignedSuperstep(move.node, move.toStep);

            setSchedule_.stepProcessorVertices[move.fromStep][move.fromProc].erase(move.node);
            setSchedule_.stepProcessorVertices[move.toStep][move.toProc].insert(move.node);
            workDatastructures.ApplyMove(move, instance_->GetComputationalDag().VertexWorkWeight(move.node));
            commDatastructures.UpdateDatastructureAfterMove(move, startStep, endStep);
            if constexpr (useMemoryConstraint) {
                memoryConstraint.ApplyMove(move.node, move.fromProc, move.fromStep, move.toProc, move.toStep);
            }
        }
    }

    void UpdateViolations(VertexType node, ThreadDataT &threadData) {
        threadData.newViolations.clear();
        threadData.resolvedViolations.clear();

        const unsigned nodeStep = vectorSchedule_.AssignedSuperstep(node);
        const unsigned nodeProc = vectorSchedule_.AssignedProcessor(node);

        for (const auto &edge : OutEdges(node, instance_->GetComputationalDag())) {
            const auto &child = Target(edge, instance_->GetComputationalDag());

            if (threadData.currentViolations.find(edge) == threadData.currentViolations.end()) {
                if ((nodeStep > vectorSchedule_.AssignedSuperstep(child))
                    || (nodeStep == vectorSchedule_.AssignedSuperstep(child)
                        && nodeProc != vectorSchedule_.AssignedProcessor(child))) {
                    threadData.currentViolations.insert(edge);
                    threadData.newViolations[child] = edge;
                }
            } else {
                if ((nodeStep < vectorSchedule_.AssignedSuperstep(child))
                    || (nodeStep == vectorSchedule_.AssignedSuperstep(child)
                        && nodeProc == vectorSchedule_.AssignedProcessor(child))) {
                    threadData.currentViolations.erase(edge);
                    threadData.resolvedViolations.insert(edge);
                }
            }
        }

        for (const auto &edge : InEdges(node, instance_->GetComputationalDag())) {
            const auto &parent = Source(edge, instance_->GetComputationalDag());

            if (threadData.currentViolations.find(edge) == threadData.currentViolations.end()) {
                if ((nodeStep < vectorSchedule_.AssignedSuperstep(parent))
                    || (nodeStep == vectorSchedule_.AssignedSuperstep(parent)
                        && nodeProc != vectorSchedule_.AssignedProcessor(parent))) {
                    threadData.currentViolations.insert(edge);
                    threadData.newViolations[parent] = edge;
                }
            } else {
                if ((nodeStep > vectorSchedule_.AssignedSuperstep(parent))
                    || (nodeStep == vectorSchedule_.AssignedSuperstep(parent)
                        && nodeProc == vectorSchedule_.AssignedProcessor(parent))) {
                    threadData.currentViolations.erase(edge);
                    threadData.resolvedViolations.insert(edge);
                }
            }
        }

#ifdef KL_DEBUG

        if (thread_data.new_violations.size() > 0) {
            std::cout << "New violations: " << std::endl;
            for (const auto &edge : thread_data.new_violations) {
                std::cout << "Edge: " << source(edge.second, instance->getComputationalDag()) << " -> "
                          << target(edge.second, instance->getComputationalDag()) << std::endl;
            }
        }

        if (thread_data.resolved_violations.size() > 0) {
            std::cout << "Resolved violations: " << std::endl;
            for (const auto &edge : thread_data.resolved_violations) {
                std::cout << "Edge: " << source(edge, instance->getComputationalDag()) << " -> "
                          << target(edge, instance->getComputationalDag()) << std::endl;
            }
        }

#endif

        if (threadData.currentViolations.size() > 0) {
            threadData.feasible = false;
        } else {
            threadData.feasible = true;
        }
    }
};

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::Clear() {
    workDatastructures.Clear();
    vectorSchedule_.Clear();
    setSchedule_.Clear();
    if constexpr (useMemoryConstraint) {
        memoryConstraint.Clear();
    }
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::ComputeViolations(ThreadDataT &threadData) {
    threadData.currentViolations.clear();
    threadData.feasible = true;

    for (const auto &edge : Edges(instance_->GetComputationalDag())) {
        const auto &sourceV = Source(edge, instance_->GetComputationalDag());
        const auto &targetV = Target(edge, instance_->GetComputationalDag());

        const unsigned sourceProc = AssignedProcessor(sourceV);
        const unsigned targetProc = AssignedProcessor(targetV);
        const unsigned sourceStep = AssignedSuperstep(sourceV);
        const unsigned targetStep = AssignedSuperstep(targetV);

        if (sourceStep > targetStep || (sourceStep == targetStep && sourceProc != targetProc)) {
            threadData.currentViolations.insert(edge);
            threadData.feasible = false;
        }
    }
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::Initialize(const IBspSchedule<GraphT> &schedule) {
    instance_ = &schedule.GetInstance();
    vectorSchedule_ = VectorSchedule(schedule);
    setSchedule_ = SetSchedule(schedule);
    workDatastructures.Initialize(setSchedule_, *instance_, NumSteps());

    cost_ = 0;
    feasible_ = true;

    if constexpr (useMemoryConstraint) {
        memoryConstraint.Initialize(setSchedule_, vectorSchedule_);
    }

    ComputeWorkMemoryDatastructures(0, NumSteps() - 1);
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::ComputeWorkMemoryDatastructures(unsigned startStep, unsigned endStep) {
    if constexpr (useMemoryConstraint) {
        memoryConstraint.ComputeMemoryDatastructure(startStep, endStep);
    }
    workDatastructures.ComputeWorkDatastructures(startStep, endStep);
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
            for (const auto node : setSchedule_.stepProcessorVertices[i + 1][proc]) {
                vectorSchedule_.SetAssignedSuperstep(node, i);
            }
        }
        std::swap(setSchedule_.stepProcessorVertices[i], setSchedule_.stepProcessorVertices[i + 1]);
        workDatastructures.SwapSteps(i, i + 1);
        if constexpr (useMemoryConstraint) {
            memoryConstraint.swap_steps(i, i + 1);
        }
    }
    vectorSchedule_.numberOfSupersteps--;
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::SwapEmptyStepFwd(const unsigned step, const unsigned toStep) {
    for (unsigned i = step; i < toStep; i++) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            for (const auto node : setSchedule_.stepProcessorVertices[i + 1][proc]) {
                vectorSchedule_.SetAssignedSuperstep(node, i);
            }
        }
        std::swap(setSchedule_.stepProcessorVertices[i], setSchedule_.stepProcessorVertices[i + 1]);
        workDatastructures.SwapSteps(i, i + 1);
        if constexpr (useMemoryConstraint) {
            memoryConstraint.SwapSteps(i, i + 1);
        }
    }
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::InsertEmptyStep(unsigned step) {
    unsigned i = vectorSchedule_.number_of_supersteps++;

    for (; i > step; i--) {
        for (unsigned proc = 0; proc < instance_->numberOfProcessors(); proc++) {
            for (const auto node : setSchedule_.step_processor_vertices[i - 1][proc]) {
                vectorSchedule_.setAssignedSuperstep(node, i);
            }
        }
        std::swap(setSchedule_.step_processor_vertices[i], setSchedule_.step_processor_vertices[i - 1]);
        workDatastructures.swap_steps(i - 1, i);
        if constexpr (useMemoryConstraint) {
            memoryConstraint.swap_steps(i - 1, i);
        }
    }
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::SwapEmptyStepBwd(const unsigned toStep, const unsigned emptyStep) {
    unsigned i = toStep;

    for (; i > emptyStep; i--) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            for (const auto node : setSchedule_.stepProcessorVertices[i - 1][proc]) {
                vectorSchedule_.SetAssignedSuperstep(node, i);
            }
        }
        std::swap(setSchedule_.stepProcessorVertices[i], setSchedule_.stepProcessorVertices[i - 1]);
        workDatastructures.SwapSteps(i - 1, i);
        if constexpr (useMemoryConstraint) {
            memoryConstraint.SwapSteps(i - 1, i);
        }
    }
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::SwapSteps(const unsigned step1, const unsigned step2) {
    if (step1 == step2) {
        return;
    }

    for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
        for (const auto node : setSchedule_.stepProcessorVertices[step1][proc]) {
            vectorSchedule_.SetAssignedSuperstep(node, step2);
        }
        for (const auto node : setSchedule_.stepProcessorVertices[step2][proc]) {
            vectorSchedule_.SetAssignedSuperstep(node, step1);
        }
    }
    std::swap(setSchedule_.stepProcessorVertices[step1], setSchedule_.stepProcessorVertices[step2]);
    workDatastructures.SwapSteps(step1, step2);
    if constexpr (useMemoryConstraint) {
        memoryConstraint.SwapSteps(step1, step2);
    }
}

}    // namespace osp
