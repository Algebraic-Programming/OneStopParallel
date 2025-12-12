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
    using work_weight_t = v_workw_t<Graph_t>;

    const BspInstance<GraphT> *instance_;
    const SetSchedule<GraphT> *setSchedule_;

    struct WeightProc {
        work_weight_t work_;
        unsigned proc_;

        WeightProc() : work(0), proc_(0) {}

        WeightProc(work_weight_t work, unsigned proc) : work(_work), proc_(proc) {}

        bool operator<(WeightProc const &rhs) const { return (work > rhs.work) or (work == rhs.work and proc < rhs.proc); }
    };

    std::vector<std::vector<WeightProc>> stepProcessorWork_;
    std::vector<std::vector<unsigned>> stepProcessorPosition_;
    std::vector<unsigned> stepMaxWorkProcessorCount_;
    work_weight_t maxWorkWeight_;
    work_weight_t totalWorkWeight_;

    inline work_weight_t StepMaxWork(unsigned step) const { return stepProcessorWork_[step][0].work; }

    inline work_weight_t StepSecondMaxWork(unsigned step) const {
        return stepProcessorWork_[step][stepMaxWorkProcessorCount_[step]].work;
    }

    inline work_weight_t StepProcWork(unsigned step, unsigned proc) const {
        return stepProcessorWork_[step][stepProcessorPosition_[step][proc]].work;
    }

    inline work_weight_t &StepProcWork(unsigned step, unsigned proc) {
        return stepProcessorWork_[step][stepProcessorPosition_[step][proc]].work;
    }

    template <typename CostT, typename VertexIdxT>
    inline pre_move_work_data<work_weight_t> GetPreMoveWorkData(KlMoveStruct<CostT, VertexIdxT> move) {
        return pre_move_work_data<work_weight_t>(step_max_work(move.from_step),
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
        const work_weight_t maxWorkTo = stepProcessorWork_[step][0].work;

        for (const auto &wp : stepProcessorWork_[step]) {
            stepProcessorPosition_[step][wp.proc] = pos++;

            if (wp.work == max_work_to && pos < instance_->NumberOfProcessors()) {
                stepMaxWorkProcessorCount_[step] = pos;
            }
        }
    }

    template <typename CostT, typename VertexIdxT>
    void ApplyMove(KlMoveStruct<CostT, VertexIdxT> move, work_weight_t workWeight) {
        if (workWeight == 0) {
            return;
        }

        if (move.to_step != move.from_step) {
            step_proc_work(move.to_step, move.to_proc) += work_weight;
            step_proc_work(move.from_step, move.from_proc) -= work_weight;

            ArrangeSuperstepData(move.to_step);
            ArrangeSuperstepData(move.from_step);

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
            step_proc_work(move.to_step, move.to_proc) += work_weight;
            step_proc_work(move.from_step, move.from_proc) -= work_weight;
            ArrangeSuperstepData(move.to_step);
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
                    const work_weight_t vertexWorkWeight = instance_->getComputationalDag().VertexWorkWeight(node);
                    total_work_weight += vertex_work_weight;
                    max_work_weight = std::max(vertex_work_weight, max_work_weight);
                    stepProcessorWork_[step][proc].work += vertex_work_weight;
                }

                if (stepProcessorWork_[step][proc].work > max_work) {
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
    using VertexType = vertex_idx_t<Graph_t>;
    using EdgeType = edge_desc_t<Graph_t>;

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
    using VertexType = vertex_idx_t<Graph_t>;
    using EdgeType = edge_desc_t<Graph_t>;
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

    inline unsigned AssignedSuperstep(VertexType node) const { return vectorSchedule_.assignedSuperstep(node); }

    inline v_workw_t<Graph_t> GetStepMaxWork(unsigned step) const { return workDatastructures_.step_max_work(step); }

    inline v_workw_t<Graph_t> GetStepSecondMaxWork(unsigned step) const { return workDatastructures_.step_second_max_work(step); }

    inline std::vector<unsigned> &GetStepMaxWorkProcessorCount() { return workDatastructures_.step_max_work_processor_count; }

    inline v_workw_t<Graph_t> GetStepProcessorWork(unsigned step, unsigned proc) const {
        return workDatastructures_.step_proc_work(step, proc);
    }

    inline pre_move_work_data<v_workw_t<Graph_t>> GetPreMoveWorkData(kl_move move) {
        return workDatastructures_.get_pre_move_work_data(move);
    }

    inline v_workw_t<Graph_t> GetMaxWorkWeight() { return workDatastructures_.max_work_weight; }

    inline v_workw_t<Graph_t> GetTotalWorkWeight() { return workDatastructures_.total_work_weight; }

    inline void SetCost(CostT cost) { cost_ = cost; }

    constexpr static bool useMemoryConstraint_ = is_local_search_memory_constraint_v<MemoryConstraintT>;

    MemoryConstraintT memoryConstraint_;

    KlActiveScheduleWorkDatastructures<GraphT> workDatastructures_;

    inline v_workw_t<Graph_t> GetStepTotalWork(unsigned step) const {
        v_workw_t<Graph_t> totalWork = 0;
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            total_work += get_step_processor_work(step, proc);
        }
        return total_work;
    }

    void ApplyMove(kl_move move, ThreadDataT &threadData) {
        vectorSchedule_.setAssignedProcessor(move.node, move.to_proc);
        vectorSchedule_.setAssignedSuperstep(move.node, move.to_step);

        setSchedule_.step_processor_vertices[move.from_step][move.from_proc].erase(move.node);
        setSchedule_.step_processor_vertices[move.to_step][move.to_proc].insert(move.node);

        update_violations(move.node, thread_data);
        threadData.applied_moves.push_back(move);

        workDatastructures_.apply_move(move, instance_->getComputationalDag().VertexWorkWeight(move.node));
        if constexpr (useMemoryConstraint_) {
            memoryConstraint_.apply_move(move.node, move.from_proc, move.from_step, move.to_proc, move.to_step);
        }
    }

    template <typename CommDatastructuresT>
    void RevertToBestSchedule(unsigned startMove,
                              unsigned insertStep,
                              CommDatastructuresT &commDatastructures,
                              ThreadDataT &threadData,
                              unsigned startStep,
                              unsigned &endStep) {
        const unsigned bound = std::max(startMove, threadData.best_schedule_idx);
        revert_moves(bound, commDatastructures, threadData, startStep, endStep);

        if (startMove > threadData.best_schedule_idx) {
            SwapEmptyStepBwd(++endStep, insertStep);
        }

        revert_moves(threadData.best_schedule_idx, commDatastructures, threadData, startStep, endStep);

#ifdef KL_DEBUG
        if (not thread_data.feasible) {
            std::cout << "Reverted to best schedule with cost: " << thread_data.best_cost << " and "
                      << vector_schedule.number_of_supersteps << " supersteps" << std::endl;
        }
#endif

        threadData.applied_moves.clear();
        threadData.best_schedule_idx = 0;
        threadData.current_violations.clear();
        threadData.feasible = true;
        threadData.cost = threadData.best_cost;
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
        while (threadData.applied_moves.size() > bound) {
            const auto move = threadData.applied_moves.back().reverse_move();
            threadData.applied_moves.pop_back();

            vectorSchedule_.setAssignedProcessor(move.node, move.to_proc);
            vectorSchedule_.setAssignedSuperstep(move.node, move.to_step);

            setSchedule_.step_processor_vertices[move.from_step][move.from_proc].erase(move.node);
            setSchedule_.step_processor_vertices[move.to_step][move.to_proc].insert(move.node);
            workDatastructures_.apply_move(move, instance_->getComputationalDag().VertexWorkWeight(move.node));
            commDatastructures.update_datastructure_after_move(move, startStep, endStep);
            if constexpr (useMemoryConstraint_) {
                memoryConstraint_.apply_move(move.node, move.from_proc, move.from_step, move.to_proc, move.to_step);
            }
        }
    }

    void UpdateViolations(VertexType node, ThreadDataT &threadData) {
        threadData.new_violations.clear();
        threadData.resolved_violations.clear();

        const unsigned nodeStep = vectorSchedule_.assignedSuperstep(node);
        const unsigned nodeProc = vectorSchedule_.assignedProcessor(node);

        for (const auto &edge : OutEdges(node, instance->getComputationalDag())) {
            const auto &child = Traget(edge, instance->getComputationalDag());

            if (thread_data.current_violations.find(edge) == thread_data.current_violations.end()) {
                if ((node_step > vector_schedule.assignedSuperstep(child))
                    || (node_step == vector_schedule.assignedSuperstep(child)
                        && node_proc != vector_schedule.assignedProcessor(child))) {
                    thread_data.current_violations.insert(edge);
                    thread_data.new_violations[child] = edge;
                }
            } else {
                if ((node_step < vector_schedule.assignedSuperstep(child))
                    || (node_step == vector_schedule.assignedSuperstep(child)
                        && node_proc == vector_schedule.assignedProcessor(child))) {
                    thread_data.current_violations.erase(edge);
                    thread_data.resolved_violations.insert(edge);
                }
            }
        }

        for (const auto &edge : InEdges(node, instance->getComputationalDag())) {
            const auto &parent = Source(edge, instance->getComputationalDag());

            if (thread_data.current_violations.find(edge) == thread_data.current_violations.end()) {
                if ((node_step < vector_schedule.assignedSuperstep(parent))
                    || (node_step == vector_schedule.assignedSuperstep(parent)
                        && node_proc != vector_schedule.assignedProcessor(parent))) {
                    thread_data.current_violations.insert(edge);
                    thread_data.new_violations[parent] = edge;
                }
            } else {
                if ((node_step > vector_schedule.assignedSuperstep(parent))
                    || (node_step == vector_schedule.assignedSuperstep(parent)
                        && node_proc == vector_schedule.assignedProcessor(parent))) {
                    thread_data.current_violations.erase(edge);
                    thread_data.resolved_violations.insert(edge);
                }
            }
        }

#ifdef KL_DEBUG

        if (thread_data.new_violations.size() > 0) {
            std::cout << "New violations: " << std::endl;
            for (const auto &edge : thread_data.new_violations) {
                std::cout << "Edge: " << Source(edge.second, instance->getComputationalDag()) << " -> "
                          << Traget(edge.second, instance->getComputationalDag()) << std::endl;
            }
        }

        if (thread_data.resolved_violations.size() > 0) {
            std::cout << "Resolved violations: " << std::endl;
            for (const auto &edge : thread_data.resolved_violations) {
                std::cout << "Edge: " << Source(edge, instance->getComputationalDag()) << " -> "
                          << Traget(edge, instance->getComputationalDag()) << std::endl;
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

    for (const auto &edge : Edges(instance_->getComputationalDag())) {
        const auto &sourceV = Source(edge, instance_->getComputationalDag());
        const auto &targetV = Traget(edge, instance_->getComputationalDag());

        const unsigned sourceProc = assigned_processor(source_v);
        const unsigned targetProc = assigned_processor(target_v);
        const unsigned sourceStep = assigned_superstep(source_v);
        const unsigned targetStep = assigned_superstep(target_v);

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
    workDatastructures_.initialize(setSchedule_, *instance_, NumSteps());

    cost_ = 0;
    feasible_ = true;

    if constexpr (useMemoryConstraint_) {
        memoryConstraint_.initialize(setSchedule_, vectorSchedule_);
    }

    ComputeWorkMemoryDatastructures(0, NumSteps() - 1);
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::ComputeWorkMemoryDatastructures(unsigned startStep, unsigned endStep) {
    if constexpr (useMemoryConstraint_) {
        memoryConstraint_.compute_memory_datastructure(startStep, endStep);
    }
    workDatastructures_.compute_work_datastructures(startStep, endStep);
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::WriteSchedule(BspSchedule<GraphT> &schedule) {
    for (const auto v : instance_->vertices()) {
        schedule.setAssignedProcessor(v, vectorSchedule_.assignedProcessor(v));
        schedule.setAssignedSuperstep(v, vectorSchedule_.assignedSuperstep(v));
    }
    schedule.updateNumberOfSupersteps();
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::RemoveEmptyStep(unsigned step) {
    for (unsigned i = step; i < NumSteps() - 1; i++) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            for (const auto node : setSchedule_.step_processor_vertices[i + 1][proc]) {
                vectorSchedule_.setAssignedSuperstep(node, i);
            }
        }
        std::swap(setSchedule_.step_processor_vertices[i], setSchedule_.step_processor_vertices[i + 1]);
        workDatastructures_.swap_steps(i, i + 1);
        if constexpr (useMemoryConstraint_) {
            memoryConstraint_.swap_steps(i, i + 1);
        }
    }
    vectorSchedule_.number_of_supersteps--;
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::SwapEmptyStepFwd(const unsigned step, const unsigned toStep) {
    for (unsigned i = step; i < toStep; i++) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            for (const auto node : setSchedule_.step_processor_vertices[i + 1][proc]) {
                vectorSchedule_.setAssignedSuperstep(node, i);
            }
        }
        std::swap(setSchedule_.step_processor_vertices[i], setSchedule_.step_processor_vertices[i + 1]);
        workDatastructures_.swap_steps(i, i + 1);
        if constexpr (useMemoryConstraint_) {
            memoryConstraint_.swap_steps(i, i + 1);
        }
    }
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::InsertEmptyStep(unsigned step) {
    unsigned i = vectorSchedule_.number_of_supersteps++;

    for (; i > step; i--) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            for (const auto node : setSchedule_.step_processor_vertices[i - 1][proc]) {
                vectorSchedule_.setAssignedSuperstep(node, i);
            }
        }
        std::swap(setSchedule_.step_processor_vertices[i], setSchedule_.step_processor_vertices[i - 1]);
        workDatastructures_.swap_steps(i - 1, i);
        if constexpr (useMemoryConstraint_) {
            memoryConstraint_.swap_steps(i - 1, i);
        }
    }
}

template <typename GraphT, typename CostT, typename MemoryConstraintT>
void KlActiveSchedule<GraphT, CostT, MemoryConstraintT>::SwapEmptyStepBwd(const unsigned toStep, const unsigned emptyStep) {
    unsigned i = toStep;

    for (; i > emptyStep; i--) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            for (const auto node : setSchedule_.step_processor_vertices[i - 1][proc]) {
                vectorSchedule_.setAssignedSuperstep(node, i);
            }
        }
        std::swap(setSchedule_.step_processor_vertices[i], setSchedule_.step_processor_vertices[i - 1]);
        workDatastructures_.swap_steps(i - 1, i);
        if constexpr (useMemoryConstraint_) {
            memoryConstraint_.swap_steps(i - 1, i);
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
            vectorSchedule_.setAssignedSuperstep(node, step2);
        }
        for (const auto node : setSchedule_.step_processor_vertices[step2][proc]) {
            vectorSchedule_.setAssignedSuperstep(node, step1);
        }
    }
    std::swap(setSchedule_.step_processor_vertices[step1], setSchedule_.step_processor_vertices[step2]);
    workDatastructures_.swap_steps(step1, step2);
    if constexpr (useMemoryConstraint_) {
        memoryConstraint_.swap_steps(step1, step2);
    }
}

}    // namespace osp
