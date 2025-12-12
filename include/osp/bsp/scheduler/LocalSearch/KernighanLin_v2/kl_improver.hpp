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

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "kl_active_schedule.hpp"
#include "kl_util.hpp"
#include "osp/auxiliary/datastructures/heaps/PairingHeap.hpp"
#include "osp/auxiliary/misc.hpp"
#include "osp/bsp/model/util/CompatibleProcessorRange.hpp"
#include "osp/bsp/scheduler/ImprovementScheduler.hpp"
#include "osp/bsp/scheduler/LocalSearch/LocalSearchMemoryConstraintModules.hpp"
#include "osp/graph_algorithms/directed_graph_edge_desc_util.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"

namespace osp {

struct KlParameter {
    double timeQuality_ = 0.8;
    double superstepRemoveStrength_ = 0.5;
    unsigned numParallelLoops_ = 4;

    unsigned maxInnerIterationsReset_ = 500;
    unsigned maxNoImprovementIterations_ = 50;

    constexpr static unsigned abortScatterNodesViolationThreshold_ = 500;
    constexpr static unsigned initialViolationThreshold_ = 250;

    unsigned maxNoVioaltionsRemovedBacktrackReset_;
    unsigned removeStepEpocs_;
    unsigned nodeMaxStepSelectionEpochs_;
    unsigned maxNoVioaltionsRemovedBacktrackForRemoveStepReset_;
    unsigned maxOuterIterations_;
    unsigned tryRemoveStepAfterNumOuterIterations_;
    unsigned minInnerIterReset_;

    unsigned threadMinRange_ = 8;
    unsigned threadRangeGap_ = 0;
};

template <typename VertexType>
struct KlUpdateInfo {
    VertexType node_ = 0;

    bool fullUpdate_ = false;
    bool updateFromStep_ = false;
    bool updateToStep_ = false;
    bool updateEntireToStep_ = false;
    bool updateEntireFromStep_ = false;

    KlUpdateInfo() = default;

    KlUpdateInfo(VertexType n) : node_(n), fullUpdate_(false), updateEntireToStep_(false), updateEntireFromStep_(false) {}

    KlUpdateInfo(VertexType n, bool full)
        : node_(n), fullUpdate_(full), updateEntireToStep_(false), updateEntireFromStep_(false) {}
};

template <typename GraphT,
          typename CommCostFunctionT,
          typename MemoryConstraintT = NoLocalSearchMemoryConstraint,
          unsigned windowSize = 1,
          typename CostT = double>
class KlImprover : public ImprovementScheduler<GraphT> {
    static_assert(is_directed_graph_edge_desc_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    static_assert(has_hashable_edge_desc_v<Graph_t>, "Graph_t must satisfy the has_hashable_edge_desc concept");
    static_assert(IsComputationalDagV<Graph_t>, "Graph_t must satisfy the computational_dag concept");

  protected:
    constexpr static unsigned windowRange_ = 2 * windowSize + 1;
    constexpr static bool enableQuickMoves_ = true;
    constexpr static bool enablePreresolvingViolations_ = true;
    constexpr static double epsilon_ = 1e-9;

    using memw_t = v_memw_t<Graph_t>;
    using commw_t = v_commw_t<Graph_t>;
    using work_weight_t = v_workw_t<Graph_t>;
    using VertexType = vertex_idx_t<Graph_t>;
    using EdgeType = edge_desc_t<Graph_t>;

    using kl_move = kl_move_struct<cost_t, VertexType>;
    using heap_datastructure = MaxPairingHeap<VertexType, kl_move>;
    using ActiveScheduleT = KlActiveSchedule<GraphT, CostT, MemoryConstraintT>;
    using NodeSelectionContainerT = AdaptiveAffinityTable<GraphT, CostT, ActiveScheduleT, windowSize>;
    using kl_gain_update_info = kl_update_info<VertexType>;

    struct ThreadSearchContext {
        unsigned threadId_ = 0;
        unsigned startStep_ = 0;
        unsigned endStep_ = 0;
        unsigned originalEndStep_ = 0;

        vector_vertex_lock_manger<VertexType> lockManager_;
        heap_datastructure maxGainHeap_;
        NodeSelectionContainerT affinityTable_;
        std::vector<std::vector<CostT>> localAffinityTable_;
        RewardPenaltyStrategy<CostT, CommCostFunctionT, ActiveScheduleT> rewardPenaltyStrat_;
        VertexSelectionStrategy<GraphT, NodeSelectionContainerT, ActiveScheduleT> selectionStrategy_;
        ThreadLocalActiveScheduleData<GraphT, CostT> activeScheduleData_;

        double averageGain_ = 0.0;
        unsigned maxInnerIterations_ = 0;
        unsigned noImprovementIterationsReducePenalty_ = 0;
        unsigned minInnerIter_ = 0;
        unsigned noImprovementIterationsIncreaseInnerIter_ = 0;
        unsigned stepSelectionEpochCounter_ = 0;
        unsigned stepSelectionCounter_ = 0;
        unsigned stepToRemove_ = 0;
        unsigned localSearchStartStep_ = 0;
        unsigned unlockEdgeBacktrackCounter_ = 0;
        unsigned unlockEdgeBacktrackCounterReset_ = 0;
        unsigned maxNoVioaltionsRemovedBacktrack_ = 0;

        inline unsigned NumSteps() const { return endStep_ - startStep_ + 1; }

        inline unsigned StartIdx(const unsigned nodeStep) const {
            return nodeStep < startStep_ + windowSize ? windowSize - (nodeStep - startStep_) : 0;
        }

        inline unsigned EndIdx(unsigned nodeStep) const {
            return nodeStep + windowSize <= endStep_ ? windowRange_ : windowRange_ - (nodeStep + windowSize - endStep_);
        }
    };

    bool computeWithTimeLimit_ = false;

    BspSchedule<GraphT> *inputSchedule_;
    const GraphT *graph_;
    const BspInstance<GraphT> *instance_;

    CompatibleProcessorRange<GraphT> procRange_;

    KlParameter parameters_;
    std::mt19937 gen_;

    ActiveScheduleT activeSchedule_;
    CommCostFunctionT commCostF_;
    std::vector<ThreadSearchContext> threadDataVec_;
    std::vector<bool> threadFinishedVec_;

    inline unsigned RelStepIdx(const unsigned nodeStep, const unsigned moveStep) const {
        return (moveStep >= nodeStep) ? ((moveStep - nodeStep) + windowSize) : (windowSize - (nodeStep - moveStep));
    }

    inline bool IsCompatible(VertexType node, unsigned proc) const {
        return activeSchedule_.getInstance().isCompatible(node, proc);
    }

    void SetStartStep(const unsigned step, ThreadSearchContext &threadData) {
        threadData.startStep_ = step;
        threadData.stepToRemove_ = step;
        threadData.stepSelectionCounter_ = step;

        threadData.averageGain_ = 0.0;
        threadData.maxInnerIterations_ = parameters_.maxInnerIterationsReset_;
        threadData.noImprovementIterationsReducePenalty_ = parameters_.maxNoImprovementIterations_ / 5;
        threadData.minInnerIter_ = parameters_.minInnerIterReset_;
        threadData.stepSelectionEpochCounter_ = 0;
        threadData.noImprovementIterationsIncreaseInnerIter_ = 10;
        threadData.unlockEdgeBacktrackCounterReset_ = 0;
        threadData.unlockEdgeBacktrackCounter_ = threadData.unlockEdgeBacktrackCounterReset_;
        threadData.maxNoVioaltionsRemovedBacktrack_ = parameters_.maxNoVioaltionsRemovedBacktrackReset_;
    }

    kl_move GetBestMove(NodeSelectionContainerT &affinityTable,
                        vector_vertex_lock_manger<VertexType> &lockManager,
                        heap_datastructure &maxGainHeap) {
        // To introduce non-determinism and help escape local optima, if there are multiple moves with the same
        // top gain, we randomly select one. We check up to `local_max` ties.
        const unsigned localMax = 50;
        std::vector<VertexType> topGainNodes = max_gain_heap.get_top_keys(localMax);

        if (topGainNodes.empty()) {
            // This case is guarded by the caller, but for safety:
            topGainNodes.push_back(max_gain_heap.top());
        }

        std::uniform_int_distribution<size_t> dis(0, top_gain_nodes.size() - 1);
        const VertexType node = top_gain_nodes[dis(gen_)];

        kl_move bestMove = max_gain_heap.get_value(node);
        maxGainHeap.erase(node);
        lockManager.lock(node);
        affinityTable.remove(node);

        return best_move;
    }

    inline void ProcessOtherStepsBestMove(const unsigned idx,
                                          const unsigned nodeStep,
                                          const VertexType &node,
                                          const CostT affinityCurrentProcStep,
                                          CostT &maxGain,
                                          unsigned &maxProc,
                                          unsigned &maxStep,
                                          const std::vector<std::vector<CostT>> &affinityTableNode) const {
        for (const unsigned p : proc_range.compatible_processors_vertex(node)) {
            if constexpr (active_schedule_t::use_memory_constraint) {
                if (not active_schedule.memory_constraint.can_move(node, p, node_step + idx - window_size)) {
                    continue;
                }
            }

            const cost_t gain = affinity_current_proc_step - affinity_table_node[p][idx];
            if (gain > max_gain) {
                max_gain = gain;
                max_proc = p;
                max_step = idx;
            }
        }
    }

    template <bool moveToSameSuperStep>
    kl_move ComputeBestMove(VertexType node,
                            const std::vector<std::vector<CostT>> &affinityTableNode,
                            ThreadSearchContext &threadData) {
        const unsigned nodeStep = activeSchedule_.assigned_superstep(node);
        const unsigned nodeProc = activeSchedule_.assigned_processor(node);

        CostT maxGain = std::numeric_limits<CostT>::lowest();

        unsigned maxProc = std::numeric_limits<unsigned>::max();
        unsigned maxStep = std::numeric_limits<unsigned>::max();

        const CostT affinityCurrentProcStep = affinityTableNode[nodeProc][windowSize];

        unsigned idx = threadData.StartIdx(nodeStep);
        for (; idx < windowSize; idx++) {
            process_other_steps_best_move(
                idx, node_step, node, affinity_current_proc_step, max_gain, max_proc, max_step, affinity_table_node);
        }

        if constexpr (moveToSameSuperStep) {
            for (const unsigned proc : proc_range.compatible_processors_vertex(node)) {
                if (proc == node_proc) {
                    continue;
                }

                if constexpr (active_schedule_t::use_memory_constraint) {
                    if (not active_schedule.memory_constraint.can_move(node, proc, node_step + idx - window_size)) {
                        continue;
                    }
                }

                const cost_t gain = affinity_current_proc_step - affinity_table_node[proc][window_size];
                if (gain > max_gain) {
                    max_gain = gain;
                    max_proc = proc;
                    max_step = idx;
                }
            }
        }

        idx++;

        const unsigned bound = threadData.EndIdx(nodeStep);
        for (; idx < bound; idx++) {
            process_other_steps_best_move(
                idx, node_step, node, affinity_current_proc_step, max_gain, max_proc, max_step, affinity_table_node);
        }

        return kl_move(node, maxGain, nodeProc, nodeStep, maxProc, nodeStep + maxStep - windowSize);
    }

    kl_gain_update_info UpdateNodeWorkAffinityAfterMove(VertexType node,
                                                        kl_move move,
                                                        const pre_move_work_data<work_weight_t> &prevWorkData,
                                                        std::vector<std::vector<CostT>> &affinityTableNode) {
        const unsigned nodeStep = activeSchedule_.assigned_superstep(node);
        const work_weight_t vertexWeight = graph_->vertex_work_weight(node);

        kl_gain_update_info updateInfo(node);

        if (move.from_step == move.to_step) {
            const unsigned lowerBound = move.from_step > windowSize ? move.from_step - windowSize : 0;
            if (lowerBound <= nodeStep && nodeStep <= move.from_step + windowSize) {
                updateInfo.update_from_step = true;
                updateInfo.update_to_step = true;

                const work_weight_t prevMaxWork = prev_work_data.from_step_max_work;
                const work_weight_t prevSecondMaxWork = prev_work_data.from_step_second_max_work;

                if (nodeStep == move.from_step) {
                    const unsigned nodeProc = activeSchedule_.assigned_processor(node);
                    const work_weight_t newMaxWeight = activeSchedule_.get_step_max_work(move.from_step);
                    const work_weight_t newSecondMaxWeight = activeSchedule_.get_step_second_max_work(move.from_step);
                    const work_weight_t newStepProcWork = activeSchedule_.get_step_processor_work(nodeStep, nodeProc);
                    const work_weight_t prevStepProcWork
                        = (nodeProc == move.from_proc) ? new_step_proc_work + graph_->vertex_work_weight(move.node)
                          : (nodeProc == move.to_proc) ? new_step_proc_work - graph_->vertex_work_weight(move.node)
                                                       : new_step_proc_work;
                    const bool prevIsSoleMaxProcessor = (prevWorkData.from_step_max_work_processor_count == 1)
                                                        && (prevMaxWork == prev_step_proc_work);
                    const CostT prevNodeProcAffinity
                        = prevIsSoleMaxProcessor ? std::min(vertex_weight, prev_max_work - prev_second_max_work) : 0.0;
                    const bool newIsSoleMaxProcessor = (activeSchedule_.get_step_max_work_processor_count()[nodeStep] == 1)
                                                       && (newMaxWeight == new_step_proc_work);
                    const CostT newNodeProcAffinity
                        = newIsSoleMaxProcessor ? std::min(vertex_weight, new_max_weight - new_second_max_weight) : 0.0;

                    const CostT diff = newNodeProcAffinity - prevNodeProcAffinity;
                    if (std::abs(diff) > epsilon_) {
                        updateInfo.full_update = true;
                        affinityTableNode[nodeProc][windowSize] += diff;    // Use the pre-calculated diff
                    }

                    if ((prevMaxWork != new_max_weight) || update_info.full_update) {
                        updateInfo.update_entire_from_step = true;

                        for (const unsigned proc : proc_range.compatible_processors_vertex(node)) {
                            if ((proc == node_proc) || (proc == move.from_proc) || (proc == move.to_proc)) {
                                continue;
                            }

                            const work_weight_t new_weight
                                = vertex_weight + active_schedule.get_step_processor_work(node_step, proc);
                            const cost_t prev_other_affinity
                                = compute_same_step_affinity(prev_max_work, new_weight, prev_node_proc_affinity);
                            const cost_t other_affinity
                                = compute_same_step_affinity(new_max_weight, new_weight, new_node_proc_affinity);

                            affinity_table_node[proc][window_size] += (other_affinity - prev_other_affinity);
                        }
                    }

                    if (node_proc != move.from_proc && is_compatible(node, move.from_proc)) {
                        const work_weight_t prevNewWeight = vertex_weight
                                                            + activeSchedule_.get_step_processor_work(nodeStep, move.from_proc)
                                                            + graph_->vertex_work_weight(move.node);
                        const CostT prevOtherAffinity
                            = compute_same_step_affinity(prev_max_work, prev_new_weight, prev_node_proc_affinity);
                        const work_weight_t newWeight
                            = vertex_weight + activeSchedule_.get_step_processor_work(nodeStep, move.from_proc);
                        const CostT otherAffinity = compute_same_step_affinity(new_max_weight, new_weight, new_node_proc_affinity);
                        affinityTableNode[move.from_proc][windowSize] += (otherAffinity - prevOtherAffinity);
                    }

                    if (node_proc != move.to_proc && is_compatible(node, move.to_proc)) {
                        const work_weight_t prevNewWeight = vertex_weight
                                                            + activeSchedule_.get_step_processor_work(nodeStep, move.to_proc)
                                                            - graph_->vertex_work_weight(move.node);
                        const CostT prevOtherAffinity
                            = compute_same_step_affinity(prev_max_work, prev_new_weight, prev_node_proc_affinity);
                        const work_weight_t newWeight
                            = vertex_weight + activeSchedule_.get_step_processor_work(nodeStep, move.to_proc);
                        const CostT otherAffinity = compute_same_step_affinity(new_max_weight, new_weight, new_node_proc_affinity);
                        affinityTableNode[move.to_proc][windowSize] += (otherAffinity - prevOtherAffinity);
                    }

                } else {
                    const work_weight_t newMaxWeight = activeSchedule_.get_step_max_work(move.from_step);
                    const unsigned idx = RelStepIdx(nodeStep, move.from_step);
                    if (prevMaxWork != new_max_weight) {
                        updateInfo.update_entire_from_step = true;
                        // update moving to all procs with special for move.from_proc
                        for (const unsigned proc : proc_range.compatible_processors_vertex(node)) {
                            const work_weight_t new_weight
                                = vertex_weight + active_schedule.get_step_processor_work(move.from_step, proc);
                            if (proc == move.from_proc) {
                                const work_weight_t prev_new_weight
                                    = vertex_weight + active_schedule.get_step_processor_work(move.from_step, proc)
                                      + graph->vertex_work_weight(move.node);
                                const cost_t prev_affinity
                                    = prev_max_work < prev_new_weight
                                          ? static_cast<cost_t>(prev_new_weight) - static_cast<cost_t>(prev_max_work)
                                          : 0.0;
                                const cost_t new_affinity = new_max_weight < new_weight ? static_cast<cost_t>(new_weight)
                                                                                              - static_cast<cost_t>(new_max_weight)
                                                                                        : 0.0;
                                affinity_table_node[proc][idx] += new_affinity - prev_affinity;
                            } else if (proc == move.to_proc) {
                                const work_weight_t prev_new_weight
                                    = vertex_weight + active_schedule.get_step_processor_work(move.to_step, proc)
                                      - graph->vertex_work_weight(move.node);
                                const cost_t prev_affinity
                                    = prev_max_work < prev_new_weight
                                          ? static_cast<cost_t>(prev_new_weight) - static_cast<cost_t>(prev_max_work)
                                          : 0.0;
                                const cost_t new_affinity = new_max_weight < new_weight ? static_cast<cost_t>(new_weight)
                                                                                              - static_cast<cost_t>(new_max_weight)
                                                                                        : 0.0;
                                affinity_table_node[proc][idx] += new_affinity - prev_affinity;
                            } else {
                                const cost_t prev_affinity = prev_max_work < new_weight ? static_cast<cost_t>(new_weight)
                                                                                              - static_cast<cost_t>(prev_max_work)
                                                                                        : 0.0;
                                const cost_t new_affinity = new_max_weight < new_weight ? static_cast<cost_t>(new_weight)
                                                                                              - static_cast<cost_t>(new_max_weight)
                                                                                        : 0.0;
                                affinity_table_node[proc][idx] += new_affinity - prev_affinity;
                            }
                        }
                    } else {
                        // update only move.from_proc and move.to_proc
                        if (is_compatible(node, move.from_proc)) {
                            const work_weight_t fromNewWeight
                                = vertex_weight + activeSchedule_.get_step_processor_work(move.from_step, move.from_proc);
                            const work_weight_t fromPrevNewWeight = from_new_weight + graph_->vertex_work_weight(move.node);
                            const CostT fromPrevAffinity
                                = prev_max_work < from_prev_new_weight
                                      ? static_cast<CostT>(from_prev_new_weight) - static_cast<CostT>(prev_max_work)
                                      : 0.0;

                            const CostT fromNewAffinity
                                = new_max_weight < from_new_weight
                                      ? static_cast<CostT>(from_new_weight) - static_cast<CostT>(new_max_weight)
                                      : 0.0;
                            affinityTableNode[move.from_proc][idx] += fromNewAffinity - fromPrevAffinity;
                        }

                        if (is_compatible(node, move.to_proc)) {
                            const work_weight_t toNewWeight
                                = vertex_weight + activeSchedule_.get_step_processor_work(move.to_step, move.to_proc);
                            const work_weight_t toPrevNewWeight = to_new_weight - graph_->vertex_work_weight(move.node);
                            const CostT toPrevAffinity
                                = prev_max_work < to_prev_new_weight
                                      ? static_cast<CostT>(to_prev_new_weight) - static_cast<CostT>(prev_max_work)
                                      : 0.0;

                            const CostT toNewAffinity = new_max_weight < to_new_weight ? static_cast<CostT>(to_new_weight)
                                                                                             - static_cast<CostT>(new_max_weight)
                                                                                       : 0.0;
                            affinityTableNode[move.to_proc][idx] += toNewAffinity - toPrevAffinity;
                        }
                    }
                }
            }

        } else {
            const unsigned nodeProc = activeSchedule_.assigned_processor(node);
            process_work_update_step(node,
                                     node_step,
                                     node_proc,
                                     vertex_weight,
                                     move.from_step,
                                     move.from_proc,
                                     graph->vertex_work_weight(move.node),
                                     prev_work_data.from_step_max_work,
                                     prev_work_data.from_step_second_max_work,
                                     prev_work_data.from_step_max_work_processor_count,
                                     update_info.update_from_step,
                                     update_info.update_entire_from_step,
                                     update_info.full_update,
                                     affinity_table_node);
            process_work_update_step(node,
                                     node_step,
                                     node_proc,
                                     vertex_weight,
                                     move.to_step,
                                     move.to_proc,
                                     -graph->vertex_work_weight(move.node),
                                     prev_work_data.to_step_max_work,
                                     prev_work_data.to_step_second_max_work,
                                     prev_work_data.to_step_max_work_processor_count,
                                     update_info.update_to_step,
                                     update_info.update_entire_to_step,
                                     update_info.full_update,
                                     affinity_table_node);
        }

        return update_info;
    }

    void ProcessWorkUpdateStep(VertexType node,
                               unsigned nodeStep,
                               unsigned nodeProc,
                               work_weight_t vertexWeight,
                               unsigned moveStep,
                               unsigned moveProc,
                               work_weight_t moveCorrectionNodeWeight,
                               const work_weight_t prevMoveStepMaxWork,
                               const work_weight_t prevMoveStepSecondMaxWork,
                               unsigned prevMoveStepMaxWorkProcessorCount,
                               bool &updateStep,
                               bool &updateEntireStep,
                               bool &fullUpdate,
                               std::vector<std::vector<CostT>> &affinityTableNode);
    void UpdateNodeWorkAffinity(NodeSelectionContainerT &nodes,
                                kl_move move,
                                const pre_move_work_data<work_weight_t> &prevWorkData,
                                std::map<VertexType, kl_gain_update_info> &recomputeMaxGain);
    void UpdateBestMove(
        VertexType node, unsigned step, unsigned proc, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData);
    void UpdateBestMove(VertexType node, unsigned step, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData);
    void UpdateMaxGain(kl_move move, std::map<VertexType, kl_gain_update_info> &recomputeMaxGain, ThreadSearchContext &threadData);
    void ComputeWorkAffinity(VertexType node, std::vector<std::vector<CostT>> &affinityTableNode, ThreadSearchContext &threadData);

    inline void RecomputeNodeMaxGain(VertexType node, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData) {
        const auto bestMove = compute_best_move<true>(node, affinityTable[node], threadData);
        threadData.maxGainHeap_.update(node, best_move);
    }

    inline CostT ComputeSameStepAffinity(const work_weight_t &maxWorkForStep,
                                         const work_weight_t &newWeight,
                                         const CostT &nodeProcAffinity) {
        const CostT maxWorkAfterRemoval = static_cast<CostT>(max_work_for_step) - nodeProcAffinity;
        if (newWeight > maxWorkAfterRemoval) {
            return new_weight - maxWorkAfterRemoval;
        }
        return 0.0;
    }

    inline CostT ApplyMove(kl_move move, ThreadSearchContext &threadData) {
        activeSchedule_.apply_move(move, threadData.activeScheduleData_);
        commCostF_.update_datastructure_after_move(move, threadData.startStep_, threadData.endStep_);
        CostT changeInCost = -move.gain;
        changeInCost += static_cast<CostT>(threadData.activeScheduleData_.resolved_violations.size())
                        * threadData.rewardPenaltyStrat_.reward;
        changeInCost
            -= static_cast<CostT>(threadData.activeScheduleData_.new_violations.size()) * threadData.rewardPenaltyStrat_.penalty;

#ifdef KL_DEBUG
        std::cout << "penalty: " << thread_data.reward_penalty_strat.penalty
                  << " num violations: " << thread_data.active_schedule_data.current_violations.size()
                  << " num new violations: " << thread_data.active_schedule_data.new_violations.size()
                  << ", num resolved violations: " << thread_data.active_schedule_data.resolved_violations.size()
                  << ", reward: " << thread_data.reward_penalty_strat.reward << std::endl;
        std::cout << "apply move, previous cost: " << thread_data.active_schedule_data.cost
                  << ", new cost: " << thread_data.active_schedule_data.cost + change_in_cost << ", "
                  << (thread_data.active_schedule_data.feasible ? "feasible," : "infeasible,") << std::endl;
#endif

        threadData.activeScheduleData_.update_cost(changeInCost);

        return changeInCost;
    }

    void RunQuickMoves(unsigned &innerIter,
                       ThreadSearchContext &threadData,
                       const CostT changeInCost,
                       const VertexType bestMoveNode) {
#ifdef KL_DEBUG
        std::cout << "Starting quick moves sequence." << std::endl;
#endif
        innerIter++;

        const size_t numAppliedMoves = threadData.activeScheduleData_.applied_moves.size() - 1;
        const CostT savedCost = threadData.activeScheduleData_.cost - changeInCost;

        std::unordered_set<VertexType> localLock;
        localLock.insert(best_move_node);
        std::vector<VertexType> quickMovesStack;
        quickMovesStack.reserve(10 + threadData.activeScheduleData_.new_violations.size() * 2);

        for (const auto &keyValuePair : threadData.activeScheduleData_.new_violations) {
            const auto &key = keyValuePair.first;
            quickMovesStack.push_back(key);
        }

        while (quickMovesStack.size() > 0) {
            auto nextNodeToMove = quick_moves_stack.back();
            quickMovesStack.pop_back();

            threadData.rewardPenaltyStrat_.init_reward_penalty(
                static_cast<double>(threadData.activeScheduleData_.current_violations.size()) + 1.0);
            compute_node_affinities(next_node_to_move, thread_data.local_affinity_table, thread_data);
            kl_move bestQuickMove = compute_best_move<true>(next_node_to_move, threadData.localAffinityTable_, threadData);

            localLock.insert(next_node_to_move);
            if (bestQuickMove.gain <= std::numeric_limits<CostT>::lowest()) {
                continue;
            }

#ifdef KL_DEBUG
            std::cout << " >>> move node " << best_quick_move.node << " with gain " << best_quick_move.gain
                      << ", from proc|step: " << best_quick_move.from_proc << "|" << best_quick_move.from_step
                      << " to: " << best_quick_move.to_proc << "|" << best_quick_move.to_step << std::endl;
#endif

            apply_move(best_quick_move, thread_data);
            innerIter++;

            if (threadData.activeScheduleData_.new_violations.size() > 0) {
                bool abort = false;

                for (const auto &keyValuePair : threadData.activeScheduleData_.new_violations) {
                    const auto &key = keyValuePair.first;
                    if (localLock.find(key) != local_lock.end()) {
                        abort = true;
                        break;
                    }
                    quickMovesStack.push_back(key);
                }

                if (abort) {
                    break;
                }

            } else if (threadData.activeScheduleData_.feasible) {
                break;
            }
        }

        if (!threadData.activeScheduleData_.feasible) {
            activeSchedule_.revert_schedule_to_bound(numAppliedMoves,
                                                     savedCost,
                                                     true,
                                                     commCostF_,
                                                     threadData.activeScheduleData_,
                                                     threadData.startStep_,
                                                     threadData.endStep_);
#ifdef KL_DEBUG
            std::cout << "Ending quick moves sequence with infeasible solution." << std::endl;
#endif
        }
#ifdef KL_DEBUG
        else {
            std::cout << "Ending quick moves sequence with feasible solution." << std::endl;
        }
#endif

        threadData.affinityTable_.trim();
        threadData.maxGainHeap_.clear();
        threadData.rewardPenaltyStrat_.init_reward_penalty(1.0);
        InsertGainHeap(threadData);    // Re-initialize the heap with the current state
    }

    void ResolveViolations(ThreadSearchContext &threadData) {
        auto &currentViolations = threadData.activeScheduleData_.current_violations;
        unsigned numViolations = static_cast<unsigned>(currentViolations.size());
        if (numViolations > 0) {
#ifdef KL_DEBUG_1
            std::cout << "thread " << thread_data.thread_id << ", Starting preresolving violations with " << num_violations
                      << " initial violations" << std::endl;
#endif
            threadData.rewardPenaltyStrat_.init_reward_penalty(static_cast<double>(numViolations) + 1.0);
            std::unordered_set<VertexType> localLock;
            unsigned numIter = 0;
            const unsigned minIter = numViolations / 4;
            while (not currentViolations.empty()) {
                std::uniform_int_distribution<size_t> dis(0, currentViolations.size() - 1);
                auto it = currentViolations.begin();
                std::advance(it, dis(gen_));
                const auto &nextEdge = *it;
                const VertexType sourceV = source(nextEdge, *graph_);
                const VertexType targetV = target(nextEdge, *graph_);
                const bool sourceLocked = local_lock.find(source_v) != local_lock.end();
                const bool targetLocked = local_lock.find(target_v) != local_lock.end();

                if (sourceLocked && targetLocked) {
#ifdef KL_DEBUG_1
                    std::cout << "source, target locked" << std::endl;
#endif
                    break;
                }

                kl_move bestMove;
                if (sourceLocked || targetLocked) {
                    const VertexType node = sourceLocked ? target_v : source_v;
                    compute_node_affinities(node, thread_data.local_affinity_table, thread_data);
                    bestMove = compute_best_move<true>(node, threadData.localAffinityTable_, threadData);
                } else {
                    compute_node_affinities(source_v, thread_data.local_affinity_table, thread_data);
                    kl_move bestSourceVMove = compute_best_move<true>(source_v, threadData.localAffinityTable_, threadData);
                    compute_node_affinities(target_v, thread_data.local_affinity_table, thread_data);
                    kl_move bestTargetVMove = compute_best_move<true>(target_v, threadData.localAffinityTable_, threadData);
                    bestMove = best_target_v_move.gain > best_source_v_move.gain ? std::move(best_target_v_move)
                                                                                 : std::move(best_source_v_move);
                }

                localLock.insert(best_move.node);
                if (bestMove.gain <= std::numeric_limits<CostT>::lowest()) {
                    continue;
                }

                apply_move(best_move, thread_data);
                threadData.affinityTable_.insert(best_move.node);
#ifdef KL_DEBUG_1
                std::cout << "move node " << best_move.node << " with gain " << best_move.gain
                          << ", from proc|step: " << best_move.from_proc << "|" << best_move.from_step
                          << " to: " << best_move.to_proc << "|" << best_move.to_step << std::endl;
#endif
                const unsigned newNumViolations = static_cast<unsigned>(currentViolations.size());
                if (newNumViolations == 0) {
                    break;
                }

                if (threadData.activeScheduleData_.new_violations.size() > 0) {
                    for (const auto &vertexEdgePair : threadData.activeScheduleData_.new_violations) {
                        const auto &vertex = vertexEdgePair.first;
                        threadData.affinityTable_.insert(vertex);
                    }
                }

                const double gain = static_cast<double>(numViolations) - static_cast<double>(newNumViolations);
                numViolations = newNumViolations;
                UpdateAvgGain(gain, numIter++, threadData.averageGain_);
#ifdef KL_DEBUG_1
                std::cout << "thread " << thread_data.thread_id << ",  preresolving violations with " << num_violations
                          << " violations, " << num_iter << " #iterations, " << thread_data.average_gain << " average gain"
                          << std::endl;
#endif
                if (numIter > minIter && threadData.averageGain_ < 0.0) {
                    break;
                }
            }
            threadData.averageGain_ = 0.0;
        }
    }

    void RunLocalSearch(ThreadSearchContext &threadData) {
#ifdef KL_DEBUG_1
        std::cout << "thread " << thread_data.thread_id
                  << ", start local search, initial schedule cost: " << thread_data.active_schedule_data.cost << " with "
                  << thread_data.num_steps() << " supersteps." << std::endl;
#endif
        std::vector<VertexType> newNodes;
        std::vector<VertexType> unlockNodes;
        std::map<VertexType, kl_gain_update_info> recomputeMaxGain;

        const auto startTime = std::chrono::high_resolution_clock::now();

        unsigned noImprovementIterCounter = 0;
        unsigned outerIter = 0;

        for (; outerIter < parameters_.maxOuterIterations_; outerIter++) {
            CostT initialInnerIterCost = threadData.activeScheduleData_.cost;

            ResetInnerSearchStructures(threadData);
            SelectActiveNodes(threadData);
            threadData.rewardPenaltyStrat_.init_reward_penalty(
                static_cast<double>(threadData.activeScheduleData_.current_violations.size()) + 1.0);
            InsertGainHeap(threadData);

            unsigned innerIter = 0;
            unsigned violationRemovedCount = 0;
            unsigned resetCounter = 0;
            bool iterInitalFeasible = threadData.activeScheduleData_.feasible;

#ifdef KL_DEBUG
            std::cout << "------ start inner loop ------" << std::endl;
            std::cout << "initial node selection: {";
            for (size_t i = 0; i < thread_data.affinity_table.size(); ++i) {
                std::cout << thread_data.affinity_table.get_selected_nodes()[i] << ", ";
            }
            std::cout << "}" << std::endl;
#endif
#ifdef KL_DEBUG_1
            if (not iter_inital_feasible) {
                std::cout << "initial solution not feasible, num violations: "
                          << thread_data.active_schedule_data.current_violations.size()
                          << ". Penalty: " << thread_data.reward_penalty_strat.penalty
                          << ", reward: " << thread_data.reward_penalty_strat.reward << std::endl;
            }
#endif
#ifdef KL_DEBUG_COST_CHECK
            active_schedule.getVectorSchedule().number_of_supersteps = thread_data_vec[0].num_steps();
            if (std::abs(comm_cost_f.compute_schedule_cost_test() - thread_data.active_schedule_data.cost) > 0.00001) {
                std::cout << "computed cost: " << comm_cost_f.compute_schedule_cost_test()
                          << ", current cost: " << thread_data.active_schedule_data.cost << std::endl;
                std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
            }
            if constexpr (active_schedule_t::use_memory_constraint) {
                if (not active_schedule.memory_constraint.satisfied_memory_constraint()) {
                    std::cout << "memory constraint not satisfied" << std::endl;
                }
            }
#endif

            while (innerIter < threadData.maxInnerIterations_ && threadData.maxGainHeap_.size() > 0) {
                kl_move bestMove
                    = get_best_move(thread_data.affinity_table,
                                    thread_data.lock_manager,
                                    thread_data.max_gain_heap);    // locks best_move.node and removes it from node_selection
                if (bestMove.gain <= std::numeric_limits<CostT>::lowest()) {
                    break;
                }
                UpdateAvgGain(best_move.gain, innerIter, threadData.averageGain_);
#ifdef KL_DEBUG
                std::cout << " >>> move node " << best_move.node << " with gain " << best_move.gain
                          << ", from proc|step: " << best_move.from_proc << "|" << best_move.from_step
                          << " to: " << best_move.to_proc << "|" << best_move.to_step << ",avg gain: " << thread_data.average_gain
                          << std::endl;
#endif
                if (innerIter > threadData.minInnerIter_ && threadData.averageGain_ < 0.0) {
#ifdef KL_DEBUG
                    std::cout << "Negative average gain: " << thread_data.average_gain << ", end local search" << std::endl;
#endif
                    break;
                }

#ifdef KL_DEBUG
                if (not active_schedule.getInstance().isCompatible(best_move.node, best_move.to_proc)) {
                    std::cout << "move to incompatibe node" << std::endl;
                }
#endif

                const auto prevWorkData = activeSchedule_.get_pre_move_work_data(best_move);
                const typename CommCostFunctionT::pre_move_comm_data_t prevCommData = commCostF_.get_pre_move_comm_data(best_move);
                const CostT changeInCost = apply_move(best_move, thread_data);
#ifdef KL_DEBUG_COST_CHECK
                active_schedule.getVectorSchedule().number_of_supersteps = thread_data_vec[0].num_steps();
                if (std::abs(comm_cost_f.compute_schedule_cost_test() - thread_data.active_schedule_data.cost) > 0.00001) {
                    std::cout << "computed cost: " << comm_cost_f.compute_schedule_cost_test()
                              << ", current cost: " << thread_data.active_schedule_data.cost << std::endl;
                    std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
                }
                if constexpr (active_schedule_t::use_memory_constraint) {
                    if (not active_schedule.memory_constraint.satisfied_memory_constraint()) {
                        std::cout << "memory constraint not satisfied" << std::endl;
                    }
                }
#endif
                if constexpr (enableQuickMoves_) {
                    if (iterInitalFeasible && threadData.activeScheduleData_.new_violations.size() > 0) {
                        run_quick_moves(inner_iter, thread_data, change_in_cost, best_move.node);
#ifdef KL_DEBUG_COST_CHECK
                        active_schedule.getVectorSchedule().number_of_supersteps = thread_data_vec[0].num_steps();
                        if (std::abs(comm_cost_f.compute_schedule_cost_test() - thread_data.active_schedule_data.cost) > 0.00001) {
                            std::cout << "computed cost: " << comm_cost_f.compute_schedule_cost_test()
                                      << ", current cost: " << thread_data.active_schedule_data.cost << std::endl;
                            std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<"
                                      << std::endl;
                        }
                        if constexpr (active_schedule_t::use_memory_constraint) {
                            if (not active_schedule.memory_constraint.satisfied_memory_constraint()) {
                                std::cout << "memory constraint not satisfied" << std::endl;
                            }
                        }
#endif
                        continue;
                    }
                }

                if (threadData.activeScheduleData_.current_violations.size() > 0) {
                    if (threadData.activeScheduleData_.resolved_violations.size() > 0) {
                        violationRemovedCount = 0;
                    } else {
                        violationRemovedCount++;

                        if (violationRemovedCount > 3) {
                            if (resetCounter < threadData.maxNoVioaltionsRemovedBacktrack_
                                && ((not iterInitalFeasible)
                                    || (threadData.activeScheduleData_.cost < threadData.activeScheduleData_.best_cost))) {
                                threadData.affinityTable_.reset_node_selection();
                                threadData.maxGainHeap_.clear();
                                threadData.lockManager_.clear();
                                threadData.selectionStrategy_.select_nodes_violations(
                                    threadData.affinityTable_,
                                    threadData.activeScheduleData_.current_violations,
                                    threadData.startStep_,
                                    threadData.endStep_);
#ifdef KL_DEBUG
                                std::cout << "Infeasible, and no violations resolved for 5 iterations, reset node selection"
                                          << std::endl;
#endif
                                threadData.rewardPenaltyStrat_.init_reward_penalty(
                                    static_cast<double>(threadData.activeScheduleData_.current_violations.size()));
                                InsertGainHeap(threadData);

                                resetCounter++;
                                innerIter++;
                                continue;
                            } else {
#ifdef KL_DEBUG
                                std::cout << "Infeasible, and no violations resolved for 5 iterations, end local search"
                                          << std::endl;
#endif
                                break;
                            }
                        }
                    }
                }

                if (IsLocalSearchBlocked(threadData)) {
                    if (not blocked_edge_strategy(best_move.node, unlock_nodes, thread_data)) {
                        break;
                    }
                }

                threadData.affinityTable_.trim();
                update_affinities(best_move, thread_data, recompute_max_gain, new_nodes, prev_work_data, prev_comm_data);

                for (const auto v : unlock_nodes) {
                    thread_data.lock_manager.unlock(v);
                }
                newNodes.insert(new_nodes.end(), unlock_nodes.begin(), unlock_nodes.end());
                unlockNodes.clear();

#ifdef KL_DEBUG
                std::cout << "recmopute max gain: {";
                for (const auto map_pair : recompute_max_gain) {
                    const auto &key = map_pair.first;
                    std::cout << key << ", ";
                }
                std::cout << "}" << std::endl;
                std::cout << "new nodes: {";
                for (const auto v : new_nodes) {
                    std::cout << v << ", ";
                }
                std::cout << "}" << std::endl;
#endif
#ifdef KL_DEBUG_COST_CHECK
                active_schedule.getVectorSchedule().number_of_supersteps = thread_data_vec[0].num_steps();
                if (std::abs(comm_cost_f.compute_schedule_cost_test() - thread_data.active_schedule_data.cost) > 0.00001) {
                    std::cout << "computed cost: " << comm_cost_f.compute_schedule_cost_test()
                              << ", current cost: " << thread_data.active_schedule_data.cost << std::endl;
                    std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
                }
                if constexpr (active_schedule_t::use_memory_constraint) {
                    if (not active_schedule.memory_constraint.satisfied_memory_constraint()) {
                        std::cout << "memory constraint not satisfied" << std::endl;
                    }
                }
#endif
                update_max_gain(best_move, recompute_max_gain, thread_data);
                insert_new_nodes_gain_heap(new_nodes, thread_data.affinity_table, thread_data);

                recomputeMaxGain.clear();
                newNodes.clear();

                innerIter++;
            }

#ifdef KL_DEBUG
            std::cout << "--- end inner loop after " << inner_iter
                      << " inner iterations, gain heap size: " << thread_data.max_gain_heap.size() << ", outer iteraion "
                      << outer_iter << "/" << parameters.max_outer_iterations
                      << ", current cost: " << thread_data.active_schedule_data.cost << ", "
                      << (thread_data.active_schedule_data.feasible ? "feasible" : "infeasible") << std::endl;
#endif
#ifdef KL_DEBUG_1
            const unsigned num_steps_tmp = thread_data.end_step;
#endif
            activeSchedule_.revert_to_best_schedule(threadData.localSearchStartStep_,
                                                    threadData.stepToRemove_,
                                                    commCostF_,
                                                    threadData.activeScheduleData_,
                                                    threadData.startStep_,
                                                    threadData.endStep_);
#ifdef KL_DEBUG_1
            if (thread_data.local_search_start_step > 0) {
                if (num_steps_tmp == thread_data.end_step) {
                    std::cout << "thread " << thread_data.thread_id << ", removing step " << thread_data.step_to_remove
                              << " succeded " << std::endl;
                } else {
                    std::cout << "thread " << thread_data.thread_id << ", removing step " << thread_data.step_to_remove
                              << " failed " << std::endl;
                }
            }
#endif

#ifdef KL_DEBUG_COST_CHECK
            active_schedule.getVectorSchedule().number_of_supersteps = thread_data_vec[0].num_steps();
            if (std::abs(comm_cost_f.compute_schedule_cost_test() - thread_data.active_schedule_data.cost) > 0.00001) {
                std::cout << "computed cost: " << comm_cost_f.compute_schedule_cost_test()
                          << ", current cost: " << thread_data.active_schedule_data.cost << std::endl;
                std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
            }
            if constexpr (active_schedule_t::use_memory_constraint) {
                if (not active_schedule.memory_constraint.satisfied_memory_constraint()) {
                    std::cout << "memory constraint not satisfied" << std::endl;
                }
            }
#endif

            if (computeWithTimeLimit_) {
                auto finishTime = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(finishTime - startTime).count();
                if (duration > ImprovementScheduler<GraphT>::timeLimitSeconds) {
                    break;
                }
            }

            if (OtherThreadsFinished(threadData.threadId_)) {
#ifdef KL_DEBUG_1
                std::cout << "thread " << thread_data.thread_id << ", other threads finished, end local search" << std::endl;
#endif
                break;
            }

            if (initialInnerIterCost <= threadData.activeScheduleData_.cost) {
                noImprovementIterCounter++;

                if (noImprovementIterCounter >= parameters_.maxNoImprovementIterations_) {
#ifdef KL_DEBUG_1
                    std::cout << "thread " << thread_data.thread_id << ", no improvement for "
                              << parameters.max_no_improvement_iterations << " iterations, end local search" << std::endl;
#endif
                    break;
                }
            } else {
                noImprovementIterCounter = 0;
            }

            AdjustLocalSearchParameters(outerIter, noImprovementIterCounter, threadData);
        }

#ifdef KL_DEBUG_1
        std::cout << "thread " << thread_data.thread_id << ", local search end after " << outer_iter
                  << " outer iterations, current cost: " << thread_data.active_schedule_data.cost << " with "
                  << thread_data.num_steps() << " supersteps, vs serial cost " << active_schedule.get_total_work_weight() << "."
                  << std::endl;
#endif
        threadFinishedVec_[threadData.threadId_] = true;
    }

    bool OtherThreadsFinished(const unsigned threadId) {
        const size_t numThreads = threadFinishedVec_.size();
        if (numThreads == 1) {
            return false;
        }

        for (size_t i = 0; i < numThreads; i++) {
            if (i != threadId && !threadFinishedVec_[i]) {
                return false;
            }
        }
        return true;
    }

    inline void UpdateAffinities(const kl_move &bestMove,
                                 ThreadSearchContext &threadData,
                                 std::map<VertexType, kl_gain_update_info> &recomputeMaxGain,
                                 std::vector<VertexType> &newNodes,
                                 const pre_move_work_data<v_workw_t<Graph_t>> &prevWorkData,
                                 const typename CommCostFunctionT::pre_move_comm_data_t &prevCommData) {
        if constexpr (CommCostFunctionT::is_max_comm_cost_function) {
            commCostF_.update_node_comm_affinity(
                best_move,
                threadData,
                threadData.rewardPenaltyStrat_.penalty,
                threadData.rewardPenaltyStrat_.reward,
                recompute_max_gain,
                new_nodes);    // this only updated reward/penalty, collects new_nodes, and fills recompute_max_gain

            // Add nodes from affected steps to new_nodes
            // {
            //     std::unordered_set<unsigned> steps_to_check;
            //     const unsigned num_steps = active_schedule.num_steps();

            //     auto add_steps_range = [&](unsigned center_step) {
            //         unsigned start = (center_step > window_size) ? center_step - window_size : 0;
            //         unsigned end = std::min(center_step + window_size, num_steps - 1);

            //         // Constrain to thread range
            //         if (start < thread_data.start_step)
            //             start = thread_data.start_step;
            //         if (end > thread_data.end_step)
            //             end = thread_data.end_step;

            //         for (unsigned s = start; s <= end; ++s) {
            //             steps_to_check.insert(s);
            //         }
            //     };

            //     add_steps_range(best_move.from_step);
            //     add_steps_range(best_move.to_step);

            //     for (unsigned step : steps_to_check) {
            //         for (unsigned proc = 0; proc < instance->numberOfProcessors(); ++proc) {
            //             const auto &nodes_in_step = active_schedule.getSetSchedule().step_processor_vertices[step][proc];
            //             for (const auto &node : nodes_in_step) {
            //                 if (!thread_data.affinity_table.is_selected(node) && !thread_data.lock_manager.is_locked(node)) {
            //                     new_nodes.push_back(node);
            //                 }
            //             }
            //         }
            //     }

            //     // Deduplicate new_nodes
            //     std::sort(new_nodes.begin(), new_nodes.end());
            //     new_nodes.erase(std::unique(new_nodes.begin(), new_nodes.end()), new_nodes.end());
            // }

            // Determine the steps where max/second_max/max_count for work/comm changed
            std::unordered_set<unsigned> changedSteps;

            // Check work changes for from_step
            if (bestMove.from_step == best_move.to_step) {
                // Same step - check if max/second_max changed
                const auto currentMax = activeSchedule_.get_step_max_work(best_move.from_step);
                const auto currentSecondMax = activeSchedule_.get_step_second_max_work(best_move.from_step);
                const auto currentCount = activeSchedule_.get_step_max_work_processor_count()[best_move.from_step];
                if (currentMax != prev_work_data.from_step_max_work
                    || current_second_max != prev_work_data.from_step_second_max_work
                    || current_count != prev_work_data.from_step_max_work_processor_count) {
                    changedSteps.insert(best_move.from_step);
                }
            } else {
                // Different steps - check both
                const auto currentFromMax = activeSchedule_.get_step_max_work(best_move.from_step);
                const auto currentFromSecondMax = activeSchedule_.get_step_second_max_work(best_move.from_step);
                const auto currentFromCount = activeSchedule_.get_step_max_work_processor_count()[best_move.from_step];
                if (currentFromMax != prev_work_data.from_step_max_work
                    || current_from_second_max != prev_work_data.from_step_second_max_work
                    || current_from_count != prev_work_data.from_step_max_work_processor_count) {
                    changedSteps.insert(best_move.from_step);
                }

                const auto currentToMax = activeSchedule_.get_step_max_work(best_move.to_step);
                const auto currentToSecondMax = activeSchedule_.get_step_second_max_work(best_move.to_step);
                const auto currentToCount = activeSchedule_.get_step_max_work_processor_count()[best_move.to_step];
                if (currentToMax != prev_work_data.to_step_max_work
                    || current_to_second_max != prev_work_data.to_step_second_max_work
                    || current_to_count != prev_work_data.to_step_max_work_processor_count) {
                    changedSteps.insert(best_move.to_step);
                }
            }

            for (const auto &[step, step_info] : prevCommData.step_data) {
                typename CommCostFunctionT::pre_move_comm_data_t::step_info currentInfo;
                // Query current values
                const auto currentMax = commCostF_.comm_ds.step_max_comm(step);
                const auto currentSecondMax = commCostF_.comm_ds.step_second_max_comm(step);
                const auto currentCount = commCostF_.comm_ds.step_max_comm_count(step);

                if (currentMax != step_info.max_comm || currentSecondMax != step_info.second_max_comm
                    || currentCount != step_info.max_comm_count) {
                    changedSteps.insert(step);
                }
            }

            // Recompute affinities for all active nodes
            const size_t activeCount = threadData.affinityTable_.size();
            for (size_t i = 0; i < activeCount; ++i) {
                const VertexType node = threadData.affinityTable_.get_selected_nodes()[i];

                // Determine if this node needs affinity recomputation
                // A node needs recomputation if it's in or adjacent to changed steps
                const unsigned nodeStep = activeSchedule_.assigned_superstep(node);

                // Calculate window bounds for this node once
                const int nodeLowerBound = static_cast<int>(nodeStep) - static_cast<int>(windowSize);
                const unsigned nodeUpperBound = nodeStep + windowSize;

                bool needsUpdate = false;
                // Check if any changed step falls within the node's window
                for (unsigned step : changedSteps) {
                    if (static_cast<int>(step) >= nodeLowerBound && step <= nodeUpperBound) {
                        needsUpdate = true;
                        break;
                    }
                }

                if (needsUpdate) {
                    auto &affinityTableNode = threadData.affinityTable_.get_affinity_table(node);

                    // Reset affinity table entries to zero
                    const unsigned numProcs = activeSchedule_.getInstance().numberOfProcessors();
                    for (unsigned p = 0; p < numProcs; ++p) {
                        for (unsigned idx = 0; idx < affinityTableNode[p].size(); ++idx) {
                            affinityTableNode[p][idx] = 0;
                        }
                    }

                    compute_node_affinities(node, affinity_table_node, thread_data);
                    recomputeMaxGain[node] = kl_gain_update_info(node, true);
                }
            }
        } else {
            update_node_work_affinity(thread_data.affinity_table, best_move, prev_work_data, recompute_max_gain);
            commCostF_.update_node_comm_affinity(best_move,
                                                 threadData,
                                                 threadData.rewardPenaltyStrat_.penalty,
                                                 threadData.rewardPenaltyStrat_.reward,
                                                 recompute_max_gain,
                                                 new_nodes);
        }
    }

    inline bool BlockedEdgeStrategy(VertexType node, std::vector<VertexType> &unlockNodes, ThreadSearchContext &threadData) {
        if (threadData.unlockEdgeBacktrackCounter_ > 1) {
            for (const auto vertexEdgePair : threadData.activeScheduleData_.new_violations) {
                const auto &e = vertexEdgePair.second;
                const auto sourceV = source(e, *graph_);
                const auto targetV = target(e, *graph_);

                if (node == sourceV && threadData.lockManager_.is_locked(targetV)) {
                    unlockNodes.push_back(targetV);
                } else if (node == targetV && threadData.lockManager_.is_locked(sourceV)) {
                    unlockNodes.push_back(sourceV);
                }
            }
#ifdef KL_DEBUG
            std::cout << "Nodes of violated edge locked, backtrack counter: " << thread_data.unlock_edge_backtrack_counter
                      << std::endl;
#endif
            threadData.unlockEdgeBacktrackCounter_--;
            return true;
        } else {
#ifdef KL_DEBUG
            std::cout << "Nodes of violated edge locked, end local search" << std::endl;
#endif
            return false;    // or reset local search and initalize with violating nodes
        }
    }

    inline void AdjustLocalSearchParameters(unsigned outerIter, unsigned noImpCounter, ThreadSearchContext &threadData) {
        if (noImpCounter >= threadData.noImprovementIterationsReducePenalty_
            && threadData.rewardPenaltyStrat_.initial_penalty > 1.0) {
            threadData.rewardPenaltyStrat_.initial_penalty
                = static_cast<CostT>(std::floor(std::sqrt(threadData.rewardPenaltyStrat_.initial_penalty)));
            threadData.unlockEdgeBacktrackCounterReset_ += 1;
            threadData.noImprovementIterationsReducePenalty_ += 15;
#ifdef KL_DEBUG_1
            std::cout << "thread " << thread_data.thread_id << ", no improvement for "
                      << thread_data.no_improvement_iterations_reduce_penalty << " iterations, reducing initial penalty to "
                      << thread_data.reward_penalty_strat.initial_penalty << std::endl;
#endif
        }

        if (parameters_.tryRemoveStepAfterNumOuterIterations_ > 0
            && ((outerIter + 1) % parameters_.tryRemoveStepAfterNumOuterIterations_) == 0) {
            threadData.stepSelectionEpochCounter_ = 0;
            ;
#ifdef KL_DEBUG
            std::cout << "reset remove epoc counter after " << outer_iter << " iterations." << std::endl;
#endif
        }

        if (noImpCounter >= threadData.noImprovementIterationsIncreaseInnerIter_) {
            threadData.minInnerIter_ = static_cast<unsigned>(std::ceil(threadData.minInnerIter_ * 2.2));
            threadData.noImprovementIterationsIncreaseInnerIter_ += 20;
#ifdef KL_DEBUG_1
            std::cout << "thread " << thread_data.thread_id << ", no improvement for "
                      << thread_data.no_improvement_iterations_increase_inner_iter << " iterations, increasing min inner iter to "
                      << thread_data.min_inner_iter << std::endl;
#endif
        }
    }

    bool IsLocalSearchBlocked(ThreadSearchContext &threadData);
    void SetParameters(vertex_idx_t<Graph_t> numNodes);
    void ResetInnerSearchStructures(ThreadSearchContext &threadData) const;
    void InitializeDatastructures(BspSchedule<GraphT> &schedule);
    void PrintHeap(heap_datastructure &maxGainHeap) const;
    void CleanupDatastructures();
    void UpdateAvgGain(const CostT gain, const unsigned numIter, double &averageGain);
    void InsertGainHeap(ThreadSearchContext &threadData);
    void InsertNewNodesGainHeap(std::vector<VertexType> &newNodes, NodeSelectionContainerT &nodes, ThreadSearchContext &threadData);

    inline void ComputeNodeAffinities(VertexType node,
                                      std::vector<std::vector<CostT>> &affinityTableNode,
                                      ThreadSearchContext &threadData) {
        compute_work_affinity(node, affinity_table_node, thread_data);
        commCostF_.compute_comm_affinity(node,
                                         affinityTableNode,
                                         threadData.rewardPenaltyStrat_.penalty,
                                         threadData.rewardPenaltyStrat_.reward,
                                         threadData.startStep_,
                                         threadData.endStep_);
    }

    void SelectActiveNodes(ThreadSearchContext &threadData) {
        if (SelectNodesCheckRemoveSuperstep(threadData.stepToRemove_, threadData)) {
            activeSchedule_.swap_empty_step_fwd(threadData.stepToRemove_, threadData.endStep_);
            threadData.endStep_--;
            threadData.localSearchStartStep_ = static_cast<unsigned>(threadData.activeScheduleData_.applied_moves.size());
            threadData.activeScheduleData_.update_cost(static_cast<CostT>(-1.0 * instance_->synchronisationCosts()));

            if constexpr (enablePreresolvingViolations_) {
                ResolveViolations(threadData);
            }

            if (threadData.activeScheduleData_.current_violations.size() > parameters_.initialViolationThreshold_) {
                activeSchedule_.revert_to_best_schedule(threadData.localSearchStartStep_,
                                                        threadData.stepToRemove_,
                                                        commCostF_,
                                                        threadData.activeScheduleData_,
                                                        threadData.startStep_,
                                                        threadData.endStep_);
            } else {
                threadData.unlockEdgeBacktrackCounter_
                    = static_cast<unsigned>(threadData.activeScheduleData_.current_violations.size());
                threadData.maxInnerIterations_
                    = std::max(threadData.unlockEdgeBacktrackCounter_ * 5u, parameters_.maxInnerIterationsReset_);
                threadData.maxNoVioaltionsRemovedBacktrack_ = parameters_.maxNoVioaltionsRemovedBacktrackForRemoveStepReset_;
#ifdef KL_DEBUG_1
                std::cout << "thread " << thread_data.thread_id << ", Trying to remove step " << thread_data.step_to_remove
                          << std::endl;
#endif
                return;
            }
        }
        // thread_data.step_to_remove = thread_data.start_step;
        threadData.localSearchStartStep_ = 0;
        threadData.selectionStrategy_.select_active_nodes(threadData.affinityTable_, threadData.startStep_, threadData.endStep_);
    }

    bool CheckRemoveSuperstep(unsigned step);
    bool SelectNodesCheckRemoveSuperstep(unsigned &step, ThreadSearchContext &threadData);

    bool ScatterNodesSuperstep(unsigned step, ThreadSearchContext &threadData) {
        assert(step <= threadData.endStep_ && threadData.startStep_ <= step);
        bool abort = false;

        for (unsigned proc = 0; proc < instance_->numberOfProcessors(); proc++) {
            const std::vector<VertexType> stepProcNodeVec(
                activeSchedule_.getSetSchedule().step_processor_vertices[step][proc].begin(),
                activeSchedule_.getSetSchedule().step_processor_vertices[step][proc].end());
            for (const auto &node : step_proc_node_vec) {
                thread_data.reward_penalty_strat.init_reward_penalty(
                    static_cast<double>(thread_data.active_schedule_data.current_violations.size()) + 1.0);
                compute_node_affinities(node, thread_data.local_affinity_table, thread_data);
                kl_move best_move = compute_best_move<false>(node, thread_data.local_affinity_table, thread_data);

                if (best_move.gain <= std::numeric_limits<double>::lowest()) {
                    abort = true;
                    break;
                }

                apply_move(best_move, thread_data);
                if (thread_data.active_schedule_data.current_violations.size()
                    > parameters.abort_scatter_nodes_violation_threshold) {
                    abort = true;
                    break;
                }

                thread_data.affinity_table.insert(node);
                // thread_data.selection_strategy.add_neighbours_to_selection(node, thread_data.affinity_table,
                // thread_data.start_step, thread_data.end_step);
                if (thread_data.active_schedule_data.new_violations.size() > 0) {
                    for (const auto &vertex_edge_pair : thread_data.active_schedule_data.new_violations) {
                        const auto &vertex = vertex_edge_pair.first;
                        thread_data.affinity_table.insert(vertex);
                    }
                }

#ifdef KL_DEBUG
                std::cout << "move node " << best_move.node << " with gain " << best_move.gain
                          << ", from proc|step: " << best_move.from_proc << "|" << best_move.from_step
                          << " to: " << best_move.to_proc << "|" << best_move.to_step << std::endl;
#endif

#ifdef KL_DEBUG_COST_CHECK
                active_schedule.getVectorSchedule().number_of_supersteps = thread_data_vec[0].num_steps();
                if (std::abs(comm_cost_f.compute_schedule_cost_test() - thread_data.active_schedule_data.cost) > 0.00001) {
                    std::cout << "computed cost: " << comm_cost_f.compute_schedule_cost_test()
                              << ", current cost: " << thread_data.active_schedule_data.cost << std::endl;
                    std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
                }
                if constexpr (active_schedule_t::use_memory_constraint) {
                    if (not active_schedule.memory_constraint.satisfied_memory_constraint()) {
                        std::cout << "memory constraint not satisfied" << std::endl;
                    }
                }
#endif
            }

            if (abort) {
                break;
            }
        }

        if (abort) {
            activeSchedule_.revert_to_best_schedule(
                0, 0, commCostF_, threadData.activeScheduleData_, threadData.startStep_, threadData.endStep_);
            threadData.affinityTable_.reset_node_selection();
            return false;
        }
        return true;
    }

    void SynchronizeActiveSchedule(const unsigned numThreads) {
        if (numThreads == 1) {    // single thread case
            activeSchedule_.set_cost(threadDataVec_[0].active_schedule_data.cost);
            activeSchedule_.getVectorSchedule().number_of_supersteps = threadDataVec_[0].num_steps();
            return;
        }

        unsigned writeCursor = threadDataVec_[0].end_step + 1;
        for (unsigned i = 1; i < numThreads; ++i) {
            auto &thread = threadDataVec_[i];
            if (thread.start_step <= thread.end_step) {
                for (unsigned j = thread.start_step; j <= thread.end_step; ++j) {
                    if (j != writeCursor) {
                        activeSchedule_.swap_steps(j, writeCursor);
                    }
                    writeCursor++;
                }
            }
        }
        activeSchedule_.getVectorSchedule().number_of_supersteps = writeCursor;
        const CostT newCost = commCostF_.compute_schedule_cost();
        activeSchedule_.set_cost(newCost);
    }

  public:
    KlImprover() : ImprovementScheduler<GraphT>() {
        std::random_device rd;
        gen_ = std::mt19937(rd());
    }

    explicit KlImprover(unsigned seed) : ImprovementScheduler<GraphT>() { gen_ = std::mt19937(seed); }

    virtual ~KlImprover() = default;

    virtual RETURN_STATUS improveSchedule(BspSchedule<GraphT> &schedule) override {
        if (schedule.getInstance().numberOfProcessors() < 2) {
            return RETURN_STATUS::BEST_FOUND;
        }

        const unsigned numThreads = 1;

        threadDataVec_.resize(numThreads);
        threadFinishedVec_.assign(numThreads, true);

        set_parameters(schedule.getInstance().numberOfVertices());
        InitializeDatastructures(schedule);
        const CostT initialCost = activeSchedule_.get_cost();
        const unsigned numSteps = schedule.numberOfSupersteps();

        SetStartStep(0, threadDataVec_[0]);
        threadDataVec_[0].end_step = (numSteps > 0) ? numSteps - 1 : 0;

        auto &threadData = this->threadDataVec_[0];
        threadData.active_schedule_data.initialize_cost(activeSchedule_.get_cost());
        threadData.selection_strategy.setup(threadData.start_step, threadData.end_step);
        RunLocalSearch(threadData);

        SynchronizeActiveSchedule(numThreads);

        if (initialCost > activeSchedule_.get_cost()) {
            activeSchedule_.write_schedule(schedule);
            CleanupDatastructures();
            return RETURN_STATUS::OSP_SUCCESS;
        } else {
            CleanupDatastructures();
            return RETURN_STATUS::BEST_FOUND;
        }
    }

    virtual RETURN_STATUS improveScheduleWithTimeLimit(BspSchedule<GraphT> &schedule) override {
        computeWithTimeLimit_ = true;
        return improveSchedule(schedule);
    }

    virtual void SetTimeQualityParameter(const double timeQuality) { this->parameters_.timeQuality_ = timeQuality; }

    virtual void SetSuperstepRemoveStrengthParameter(const double superstepRemoveStrength) {
        this->parameters_.superstepRemoveStrength_ = superstepRemoveStrength;
    }

    virtual std::string GetScheduleName() const { return "kl_improver_" + commCostF_.name(); }
};

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::SetParameters(vertex_idx_t<Graph_t> numNodes) {
    const unsigned logNumNodes = (numNodes > 1) ? static_cast<unsigned>(std::log(num_nodes)) : 1;

    // Total number of outer iterations. Proportional to sqrt N.
    parameters_.maxOuterIterations_
        = static_cast<unsigned>(std::sqrt(num_nodes) * (parameters_.timeQuality_ * 10.0) / parameters_.numParallelLoops_);

    // Number of times to reset the search for violations before giving up.
    parameters_.maxNoVioaltionsRemovedBacktrackReset_ = parameters_.timeQuality_ < 0.75  ? 1
                                                        : parameters_.timeQuality_ < 1.0 ? 2
                                                                                         : 3;

    // Parameters for the superstep removal heuristic.
    parameters_.maxNoVioaltionsRemovedBacktrackForRemoveStepReset_
        = 3 + static_cast<unsigned>(parameters_.superstepRemoveStrength_ * 7);
    parameters_.nodeMaxStepSelectionEpochs_ = parameters_.superstepRemoveStrength_ < 0.75  ? 1
                                              : parameters_.superstepRemoveStrength_ < 1.0 ? 2
                                                                                           : 3;
    parameters_.removeStepEpocs_ = static_cast<unsigned>(parameters_.superstepRemoveStrength_ * 4.0);

    parameters_.minInnerIterReset_ = static_cast<unsigned>(logNumNodes + logNumNodes * (1.0 + parameters_.timeQuality_));

    if (parameters_.removeStepEpocs_ > 0) {
        parameters_.tryRemoveStepAfterNumOuterIterations_ = parameters_.maxOuterIterations_ / parameters_.removeStepEpocs_;
    } else {
        // Effectively disable superstep removal if remove_step_epocs is 0.
        parameters_.tryRemoveStepAfterNumOuterIterations_ = parameters_.maxOuterIterations_ + 1;
    }

    unsigned i = 0;
    for (auto &thread : threadDataVec_) {
        thread.thread_id = i++;
        // The number of nodes to consider in each inner iteration. Proportional to log(N).
        thread.selection_strategy.selection_threshold
            = static_cast<std::size_t>(std::ceil(parameters_.timeQuality_ * 10 * logNumNodes + logNumNodes));
    }

#ifdef KL_DEBUG_1
    std::cout << "kl set parameter, number of nodes: " << num_nodes << std::endl;
    std::cout << "max outer iterations: " << parameters.max_outer_iterations << std::endl;
    std::cout << "max inner iterations: " << parameters.max_inner_iterations_reset << std::endl;
    std::cout << "no improvement iterations reduce penalty: " << thread_data_vec[0].no_improvement_iterations_reduce_penalty
              << std::endl;
    std::cout << "selction threshold: " << thread_data_vec[0].selection_strategy.selection_threshold << std::endl;
    std::cout << "remove step epocs: " << parameters.remove_step_epocs << std::endl;
    std::cout << "try remove step after num outer iterations: " << parameters.try_remove_step_after_num_outer_iterations
              << std::endl;
    std::cout << "number of parallel loops: " << parameters.num_parallel_loops << std::endl;
#endif
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::UpdateNodeWorkAffinity(
    NodeSelectionContainerT &nodes,
    kl_move move,
    const pre_move_work_data<work_weight_t> &prevWorkData,
    std::map<VertexType, kl_gain_update_info> &recomputeMaxGain) {
    const size_t activeCount = nodes.size();

    for (size_t i = 0; i < activeCount; ++i) {
        const VertexType node = nodes.get_selected_nodes()[i];

        kl_gain_update_info updateInfo = update_node_work_affinity_after_move(node, move, prev_work_data, nodes.at(node));
        if (updateInfo.update_from_step || update_info.update_to_step) {
            recomputeMaxGain[node] = update_info;
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::UpdateMaxGain(
    kl_move move, std::map<VertexType, kl_gain_update_info> &recomputeMaxGain, ThreadSearchContext &threadData) {
    for (auto &pair : recompute_max_gain) {
        if (pair.second.full_update) {
            recompute_node_max_gain(pair.first, thread_data.affinity_table, thread_data);
        } else {
            if (pair.second.update_entire_from_step) {
                update_best_move(pair.first, move.from_step, thread_data.affinity_table, thread_data);
            } else if (pair.second.update_from_step && is_compatible(pair.first, move.from_proc)) {
                update_best_move(pair.first, move.from_step, move.from_proc, thread_data.affinity_table, thread_data);
            }

            if (move.from_step != move.to_step || not pair.second.update_entire_from_step) {
                if (pair.second.update_entire_to_step) {
                    update_best_move(pair.first, move.to_step, thread_data.affinity_table, thread_data);
                } else if (pair.second.update_to_step && is_compatible(pair.first, move.to_proc)) {
                    update_best_move(pair.first, move.to_step, move.to_proc, thread_data.affinity_table, thread_data);
                }
            }
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::ComputeWorkAffinity(
    VertexType node, std::vector<std::vector<CostT>> &affinityTableNode, ThreadSearchContext &threadData) {
    const unsigned nodeStep = activeSchedule_.assigned_superstep(node);
    const work_weight_t vertexWeight = graph_->vertex_work_weight(node);

    unsigned step = (nodeStep > windowSize) ? (nodeStep - windowSize) : 0;
    for (unsigned idx = threadData.StartIdx(nodeStep); idx < threadData.EndIdx(nodeStep); ++idx, ++step) {
        if (idx == windowSize) {
            continue;
        }

        const CostT maxWorkForStep = static_cast<CostT>(activeSchedule_.get_step_max_work(step));

        for (const unsigned proc : proc_range.compatible_processors_vertex(node)) {
            const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(step, proc);
            const cost_t work_diff = static_cast<cost_t>(new_weight) - max_work_for_step;
            affinity_table_node[proc][idx] = std::max(0.0, work_diff);
        }
    }

    const unsigned nodeProc = activeSchedule_.assigned_processor(node);
    const work_weight_t maxWorkForStep = activeSchedule_.get_step_max_work(nodeStep);
    const bool isSoleMaxProcessor = (activeSchedule_.get_step_max_work_processor_count()[nodeStep] == 1)
                                    && (maxWorkForStep == activeSchedule_.get_step_processor_work(nodeStep, nodeProc));

    const CostT nodeProcAffinity
        = isSoleMaxProcessor ? std::min(vertex_weight, max_work_for_step - activeSchedule_.get_step_second_max_work(nodeStep))
                             : 0.0;
    affinityTableNode[nodeProc][windowSize] = nodeProcAffinity;

    for (const unsigned proc : proc_range.compatible_processors_vertex(node)) {
        if (proc == node_proc) {
            continue;
        }

        const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(node_step, proc);
        affinity_table_node[proc][window_size] = compute_same_step_affinity(max_work_for_step, new_weight, node_proc_affinity);
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::ProcessWorkUpdateStep(
    VertexType node,
    unsigned nodeStep,
    unsigned nodeProc,
    work_weight_t vertexWeight,
    unsigned moveStep,
    unsigned moveProc,
    work_weight_t moveCorrectionNodeWeight,
    const work_weight_t prevMoveStepMaxWork,
    const work_weight_t prevMoveStepSecondMaxWork,
    unsigned prevMoveStepMaxWorkProcessorCount,
    bool &updateStep,
    bool &updateEntireStep,
    bool &fullUpdate,
    std::vector<std::vector<CostT>> &affinityTableNode) {
    const unsigned lowerBound = moveStep > windowSize ? moveStep - windowSize : 0;
    if (lowerBound <= nodeStep && nodeStep <= moveStep + windowSize) {
        updateStep = true;
        if (nodeStep == moveStep) {
            const work_weight_t newMaxWeight = activeSchedule_.get_step_max_work(moveStep);
            const work_weight_t newSecondMaxWeight = activeSchedule_.get_step_second_max_work(moveStep);
            const work_weight_t newStepProcWork = activeSchedule_.get_step_processor_work(nodeStep, nodeProc);

            const work_weight_t prevStepProcWork = (nodeProc == moveProc) ? new_step_proc_work + move_correction_node_weight
                                                                          : new_step_proc_work;
            const bool prevIsSoleMaxProcessor = (prevMoveStepMaxWorkProcessorCount == 1)
                                                && (prevMoveStepMaxWork == prev_step_proc_work);
            const CostT prevNodeProcAffinity
                = prevIsSoleMaxProcessor ? std::min(vertex_weight, prev_move_step_max_work - prev_move_step_second_max_work) : 0.0;

            const bool newIsSoleMaxProcessor = (activeSchedule_.get_step_max_work_processor_count()[nodeStep] == 1)
                                               && (newMaxWeight == new_step_proc_work);
            const CostT newNodeProcAffinity
                = newIsSoleMaxProcessor ? std::min(vertex_weight, new_max_weight - new_second_max_weight) : 0.0;

            const CostT diff = newNodeProcAffinity - prevNodeProcAffinity;
            const bool updateNodeProcAffinity = std::abs(diff) > epsilon_;
            if (updateNodeProcAffinity) {
                fullUpdate = true;
                affinityTableNode[nodeProc][windowSize] += diff;
            }

            if ((prevMoveStepMaxWork != new_max_weight) || updateNodeProcAffinity) {
                updateEntireStep = true;

                for (const unsigned proc : proc_range.compatible_processors_vertex(node)) {
                    if ((proc == node_proc) || (proc == move_proc)) {
                        continue;
                    }

                    const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(node_step, proc);
                    const cost_t prev_other_affinity
                        = compute_same_step_affinity(prev_move_step_max_work, new_weight, prev_node_proc_affinity);
                    const cost_t other_affinity = compute_same_step_affinity(new_max_weight, new_weight, new_node_proc_affinity);

                    affinity_table_node[proc][window_size] += (other_affinity - prev_other_affinity);
                }
            }

            if (node_proc != move_proc && is_compatible(node, move_proc)) {
                const work_weight_t prevNewWeight
                    = vertex_weight + activeSchedule_.get_step_processor_work(nodeStep, moveProc) + move_correction_node_weight;
                const CostT prevOtherAffinity
                    = compute_same_step_affinity(prev_move_step_max_work, prev_new_weight, prev_node_proc_affinity);
                const work_weight_t newWeight = vertex_weight + activeSchedule_.get_step_processor_work(nodeStep, moveProc);
                const CostT otherAffinity = compute_same_step_affinity(new_max_weight, new_weight, new_node_proc_affinity);

                affinityTableNode[moveProc][windowSize] += (otherAffinity - prevOtherAffinity);
            }

        } else {
            const work_weight_t newMaxWeight = activeSchedule_.get_step_max_work(moveStep);
            const unsigned idx = RelStepIdx(nodeStep, moveStep);
            if (prevMoveStepMaxWork != new_max_weight) {
                updateEntireStep = true;

                // update moving to all procs with special for move_proc
                for (const unsigned proc : proc_range.compatible_processors_vertex(node)) {
                    const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(move_step, proc);
                    if (proc != move_proc) {
                        const cost_t prev_affinity
                            = prev_move_step_max_work < new_weight
                                  ? static_cast<cost_t>(new_weight) - static_cast<cost_t>(prev_move_step_max_work)
                                  : 0.0;
                        const cost_t new_affinity = new_max_weight < new_weight
                                                        ? static_cast<cost_t>(new_weight) - static_cast<cost_t>(new_max_weight)
                                                        : 0.0;
                        affinity_table_node[proc][idx] += new_affinity - prev_affinity;

                    } else {
                        const work_weight_t prev_new_weight = vertex_weight
                                                              + active_schedule.get_step_processor_work(move_step, proc)
                                                              + move_correction_node_weight;
                        const cost_t prev_affinity
                            = prev_move_step_max_work < prev_new_weight
                                  ? static_cast<cost_t>(prev_new_weight) - static_cast<cost_t>(prev_move_step_max_work)
                                  : 0.0;

                        const cost_t new_affinity = new_max_weight < new_weight
                                                        ? static_cast<cost_t>(new_weight) - static_cast<cost_t>(new_max_weight)
                                                        : 0.0;
                        affinity_table_node[proc][idx] += new_affinity - prev_affinity;
                    }
                }
            } else {
                // update only move_proc
                if (is_compatible(node, move_proc)) {
                    const work_weight_t newWeight = vertex_weight + activeSchedule_.get_step_processor_work(moveStep, moveProc);
                    const work_weight_t prevNewWeight = new_weight + move_correction_node_weight;
                    const CostT prevAffinity
                        = prev_move_step_max_work < prev_new_weight
                              ? static_cast<CostT>(prev_new_weight) - static_cast<CostT>(prev_move_step_max_work)
                              : 0.0;

                    const CostT newAffinity
                        = new_max_weight < new_weight ? static_cast<CostT>(new_weight) - static_cast<CostT>(new_max_weight) : 0.0;
                    affinityTableNode[moveProc][idx] += newAffinity - prevAffinity;
                }
            }
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
bool KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::SelectNodesCheckRemoveSuperstep(
    unsigned &stepToRemove, ThreadSearchContext &threadData) {
    if (threadData.stepSelectionEpochCounter_ >= parameters_.nodeMaxStepSelectionEpochs_ || threadData.NumSteps() < 3) {
        return false;
    }

    for (stepToRemove = threadData.stepSelectionCounter_; stepToRemove <= threadData.endStep_; stepToRemove++) {
        assert(stepToRemove >= threadData.startStep_ && stepToRemove <= threadData.endStep_);
#ifdef KL_DEBUG
        std::cout << "Checking to remove step " << step_to_remove << "/" << thread_data.end_step << std::endl;
#endif
        if (CheckRemoveSuperstep(stepToRemove)) {
#ifdef KL_DEBUG
            std::cout << "Checking to scatter step " << step_to_remove << "/" << thread_data.end_step << std::endl;
#endif
            assert(stepToRemove >= threadData.startStep_ && stepToRemove <= threadData.endStep_);
            if (ScatterNodesSuperstep(stepToRemove, threadData)) {
                threadData.stepSelectionCounter_ = stepToRemove + 1;

                if (threadData.stepSelectionCounter_ > threadData.endStep_) {
                    threadData.stepSelectionCounter_ = threadData.startStep_;
                    threadData.stepSelectionEpochCounter_++;
                }
                return true;
            }
        }
    }

    threadData.stepSelectionEpochCounter_++;
    threadData.stepSelectionCounter_ = threadData.startStep_;
    return false;
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
bool KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::CheckRemoveSuperstep(unsigned step) {
    if (activeSchedule_.num_steps() < 2) {
        return false;
    }

    if (activeSchedule_.get_step_max_work(step) < instance_->synchronisationCosts()) {
        return true;
    }

    return false;
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::ResetInnerSearchStructures(
    ThreadSearchContext &threadData) const {
    threadData.unlockEdgeBacktrackCounter_ = threadData.unlockEdgeBacktrackCounterReset_;
    threadData.maxInnerIterations_ = parameters_.maxInnerIterationsReset_;
    threadData.maxNoVioaltionsRemovedBacktrack_ = parameters_.maxNoVioaltionsRemovedBacktrackReset_;
    threadData.averageGain_ = 0.0;
    threadData.affinityTable_.reset_node_selection();
    threadData.maxGainHeap_.clear();
    threadData.lockManager_.clear();
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
bool KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::IsLocalSearchBlocked(
    ThreadSearchContext &threadData) {
    for (const auto &pair : threadData.activeScheduleData_.new_violations) {
        if (threadData.lockManager_.is_locked(pair.first)) {
            return true;
        }
    }
    return false;
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::InitializeDatastructures(
    BspSchedule<GraphT> &schedule) {
    inputSchedule_ = &schedule;
    instance_ = &schedule.getInstance();
    graph_ = &instance_->getComputationalDag();

    activeSchedule_.initialize(schedule);

    procRange_.initialize(*instance_);
    commCostF_.initialize(activeSchedule_, procRange_);
    const CostT initialCost = commCostF_.compute_schedule_cost();
    activeSchedule_.set_cost(initialCost);

    for (auto &tData : threadDataVec_) {
        tData.affinity_table.initialize(activeSchedule_, tData.selection_strategy.selection_threshold);
        tData.lock_manager.initialize(graph_->num_vertices());
        tData.reward_penalty_strat.initialize(
            activeSchedule_, commCostF_.get_max_comm_weight_multiplied(), activeSchedule_.get_max_work_weight());
        tData.selection_strategy.initialize(activeSchedule_, gen_, tData.start_step, tData.end_step);

        tData.local_affinity_table.resize(instance_->numberOfProcessors());
        for (unsigned i = 0; i < instance_->numberOfProcessors(); ++i) {
            tData.local_affinity_table[i].resize(windowRange_);
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::UpdateAvgGain(const CostT gain,
                                                                                                const unsigned numIter,
                                                                                                double &averageGain) {
    averageGain = static_cast<double>((averageGain * numIter + gain)) / (numIter + 1.0);
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::InsertGainHeap(ThreadSearchContext &threadData) {
    const size_t activeCount = threadData.affinityTable_.size();

    for (size_t i = 0; i < activeCount; ++i) {
        const VertexType node = threadData.affinityTable_.get_selected_nodes()[i];
        compute_node_affinities(node, thread_data.affinity_table.at(node), thread_data);
        const auto bestMove = compute_best_move<true>(node, threadData.affinityTable_[node], threadData);
        threadData.maxGainHeap_.push(node, best_move);
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::InsertNewNodesGainHeap(
    std::vector<VertexType> &newNodes, NodeSelectionContainerT &nodes, ThreadSearchContext &threadData) {
    for (const auto &node : new_nodes) {
        nodes.insert(node);
        compute_node_affinities(node, thread_data.affinity_table.at(node), thread_data);
        const auto best_move = compute_best_move<true>(node, thread_data.affinity_table[node], thread_data);
        thread_data.max_gain_heap.push(node, best_move);
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::CleanupDatastructures() {
    threadDataVec_.clear();
    activeSchedule_.clear();
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::PrintHeap(heap_datastructure &maxGainHeap) const {
    if (maxGainHeap.is_empty()) {
        std::cout << "heap is empty" << std::endl;
        return;
    }
    heap_datastructure tempHeap = max_gain_heap;    // requires copy constructor

    std::cout << "heap current size: " << temp_heap.size() << std::endl;
    const auto &topVal = temp_heap.get_value(temp_heap.top());
    std::cout << "heap top node " << top_val.node << " gain " << top_val.gain << std::endl;

    unsigned count = 0;
    while (!temp_heap.is_empty() && count++ < 15) {
        const auto &val = temp_heap.get_value(temp_heap.top());
        std::cout << "node " << val.node << " gain " << val.gain << " to proc " << val.to_proc << " to step " << val.to_step
                  << std::endl;
        tempHeap.pop();
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::UpdateBestMove(
    VertexType node, unsigned step, unsigned proc, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData) {
    const unsigned nodeProc = activeSchedule_.assigned_processor(node);
    const unsigned nodeStep = activeSchedule_.assigned_superstep(node);

    if ((nodeProc == proc) && (nodeStep == step)) {
        return;
    }

    kl_move nodeMove = threadData.maxGainHeap_.get_value(node);
    CostT maxGain = node_move.gain;

    unsigned maxProc = node_move.to_proc;
    unsigned maxStep = node_move.to_step;

    if ((maxStep == step) && (maxProc == proc)) {
        recompute_node_max_gain(node, affinity_table, thread_data);
    } else {
        if constexpr (ActiveScheduleT::use_memory_constraint) {
            if (not activeSchedule_.memory_constraint.can_move(node, proc, step)) {
                return;
            }
        }
        const unsigned idx = RelStepIdx(nodeStep, step);
        const CostT gain = affinityTable[node][nodeProc][windowSize] - affinityTable[node][proc][idx];
        if (gain > maxGain) {
            maxGain = gain;
            maxProc = proc;
            maxStep = step;
        }

        const CostT diff = maxGain - node_move.gain;
        if ((std::abs(diff) > epsilon_) || (maxProc != node_move.to_proc) || (maxStep != node_move.to_step)) {
            nodeMove.gain = maxGain;
            nodeMove.to_proc = maxProc;
            nodeMove.to_step = maxStep;
            threadData.maxGainHeap_.update(node, node_move);
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::UpdateBestMove(
    VertexType node, unsigned step, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData) {
    const unsigned nodeProc = activeSchedule_.assigned_processor(node);
    const unsigned nodeStep = activeSchedule_.assigned_superstep(node);

    kl_move nodeMove = threadData.maxGainHeap_.get_value(node);
    CostT maxGain = node_move.gain;

    unsigned maxProc = node_move.to_proc;
    unsigned maxStep = node_move.to_step;

    if (maxStep == step) {
        recompute_node_max_gain(node, affinity_table, thread_data);
    } else {
        if (nodeStep != step) {
            const unsigned idx = RelStepIdx(nodeStep, step);
            for (const unsigned p : proc_range.compatible_processors_vertex(node)) {
                if constexpr (active_schedule_t::use_memory_constraint) {
                    if (not active_schedule.memory_constraint.can_move(node, p, step)) {
                        continue;
                    }
                }
                const cost_t gain = affinity_table[node][node_proc][window_size] - affinity_table[node][p][idx];
                if (gain > max_gain) {
                    max_gain = gain;
                    max_proc = p;
                    max_step = step;
                }
            }
        } else {
            for (const unsigned proc : proc_range.compatible_processors_vertex(node)) {
                if (proc == node_proc) {
                    continue;
                }
                if constexpr (active_schedule_t::use_memory_constraint) {
                    if (not active_schedule.memory_constraint.can_move(node, proc, step)) {
                        continue;
                    }
                }
                const cost_t gain = affinity_table[node][node_proc][window_size] - affinity_table[node][proc][window_size];
                if (gain > max_gain) {
                    max_gain = gain;
                    max_proc = proc;
                    max_step = step;
                }
            }
        }

        const CostT diff = maxGain - node_move.gain;
        if ((std::abs(diff) > epsilon_) || (maxProc != node_move.to_proc) || (maxStep != node_move.to_step)) {
            nodeMove.gain = maxGain;
            nodeMove.to_proc = maxProc;
            nodeMove.to_step = maxStep;
            threadData.maxGainHeap_.update(node, node_move);
        }
    }
}

}    // namespace osp
