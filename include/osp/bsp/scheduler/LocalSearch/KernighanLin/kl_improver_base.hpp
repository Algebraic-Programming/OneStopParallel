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
// #define KL_DEBUG_COST_CHECK
#include <algorithm>
#include <chrono>
#include <cstdlib>
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

    unsigned maxNoViolationsRemovedBacktrackReset_;
    unsigned removeStepEpocs_;
    unsigned nodeMaxStepSelectionEpochs_;
    unsigned maxNoViolationsRemovedBacktrackForRemoveStepReset_;
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

// =============================================================================
// BASE CLASS — shared logic (~90% of code)
// =============================================================================
template <typename Derived,
          typename GraphT,
          typename CommCostFunctionT,
          typename MemoryConstraintT = NoLocalSearchMemoryConstraint,
          unsigned windowSize = 1,
          typename CostT = double>
class KlImproverBase : public ImprovementScheduler<GraphT> {
    static_assert(isDirectedGraphEdgeDescV<GraphT>, "GraphT must satisfy the directed_graph concept");
    static_assert(hasHashableEdgeDescV<GraphT>, "GraphT must satisfy the HasHashableEdgeDesc concept");
    static_assert(isComputationalDagV<GraphT>, "GraphT must satisfy the computational_dag concept");

  protected:
    constexpr static unsigned windowRange_ = 2 * windowSize + 1;
    constexpr static bool enableQuickMoves_ = true;
    constexpr static bool enablePreresolvingViolations_ = true;
    constexpr static double epsilon_ = 1e-9;

    using VertexMemWeightT = osp::VMemwT<GraphT>;
    using VertexCommWeightT = osp::VCommwT<GraphT>;
    using VertexWorkWeightT = osp::VWorkwT<GraphT>;
    using VertexType = VertexIdxT<GraphT>;
    using EdgeType = EdgeDescT<GraphT>;

    using KlMove = KlMoveStruct<CostT, VertexType>;
    using HeapDatastructure = MaxPairingHeap<VertexType, KlMove>;
    using ActiveScheduleT = KlActiveSchedule<GraphT, CostT, MemoryConstraintT>;
    using NodeSelectionContainerT = AdaptiveAffinityTable<GraphT, CostT, ActiveScheduleT, windowSize>;
    using KlGainUpdateInfo = KlUpdateInfo<VertexType>;

    // --- CRTP ---
    Derived &derived() { return static_cast<Derived &>(*this); }

    const Derived &derived() const { return static_cast<const Derived &>(*this); }

    // --- ThreadSearchContext (shared, no heap or reverse index) ---

    struct ThreadSearchContext {
        unsigned threadId_ = 0;
        unsigned startStep_ = 0;
        unsigned endStep_ = 0;
        unsigned originalEndStep_ = 0;

        VectorVertexLockManager<VertexType> lockManager_;
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
        unsigned unlockEdgeBacktrackCounter_ = 0;
        unsigned unlockEdgeBacktrackCounterReset_ = 0;
        unsigned maxNoViolationsRemovedBacktrack_ = 0;

        inline unsigned NumSteps() const { return endStep_ - startStep_ + 1; }

        inline unsigned StartIdx(const unsigned nodeStep) const {
            return nodeStep < startStep_ + windowSize ? windowSize - (nodeStep - startStep_) : 0;
        }

        inline unsigned EndIdx(unsigned nodeStep) const {
            return nodeStep + windowSize <= endStep_ ? windowRange_ : windowRange_ - (nodeStep + windowSize - endStep_);
        }
    };

    // --- Shared members ---

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

    // --- Shared utility methods ---

    inline unsigned RelStepIdx(const unsigned nodeStep, const unsigned moveStep) const {
        return (moveStep >= nodeStep) ? ((moveStep - nodeStep) + windowSize) : (windowSize - (nodeStep - moveStep));
    }

    inline bool IsCompatible(VertexType node, unsigned proc) const {
        return activeSchedule_.GetInstance().IsCompatible(node, proc);
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
        threadData.maxNoViolationsRemovedBacktrack_ = parameters_.maxNoViolationsRemovedBacktrackReset_;
    }

    // --- Shared helper: collect new neighbor nodes ---

    void CollectNewNodes(const KlMove &bestMove, ThreadSearchContext &threadData, std::vector<VertexType> &newNodes) {
        const auto &dag = *graph_;
        for (const auto &child : dag.Children(bestMove.node_)) {
            if (activeSchedule_.AssignedSuperstep(child) < threadData.startStep_
                || activeSchedule_.AssignedSuperstep(child) > threadData.endStep_) {
                continue;
            }
            if (threadData.lockManager_.IsLocked(child)) {
                continue;
            }
            if (!threadData.affinityTable_.IsSelected(child)) {
                newNodes.push_back(child);
            }
        }
        for (const auto &parent : dag.Parents(bestMove.node_)) {
            if (activeSchedule_.AssignedSuperstep(parent) < threadData.startStep_
                || activeSchedule_.AssignedSuperstep(parent) > threadData.endStep_) {
                continue;
            }
            if (threadData.lockManager_.IsLocked(parent)) {
                continue;
            }
            if (!threadData.affinityTable_.IsSelected(parent)) {
                newNodes.push_back(parent);
            }
        }
    }

    // --- ComputeBestMove ---

    inline void ProcessOtherStepsBestMove(const unsigned idx,
                                          const unsigned nodeStep,
                                          const VertexType &node,
                                          const CostT affinityCurrentProcStep,
                                          CostT &maxGain,
                                          unsigned &maxProc,
                                          unsigned &maxStep,
                                          const std::vector<std::vector<CostT>> &affinityTableNode) const {
        for (const unsigned p : procRange_.CompatibleProcessorsVertex(node)) {
            if constexpr (ActiveScheduleT::useMemoryConstraint_) {
                if (not activeSchedule_.memoryConstraint_.CanMove(node, p, nodeStep + idx - windowSize)) {
                    continue;
                }
            }

            const CostT gain = affinityCurrentProcStep - affinityTableNode[p][idx];
            if (gain > maxGain) {
                maxGain = gain;
                maxProc = p;
                maxStep = idx;
            }
        }
    }

    template <bool moveToSameSuperStep>
    KlMove ComputeBestMove(VertexType node,
                           const std::vector<std::vector<CostT>> &affinityTableNode,
                           ThreadSearchContext &threadData) {
        const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);
        const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);

        CostT maxGain = std::numeric_limits<CostT>::lowest();

        unsigned maxProc = std::numeric_limits<unsigned>::max();
        unsigned maxStep = std::numeric_limits<unsigned>::max();

        const CostT affinityCurrentProcStep = affinityTableNode[nodeProc][windowSize];

        unsigned idx = threadData.StartIdx(nodeStep);
        for (; idx < windowSize; idx++) {
            ProcessOtherStepsBestMove(idx, nodeStep, node, affinityCurrentProcStep, maxGain, maxProc, maxStep, affinityTableNode);
        }

        if constexpr (moveToSameSuperStep) {
            for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
                if (proc == nodeProc) {
                    continue;
                }

                if constexpr (ActiveScheduleT::useMemoryConstraint_) {
                    if (not activeSchedule_.memoryConstraint_.CanMove(node, proc, nodeStep + idx - windowSize)) {
                        continue;
                    }
                }

                const CostT gain = affinityCurrentProcStep - affinityTableNode[proc][windowSize];
                if (gain > maxGain) {
                    maxGain = gain;
                    maxProc = proc;
                    maxStep = idx;
                }
            }
        }

        idx++;

        const unsigned bound = threadData.EndIdx(nodeStep);
        for (; idx < bound; idx++) {
            ProcessOtherStepsBestMove(idx, nodeStep, node, affinityCurrentProcStep, maxGain, maxProc, maxStep, affinityTableNode);
        }

        return KlMove(node, maxGain, nodeProc, nodeStep, maxProc, nodeStep + maxStep - windowSize);
    }

    // --- Shared affinity computation ---

    /// Work-cost delta when placing a node on a DIFFERENT step.
    /// Used by KlImproverHeap for incremental affinity updates.
    inline CostT ComputeDiffStepAffinity(const VertexWorkWeightT maxWork, const VertexWorkWeightT newWeight) const {
        return maxWork < newWeight ? static_cast<CostT>(newWeight) - static_cast<CostT>(maxWork) : 0.0;
    }

    /// Work-cost delta when placing a node on the SAME step (after removal).
    /// Used by KlImproverHeap for incremental affinity updates.
    inline CostT ComputeSameStepAffinity(const VertexWorkWeightT &maxWorkForStep,
                                         const VertexWorkWeightT &newWeight,
                                         const CostT &nodeProcAffinity) {
        const CostT maxWorkAfterRemoval = static_cast<CostT>(maxWorkForStep) - nodeProcAffinity;
        if (newWeight > maxWorkAfterRemoval) {
            return newWeight - maxWorkAfterRemoval;
        }
        return 0.0;
    }

    inline void ComputeNodeAffinities(VertexType node,
                                      std::vector<std::vector<CostT>> &affinityTableNode,
                                      ThreadSearchContext &threadData) {
        commCostF_.ComputeNodeAffinity(node,
                                       affinityTableNode,
                                       threadData.rewardPenaltyStrat_.penalty_,
                                       threadData.rewardPenaltyStrat_.reward_,
                                       threadData.startStep_,
                                       threadData.endStep_);
    }

    // --- ApplyMove ---

    inline CostT ApplyMove(KlMove move, ThreadSearchContext &threadData) {
#ifdef KL_DEBUG_COST_CHECK
        // Measure TRUE cost before move — separate work and comm per step
        activeSchedule_.GetVectorSchedule().numberOfSupersteps_ = threadData.NumSteps();
        const unsigned numStepsCheck = threadData.endStep_ + 1;
        std::vector<double> perStepCommBefore(numStepsCheck);
        std::vector<double> perStepWorkBefore(numStepsCheck);
        for (unsigned s = 0; s < numStepsCheck; s++) {
            perStepCommBefore[s] = commCostF_.StepMaxComm(s);
            perStepWorkBefore[s] = activeSchedule_.GetStepMaxWork(s);
        }
#endif

        activeSchedule_.ApplyMove(move, threadData.activeScheduleData_);
        commCostF_.UpdateDatastructureAfterMove(move, threadData.startStep_, threadData.endStep_);

        CostT changeInCost = -move.gain_;
        changeInCost += static_cast<CostT>(threadData.activeScheduleData_.resolvedViolations_.size())
                        * threadData.rewardPenaltyStrat_.reward_;
        changeInCost
            -= static_cast<CostT>(threadData.activeScheduleData_.newViolations_.size()) * threadData.rewardPenaltyStrat_.penalty_;

#ifdef KL_DEBUG_COST_CHECK
        {
            const CostT computedCost = commCostF_.ComputeScheduleCostTest();
            const CostT expectedCost = threadData.activeScheduleData_.cost_ + changeInCost;

            if (std::abs(computedCost - expectedCost) > 0.00001) {
                std::cout << "[GAIN DIVERGENCE] node=" << move.node_ << " (" << move.fromProc_ << ",S" << move.fromStep_
                          << ") -> (" << move.toProc_ << ",S" << move.toStep_ << ")" << std::endl;
                std::cout << "  gain=" << move.gain_ << " computedCost=" << computedCost << " expectedCost=" << expectedCost
                          << " error=" << (computedCost - expectedCost) << std::endl;
                // Per-step comm changes
                for (unsigned s = 0; s < numStepsCheck; s++) {
                    double csAfter = commCostF_.StepMaxComm(s);
                    if (std::abs(csAfter - perStepCommBefore[s]) > 0.00001) {
                        std::cout << "    step " << s << ": commMax " << perStepCommBefore[s] << " -> " << csAfter
                                  << " (delta=" << (csAfter - perStepCommBefore[s]) << ")" << std::endl;
                    }
                }
                // Per-step work changes
                for (unsigned s = 0; s < numStepsCheck; s++) {
                    double wsAfter = activeSchedule_.GetStepMaxWork(s);
                    if (std::abs(wsAfter - perStepWorkBefore[s]) > 0.00001) {
                        std::cout << "    step " << s << ": maxWork " << perStepWorkBefore[s] << " -> " << wsAfter
                                  << " (delta=" << (wsAfter - perStepWorkBefore[s]) << ")" << std::endl;
                    }
                }
                // Parent/child context
                const auto &graph = instance_->GetComputationalDag();
                std::cout << "  Parents:";
                for (const auto &p : graph.Parents(move.node_)) {
                    std::cout << " " << p << "(P" << activeSchedule_.AssignedProcessor(p) << ",S"
                              << activeSchedule_.AssignedSuperstep(p) << ",cw=" << graph.VertexCommWeight(p) << ")";
                }
                std::cout << "\n  Children:";
                for (const auto &c : graph.Children(move.node_)) {
                    std::cout << " " << c << "(P" << activeSchedule_.AssignedProcessor(c) << ",S"
                              << activeSchedule_.AssignedSuperstep(c) << ")";
                }
                std::cout << "\n  Node commW=" << graph.VertexCommWeight(move.node_) << std::endl;
                // std::abort();
            }
        }
#endif

#ifdef KL_DEBUG
        std::cout << "penalty: " << threadData.rewardPenaltyStrat_.penalty_
                  << " num violations: " << threadData.activeScheduleData_.currentViolations_.size()
                  << " num new violations: " << threadData.activeScheduleData_.newViolations_.size()
                  << ", num resolved violations: " << threadData.activeScheduleData_.resolvedViolations_.size()
                  << ", reward: " << threadData.rewardPenaltyStrat_.reward_ << std::endl;
        std::cout << "apply move, previous cost: " << threadData.activeScheduleData_.cost_
                  << ", new cost: " << threadData.activeScheduleData_.cost_ + changeInCost << ", "
                  << (threadData.activeScheduleData_.feasible_ ? "feasible," : "infeasible,") << std::endl;
#endif

        threadData.activeScheduleData_.UpdateCost(changeInCost);

        return changeInCost;
    }

    // --- Violation handling ---

    enum class ViolationAction { Continue, Break, Proceed };

    ViolationAction HandleViolationBacktracking(unsigned &violationRemovedCount,
                                                unsigned &resetCounter,
                                                unsigned &innerIter,
                                                bool iterInitalFeasible,
                                                ThreadSearchContext &threadData) {
        if (threadData.activeScheduleData_.currentViolations_.size() == 0) {
            return ViolationAction::Proceed;
        }

        if (threadData.activeScheduleData_.resolvedViolations_.size() > 0) {
            violationRemovedCount = 0;
            return ViolationAction::Proceed;
        }

        violationRemovedCount++;
        if (violationRemovedCount <= 3) {
            return ViolationAction::Proceed;
        }

        if (resetCounter < threadData.maxNoViolationsRemovedBacktrack_
            && ((not iterInitalFeasible) || (threadData.activeScheduleData_.cost_ < threadData.activeScheduleData_.bestCost_))) {
            threadData.affinityTable_.ResetNodeSelection();
            threadData.lockManager_.Clear();
            threadData.selectionStrategy_.SelectNodesViolations(threadData.affinityTable_,
                                                                threadData.activeScheduleData_.currentViolations_,
                                                                threadData.startStep_,
                                                                threadData.endStep_);
#ifdef KL_DEBUG
            std::cout << "Infeasible, and no violations resolved for 5 iterations, reset node selection" << std::endl;
#endif
            threadData.rewardPenaltyStrat_.InitRewardPenalty(
                static_cast<double>(threadData.activeScheduleData_.currentViolations_.size()));

            derived().ReinitializeMoveFinding(threadData);    // DISPATCH

            resetCounter++;
            innerIter++;
            return ViolationAction::Continue;
        }

#ifdef KL_DEBUG
        std::cout << "Infeasible, and no violations resolved for 5 iterations, end local search" << std::endl;
#endif
        return ViolationAction::Break;
    }

    // --- QuickMoves ---

    void RunQuickMoves(unsigned &innerIter,
                       ThreadSearchContext &threadData,
                       const CostT changeInCost,
                       const VertexType bestMoveNode) {
#ifdef KL_DEBUG
        std::cout << "Starting quick moves sequence." << std::endl;
#endif
        innerIter++;

        const size_t numAppliedMoves = threadData.activeScheduleData_.appliedMoves_.size() - 1;
        const CostT savedCost = threadData.activeScheduleData_.cost_ - changeInCost;

        std::unordered_set<VertexType> localLock;
        localLock.insert(bestMoveNode);
        std::vector<VertexType> quickMovesStack;
        quickMovesStack.reserve(10 + threadData.activeScheduleData_.newViolations_.size() * 2);

        for (const auto &keyValuePair : threadData.activeScheduleData_.newViolations_) {
            const auto &key = keyValuePair.first;
            const unsigned keyStep = activeSchedule_.AssignedSuperstep(key);
            if (keyStep >= threadData.startStep_ && keyStep <= threadData.endStep_) {
                quickMovesStack.push_back(key);
            }
        }

        while (quickMovesStack.size() > 0) {
            auto nextNodeToMove = quickMovesStack.back();
            quickMovesStack.pop_back();

            threadData.rewardPenaltyStrat_.InitRewardPenalty(
                static_cast<double>(threadData.activeScheduleData_.currentViolations_.size()) + 1.0);
            ComputeNodeAffinities(nextNodeToMove, threadData.localAffinityTable_, threadData);
            KlMove bestQuickMove = ComputeBestMove<true>(nextNodeToMove, threadData.localAffinityTable_, threadData);

            localLock.insert(nextNodeToMove);
            if (bestQuickMove.gain_ <= std::numeric_limits<CostT>::lowest()) {
                continue;
            }

#ifdef KL_DEBUG
            std::cout << " >>> move node " << bestQuickMove.node_ << " with gain " << bestQuickMove.gain_
                      << ", from proc|step: " << bestQuickMove.fromProc_ << "|" << bestQuickMove.fromStep_
                      << " to: " << bestQuickMove.toProc_ << "|" << bestQuickMove.toStep_ << std::endl;
#endif

            ApplyMove(bestQuickMove, threadData);
            DebugCostCheck(threadData, "RunQuickMoves_after_ApplyMove");
            innerIter++;

            if (threadData.activeScheduleData_.newViolations_.size() > 0) {
                bool abort = false;

                for (const auto &keyValuePair : threadData.activeScheduleData_.newViolations_) {
                    const auto &key = keyValuePair.first;
                    if (localLock.find(key) != localLock.end()) {
                        abort = true;
                        break;
                    }
                    const unsigned keyStep = activeSchedule_.AssignedSuperstep(key);
                    if (keyStep >= threadData.startStep_ && keyStep <= threadData.endStep_) {
                        quickMovesStack.push_back(key);
                    }
                }

                if (abort) {
                    break;
                }

            } else if (threadData.activeScheduleData_.feasible_) {
                break;
            }
        }

        if (!threadData.activeScheduleData_.feasible_) {
            activeSchedule_.RevertScheduleToBound(numAppliedMoves,
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

        threadData.affinityTable_.Trim();
        threadData.rewardPenaltyStrat_.InitRewardPenalty(1.0);
        derived().ReinitializeMoveFinding(threadData);    // DISPATCH
    }

    // --- ResolveViolations ---

    void ResolveViolations(ThreadSearchContext &threadData) {
        auto &currentViolations = threadData.activeScheduleData_.currentViolations_;
        unsigned numViolations = static_cast<unsigned>(currentViolations.size());
        if (numViolations > 0) {
#ifdef KL_DEBUG_1
            std::cout << "thread " << threadData.threadId_ << ", Starting preresolving violations with " << numViolations
                      << " initial violations" << std::endl;
#endif
            threadData.rewardPenaltyStrat_.InitRewardPenalty(static_cast<double>(numViolations) + 1.0);
            std::unordered_set<VertexType> localLock;
            unsigned numIter = 0;
            const unsigned minIter = numViolations / 4;

            // Shuffled vector for O(1) sequential access without replacement.
            // Stale entries (resolved by earlier moves) are skipped via the
            // authoritative currentViolations set.  New violations created by
            // ApplyMove are appended so they become reachable without a rebuild.
            using EdgeType = typename std::decay_t<decltype(currentViolations)>::value_type;
            std::vector<EdgeType> violationVec(currentViolations.begin(), currentViolations.end());
            std::shuffle(violationVec.begin(), violationVec.end(), gen_);
            size_t vecIdx = 0;

            while (not currentViolations.empty()) {
                // Rebuild lazily when the vector is exhausted
                if (vecIdx >= violationVec.size()) {
                    violationVec.assign(currentViolations.begin(), currentViolations.end());
                    std::shuffle(violationVec.begin(), violationVec.end(), gen_);
                    vecIdx = 0;
                    if (violationVec.empty()) {
                        break;
                    }
                }

                // Skip stale entries that were resolved by earlier moves
                const auto &nextEdge = violationVec[vecIdx++];
                if (currentViolations.find(nextEdge) == currentViolations.end()) {
                    continue;
                }
                const VertexType sourceV = Source(nextEdge, *graph_);
                const VertexType targetV = Target(nextEdge, *graph_);

                // Thread safety: treat out-of-range vertices as locked (unmovable).
                const unsigned sourceStep = activeSchedule_.AssignedSuperstep(sourceV);
                const unsigned targetStep = activeSchedule_.AssignedSuperstep(targetV);
                const bool sourceOutOfRange = sourceStep < threadData.startStep_ || sourceStep > threadData.endStep_;
                const bool targetOutOfRange = targetStep < threadData.startStep_ || targetStep > threadData.endStep_;

                const bool sourceLocked = sourceOutOfRange || localLock.find(sourceV) != localLock.end();
                const bool targetLocked = targetOutOfRange || localLock.find(targetV) != localLock.end();

                if (sourceLocked && targetLocked) {
#ifdef KL_DEBUG_1
                    std::cout << "source, target locked" << std::endl;
#endif
                    break;
                }

                KlMove bestMove;
                if (sourceLocked || targetLocked) {
                    const VertexType node = sourceLocked ? targetV : sourceV;
                    ComputeNodeAffinities(node, threadData.localAffinityTable_, threadData);
                    bestMove = ComputeBestMove<true>(node, threadData.localAffinityTable_, threadData);
                } else {
                    ComputeNodeAffinities(sourceV, threadData.localAffinityTable_, threadData);
                    KlMove bestSourceVMove = ComputeBestMove<true>(sourceV, threadData.localAffinityTable_, threadData);
                    ComputeNodeAffinities(targetV, threadData.localAffinityTable_, threadData);
                    KlMove bestTargetVMove = ComputeBestMove<true>(targetV, threadData.localAffinityTable_, threadData);
                    bestMove = bestTargetVMove.gain_ > bestSourceVMove.gain_ ? std::move(bestTargetVMove)
                                                                             : std::move(bestSourceVMove);
                }

                localLock.insert(bestMove.node_);
                if (bestMove.gain_ <= std::numeric_limits<CostT>::lowest()) {
                    continue;
                }

                ApplyMove(bestMove, threadData);
                DebugCostCheck(threadData, "ResolveViolations_after_ApplyMove");
                threadData.affinityTable_.Insert(bestMove.node_);
#ifdef KL_DEBUG_1
                std::cout << "move node " << bestMove.node_ << " with gain " << bestMove.gain_
                          << ", from proc|step: " << bestMove.fromProc_ << "|" << bestMove.fromStep_
                          << " to: " << bestMove.toProc_ << "|" << bestMove.toStep_ << std::endl;
#endif
                const unsigned newNumViolations = static_cast<unsigned>(currentViolations.size());
                if (newNumViolations == 0) {
                    break;
                }

                if (threadData.activeScheduleData_.newViolations_.size() > 0) {
                    for (const auto &vertexEdgePair : threadData.activeScheduleData_.newViolations_) {
                        const auto &vertex = vertexEdgePair.first;
                        const unsigned vertexStep = activeSchedule_.AssignedSuperstep(vertex);
                        if (vertexStep >= threadData.startStep_ && vertexStep <= threadData.endStep_) {
                            threadData.affinityTable_.Insert(vertex);
                        }
                        // Append new violation edges so the scan can reach them
                        violationVec.push_back(vertexEdgePair.second);
                    }
                }

                const double gain = static_cast<double>(numViolations) - static_cast<double>(newNumViolations);
                numViolations = newNumViolations;
                UpdateAvgGain(gain, numIter++, threadData.averageGain_);
#ifdef KL_DEBUG_1
                std::cout << "thread " << threadData.threadId_ << ",  preresolving violations with " << numViolations
                          << " violations, " << numIter << " #iterations, " << threadData.averageGain_ << " average gain"
                          << std::endl;
#endif
                if (numIter > minIter && threadData.averageGain_ < 0.0) {
                    break;
                }
            }
            threadData.averageGain_ = 0.0;
        }
    }

    // --- DebugCostCheck ---

    inline void DebugCostCheck([[maybe_unused]] const ThreadSearchContext &threadData,
                               [[maybe_unused]] const char *label = "unknown") {
#ifdef KL_DEBUG_COST_CHECK
        activeSchedule_.GetVectorSchedule().numberOfSupersteps_ = threadDataVec_[0].NumSteps();
        const CostT computedCost = commCostF_.ComputeScheduleCostTest();
        const CostT currentCost = threadData.activeScheduleData_.cost_;
        if (std::abs(computedCost - currentCost) > 0.00001) {
            const size_t numViolations = threadData.activeScheduleData_.currentViolations_.size();
            std::cout << "\n[COST DIVERGENCE at " << label << "] "
                      << "computed=" << computedCost << " tracked=" << currentCost << " error=" << (computedCost - currentCost)
                      << " violations=" << numViolations
                      << " feasible=" << (threadData.activeScheduleData_.feasible_ ? "true" : "false") << std::endl;
            // std::abort();
        }
        if constexpr (ActiveScheduleT::useMemoryConstraint_) {
            if (not activeSchedule_.memoryConstraint_.SatisfiedMemoryConstraint()) {
                std::cout << "[" << label << "] memory constraint not satisfied" << std::endl;
                // std::abort();
            }
        }
#endif
    }

    // --- BlockedEdgeStrategy ---

    inline bool BlockedEdgeStrategy(VertexType node, std::vector<VertexType> &unlockNodes, ThreadSearchContext &threadData) {
        if (threadData.unlockEdgeBacktrackCounter_ > 1) {
            for (const auto vertexEdgePair : threadData.activeScheduleData_.newViolations_) {
                const auto &e = vertexEdgePair.second;
                const auto sourceV = Source(e, *graph_);
                const auto targetV = Target(e, *graph_);

                if (node == sourceV && threadData.lockManager_.IsLocked(targetV)) {
                    const unsigned targetStep = activeSchedule_.AssignedSuperstep(targetV);
                    if (targetStep >= threadData.startStep_ && targetStep <= threadData.endStep_) {
                        unlockNodes.push_back(targetV);
                    }
                } else if (node == targetV && threadData.lockManager_.IsLocked(sourceV)) {
                    const unsigned sourceStep = activeSchedule_.AssignedSuperstep(sourceV);
                    if (sourceStep >= threadData.startStep_ && sourceStep <= threadData.endStep_) {
                        unlockNodes.push_back(sourceV);
                    }
                }
            }
#ifdef KL_DEBUG
            std::cout << "Nodes of violated edge locked, backtrack counter: " << threadData.unlockEdgeBacktrackCounter_
                      << std::endl;
#endif
            threadData.unlockEdgeBacktrackCounter_--;
            return true;
        } else {
#ifdef KL_DEBUG
            std::cout << "Nodes of violated edge locked, end local search" << std::endl;
#endif
            return false;
        }
    }

    // --- AdjustLocalSearchParameters ---

    inline void AdjustLocalSearchParameters(unsigned outerIter, unsigned noImpCounter, ThreadSearchContext &threadData) {
        if (noImpCounter >= threadData.noImprovementIterationsReducePenalty_
            && threadData.rewardPenaltyStrat_.initialPenalty_ > 1.0) {
            threadData.rewardPenaltyStrat_.initialPenalty_
                = static_cast<CostT>(std::floor(std::sqrt(threadData.rewardPenaltyStrat_.initialPenalty_)));
            threadData.unlockEdgeBacktrackCounterReset_ += 1;
            threadData.noImprovementIterationsReducePenalty_ += 15;
#ifdef KL_DEBUG_1
            std::cout << "thread " << threadData.threadId_ << ", no improvement for "
                      << threadData.noImprovementIterationsReducePenalty_ << " iterations, reducing initial penalty to "
                      << threadData.rewardPenaltyStrat_.initialPenalty_ << std::endl;
#endif
        }

        if (parameters_.tryRemoveStepAfterNumOuterIterations_ > 0
            && ((outerIter + 1) % parameters_.tryRemoveStepAfterNumOuterIterations_) == 0) {
            threadData.stepSelectionEpochCounter_ = 0;
#ifdef KL_DEBUG
            std::cout << "reset remove epoc counter after " << outerIter << " iterations." << std::endl;
#endif
        }

        if (noImpCounter >= threadData.noImprovementIterationsIncreaseInnerIter_) {
            threadData.minInnerIter_ = static_cast<unsigned>(std::ceil(threadData.minInnerIter_ * 2.2));
            threadData.noImprovementIterationsIncreaseInnerIter_ += 20;
#ifdef KL_DEBUG_1
            std::cout << "thread " << threadData.threadId_ << ", no improvement for "
                      << threadData.noImprovementIterationsIncreaseInnerIter_ << " iterations, increasing min inner iter to "
                      << threadData.minInnerIter_ << std::endl;
#endif
        }
    }

    // --- Other shared methods ---

    bool IsLocalSearchBlocked(ThreadSearchContext &threadData);
    bool OtherThreadsFinished(const unsigned threadId);
    void SetParameters(VertexIdxT<GraphT> numNodes);
    void ResetInnerSearchStructures(ThreadSearchContext &threadData);
    void InitializeDatastructures(BspSchedule<GraphT> &schedule);
    void CleanupDatastructures();
    void UpdateAvgGain(const CostT gain, const unsigned numIter, double &averageGain);

    void SelectActiveNodes(ThreadSearchContext &threadData);
    bool CheckRemoveSuperstep(unsigned step);
    bool SelectNodesCheckRemoveSuperstep(unsigned &step, ThreadSearchContext &threadData);
    bool ScatterNodesSuperstep(unsigned step, ThreadSearchContext &threadData);
    void SynchronizeActiveSchedule(const unsigned numThreads);

    // --- The inner loop — shared skeleton with 3 dispatch points ---

    void RunLocalSearch(ThreadSearchContext &threadData) {
#ifdef KL_DEBUG_1
        std::cout << "thread " << threadData.threadId_
                  << ", start local search, initial schedule cost: " << threadData.activeScheduleData_.cost_ << " with "
                  << threadData.NumSteps() << " supersteps." << std::endl;
#endif
        std::vector<VertexType> newNodes;
        std::vector<VertexType> unlockNodes;

        DebugCostCheck(threadData, "RunLocalSearch_entry");

        const auto startTime = std::chrono::high_resolution_clock::now();

        unsigned noImprovementIterCounter = 0;
        unsigned outerIter = 0;

        for (; outerIter < parameters_.maxOuterIterations_; outerIter++) {
            DebugCostCheck(threadData, "outer_loop_start");
            CostT initialInnerIterCost = threadData.activeScheduleData_.cost_;

            ResetInnerSearchStructures(threadData);
#ifdef KL_DEBUG_1
            const unsigned numStepsBeforeSelect = threadData.endStep_;
#endif
            SelectActiveNodes(threadData);
            DebugCostCheck(threadData, "after_SelectActiveNodes");
            threadData.rewardPenaltyStrat_.InitRewardPenalty(
                static_cast<double>(threadData.activeScheduleData_.currentViolations_.size()) + 1.0);

            // DISPATCH: initialize move-finding (heap or scan)
            derived().ReinitializeMoveFinding(threadData);

            unsigned innerIter = 0;
            unsigned violationRemovedCount = 0;
            unsigned resetCounter = 0;
            bool iterInitalFeasible = threadData.activeScheduleData_.feasible_;

#ifdef KL_DEBUG
            std::cout << "------ start inner loop ------" << std::endl;
            std::cout << "initial node selection: {";
            for (size_t i = 0; i < threadData.affinityTable_.size(); ++i) {
                std::cout << threadData.affinityTable_.GetSelectedNodes()[i] << ", ";
            }
            std::cout << "}" << std::endl;
#endif
#ifdef KL_DEBUG_1
            if (not iterInitalFeasible) {
                std::cout << "initial solution not feasible, num violations: "
                          << threadData.activeScheduleData_.currentViolations_.size()
                          << ". Penalty: " << threadData.rewardPenaltyStrat_.penalty_
                          << ", reward: " << threadData.rewardPenaltyStrat_.reward_ << std::endl;
            }
#endif
            DebugCostCheck(threadData, "before_inner_loop");

            while (innerIter < threadData.maxInnerIterations_) {
                // DISPATCH: get best move
                KlMove bestMove = derived().GetBestMove(threadData);
                if (bestMove.gain_ <= std::numeric_limits<CostT>::lowest()) {
                    break;
                }
                UpdateAvgGain(bestMove.gain_, innerIter, threadData.averageGain_);
#ifdef KL_DEBUG
                std::cout << " >>> move node " << bestMove.node_ << " with gain " << bestMove.gain_
                          << ", from proc|step: " << bestMove.fromProc_ << "|" << bestMove.fromStep_ << " to: " << bestMove.toProc_
                          << "|" << bestMove.toStep_ << ",avg gain: " << threadData.averageGain_ << std::endl;
#endif
                if (innerIter > threadData.minInnerIter_ && threadData.averageGain_ < 0.0) {
#ifdef KL_DEBUG
                    std::cout << "Negative average gain: " << threadData.averageGain_ << ", end local search" << std::endl;
#endif
                    break;
                }

#ifdef KL_DEBUG
                if (not activeSchedule_.GetInstance().IsCompatible(bestMove.node_, bestMove.toProc_)) {
                    std::cout << "move to incompatibe node" << std::endl;
                }
#endif

                const auto prevWorkData = activeSchedule_.GetPreMoveWorkData(bestMove);
                const CostT changeInCost = ApplyMove(bestMove, threadData);
                DebugCostCheck(threadData, "after_ApplyMove");

                if constexpr (enableQuickMoves_) {
                    if (iterInitalFeasible && threadData.activeScheduleData_.newViolations_.size() > 0) {
                        RunQuickMoves(innerIter, threadData, changeInCost, bestMove.node_);
                        DebugCostCheck(threadData, "after_RunQuickMoves");
                        continue;    // ReinitializeMoveFinding already called inside
                    }
                }

                {
                    const auto violationAction = HandleViolationBacktracking(
                        violationRemovedCount, resetCounter, innerIter, iterInitalFeasible, threadData);
                    if (violationAction == ViolationAction::Continue) {
                        continue;    // ReinitializeMoveFinding already called inside
                    } else if (violationAction == ViolationAction::Break) {
                        break;
                    }
                }

                if (IsLocalSearchBlocked(threadData)) {
                    if (not BlockedEdgeStrategy(bestMove.node_, unlockNodes, threadData)) {
                        break;
                    }
                }

                threadData.affinityTable_.Trim();

                // DISPATCH: post-move update (affinity updates + heap/scan maintenance)
                // Note: unlockNodes are still LOCKED here, so UpdateNodeCommAffinity
                // skips them (important: avoids duplicate insertion into heap).
                derived().PostMoveUpdate(bestMove, threadData, newNodes, unlockNodes, prevWorkData);

                newNodes.clear();
                unlockNodes.clear();

                DebugCostCheck(threadData, "after_PostMoveUpdate");
                innerIter++;
            }

#ifdef KL_DEBUG
            std::cout << "--- end inner loop after " << innerIter << " inner iterations, outer iteraion " << outerIter << "/"
                      << parameters_.maxOuterIterations_ << ", current cost: " << threadData.activeScheduleData_.cost_ << ", "
                      << (threadData.activeScheduleData_.feasible_ ? "feasible" : "infeasible") << std::endl;
#endif
#ifdef KL_DEBUG_1
            const unsigned numStepsBeforeRevert = threadData.endStep_;
#endif
            activeSchedule_.RevertToBestSchedule(
                commCostF_, threadData.activeScheduleData_, threadData.startStep_, threadData.endStep_);
#ifdef KL_DEBUG_1
            if (numStepsBeforeSelect != numStepsBeforeRevert) {
                if (numStepsBeforeRevert == threadData.endStep_) {
                    std::cout << "thread " << threadData.threadId_ << ", removing step " << threadData.stepToRemove_
                              << " succeded " << std::endl;
                } else {
                    std::cout << "thread " << threadData.threadId_ << ", removing step " << threadData.stepToRemove_ << " failed "
                              << std::endl;
                }
            }
#endif
            DebugCostCheck(threadData, "after_RevertToBestSchedule");

            if (computeWithTimeLimit_) {
                auto finishTime = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(finishTime - startTime).count();
                if (duration > ImprovementScheduler<GraphT>::timeLimitSeconds_) {
                    break;
                }
            }

            if (OtherThreadsFinished(threadData.threadId_)) {
#ifdef KL_DEBUG_1
                std::cout << "thread " << threadData.threadId_ << ", other threads finished, end local search" << std::endl;
#endif
                break;
            }

            if (initialInnerIterCost <= threadData.activeScheduleData_.cost_) {
                noImprovementIterCounter++;

                if (noImprovementIterCounter >= parameters_.maxNoImprovementIterations_) {
#ifdef KL_DEBUG_1
                    std::cout << "thread " << threadData.threadId_ << ", no improvement for "
                              << parameters_.maxNoImprovementIterations_ << " iterations, end local search" << std::endl;
#endif
                    break;
                }
            } else {
                noImprovementIterCounter = 0;
            }

            AdjustLocalSearchParameters(outerIter, noImprovementIterCounter, threadData);
        }

#ifdef KL_DEBUG_1
        std::cout << "thread " << threadData.threadId_ << ", local search end after " << outerIter
                  << " outer iterations, current cost: " << threadData.activeScheduleData_.cost_ << " with "
                  << threadData.NumSteps() << " supersteps, vs serial cost " << activeSchedule_.GetTotalWorkWeight() << "."
                  << std::endl;
#endif
        threadFinishedVec_[threadData.threadId_] = true;
    }

  public:
    KlImproverBase() : ImprovementScheduler<GraphT>() {
        std::random_device rd;
        gen_ = std::mt19937(rd());
    }

    explicit KlImproverBase(unsigned seed) : ImprovementScheduler<GraphT>() { gen_ = std::mt19937(seed); }

    virtual ~KlImproverBase() = default;

    virtual ReturnStatus ImproveSchedule(BspSchedule<GraphT> &schedule) override {
        if (schedule.GetInstance().NumberOfProcessors() < 2) {
            return ReturnStatus::BEST_FOUND;
        }

        const unsigned numThreads = 1;

        threadDataVec_.resize(numThreads);
        threadFinishedVec_.assign(numThreads, true);

        SetParameters(schedule.GetInstance().NumberOfVertices());
        InitializeDatastructures(schedule);
        const CostT initialCost = activeSchedule_.GetCost();
        const unsigned numSteps = schedule.NumberOfSupersteps();

        SetStartStep(0, threadDataVec_[0]);
        threadDataVec_[0].endStep_ = (numSteps > 0) ? numSteps - 1 : 0;

        auto &threadData = this->threadDataVec_[0];
        threadData.activeScheduleData_.InitializeCost(activeSchedule_.GetCost());
        threadData.selectionStrategy_.Setup(threadData.startStep_, threadData.endStep_);
        RunLocalSearch(threadData);

        SynchronizeActiveSchedule(numThreads);

        if (initialCost > activeSchedule_.GetCost()) {
            activeSchedule_.WriteSchedule(schedule);
            CleanupDatastructures();
            return ReturnStatus::OSP_SUCCESS;
        } else {
            CleanupDatastructures();
            return ReturnStatus::BEST_FOUND;
        }
    }

    virtual ReturnStatus ImproveScheduleWithTimeLimit(BspSchedule<GraphT> &schedule) override {
        computeWithTimeLimit_ = true;
        return ImproveSchedule(schedule);
    }

    virtual void SetTimeQualityParameter(const double timeQuality) { this->parameters_.timeQuality_ = timeQuality; }

    virtual void SetSuperstepRemoveStrengthParameter(const double superstepRemoveStrength) {
        this->parameters_.superstepRemoveStrength_ = superstepRemoveStrength;
    }

    virtual std::string GetScheduleName() const { return "kl_improver_" + commCostF_.Name(); }
};

// =============================================================================
// OUT-OF-LINE DEFINITIONS — Base
// =============================================================================

template <typename Derived, typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImproverBase<Derived, GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::SetParameters(
    VertexIdxT<GraphT> numNodes) {
    const unsigned logNumNodes = (numNodes > 1) ? static_cast<unsigned>(std::log(numNodes)) : 1;

    parameters_.maxOuterIterations_
        = static_cast<unsigned>(std::sqrt(numNodes) * (parameters_.timeQuality_ * 10.0) / parameters_.numParallelLoops_);

    parameters_.maxNoViolationsRemovedBacktrackReset_ = parameters_.timeQuality_ < 0.75  ? 1
                                                        : parameters_.timeQuality_ < 1.0 ? 2
                                                                                         : 3;

    parameters_.maxNoViolationsRemovedBacktrackForRemoveStepReset_
        = 3 + static_cast<unsigned>(parameters_.superstepRemoveStrength_ * 7);
    parameters_.nodeMaxStepSelectionEpochs_ = parameters_.superstepRemoveStrength_ < 0.75  ? 1
                                              : parameters_.superstepRemoveStrength_ < 1.0 ? 2
                                                                                           : 3;
    parameters_.removeStepEpocs_ = static_cast<unsigned>(parameters_.superstepRemoveStrength_ * 4.0);

    parameters_.minInnerIterReset_ = static_cast<unsigned>(logNumNodes + logNumNodes * (1.0 + parameters_.timeQuality_));

    if (parameters_.removeStepEpocs_ > 0) {
        parameters_.tryRemoveStepAfterNumOuterIterations_ = parameters_.maxOuterIterations_ / parameters_.removeStepEpocs_;
    } else {
        parameters_.tryRemoveStepAfterNumOuterIterations_ = parameters_.maxOuterIterations_ + 1;
    }

    unsigned i = 0;
    for (auto &thread : threadDataVec_) {
        thread.threadId_ = i++;
        thread.selectionStrategy_.selectionThreshold_
            = static_cast<std::size_t>(std::ceil(parameters_.timeQuality_ * 10 * logNumNodes + logNumNodes));
    }

#ifdef KL_DEBUG_1
    std::cout << "kl set parameter, number of nodes: " << numNodes << std::endl;
    std::cout << "max outer iterations: " << parameters_.maxOuterIterations_ << std::endl;
    std::cout << "max inner iterations: " << parameters_.maxInnerIterationsReset_ << std::endl;
    std::cout << "selction threshold: " << threadDataVec_[0].selectionStrategy_.selectionThreshold_ << std::endl;
    std::cout << "remove step epocs: " << parameters_.removeStepEpocs_ << std::endl;
    std::cout << "try remove step after num outer iterations: " << parameters_.tryRemoveStepAfterNumOuterIterations_ << std::endl;
    std::cout << "number of parallel loops: " << parameters_.numParallelLoops_ << std::endl;
#endif
}

template <typename Derived, typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
bool KlImproverBase<Derived, GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::OtherThreadsFinished(
    const unsigned threadId) {
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

template <typename Derived, typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
bool KlImproverBase<Derived, GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::IsLocalSearchBlocked(
    ThreadSearchContext &threadData) {
    for (const auto &pair : threadData.activeScheduleData_.newViolations_) {
        if (threadData.lockManager_.IsLocked(pair.first)) {
            return true;
        }
    }
    return false;
}

template <typename Derived, typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImproverBase<Derived, GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::ResetInnerSearchStructures(
    ThreadSearchContext &threadData) {
    threadData.unlockEdgeBacktrackCounter_ = threadData.unlockEdgeBacktrackCounterReset_;
    threadData.maxInnerIterations_ = parameters_.maxInnerIterationsReset_;
    threadData.maxNoViolationsRemovedBacktrack_ = parameters_.maxNoViolationsRemovedBacktrackReset_;
    threadData.averageGain_ = 0.0;
    threadData.affinityTable_.ResetNodeSelection();
    threadData.lockManager_.Clear();
    // Variant-specific state (heap, scan best) is reset via ReinitializeMoveFinding
}

template <typename Derived, typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImproverBase<Derived, GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::InitializeDatastructures(
    BspSchedule<GraphT> &schedule) {
    inputSchedule_ = &schedule;
    instance_ = &schedule.GetInstance();
    graph_ = &instance_->GetComputationalDag();

    activeSchedule_.Initialize(schedule);

    procRange_.Initialize(*instance_);
    commCostF_.Initialize(activeSchedule_, procRange_);
    const CostT initialCost = commCostF_.ComputeScheduleCost();
    activeSchedule_.SetCost(initialCost);

    for (auto &tData : threadDataVec_) {
        tData.affinityTable_.Initialize(activeSchedule_, tData.selectionStrategy_.selectionThreshold_);
        tData.lockManager_.Initialize(graph_->NumVertices());
        tData.rewardPenaltyStrat_.Initialize(
            activeSchedule_, commCostF_.GetMaxCommWeightMultiplied(), activeSchedule_.GetMaxWorkWeight());
        tData.selectionStrategy_.Initialize(activeSchedule_, gen_, tData.startStep_, tData.endStep_);

        tData.localAffinityTable_.resize(instance_->NumberOfProcessors());
        for (unsigned i = 0; i < instance_->NumberOfProcessors(); ++i) {
            tData.localAffinityTable_[i].resize(windowRange_);
        }
    }

    // Initialize variant-specific per-thread data
    derived().InitializeVariantData();
}

template <typename Derived, typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImproverBase<Derived, GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::CleanupDatastructures() {
    threadDataVec_.clear();
    activeSchedule_.Clear();
}

template <typename Derived, typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImproverBase<Derived, GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::UpdateAvgGain(
    const CostT gain, const unsigned numIter, double &averageGain) {
    averageGain = static_cast<double>((averageGain * numIter + gain)) / (numIter + 1.0);
}

template <typename Derived, typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImproverBase<Derived, GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::SelectActiveNodes(
    ThreadSearchContext &threadData) {
    if (SelectNodesCheckRemoveSuperstep(threadData.stepToRemove_, threadData)) {
        const unsigned r = threadData.stepToRemove_;

        // MaxBSP: capture work[r+1] BEFORE SwapEmptyStepFwd rearranges work data.
        // The removed step is empty (work[r]=0). Only two cost terms are affected:
        //   term at s=r  (removed)  and  term at s=r+1 (merged with r-1's comm).
        CostT maxBspWorkNext = 0;
        bool maxBspHasNext = false;
        if constexpr (CommCostFunctionT::coupledWorkComm_) {
            maxBspHasNext = (r < threadData.endStep_);
            if (maxBspHasNext) {
                maxBspWorkNext = static_cast<CostT>(activeSchedule_.GetStepMaxWork(r + 1));
            }
        }

        activeSchedule_.SwapEmptyStepFwd(r, threadData.endStep_);
        const unsigned oldEndStep = threadData.endStep_;

        // Capture comm costs BEFORE the comm swap loop.
        // SwapEmptyStepFwd only moves work data; comm indices are still original.
        CostT oldCommR_1 = 0;    // comm[r-1] — used by both BSP and MaxBSP
        CostT oldCommR = 0;      // comm[r]   — BSP: removedStepMaxComm; MaxBSP: paired with step r+1
        if (r > 0) {
            oldCommR_1 = static_cast<CostT>(commCostF_.StepMaxComm(r - 1));
        }
        oldCommR = static_cast<CostT>(commCostF_.StepMaxComm(r));

        for (unsigned i = r; i < threadData.endStep_; i++) {
            commCostF_.SwapCommSteps(i, i + 1);
        }
        threadData.endStep_--;
        commCostF_.UpdateLambdaAfterStepRemoval(r, threadData.startStep_, oldEndStep);
        commCostF_.FixupSendRecvAfterStepRemoval(r, oldEndStep);

        const CostT syncCost = static_cast<CostT>(instance_->SynchronisationCosts());
        threadData.activeScheduleData_.appliedMoves_.push_back(KlMove::MakeRemoveStep(r, syncCost));

        if (activeSchedule_.GetStaleness() > 1) {
            activeSchedule_.UpdateViolationsAfterStepRemoval(r, threadData.activeScheduleData_);
        }

        if constexpr (CommCostFunctionT::coupledWorkComm_) {
            // MaxBSP analytical delta.
            // cost = work[0] + Σ_{s≥1} max(work[s], comm[s-1]·g) + (S-1)·L
            // Removing empty step r touches only terms at s=r and s=r+1.
            const CostT g = static_cast<CostT>(instance_->CommunicationCosts());

            // Old term at s=r: max(0, comm[r-1]·g) = comm[r-1]·g  (or 0 if r=0)
            const CostT oldRemovedTerm = (r > 0) ? oldCommR_1 * g : CostT(0);

            // Old term at s=r+1: max(work[r+1], comm[r]·g)  (or 0 if r+1 out of range)
            const CostT oldNextTerm = maxBspHasNext ? std::max(maxBspWorkNext, oldCommR * g) : CostT(0);

            // New merged term: step r+1 takes r's slot, paired with merged comm at r-1
            CostT newMergedTerm = CostT(0);
            if (maxBspHasNext) {
                if (r == 0) {
                    // New step 0 is pure work (no paired comm from below)
                    newMergedTerm = maxBspWorkNext;
                } else {
                    // comm[r-1] now holds merged comm after FixupSendRecv
                    const CostT newCommR_1 = static_cast<CostT>(commCostF_.StepMaxComm(r - 1));
                    newMergedTerm = std::max(maxBspWorkNext, newCommR_1 * g);
                }
            }

            threadData.activeScheduleData_.UpdateCost(newMergedTerm - oldRemovedTerm - oldNextTerm - syncCost);
        } else {
            // BSP/Total: analytical comm delta from the merge.
            const CostT commCostMultiplier = static_cast<CostT>(instance_->CommunicationCosts());
            CostT commDelta = -oldCommR * commCostMultiplier;
            if (r > 0) {
                const CostT newCommR_1 = static_cast<CostT>(commCostF_.StepMaxComm(r - 1));
                commDelta += (newCommR_1 - oldCommR_1) * commCostMultiplier;
            }
            threadData.activeScheduleData_.UpdateCost(static_cast<CostT>(-1.0 * syncCost) + commDelta);
        }
        DebugCostCheck(threadData, "SelectActiveNodes_after_StepRemoval_UpdateCost");

        if constexpr (enablePreresolvingViolations_) {
            ResolveViolations(threadData);
            DebugCostCheck(threadData, "SelectActiveNodes_after_ResolveViolations");
        }

        if (threadData.activeScheduleData_.currentViolations_.size() > parameters_.initialViolationThreshold_) {
            activeSchedule_.RevertToBestSchedule(
                commCostF_, threadData.activeScheduleData_, threadData.startStep_, threadData.endStep_);
            DebugCostCheck(threadData, "SelectActiveNodes_after_Revert_tooManyViolations");
        } else {
            threadData.unlockEdgeBacktrackCounter_
                = static_cast<unsigned>(threadData.activeScheduleData_.currentViolations_.size());
            threadData.maxInnerIterations_
                = std::max(threadData.unlockEdgeBacktrackCounter_ * 5u, parameters_.maxInnerIterationsReset_);
            threadData.maxNoViolationsRemovedBacktrack_ = parameters_.maxNoViolationsRemovedBacktrackForRemoveStepReset_;
#ifdef KL_DEBUG_1
            std::cout << "thread " << threadData.threadId_ << ", Trying to remove step " << threadData.stepToRemove_ << std::endl;
#endif
            return;
        }
    }
    threadData.selectionStrategy_.SelectActiveNodes(threadData.affinityTable_, threadData.startStep_, threadData.endStep_);
}

template <typename Derived, typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
bool KlImproverBase<Derived, GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::CheckRemoveSuperstep(unsigned step) {
    if (activeSchedule_.NumSteps() < 2) {
        return false;
    }
    if (activeSchedule_.GetStepMaxWork(step) < instance_->SynchronisationCosts()) {
        return true;
    }
    return false;
}

template <typename Derived, typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
bool KlImproverBase<Derived, GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::SelectNodesCheckRemoveSuperstep(
    unsigned &stepToRemove, ThreadSearchContext &threadData) {
    if (threadData.stepSelectionEpochCounter_ >= parameters_.nodeMaxStepSelectionEpochs_ || threadData.NumSteps() < 3) {
        return false;
    }

    for (stepToRemove = threadData.stepSelectionCounter_; stepToRemove <= threadData.endStep_; stepToRemove++) {
        assert(stepToRemove >= threadData.startStep_ && stepToRemove <= threadData.endStep_);

        // In MT mode, skip boundary steps — they buffer against gap zones.
        // Removing a boundary step scatters nodes toward the frozen gap,
        // easily creating cross-boundary violations that can't be resolved.
        // Exception: the globally first/last step has no adjacent thread.
        // Use originalEndStep_ because endStep_ shrinks during step removal.
        if (stepToRemove == threadData.startStep_ && threadData.startStep_ != 0) {
            continue;
        }
        if (stepToRemove == threadData.originalEndStep_ && threadData.originalEndStep_ != activeSchedule_.NumSteps() - 1) {
            continue;
        }

#ifdef KL_DEBUG
        std::cout << "Checking to remove step " << stepToRemove << "/" << threadData.endStep_ << std::endl;
#endif
        if (CheckRemoveSuperstep(stepToRemove)) {
#ifdef KL_DEBUG
            std::cout << "Checking to scatter step " << stepToRemove << "/" << threadData.endStep_ << std::endl;
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

template <typename Derived, typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
bool KlImproverBase<Derived, GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::ScatterNodesSuperstep(
    unsigned step, ThreadSearchContext &threadData) {
    assert(step <= threadData.endStep_ && threadData.startStep_ <= step);
    bool abort = false;

    for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
        const std::vector<VertexType> stepProcNodeVec(
            activeSchedule_.GetSetSchedule().GetProcessorStepVertices()[step][proc].begin(),
            activeSchedule_.GetSetSchedule().GetProcessorStepVertices()[step][proc].end());
        for (const auto &node : stepProcNodeVec) {
            threadData.rewardPenaltyStrat_.InitRewardPenalty(
                static_cast<double>(threadData.activeScheduleData_.currentViolations_.size()) + 1.0);
            ComputeNodeAffinities(node, threadData.localAffinityTable_, threadData);
            KlMove bestMove = ComputeBestMove<false>(node, threadData.localAffinityTable_, threadData);

            if (bestMove.gain_ <= std::numeric_limits<double>::lowest()) {
                abort = true;
                break;
            }

            ApplyMove(bestMove, threadData);
            DebugCostCheck(threadData, "ScatterNodes_after_ApplyMove_immediate");
            if (threadData.activeScheduleData_.currentViolations_.size() > parameters_.abortScatterNodesViolationThreshold_) {
                abort = true;
                break;
            }

            threadData.affinityTable_.Insert(node);
            if (threadData.activeScheduleData_.newViolations_.size() > 0) {
                for (const auto &vertexEdgePair : threadData.activeScheduleData_.newViolations_) {
                    const auto &vertex = vertexEdgePair.first;
                    const unsigned vertexStep = activeSchedule_.AssignedSuperstep(vertex);
                    if (vertexStep >= threadData.startStep_ && vertexStep <= threadData.endStep_) {
                        threadData.affinityTable_.Insert(vertex);
                    }
                }
            }

#ifdef KL_DEBUG
            std::cout << "move node " << bestMove.node_ << " with gain " << bestMove.gain_
                      << ", from proc|step: " << bestMove.fromProc_ << "|" << bestMove.fromStep_ << " to: " << bestMove.toProc_
                      << "|" << bestMove.toStep_ << std::endl;
#endif
            DebugCostCheck(threadData, "ScatterNodes_after_ApplyMove");
        }

        if (abort) {
            break;
        }
    }

    if (abort) {
        activeSchedule_.RevertToBestSchedule(
            commCostF_, threadData.activeScheduleData_, threadData.startStep_, threadData.endStep_);
        threadData.affinityTable_.ResetNodeSelection();
        return false;
    }
    return true;
}

template <typename Derived, typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImproverBase<Derived, GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::SynchronizeActiveSchedule(
    const unsigned numThreads) {
    if (numThreads == 1) {
        activeSchedule_.SetCost(threadDataVec_[0].activeScheduleData_.cost_);
        activeSchedule_.GetVectorSchedule().numberOfSupersteps_ = threadDataVec_[0].NumSteps();
        return;
    }

    // Compact the schedule by closing gaps created by step removals.
    //
    // Layout before compaction (example with 2 threads, gap=2, T0 removed 2 steps):
    //   T0 active: [0..6]  empty: [7,8]  gap: [9,10]  T1 active: [11..18]  empty: [19,20]
    //
    // We must preserve: T0 content | gap steps | T1 content | gap steps | T2 content ...
    // The gap steps contain frozen nodes whose relative position between
    // thread ranges is essential for staleness feasibility.

    unsigned writeCursor = threadDataVec_[0].endStep_ + 1;
    for (unsigned i = 1; i < numThreads; ++i) {
        auto &thread = threadDataVec_[i];

        // 1. Place the gap steps between thread i-1 and thread i.
        //    Gap occupies [prevThread.originalEndStep_+1 .. thread.startStep_-1].
        const unsigned gapStart = threadDataVec_[i - 1].originalEndStep_ + 1;
        const unsigned gapEnd = thread.startStep_;    // exclusive
        for (unsigned g = gapStart; g < gapEnd; ++g) {
            if (g != writeCursor) {
                activeSchedule_.SwapSteps(g, writeCursor);
            }
            writeCursor++;
        }

        // 2. Place thread i's active steps.
        if (thread.startStep_ <= thread.endStep_) {
            for (unsigned j = thread.startStep_; j <= thread.endStep_; ++j) {
                if (j != writeCursor) {
                    activeSchedule_.SwapSteps(j, writeCursor);
                }
                writeCursor++;
            }
        }
    }
    activeSchedule_.GetVectorSchedule().numberOfSupersteps_ = writeCursor;
    const CostT newCost = commCostF_.ComputeScheduleCost();
    activeSchedule_.SetCost(newCost);
}

}    // namespace osp
