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
#include <iostream>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "comm_cost_policies.hpp"
#include "generic_lambda_container.hpp"
#include "lambda_container.hpp"
#include "osp/bsp/model/BspInstance.hpp"

namespace osp {

template <typename CommWeightT>
struct PreMoveCommData {
    struct StepInfo {
        CommWeightT maxComm_;
        CommWeightT secondMaxComm_;
        unsigned maxCommCount_;
    };

    std::unordered_map<unsigned, StepInfo> stepData_;

    PreMoveCommData() = default;

    void AddStep(unsigned step, CommWeightT max, CommWeightT second, unsigned count) { stepData_[step] = {max, second, count}; }

    bool GetStep(unsigned step, StepInfo &info) const {
        auto it = stepData_.find(step);
        if (it != stepData_.end()) {
            info = it->second;
            return true;
        }
        return false;
    }
};

template <typename GraphT, typename CostT, typename KlActiveScheduleT, typename CommPolicy = EagerCommCostPolicy>
struct MaxCommDatastructure {
    using CommWeightT = VCommwT<GraphT>;
    using VertexType = VertexIdxT<GraphT>;
    using KlMove = KlMoveStruct<CostT, VertexType>;

    const BspInstance<GraphT> *instance_;
    const KlActiveScheduleT *activeSchedule_;

    std::vector<std::vector<CommWeightT>> stepProcSend_;
    std::vector<std::vector<CommWeightT>> stepProcReceive_;

    // Caches for fast cost calculation (Global Max/Second Max per step)
    std::vector<CommWeightT> stepMaxCommCache_;
    std::vector<CommWeightT> stepSecondMaxCommCache_;
    std::vector<unsigned> stepMaxCommCountCache_;

    CommWeightT maxCommWeight_ = 0;

    // Select the appropriate container type based on the policy's ValueType
    using ContainerType = typename std::conditional<std::is_same<typename CommPolicy::ValueType, unsigned>::value,
                                                    LambdaVectorContainer<VertexType>,
                                                    GenericLambdaVectorContainer<VertexType, typename CommPolicy::ValueType>>::type;

    ContainerType nodeLambdaMap_;

    // Optimization: Scratchpad for update_datastructure_after_move to avoid allocations
    std::vector<unsigned> affectedStepsList_;
    std::vector<bool> stepIsAffected_;

    inline CommWeightT StepProcSend(unsigned step, unsigned proc) const { return stepProcSend_[step][proc]; }

    inline CommWeightT &StepProcSend(unsigned step, unsigned proc) { return stepProcSend_[step][proc]; }

    inline CommWeightT StepProcReceive(unsigned step, unsigned proc) const { return stepProcReceive_[step][proc]; }

    inline CommWeightT &StepProcReceive(unsigned step, unsigned proc) { return stepProcReceive_[step][proc]; }

    inline CommWeightT StepMaxComm(unsigned step) const { return stepMaxCommCache_[step]; }

    inline CommWeightT StepSecondMaxComm(unsigned step) const { return stepSecondMaxCommCache_[step]; }

    inline unsigned StepMaxCommCount(unsigned step) const { return stepMaxCommCountCache_[step]; }

    /// Returns the list of steps where send/recv arrays were modified by the last
    /// UpdateDatastructureAfterMove call.  For Lazy/Buffered policies these include
    /// the min(child_steps)-1 steps where communication is actually placed, which
    /// may differ from the node positions used by the higher-level changedSteps set.
    inline const std::vector<unsigned> &GetLastAffectedCommSteps() const { return affectedStepsList_; }

    inline void Initialize(KlActiveScheduleT &klSched) {
        activeSchedule_ = &klSched;
        instance_ = &activeSchedule_->GetInstance();
        const unsigned numSteps = activeSchedule_->NumSteps();
        const unsigned numProcs = instance_->NumberOfProcessors();
        maxCommWeight_ = 0;

        stepProcSend_.assign(numSteps, std::vector<CommWeightT>(numProcs, 0));
        stepProcReceive_.assign(numSteps, std::vector<CommWeightT>(numProcs, 0));

        stepMaxCommCache_.assign(numSteps, 0);
        stepSecondMaxCommCache_.assign(numSteps, 0);
        stepMaxCommCountCache_.assign(numSteps, 0);

        nodeLambdaMap_.Initialize(instance_->GetComputationalDag().NumVertices(), numProcs);

        // Initialize scratchpad
        stepIsAffected_.assign(numSteps, false);
        affectedStepsList_.reserve(numSteps);
    }

    inline void Clear() {
        stepProcSend_.clear();
        stepProcReceive_.clear();
        stepMaxCommCache_.clear();
        stepSecondMaxCommCache_.clear();
        stepMaxCommCountCache_.clear();
        nodeLambdaMap_.clear();
        affectedStepsList_.clear();
        stepIsAffected_.clear();
    }

    inline void ArrangeSuperstepCommData(const unsigned step) {
        CommWeightT maxSend = 0;
        CommWeightT secondMaxSend = 0;
        unsigned maxSendCount = 0;

        const auto &sends = stepProcSend_[step];
        for (const auto val : sends) {
            if (val > maxSend) {
                secondMaxSend = maxSend;
                maxSend = val;
                maxSendCount = 1;
            } else if (val == maxSend) {
                maxSendCount++;
            } else if (val > secondMaxSend) {
                secondMaxSend = val;
            }
        }

        CommWeightT maxReceive = 0;
        CommWeightT secondMaxReceive = 0;
        unsigned maxReceiveCount = 0;

        const auto &receives = stepProcReceive_[step];
        for (const auto val : receives) {
            if (val > maxReceive) {
                secondMaxReceive = maxReceive;
                maxReceive = val;
                maxReceiveCount = 1;
            } else if (val == maxReceive) {
                maxReceiveCount++;
            } else if (val > secondMaxReceive) {
                secondMaxReceive = val;
            }
        }

        const CommWeightT globalMax = std::max(maxSend, maxReceive);
        stepMaxCommCache_[step] = globalMax;

        unsigned globalCount = 0;
        if (maxSend == globalMax) {
            globalCount += maxSendCount;
        }
        if (maxReceive == globalMax) {
            globalCount += maxReceiveCount;
        }
        stepMaxCommCountCache_[step] = globalCount;

        CommWeightT candSend = (maxSend == globalMax) ? secondMaxSend : maxSend;
        CommWeightT candRecv = (maxReceive == globalMax) ? secondMaxReceive : maxReceive;

        stepSecondMaxCommCache_[step] = std::max(candSend, candRecv);
    }

    void RecomputeMaxSendReceive(unsigned step) { ArrangeSuperstepCommData(step); }

    inline PreMoveCommData<CommWeightT> GetPreMoveCommData(const KlMove &move) {
        PreMoveCommData<CommWeightT> data;
        std::unordered_set<unsigned> affectedSteps;

        affectedSteps.insert(move.fromStep_);
        affectedSteps.insert(move.toStep_);

        const auto &graph = instance_->GetComputationalDag();

        for (const auto &parent : graph.Parents(move.node_)) {
            affectedSteps.insert(activeSchedule_->AssignedSuperstep(parent));
        }

        for (unsigned step : affectedSteps) {
            data.AddStep(step, StepMaxComm(step), StepSecondMaxComm(step), StepMaxCommCount(step));
        }

        return data;
    }

    void UpdateDatastructureAfterMove(const KlMove &move, unsigned, unsigned) {
        const auto &graph = instance_->GetComputationalDag();

        // Prepare Scratchpad (Avoids Allocations) ---
        for (unsigned step : affectedStepsList_) {
            if (step < stepIsAffected_.size()) {
                stepIsAffected_[step] = false;
            }
        }
        affectedStepsList_.clear();

        auto MarkStep = [&](unsigned step) {
            if (step < stepIsAffected_.size() && !stepIsAffected_[step]) {
                stepIsAffected_[step] = true;
                affectedStepsList_.push_back(step);
            }
        };

        const VertexType node = move.node_;
        const unsigned fromStep = move.fromStep_;
        const unsigned toStep = move.toStep_;
        const unsigned fromProc = move.fromProc_;
        const unsigned toProc = move.toProc_;
        const CommWeightT commWNode = graph.VertexCommWeight(node);

        // Handle Node Movement (Outgoing Edges: Node -> Children)

        if (fromStep != toStep) {
            // Case 1: Node changes Step
            for (const auto [proc, val] : nodeLambdaMap_.IterateProcEntries(node)) {
                // A. Remove Old (Sender: fromProc, Receiver: proc)
                if (proc != fromProc) {
                    const CommWeightT cost = commWNode * instance_->SendCosts(fromProc, proc);
                    if (cost > 0) {
                        CommPolicy::RemoveOutgoingComm(*this, cost, fromStep, fromProc, proc, val, MarkStep);
                    }
                }

                // B. Add New (Sender: toProc, Receiver: proc)
                if (proc != toProc) {
                    const CommWeightT cost = commWNode * instance_->SendCosts(toProc, proc);
                    if (cost > 0) {
                        CommPolicy::AddOutgoingComm(*this, cost, toStep, toProc, proc, val, MarkStep);
                    }
                }
            }

        } else if (fromProc != toProc) {
            // Case 2: Node stays in same Step, but changes Processor

            for (const auto [proc, val] : nodeLambdaMap_.IterateProcEntries(node)) {
                // Remove Old (Sender: fromProc, Receiver: proc)
                if (proc != fromProc) {
                    const CommWeightT cost = commWNode * instance_->SendCosts(fromProc, proc);
                    if (cost > 0) {
                        CommPolicy::RemoveOutgoingComm(*this, cost, fromStep, fromProc, proc, val, MarkStep);
                    }
                }

                // Add New (Sender: toProc, Receiver: proc)
                if (proc != toProc) {
                    const CommWeightT cost = commWNode * instance_->SendCosts(toProc, proc);
                    if (cost > 0) {
                        CommPolicy::AddOutgoingComm(*this, cost, fromStep, toProc, proc, val, MarkStep);
                    }
                }
            }
        }

        // Update Parents' Outgoing Communication (Parents → Node)

        for (const auto &parent : graph.Parents(node)) {
            const unsigned parentStep = activeSchedule_->AssignedSuperstep(parent);
            // Fast boundary check
            if (parentStep >= stepProcSend_.size()) {
                continue;
            }

            const unsigned parentProc = activeSchedule_->AssignedProcessor(parent);
            const CommWeightT commWParent = graph.VertexCommWeight(parent);

            auto &val = nodeLambdaMap_.GetProcEntry(parent, fromProc);
            const bool removedFromProc = CommPolicy::RemoveChild(val, fromStep);

            // 1. Handle Removal from fromProc
            if (removedFromProc) {
                if (fromProc != parentProc) {
                    const CommWeightT cost = commWParent * instance_->SendCosts(parentProc, fromProc);
                    if (cost > 0) {
                        CommPolicy::UnattributeCommunication(
                            *this, cost, parentStep, parentProc, fromProc, fromStep, val, MarkStep);
                    }
                }
            }

            auto &valTo = nodeLambdaMap_.GetProcEntry(parent, toProc);
            const bool addedToProc = CommPolicy::AddChild(valTo, toStep);

            // 2. Handle Addition to toProc
            if (addedToProc) {
                if (toProc != parentProc) {
                    const CommWeightT cost = commWParent * instance_->SendCosts(parentProc, toProc);
                    if (cost > 0) {
                        CommPolicy::AttributeCommunication(*this, cost, parentStep, parentProc, toProc, toStep, valTo, MarkStep);
                    }
                }
            }
        }

        // Re-arrange Affected Steps
        for (unsigned step : affectedStepsList_) {
            ArrangeSuperstepCommData(step);
        }
    }

    /// Validate incremental send/recv AND lambda against from-scratch recomputation.
    /// Saves current state, recomputes everything from scratch, compares cell-by-cell
    /// and lambda entry-by-entry for the moved node and its parents.
    /// On divergence: prints full diagnostic and aborts.
    /// After return (if no divergence), the datastructure is in the from-scratch
    /// (correct) state so each subsequent move is independently testable.
    /// Returns true if incremental state matches from-scratch computation.
    bool ValidateCommDs(unsigned moveCounter, const KlMove &move) {
        const unsigned numSteps = stepProcSend_.size();
        const unsigned numProcs = numSteps > 0 ? stepProcSend_[0].size() : 0;
        const auto &graph = instance_->GetComputationalDag();
        const auto &vecSched = activeSchedule_->GetVectorSchedule();

        // Determine active step range (max step any node is assigned to)
        unsigned activeEndStep = 0;
        for (const auto &u : graph.Vertices()) {
            activeEndStep = std::max(activeEndStep, vecSched.AssignedSuperstep(u));
        }
        const unsigned compareSteps = activeEndStep + 1;

        // 1. Snapshot the incremental state (send/recv + lambda)
        auto savedSend = stepProcSend_;
        auto savedRecv = stepProcReceive_;

        // Snapshot lambda for moved node and its parents
        using LambdaEntry = typename CommPolicy::ValueType;

        struct NodeLambdaSnapshot {
            VertexType node;
            std::vector<LambdaEntry> procEntries;    // indexed by proc
        };

        auto snapshotLambda = [&](VertexType n) -> NodeLambdaSnapshot {
            NodeLambdaSnapshot snap;
            snap.node = n;
            snap.procEntries.resize(numProcs);
            for (unsigned p = 0; p < numProcs; p++) {
                snap.procEntries[p] = nodeLambdaMap_.GetProcEntry(n, p);
            }
            return snap;
        };

        // Snapshot moved node's lambda (its children grouped by proc)
        NodeLambdaSnapshot movedNodeLambda = snapshotLambda(move.node_);

        // Snapshot all parents' lambdas (parents of the moved node)
        std::vector<NodeLambdaSnapshot> parentLambdas;
        for (const auto &parent : graph.Parents(move.node_)) {
            parentLambdas.push_back(snapshotLambda(parent));
        }

        // 2. Recompute from scratch (resets nodeLambdaMap_ + stepProcSend_/Receive_)
        ComputeCommDatastructures(0, activeEndStep);

        // 3. Compare send/recv cell-by-cell (active range only)
        bool sendRecvOk = true;
        for (unsigned s = 0; s < compareSteps; s++) {
            for (unsigned p = 0; p < numProcs; p++) {
                if (savedSend[s][p] != stepProcSend_[s][p] || savedRecv[s][p] != stepProcReceive_[s][p]) {
                    sendRecvOk = false;
                }
            }
        }

        // 4. Compare lambda for moved node and parents
        auto compareLambdaEntry = [](const LambdaEntry &a, const LambdaEntry &b) -> bool {
            if constexpr (std::is_same_v<LambdaEntry, unsigned>) {
                return a == b;
            } else {
                // For vector<unsigned>: compare as sorted multisets
                if (a.size() != b.size()) {
                    return false;
                }
                auto sa = a;
                std::sort(sa.begin(), sa.end());
                auto sb = b;
                std::sort(sb.begin(), sb.end());
                return sa == sb;
            }
        };

        auto printLambdaEntry = [](const LambdaEntry &entry, std::ostream &os) {
            if constexpr (std::is_same_v<LambdaEntry, unsigned>) {
                os << entry;
            } else {
                os << "[";
                for (size_t i = 0; i < entry.size(); i++) {
                    if (i > 0) {
                        os << ",";
                    }
                    os << entry[i];
                }
                os << "]";
            }
        };

        bool lambdaOk = true;
        // Check moved node
        for (unsigned p = 0; p < numProcs; p++) {
            auto freshEntry = nodeLambdaMap_.GetProcEntry(move.node_, p);
            if (!compareLambdaEntry(movedNodeLambda.procEntries[p], freshEntry)) {
                lambdaOk = false;
            }
        }
        // Check parents
        for (const auto &parentSnap : parentLambdas) {
            for (unsigned p = 0; p < numProcs; p++) {
                auto freshEntry = nodeLambdaMap_.GetProcEntry(parentSnap.node, p);
                if (!compareLambdaEntry(parentSnap.procEntries[p], freshEntry)) {
                    lambdaOk = false;
                }
            }
        }

        if (!sendRecvOk || !lambdaOk) {
            std::cout << "\n========== COMM DS DIVERGENCE at move #" << moveCounter << ": node=" << move.node_ << " ("
                      << move.fromProc_ << "," << move.fromStep_ << ")"
                      << "->(" << move.toProc_ << "," << move.toStep_ << ")"
                      << "  sendRecvOk=" << sendRecvOk << " lambdaOk=" << lambdaOk << " activeSteps=0.." << activeEndStep
                      << " arraySize=" << numSteps << " ==========" << std::endl;

            // Print send/recv divergences
            if (!sendRecvOk) {
                std::cout << "  --- Send/Recv mismatches ---" << std::endl;
                for (unsigned s = 0; s < compareSteps; s++) {
                    for (unsigned p = 0; p < numProcs; p++) {
                        if (savedSend[s][p] != stepProcSend_[s][p]) {
                            std::cout << "  SEND[s=" << s << "][p=" << p << "]"
                                      << "  inc=" << savedSend[s][p] << "  fresh=" << stepProcSend_[s][p] << "  delta="
                                      << (static_cast<long long>(savedSend[s][p]) - static_cast<long long>(stepProcSend_[s][p]))
                                      << std::endl;
                        }
                        if (savedRecv[s][p] != stepProcReceive_[s][p]) {
                            std::cout << "  RECV[s=" << s << "][p=" << p << "]"
                                      << "  inc=" << savedRecv[s][p] << "  fresh=" << stepProcReceive_[s][p] << "  delta="
                                      << (static_cast<long long>(savedRecv[s][p]) - static_cast<long long>(stepProcReceive_[s][p]))
                                      << std::endl;
                        }
                    }
                }
            }

            // Print lambda divergences for moved node
            std::cout << "  --- Lambda for moved node " << move.node_ << " (now at P" << vecSched.AssignedProcessor(move.node_)
                      << ",S" << vecSched.AssignedSuperstep(move.node_) << ") ---" << std::endl;
            for (unsigned p = 0; p < numProcs; p++) {
                auto freshEntry = nodeLambdaMap_.GetProcEntry(move.node_, p);
                bool differs = !compareLambdaEntry(movedNodeLambda.procEntries[p], freshEntry);
                if (differs || CommPolicy::HasEntry(movedNodeLambda.procEntries[p]) || CommPolicy::HasEntry(freshEntry)) {
                    std::cout << "    lambda[" << move.node_ << "][P" << p << "]" << (differs ? " *** DIFFERS *** " : "  ");
                    std::cout << "inc=";
                    printLambdaEntry(movedNodeLambda.procEntries[p], std::cout);
                    std::cout << "  fresh=";
                    printLambdaEntry(freshEntry, std::cout);
                    std::cout << std::endl;
                }
            }

            // Print lambda divergences for parents
            for (const auto &parentSnap : parentLambdas) {
                unsigned parentProc = vecSched.AssignedProcessor(parentSnap.node);
                unsigned parentStep = vecSched.AssignedSuperstep(parentSnap.node);
                std::cout << "  --- Lambda for parent " << parentSnap.node << " (at P" << parentProc << ",S" << parentStep
                          << ") ---" << std::endl;
                for (unsigned p = 0; p < numProcs; p++) {
                    auto freshEntry = nodeLambdaMap_.GetProcEntry(parentSnap.node, p);
                    bool differs = !compareLambdaEntry(parentSnap.procEntries[p], freshEntry);
                    if (differs || CommPolicy::HasEntry(parentSnap.procEntries[p]) || CommPolicy::HasEntry(freshEntry)) {
                        std::cout << "    lambda[" << parentSnap.node << "][P" << p << "]"
                                  << (differs ? " *** DIFFERS *** " : "  ");
                        std::cout << "inc=";
                        printLambdaEntry(parentSnap.procEntries[p], std::cout);
                        std::cout << "  fresh=";
                        printLambdaEntry(freshEntry, std::cout);
                        std::cout << std::endl;
                    }
                }
            }

            // Print children of moved node for context
            std::cout << "  --- Children of moved node " << move.node_ << " ---" << std::endl;
            for (const auto &child : graph.Children(move.node_)) {
                std::cout << "    child " << child << " -> (P" << vecSched.AssignedProcessor(child) << ", S"
                          << vecSched.AssignedSuperstep(child) << ")" << std::endl;
            }

            // Print parents of moved node for context
            std::cout << "  --- Parents of moved node " << move.node_ << " ---" << std::endl;
            for (const auto &parent : graph.Parents(move.node_)) {
                std::cout << "    parent " << parent << " -> (P" << vecSched.AssignedProcessor(parent) << ", S"
                          << vecSched.AssignedSuperstep(parent) << ")" << std::endl;
            }

            std::cout << "  ========== END DIVERGENCE ==========\n" << std::endl;
        }

        // State is now fresh-computed (correct) — next incremental update starts clean
        return sendRecvOk && lambdaOk;
    }

    /// After a step removal (bubble empty step forward from removedStep to endStep),
    /// all nodes that were at step S > removedStep are now at step S-1.
    /// Update lambda entries to match the new step numbering.
    /// Only needed for policies that store step values (Lazy, Buffered).
    void UpdateLambdaAfterStepRemoval(unsigned removedStep) {
        if constexpr (std::is_same_v<typename CommPolicy::ValueType, std::vector<unsigned>>) {
            for (auto &nodeEntries : nodeLambdaMap_.nodeLambdaVec_) {
                for (auto &procEntry : nodeEntries) {
                    for (auto &step : procEntry) {
                        if (step > removedStep) {
                            step--;
                        }
                    }
                }
            }
        }
    }

    /// After step removal, the SwapSteps loop bubbled the removed step's data to
    /// oldEndStep. For Lazy/Buffered, that empty step can carry comm data (from
    /// min(child_steps)-1 attribution). Merge it into removedStep-1 (the new
    /// correct position) but KEEP oldEndStep as a backup so that insertion can
    /// reverse this merge in O(P).
    /// Call AFTER the SwapSteps loop and AFTER UpdateLambdaAfterStepRemoval.
    void FixupSendRecvAfterStepRemoval(unsigned removedStep, unsigned oldEndStep) {
        if constexpr (std::is_same_v<typename CommPolicy::ValueType, std::vector<unsigned>>) {
            if (removedStep == 0) {
                // No position -1 to merge into. Clear backup — data is lost.
                // Insertion at step 0 will need a full recompute (extremely rare).
                std::fill(stepProcSend_[oldEndStep].begin(), stepProcSend_[oldEndStep].end(), 0);
                std::fill(stepProcReceive_[oldEndStep].begin(), stepProcReceive_[oldEndStep].end(), 0);
                ArrangeSuperstepCommData(oldEndStep);
                return;
            }
            const unsigned numProcs = stepProcSend_[0].size();
            for (unsigned p = 0; p < numProcs; p++) {
                stepProcSend_[removedStep - 1][p] += stepProcSend_[oldEndStep][p];
                stepProcReceive_[removedStep - 1][p] += stepProcReceive_[oldEndStep][p];
                // DON'T clear oldEndStep — it serves as backup for insertion reversal
            }
            ArrangeSuperstepCommData(removedStep - 1);
        }
    }

    /// After a step insertion (reverting a removal), increment all lambda entries
    /// >= insertedStep to match the new step numbering.
    void UpdateLambdaAfterStepInsertion(unsigned insertedStep) {
        if constexpr (std::is_same_v<typename CommPolicy::ValueType, std::vector<unsigned>>) {
            for (auto &nodeEntries : nodeLambdaMap_.nodeLambdaVec_) {
                for (auto &procEntry : nodeEntries) {
                    for (auto &step : procEntry) {
                        if (step >= insertedStep) {
                            step++;
                        }
                    }
                }
            }
        }
    }

    /// After step insertion, the SwapSteps loop brought the backup data from
    /// beyond endStep back to position insertedStep. Position insertedStep-1
    /// still has the merged data (original + backup from removal). Subtract
    /// the backup to un-merge, restoring both positions to their correct state.
    /// Call AFTER the SwapSteps loop and AFTER UpdateLambdaAfterStepInsertion.
    void FixupSendRecvAfterStepInsertion(unsigned insertedStep, unsigned startStep, unsigned endStep) {
        if constexpr (std::is_same_v<typename CommPolicy::ValueType, std::vector<unsigned>>) {
            if (insertedStep == 0) {
                // Backup was lost during removal (cleared). Full recompute needed.
                ComputeCommDatastructures(startStep, endStep);
                return;
            }
            const unsigned numProcs = stepProcSend_[0].size();
            for (unsigned p = 0; p < numProcs; p++) {
                stepProcSend_[insertedStep - 1][p] -= stepProcSend_[insertedStep][p];
                stepProcReceive_[insertedStep - 1][p] -= stepProcReceive_[insertedStep][p];
            }
            ArrangeSuperstepCommData(insertedStep - 1);
            ArrangeSuperstepCommData(insertedStep);
        }
    }

    void SwapSteps(const unsigned step1, const unsigned step2) {
        std::swap(stepProcSend_[step1], stepProcSend_[step2]);
        std::swap(stepProcReceive_[step1], stepProcReceive_[step2]);
        std::swap(stepMaxCommCache_[step1], stepMaxCommCache_[step2]);
        std::swap(stepSecondMaxCommCache_[step1], stepSecondMaxCommCache_[step2]);
        std::swap(stepMaxCommCountCache_[step1], stepMaxCommCountCache_[step2]);
    }

    void ResetSuperstep(unsigned step) {
        std::fill(stepProcSend_[step].begin(), stepProcSend_[step].end(), 0);
        std::fill(stepProcReceive_[step].begin(), stepProcReceive_[step].end(), 0);
        ArrangeSuperstepCommData(step);
    }

    void ComputeCommDatastructures(unsigned startStep, unsigned endStep) {
        for (unsigned step = startStep; step <= endStep; step++) {
            std::fill(stepProcSend_[step].begin(), stepProcSend_[step].end(), 0);
            std::fill(stepProcReceive_[step].begin(), stepProcReceive_[step].end(), 0);
        }

        const auto &vecSched = activeSchedule_->GetVectorSchedule();
        const auto &graph = instance_->GetComputationalDag();

        for (const auto &u : graph.Vertices()) {
            nodeLambdaMap_.ResetNode(u);
            const unsigned uProc = vecSched.AssignedProcessor(u);
            const unsigned uStep = vecSched.AssignedSuperstep(u);
            const CommWeightT commW = graph.VertexCommWeight(u);
            maxCommWeight_ = std::max(maxCommWeight_, commW);

            for (const auto &v : graph.Children(u)) {
                const unsigned vProc = vecSched.AssignedProcessor(v);
                const unsigned vStep = vecSched.AssignedSuperstep(v);

                const CommWeightT commWSendCost = (uProc != vProc) ? commW * instance_->SendCosts(uProc, vProc) : 0;

                auto &val = nodeLambdaMap_.GetProcEntry(u, vProc);
                if (CommPolicy::AddChild(val, vStep)) {
                    if (uProc != vProc && commWSendCost > 0) {
                        CommPolicy::AttributeCommunication(*this, commWSendCost, uStep, uProc, vProc, vStep, val, [](unsigned) {});
                    }
                }
            }
        }

        for (unsigned step = startStep; step <= endStep; step++) {
            if (step >= stepProcSend_.size()) {
                continue;
            }
            ArrangeSuperstepCommData(step);
        }
    }
};

}    // namespace osp
