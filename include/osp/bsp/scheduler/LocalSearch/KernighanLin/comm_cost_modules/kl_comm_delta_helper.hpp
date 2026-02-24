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
#include <vector>

#include "FastDeltaTacker.hpp"

namespace osp {

// =========================================================================
// CommDeltaScratchData — thread-local scratchpad for incremental
// per-step, per-processor send/recv delta tracking.
//
// Shared by KlBspCommCostFunction and KlMaxBspCommCostFunction.
// =========================================================================

template <typename CommWeightT>
struct CommDeltaScratchData {
    std::vector<FastDeltaTracker<CommWeightT>> sendDeltas_;
    std::vector<FastDeltaTracker<CommWeightT>> recvDeltas_;

    std::vector<unsigned> activeSteps_;
    std::vector<bool> stepIsActive_;

    unsigned lastNumProcs_ = 0;

    void Init(unsigned nSteps, unsigned nProcs) {
        if (sendDeltas_.size() < nSteps) {
            const size_t oldSize = sendDeltas_.size();
            sendDeltas_.resize(nSteps);
            recvDeltas_.resize(nSteps);
            stepIsActive_.resize(nSteps, false);
            activeSteps_.reserve(nSteps);

            // Initialize only the newly-added trackers
            for (size_t i = oldSize; i < nSteps; ++i) {
                sendDeltas_[i].Initialize(nProcs);
                recvDeltas_[i].Initialize(nProcs);
            }

            // If nProcs also changed, reinitialize existing trackers
            if (nProcs != lastNumProcs_) {
                for (size_t i = 0; i < oldSize; ++i) {
                    sendDeltas_[i].Initialize(nProcs);
                    recvDeltas_[i].Initialize(nProcs);
                }
            }
            lastNumProcs_ = nProcs;
        } else if (nProcs != lastNumProcs_) {
            for (auto &t : sendDeltas_) {
                t.Initialize(nProcs);
            }
            for (auto &t : recvDeltas_) {
                t.Initialize(nProcs);
            }
            lastNumProcs_ = nProcs;
        }
        // Common case (nSteps/nProcs unchanged): O(1) — trackers
        // are already clean after the previous ClearAll().
    }

    void ClearAll() {
        for (unsigned step : activeSteps_) {
            sendDeltas_[step].Clear();
            recvDeltas_[step].Clear();
            stepIsActive_[step] = false;
        }
        activeSteps_.clear();
    }

    void MarkActive(unsigned step) {
        if (!stepIsActive_[step]) {
            stepIsActive_[step] = true;
            activeSteps_.push_back(step);
        }
    }
};

// =========================================================================
// ComputeCommAffinityDeltas — shared scaffold for BSP & MaxBSP
//
// Handles Phase 1 (removal of node from current position) and Phase 2
// (per-candidate apply → evaluate → revert). The actual cost evaluation
// is delegated to `evaluatorFn(pTo, sToIdx, sTo, scratch) -> CostT`.
//
// BSP passes an evaluator that sums per-step ΔmaxComm × g.
// MaxBSP passes an evaluator that computes coupled max(work, comm×g).
// =========================================================================

template <typename GraphT,
          typename CostT,
          typename CommWeightT,
          typename CommPolicy,
          typename CommDsT,
          typename ScheduleT,
          typename InstanceT,
          typename ProcRangeT,
          typename AffinityTableT,
          typename EvaluatorFn>
void ComputeCommAffinityDeltas(VertexIdxT<GraphT> node,
                               AffinityTableT &affinityTableNode,
                               CommDsT &commDs,
                               ScheduleT &activeSchedule,
                               const GraphT &graph,
                               const InstanceT &instance,
                               const ProcRangeT &procRange,
                               unsigned nodeStep,
                               unsigned nodeProc,
                               unsigned nodeStartIdx,
                               unsigned windowBound,
                               unsigned numSteps,
                               unsigned windowSize,
                               unsigned startStep,
                               unsigned endStep,
                               EvaluatorFn &&evaluatorFn) {
    static thread_local CommDeltaScratchData<CommWeightT> scratch;
    scratch.Init(numSteps, instance.NumberOfProcessors());
    scratch.ClearAll();

    const CommWeightT commWNode = graph.VertexCommWeight(node);
    const auto &currentVecSchedule = activeSchedule.GetVectorSchedule();

    // --- Delta accumulation helpers ---
    // THREAD SAFETY: Only accumulate deltas for steps within [startStep, endStep].
    // Steps outside the thread's range may be concurrently modified by other threads.

    auto AddDelta = [&](bool isRecv, unsigned step, unsigned proc, CommWeightT val) {
        if (val == 0) {
            return;
        }
        if (step >= startStep && step <= endStep && step < numSteps) {
            scratch.MarkActive(step);
            if (isRecv) {
                scratch.recvDeltas_[step].Add(proc, val);
            } else {
                scratch.sendDeltas_[step].Add(proc, val);
            }
        }
    };

    struct DeltaAdapterT {
        decltype(AddDelta) &fn;

        void Add(bool isRecv, unsigned step, unsigned proc, CommWeightT v) { fn(isRecv, step, proc, v); }
    };

    struct NegDeltaAdapterT {
        decltype(AddDelta) &fn;

        void Add(bool isRecv, unsigned step, unsigned proc, CommWeightT v) { fn(isRecv, step, proc, -v); }
    };

    DeltaAdapterT deltaAdapter{AddDelta};
    NegDeltaAdapterT negDeltaAdapter{AddDelta};

    // ========== Phase 1: Remove Node from Current State ==========
    // (Invariant for all candidates)

    // Phase 1 Outgoing: node stops sending to children
    auto nodeLambdaEntries = commDs.nodeLambdaMap_.IterateProcEntries(node);

    for (const auto [proc, val] : nodeLambdaEntries) {
        if (proc != nodeProc && CommPolicy::HasEntry(val)) {
            const CommWeightT cost = commWNode * instance.SendCosts(nodeProc, proc);
            if (cost > 0) {
                int recvStep = CommPolicy::OutgoingRecvStep(nodeStep, val);
                int sendStep = CommPolicy::OutgoingSendStep(nodeStep, val);
                if (recvStep >= 0) {
                    AddDelta(true, static_cast<unsigned>(recvStep), proc, -cost);
                }
                if (sendStep >= 0) {
                    AddDelta(false, static_cast<unsigned>(sendStep), nodeProc, -cost);
                }
            }
        }
    }

    // Phase 1 Incoming: parents stop sending to node on nodeProc
    // THREAD SAFETY: Skip parents outside [startStep, endStep] — their lambda
    // entries may be concurrently modified by other threads.
    for (const auto &u : graph.Parents(node)) {
        const unsigned uStep = currentVecSchedule.AssignedSuperstep(u);
        if (uStep < startStep || uStep > endStep) {
            continue;
        }

        const unsigned uProc = activeSchedule.AssignedProcessor(u);
        const CommWeightT commWU = graph.VertexCommWeight(u);

        if (uProc != nodeProc) {
            const auto &lambdaVal = commDs.nodeLambdaMap_.GetProcEntry(u, nodeProc);
            if (CommPolicy::HasEntry(lambdaVal)) {
                const CommWeightT cost = commWU * instance.SendCosts(uProc, nodeProc);
                if (cost > 0) {
                    CommPolicy::CalculateDeltaRemove(lambdaVal, nodeStep, uStep, uProc, nodeProc, cost, deltaAdapter);
                }
            }
        }
    }

    // ========== Phase 2: Add Node to Each Candidate ==========

    // Helper: compute effective val after conceptually removing one instance of nodeStep.
    // Used for Phase 2A when pTo == nodeProc.
    auto ComputeEffectiveVal = [&](const typename CommPolicy::ValueType &val) -> typename CommPolicy::ValueType {
        if constexpr (std::is_same_v<typename CommPolicy::ValueType, unsigned>) {
            return val > 0 ? val - 1 : 0;
        } else {
            auto result = val;
            auto it = std::find(result.begin(), result.end(), nodeStep);
            if (it != result.end()) {
                result.erase(it);
            }
            return result;
        }
    };

    // Per-parent precomputed data for Phase 2A incoming additions
    struct ParentAddInfo {
        unsigned uProc;
        unsigned uStep;
        CommWeightT cost;
        typename CommPolicy::ValueType effectiveVal;
    };

    // Per-dest-proc precomputed data for Phase 2B outgoing
    struct OutgoingInfo {
        unsigned vProc;
        CommWeightT cost;
        int recvStep;
        int sendStep;
    };

    static thread_local std::vector<ParentAddInfo> parentAddInfos;
    static thread_local std::vector<OutgoingInfo> outgoingInfos;

    for (const unsigned pTo : procRange.CompatibleProcessorsVertex(node)) {
        // --- Precompute Phase 2A: parent effective vals ---
        // THREAD SAFETY: Skip parents outside [startStep, endStep].
        parentAddInfos.clear();
        for (const auto &u : graph.Parents(node)) {
            const unsigned uStep = currentVecSchedule.AssignedSuperstep(u);
            if (uStep < startStep || uStep > endStep) {
                continue;
            }

            const unsigned uProc = activeSchedule.AssignedProcessor(u);
            if (uProc == pTo) {
                continue;
            }

            const CommWeightT commWU = graph.VertexCommWeight(u);
            const CommWeightT cost = commWU * instance.SendCosts(uProc, pTo);
            if (cost <= 0) {
                continue;
            }

            const auto &valOnPTo = commDs.nodeLambdaMap_.GetProcEntry(u, pTo);
            typename CommPolicy::ValueType effectiveVal;
            if (pTo == nodeProc) {
                effectiveVal = ComputeEffectiveVal(valOnPTo);
            } else {
                effectiveVal = valOnPTo;
            }
            parentAddInfos.push_back({uProc, uStep, cost, std::move(effectiveVal)});
        }

        // --- Precompute Phase 2B: outgoing (node -> children) ---
        outgoingInfos.clear();
        for (const auto [vProc, val] : commDs.nodeLambdaMap_.IterateProcEntries(node)) {
            if (vProc != pTo && CommPolicy::HasEntry(val)) {
                const CommWeightT cost = commWNode * instance.SendCosts(pTo, vProc);
                if (cost > 0) {
                    int recvStep = -1;
                    int sendStep = -1;
                    if constexpr (!CommPolicy::outgoing_recv_at_parent_step) {
                        recvStep = CommPolicy::OutgoingRecvStep(0, val);
                    }
                    if constexpr (!CommPolicy::outgoing_send_at_parent_step) {
                        sendStep = CommPolicy::OutgoingSendStep(0, val);
                    }
                    outgoingInfos.push_back({vProc, cost, recvStep, sendStep});
                }
            }
        }

        // --- Iterate Window (sTo) ---
        for (unsigned sToIdx = nodeStartIdx; sToIdx < windowBound; ++sToIdx) {
            unsigned sTo = nodeStep + sToIdx - windowSize;

            // Apply Phase 2A: incoming deltas (policy-aware, sTo-dependent)
            for (const auto &info : parentAddInfos) {
                CommPolicy::CalculateDeltaAdd(info.effectiveVal, sTo, info.uStep, info.uProc, pTo, info.cost, deltaAdapter);
            }

            // Apply Phase 2B: outgoing deltas (policy-aware)
            for (const auto &info : outgoingInfos) {
                if constexpr (CommPolicy::outgoing_recv_at_parent_step) {
                    AddDelta(true, sTo, info.vProc, info.cost);
                } else {
                    if (info.recvStep >= 0) {
                        AddDelta(true, static_cast<unsigned>(info.recvStep), info.vProc, info.cost);
                    }
                }
                if constexpr (CommPolicy::outgoing_send_at_parent_step) {
                    AddDelta(false, sTo, pTo, info.cost);
                } else {
                    if (info.sendStep >= 0) {
                        AddDelta(false, static_cast<unsigned>(info.sendStep), pTo, info.cost);
                    }
                }
            }

            // --- Evaluate cost change via model-specific callback ---
            affinityTableNode[pTo][sToIdx] += evaluatorFn(pTo, sToIdx, sTo, scratch);

            // Revert Phase 2B: outgoing deltas
            for (const auto &info : outgoingInfos) {
                if constexpr (CommPolicy::outgoing_recv_at_parent_step) {
                    AddDelta(true, sTo, info.vProc, -info.cost);
                } else {
                    if (info.recvStep >= 0) {
                        AddDelta(true, static_cast<unsigned>(info.recvStep), info.vProc, -info.cost);
                    }
                }
                if constexpr (CommPolicy::outgoing_send_at_parent_step) {
                    AddDelta(false, sTo, pTo, -info.cost);
                } else {
                    if (info.sendStep >= 0) {
                        AddDelta(false, static_cast<unsigned>(info.sendStep), pTo, -info.cost);
                    }
                }
            }

            // Revert Phase 2A: incoming deltas
            for (const auto &info : parentAddInfos) {
                CommPolicy::CalculateDeltaAdd(info.effectiveVal, sTo, info.uStep, info.uProc, pTo, info.cost, negDeltaAdapter);
            }
        }
    }
}

}    // namespace osp
