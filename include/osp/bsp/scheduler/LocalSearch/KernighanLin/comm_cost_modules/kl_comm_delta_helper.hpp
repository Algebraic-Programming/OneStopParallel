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

            for (size_t i = oldSize; i < nSteps; ++i) {
                sendDeltas_[i].Initialize(nProcs);
                recvDeltas_[i].Initialize(nProcs);
            }

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

    struct ParentAddInfo {
        unsigned uProc;
        unsigned uStep;
        CommWeightT cost;
        typename CommPolicy::ValueType effectiveVal;
    };

    struct OutgoingInfo {
        unsigned vProc;
        CommWeightT cost;
        int recvStep;
        int sendStep;
    };

    static thread_local std::vector<ParentAddInfo> parentAddInfos;
    static thread_local std::vector<OutgoingInfo> outgoingInfos;

    for (const unsigned pTo : procRange.CompatibleProcessorsVertex(node)) {
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

        for (unsigned sToIdx = nodeStartIdx; sToIdx < windowBound; ++sToIdx) {
            unsigned sTo = nodeStep + sToIdx - windowSize;

            for (const auto &info : parentAddInfos) {
                CommPolicy::CalculateDeltaAdd(info.effectiveVal, sTo, info.uStep, info.uProc, pTo, info.cost, deltaAdapter);
            }

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

            affinityTableNode[pTo][sToIdx] += evaluatorFn(pTo, sToIdx, sTo, scratch);

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

            for (const auto &info : parentAddInfos) {
                CommPolicy::CalculateDeltaAdd(info.effectiveVal, sTo, info.uStep, info.uProc, pTo, info.cost, negDeltaAdapter);
            }
        }
    }
}

}    // namespace osp
