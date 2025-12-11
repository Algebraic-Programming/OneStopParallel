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
        CommWeightT maxComm;
        CommWeightT secondMaxComm;
        unsigned maxCommCount;
    };

    std::unordered_map<unsigned, StepInfo> stepData;

    PreMoveCommData() = default;

    void AddStep(unsigned step, CommWeightT max, CommWeightT second, unsigned count) { stepData[step] = {max, second, count}; }

    bool GetStep(unsigned step, StepInfo &info) const {
        auto it = stepData.find(step);
        if (it != stepData.end()) {
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

    const BspInstance<GraphT> *instance;
    const KlActiveScheduleT *activeSchedule;

    std::vector<std::vector<CommWeightT>> stepProcSend;
    std::vector<std::vector<CommWeightT>> stepProcReceive;

    // Caches for fast cost calculation (Global Max/Second Max per step)
    std::vector<CommWeightT> stepMaxCommCache;
    std::vector<CommWeightT> stepSecondMaxCommCache;
    std::vector<unsigned> stepMaxCommCountCache;

    CommWeightT maxCommWeight = 0;

    // Select the appropriate container type based on the policy's ValueType
    using ContainerType = typename std::conditional<std::is_same<typename CommPolicy::ValueType, unsigned>::value,
                                                    LambdaVectorContainer<VertexType>,
                                                    GenericLambdaVectorContainer<VertexType, typename CommPolicy::ValueType>>::type;

    ContainerType nodeLambdaMap;

    // Optimization: Scratchpad for update_datastructure_after_move to avoid allocations
    std::vector<unsigned> affectedStepsList;
    std::vector<bool> stepIsAffected;

    inline CommWeightT StepProcSend(unsigned step, unsigned proc) const { return stepProcSend[step][proc]; }

    inline CommWeightT &StepProcSend(unsigned step, unsigned proc) { return stepProcSend[step][proc]; }

    inline CommWeightT StepProcReceive(unsigned step, unsigned proc) const { return stepProcReceive[step][proc]; }

    inline CommWeightT &StepProcReceive(unsigned step, unsigned proc) { return stepProcReceive[step][proc]; }

    inline CommWeightT StepMaxComm(unsigned step) const { return stepMaxCommCache[step]; }

    inline CommWeightT StepSecondMaxComm(unsigned step) const { return stepSecondMaxCommCache[step]; }

    inline unsigned StepMaxCommCount(unsigned step) const { return stepMaxCommCountCache[step]; }

    inline void Initialize(KlActiveScheduleT &klSched) {
        activeSchedule = &klSched;
        instance = &activeSchedule->GetInstance();
        const unsigned numSteps = activeSchedule->NumSteps();
        const unsigned numProcs = instance->NumberOfProcessors();
        maxCommWeight = 0;

        stepProcSend.assign(numSteps, std::vector<CommWeightT>(numProcs, 0));
        stepProcReceive.assign(numSteps, std::vector<CommWeightT>(numProcs, 0));

        stepMaxCommCache.assign(numSteps, 0);
        stepSecondMaxCommCache.assign(numSteps, 0);
        stepMaxCommCountCache.assign(numSteps, 0);

        nodeLambdaMap.Initialize(instance->GetComputationalDag().NumVertices(), numProcs);

        // Initialize scratchpad
        stepIsAffected.assign(numSteps, false);
        affectedStepsList.reserve(numSteps);
    }

    inline void Clear() {
        stepProcSend.clear();
        stepProcReceive.clear();
        stepMaxCommCache.clear();
        stepSecondMaxCommCache.clear();
        stepMaxCommCountCache.clear();
        nodeLambdaMap.clear();
        affectedStepsList.clear();
        stepIsAffected.clear();
    }

    inline void ArrangeSuperstepCommData(const unsigned step) {
        CommWeightT maxSend = 0;
        CommWeightT secondMaxSend = 0;
        unsigned maxSendCount = 0;

        const auto &sends = stepProcSend[step];
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

        const auto &receives = stepProcReceive[step];
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
        stepMaxCommCache[step] = globalMax;

        unsigned globalCount = 0;
        if (maxSend == globalMax) {
            globalCount += maxSendCount;
        }
        if (maxReceive == globalMax) {
            globalCount += maxReceiveCount;
        }
        stepMaxCommCountCache[step] = globalCount;

        CommWeightT candSend = (maxSend == globalMax) ? secondMaxSend : maxSend;
        CommWeightT candRecv = (maxReceive == globalMax) ? secondMaxReceive : maxReceive;

        stepSecondMaxCommCache[step] = std::max(candSend, candRecv);
    }

    void RecomputeMaxSendReceive(unsigned step) { ArrangeSuperstepCommData(step); }

    inline PreMoveCommData<CommWeightT> GetPreMoveCommData(const KlMove &move) {
        PreMoveCommData<CommWeightT> data;
        std::unordered_set<unsigned> affectedSteps;

        affectedSteps.insert(move.fromStep);
        affectedSteps.insert(move.toStep);

        const auto &graph = instance->GetComputationalDag();

        for (const auto &parent : graph.Parents(move.node)) {
            affectedSteps.insert(activeSchedule->AssignedSuperstep(parent));
        }

        for (unsigned step : affectedSteps) {
            data.AddStep(step, StepMaxComm(step), StepSecondMaxComm(step), StepMaxCommCount(step));
        }

        return data;
    }

    void UpdateDatastructureAfterMove(const KlMove &move, unsigned, unsigned) {
        const auto &graph = instance->GetComputationalDag();

        // Prepare Scratchpad (Avoids Allocations) ---
        for (unsigned step : affectedStepsList) {
            if (step < stepIsAffected.size()) {
                stepIsAffected[step] = false;
            }
        }
        affectedStepsList.clear();

        auto markStep = [&](unsigned step) {
            if (step < stepIsAffected.size() && !stepIsAffected[step]) {
                stepIsAffected[step] = true;
                affectedStepsList.push_back(step);
            }
        };

        const VertexType node = move.node;
        const unsigned fromStep = move.fromStep;
        const unsigned toStep = move.toStep;
        const unsigned fromProc = move.fromProc;
        const unsigned toProc = move.toProc;
        const CommWeightT commWNode = graph.VertexCommWeight(node);

        // Handle Node Movement (Outgoing Edges: Node -> Children)

        if (fromStep != toStep) {
            // Case 1: Node changes Step
            for (const auto [proc, val] : nodeLambdaMap.IterateProcEntries(node)) {
                // A. Remove Old (Sender: from_proc, Receiver: proc)
                if (proc != fromProc) {
                    const CommWeightT cost = commWNode * instance->SendCosts(fromProc, proc);
                    if (cost > 0) {
                        CommPolicy::UnattributeCommunication(*this, cost, fromStep, fromProc, proc, 0, val);
                    }
                }

                // B. Add New (Sender: to_proc, Receiver: proc)
                if (proc != toProc) {
                    const CommWeightT cost = commWNode * instance->SendCosts(toProc, proc);
                    if (cost > 0) {
                        CommPolicy::AttributeCommunication(*this, cost, toStep, toProc, proc, 0, val);
                    }
                }
            }
            markStep(fromStep);
            markStep(toStep);

        } else if (fromProc != toProc) {
            // Case 2: Node stays in same Step, but changes Processor

            for (const auto [proc, val] : nodeLambdaMap.IterateProcEntries(node)) {
                // Remove Old (Sender: from_proc, Receiver: proc)
                if (proc != fromProc) {
                    const CommWeightT cost = commWNode * instance->SendCosts(fromProc, proc);
                    if (cost > 0) {
                        CommPolicy::UnattributeCommunication(*this, cost, fromStep, fromProc, proc, 0, val);
                    }
                }

                // Add New (Sender: to_proc, Receiver: proc)
                if (proc != toProc) {
                    const CommWeightT cost = commWNode * instance->SendCosts(toProc, proc);
                    if (cost > 0) {
                        CommPolicy::AttributeCommunication(*this, cost, fromStep, toProc, proc, 0, val);
                    }
                }
            }
            markStep(fromStep);
        }

        // Update Parents' Outgoing Communication (Parents → Node)

        for (const auto &parent : graph.Parents(node)) {
            const unsigned parentStep = activeSchedule->AssignedSuperstep(parent);
            // Fast boundary check
            if (parentStep >= stepProcSend.size()) {
                continue;
            }

            const unsigned parentProc = activeSchedule->AssignedProcessor(parent);
            const CommWeightT commWParent = graph.VertexCommWeight(parent);

            auto &val = nodeLambdaMap.GetProcEntry(parent, fromProc);
            const bool removedFromProc = CommPolicy::RemoveChild(val, fromStep);

            // 1. Handle Removal from from_proc
            if (removedFromProc) {
                if (fromProc != parentProc) {
                    const CommWeightT cost = commWParent * instance->SendCosts(parentProc, fromProc);
                    if (cost > 0) {
                        CommPolicy::UnattributeCommunication(*this, cost, parentStep, parentProc, fromProc, fromStep, val);
                    }
                }
            }

            auto &valTo = nodeLambdaMap.GetProcEntry(parent, toProc);
            const bool addedToProc = CommPolicy::AddChild(valTo, toStep);

            // 2. Handle Addition to to_proc
            if (addedToProc) {
                if (toProc != parentProc) {
                    const CommWeightT cost = commWParent * instance->SendCosts(parentProc, toProc);
                    if (cost > 0) {
                        CommPolicy::AttributeCommunication(*this, cost, parentStep, parentProc, toProc, toStep, valTo);
                    }
                }
            }

            markStep(parentStep);
        }

        // Re-arrange Affected Steps
        for (unsigned step : affectedStepsList) {
            ArrangeSuperstepCommData(step);
        }
    }

    void SwapSteps(const unsigned step1, const unsigned step2) {
        std::swap(stepProcSend[step1], stepProcSend[step2]);
        std::swap(stepProcReceive[step1], stepProcReceive[step2]);
        std::swap(stepMaxCommCache[step1], stepMaxCommCache[step2]);
        std::swap(stepSecondMaxCommCache[step1], stepSecondMaxCommCache[step2]);
        std::swap(stepMaxCommCountCache[step1], stepMaxCommCountCache[step2]);
    }

    void ResetSuperstep(unsigned step) {
        std::fill(stepProcSend[step].begin(), stepProcSend[step].end(), 0);
        std::fill(stepProcReceive[step].begin(), stepProcReceive[step].end(), 0);
        ArrangeSuperstepCommData(step);
    }

    void ComputeCommDatastructures(unsigned startStep, unsigned endStep) {
        for (unsigned step = startStep; step <= endStep; step++) {
            std::fill(stepProcSend[step].begin(), stepProcSend[step].end(), 0);
            std::fill(stepProcReceive[step].begin(), stepProcReceive[step].end(), 0);
        }

        const auto &vecSched = activeSchedule->GetVectorSchedule();
        const auto &graph = instance->GetComputationalDag();

        for (const auto &u : graph.Vertices()) {
            nodeLambdaMap.ResetNode(u);
            const unsigned uProc = vecSched.AssignedProcessor(u);
            const unsigned uStep = vecSched.AssignedSuperstep(u);
            const CommWeightT commW = graph.VertexCommWeight(u);
            maxCommWeight = std::max(maxCommWeight, commW);

            for (const auto &v : graph.Children(u)) {
                const unsigned vProc = vecSched.AssignedProcessor(v);
                const unsigned vStep = vecSched.AssignedSuperstep(v);

                const CommWeightT commWSendCost = (uProc != vProc) ? commW * instance->SendCosts(uProc, vProc) : 0;

                auto &val = nodeLambdaMap.GetProcEntry(u, vProc);
                if (CommPolicy::AddChild(val, vStep)) {
                    if (uProc != vProc && commWSendCost > 0) {
                        CommPolicy::AttributeCommunication(*this, commWSendCost, uStep, uProc, vProc, vStep, val);
                    }
                }
            }
        }

        for (unsigned step = startStep; step <= endStep; step++) {
            if (step >= stepProcSend.size()) {
                continue;
            }
            ArrangeSuperstepCommData(step);
        }
    }
};

}    // namespace osp
