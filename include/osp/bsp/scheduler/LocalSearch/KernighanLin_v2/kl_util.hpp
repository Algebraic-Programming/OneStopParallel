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

#include <unordered_set>

#include "kl_active_schedule.hpp"

namespace osp {

template <typename CostT, typename CommCostFunctionT, typename KlActiveScheduleT>
struct RewardPenaltyStrategy {
    KlActiveScheduleT *activeSchedule;
    CostT maxWeight;

    unsigned violationsThreshold = 0;
    CostT initialPenalty = 10.0;
    CostT penalty = 0;
    CostT reward = 0;

    void Initialize(KlActiveScheduleT &sched, const CostT maxComm, const CostT maxWork) {
        maxWeight = std::max(maxWork, maxComm * sched.GetInstance().CommunicationCosts());
        activeSchedule = &sched;
        initialPenalty = static_cast<CostT>(std::sqrt(maxWeight));
    }

    void InitRewardPenalty(double multiplier = 1.0) {
        multiplier = std::min(multiplier, 10.0);
        penalty = static_cast<CostT>(initialPenalty * multiplier);
        reward = static_cast<CostT>(maxWeight * multiplier);
    }
};

template <typename VertexType>
struct SetVertexLockManger {
    std::unordered_set<VertexType> lockedNodes;

    void Initialize(size_t) {}

    void Lock(VertexType node) { lockedNodes.insert(node); }

    void Unlock(VertexType node) { lockedNodes.erase(node); }

    bool IsLocked(VertexType node) { return lockedNodes.find(node) != lockedNodes.end(); }

    void Clear() { lockedNodes.clear(); }
};

template <typename VertexType>
struct VectorVertexLockManger {
    std::vector<bool> lockedNodes;

    void Initialize(size_t numNodes) { lockedNodes.resize(numNodes); }

    void Lock(VertexType node) { lockedNodes[node] = true; }

    void Unlock(VertexType node) { lockedNodes[node] = false; }

    bool IsLocked(VertexType node) { return lockedNodes[node]; }

    void Clear() { lockedNodes.assign(lockedNodes.size(), false); }
};

template <typename GraphT, typename CostT, typename KlActiveScheduleT, unsigned WindowSize>
struct AdaptiveAffinityTable {
    constexpr static unsigned windowRange = 2 * WindowSize + 1;
    using VertexType = VertexIdxT<GraphT>;

  private:
    const KlActiveScheduleT *activeSchedule_;
    const GraphT *graph_;

    std::vector<bool> nodeIsSelected_;
    std::vector<size_t> selectedNodesIdx_;

    std::vector<std::vector<std::vector<CostT>>> affinityTable_;
    std::vector<VertexType> selectedNodes_;

    std::vector<size_t> gaps_;
    size_t lastIdx_;

  public:
    void Initialize(const KlActiveScheduleT &sche, const std::size_t initialTableSize) {
        activeSchedule_ = &sche;
        graph_ = &(sche.GetInstance().GetComputationalDag());

        lastIdx_ = 0;

        nodeIsSelected_.resize(graph_->NumVertices());
        selectedNodesIdx_.resize(graph_->NumVertices());
        selectedNodes_.resize(initialTableSize);

        nodeIsSelected_.assign(nodeIsSelected_.size(), false);

        affinityTable_.resize(initialTableSize);
        const unsigned numProcs = sche.GetInstance().NumberOfProcessors();
        for (auto &table : affinityTable_) {
            table.resize(numProcs);
            for (auto &row : table) {
                row.resize(windowRange);
            }
        }
    }

    inline std::vector<VertexType> &GetSelectedNodes() { return selectedNodes_; }

    inline const std::vector<VertexType> &GetSelectedNodes() const { return selectedNodes_; }

    inline size_t Size() const { return lastIdx_ - gaps_.size(); }

    inline bool IsSelected(VertexType node) const { return nodeIsSelected_[node]; }

    inline const std::vector<size_t> &GetSelectedNodesIndices() const { return selectedNodesIdx_; }

    inline size_t GetSelectedNodesIdx(VertexType node) const { return selectedNodesIdx_[node]; }

    inline std::vector<std::vector<CostT>> &operator[](VertexType node) {
        assert(nodeIsSelected_[node]);
        return affinityTable_[selectedNodesIdx_[node]];
    }

    inline std::vector<std::vector<CostT>> &At(VertexType node) {
        assert(nodeIsSelected_[node]);
        return affinityTable_[selectedNodesIdx_[node]];
    }

    inline const std::vector<std::vector<CostT>> &At(VertexType node) const {
        assert(nodeIsSelected_[node]);
        return affinityTable_[selectedNodesIdx_[node]];
    }

    inline std::vector<std::vector<CostT>> &GetAffinityTable(VertexType node) {
        assert(nodeIsSelected_[node]);
        return affinityTable_[selectedNodesIdx_[node]];
    }

    bool Insert(VertexType node) {
        if (nodeIsSelected_[node]) {
            return false;    // Node is already in the table.
        }

        size_t insertLocation;
        if (!gaps_.empty()) {
            insertLocation = gaps_.back();
            gaps_.pop_back();
        } else {
            insertLocation = lastIdx_;

            if (insertLocation >= selectedNodes_.size()) {
                const size_t oldSize = selectedNodes_.size();
                const size_t newSize = std::min(oldSize * 2, static_cast<size_t>(graph_->NumVertices()));

                selectedNodes_.resize(newSize);
                affinityTable_.resize(newSize);

                const unsigned numProcs = activeSchedule_->GetInstance().NumberOfProcessors();
                for (size_t i = oldSize; i < newSize; ++i) {
                    affinityTable_[i].resize(numProcs);
                    for (auto &row : affinityTable_[i]) {
                        row.resize(windowRange);
                    }
                }
            }
            lastIdx_++;
        }

        nodeIsSelected_[node] = true;
        selectedNodesIdx_[node] = insertLocation;
        selectedNodes_[insertLocation] = node;

        return true;
    }

    void Remove(VertexType node) {
        assert(nodeIsSelected_[node]);
        nodeIsSelected_[node] = false;

        gaps_.push_back(selectedNodesIdx_[node]);
    }

    void ResetNodeSelection() {
        nodeIsSelected_.assign(nodeIsSelected_.size(), false);
        gaps_.clear();
        lastIdx_ = 0;
    }

    void Clear() {
        nodeIsSelected_.clear();
        selectedNodesIdx_.clear();
        affinityTable_.clear();
        selectedNodes_.clear();
        gaps_.clear();
        lastIdx_ = 0;
    }

    void Trim() {
        while (!gaps_.empty() && lastIdx_ > 0) {
            size_t lastElementIdx = lastIdx_ - 1;

            // The last element could be a gap itself. If so, just shrink the size.
            // We don't need to touch the `gaps` vector, as it will be cleared.
            if (!nodeIsSelected_[selectedNodes_[lastElementIdx]]) {
                lastIdx_--;
                continue;
            }

            size_t gapIdx = gaps_.back();
            gaps_.pop_back();

            // If the gap we picked is now at or after the end, we can ignore it.
            if (gapIdx >= lastIdx_) {
                continue;
            }

            VertexType nodeToMove = selectedNodes_[lastElementIdx];

            std::swap(affinityTable_[gapIdx], affinityTable_[lastElementIdx]);
            std::swap(selectedNodes_[gapIdx], selectedNodes_[lastElementIdx]);
            selectedNodesIdx_[nodeToMove] = gapIdx;

            lastIdx_--;
        }
        gaps_.clear();
    }
};

template <typename GraphT, typename CostT, typename KlActiveScheduleT, unsigned WindowSize>
struct StaticAffinityTable {
    constexpr static unsigned windowRange = 2 * WindowSize + 1;
    using VertexType = VertexIdxT<GraphT>;

  private:
    const KlActiveScheduleT *activeSchedule_;
    const GraphT *graph_;

    std::unordered_set<VertexType> selectedNodes_;

    std::vector<std::vector<std::vector<CostT>>> affinityTable_;

  public:
    void Initialize(const KlActiveScheduleT &sche, const std::size_t) {
        activeSchedule_ = &sche;
        graph_ = &(sche.GetInstance().GetComputationalDag());

        affinityTable_.resize(graph_->NumVertices());
        const unsigned numProcs = sche.GetInstance().NumberOfProcessors();
        for (auto &table : affinityTable_) {
            table.resize(numProcs);
            for (auto &row : table) {
                row.resize(windowRange);
            }
        }
    }

    inline std::vector<VertexType> GetSelectedNodes() const { return {selectedNodes_.begin(), selectedNodes_.end()}; }

    inline size_t Size() const { return selectedNodes_.size(); }

    inline bool IsSelected(VertexType node) const { return selectedNodes_.find(node) != selectedNodes_.end(); }

    inline std::vector<std::vector<CostT>> &operator[](VertexType node) { return affinityTable_[node]; }

    inline std::vector<std::vector<CostT>> &At(VertexType node) { return affinityTable_[node]; }

    inline const std::vector<std::vector<CostT>> &At(VertexType node) const { return affinityTable_[node]; }

    inline std::vector<std::vector<CostT>> &GetAffinityTable(VertexType node) { return affinityTable_[node]; }

    bool Insert(VertexType node) {
        const auto pair = selectedNodes_.insert(node);
        return pair.second;
    }

    void Remove(VertexType node) { selectedNodes_.erase(node); }

    void ResetNodeSelection() { selectedNodes_.clear(); }

    void Clear() {
        affinityTable_.clear();
        selectedNodes_.clear();
    }

    void Trim() {}
};

template <typename GraphT, typename ContainerT, typename KlActiveScheduleT>
struct VertexSelectionStrategy {
    using EdgeType = EdgeDescT<GraphT>;

    const KlActiveScheduleT *activeSchedule;
    const GraphT *graph;
    std::mt19937 *gen;
    std::size_t selectionThreshold = 0;
    unsigned strategyCounter = 0;

    std::vector<VertexIdxT<GraphT>> permutation;
    std::size_t permutationIdx;

    unsigned maxWorkCounter = 0;

    inline void Initialize(const KlActiveScheduleT &sche, std::mt19937 &gen_, const unsigned startStep, const unsigned endStep) {
        activeSchedule = &sche;
        graph = &(sche.GetInstance().GetComputationalDag());
        gen = &gen_;

        permutation.reserve(graph->NumVertices() / activeSchedule->NumSteps() * (endStep - startStep));
    }

    inline void Setup(const unsigned startStep, const unsigned endStep) {
        maxWorkCounter = startStep;
        strategyCounter = 0;
        permutation.clear();

        const unsigned numProcs = activeSchedule->GetInstance().NumberOfProcessors();
        for (unsigned step = startStep; step <= endStep; ++step) {
            const auto &processorVertices = activeSchedule->GetSetSchedule().stepProcessorVertices[step];
            for (unsigned proc = 0; proc < numProcs; ++proc) {
                for (const auto node : processorVertices[proc]) {
                    permutation.push_back(node);
                }
            }
        }

        permutationIdx = 0;
        std::shuffle(permutation.begin(), permutation.end(), *gen);
    }

    void AddNeighboursToSelection(VertexIdxT<GraphT> node, ContainerT &nodes, const unsigned startStep, const unsigned endStep) {
        for (const auto parent : graph->parents(node)) {
            const unsigned parentStep = activeSchedule->assigned_superstep(parent);
            if (parentStep >= startStep && parentStep <= endStep) {
                nodes.insert(parent);
            }
        }

        for (const auto child : graph->children(node)) {
            const unsigned childStep = activeSchedule->assigned_superstep(child);
            if (childStep >= startStep && childStep <= endStep) {
                nodes.insert(child);
            }
        }
    }

    inline void SelectActiveNodes(ContainerT &nodeSelection, const unsigned startStep, const unsigned endStep) {
        if (strategyCounter < 3) {
            SelectNodesPermutationThreshold(selectionThreshold, nodeSelection);
        } else if (strategyCounter == 4) {
            SelectNodesMaxWorkProc(selectionThreshold, nodeSelection, startStep, endStep);
        }

        strategyCounter++;
        strategyCounter %= 5;
    }

    void SelectNodesViolations(ContainerT &nodeSelection,
                               std::unordered_set<EdgeType> &currentViolations,
                               const unsigned startStep,
                               const unsigned endStep) {
        for (const auto &edge : currentViolations) {
            const auto sourceV = Source(edge, *graph);
            const auto targetV = Target(edge, *graph);

            const unsigned sourceStep = activeSchedule->AssignedSuperstep(sourceV);
            if (sourceStep >= startStep && sourceStep <= endStep) {
                nodeSelection.Insert(sourceV);
            }

            const unsigned targetStep = activeSchedule->AssignedSuperstep(targetV);
            if (targetStep >= startStep && targetStep <= endStep) {
                nodeSelection.Insert(targetV);
            }
        }
    }

    void SelectNodesPermutationThreshold(const std::size_t &threshold, ContainerT &nodeSelection) {
        const size_t bound = std::min(threshold + permutationIdx, permutation.size());
        for (std::size_t i = permutationIdx; i < bound; i++) {
            nodeSelection.Insert(permutation[i]);
        }

        permutationIdx = bound;
        if (permutationIdx + threshold >= permutation.size()) {
            permutationIdx = 0;
            std::shuffle(permutation.begin(), permutation.end(), *gen);
        }
    }

    void SelectNodesMaxWorkProc(const std::size_t &threshold,
                                ContainerT &nodeSelection,
                                const unsigned startStep,
                                const unsigned endStep) {
        while (nodeSelection.Size() < threshold) {
            if (maxWorkCounter > endStep) {
                maxWorkCounter = startStep;    // wrap around
                break;                         // stop after one full pass
            }

            SelectNodesMaxWorkProcHelper(threshold - nodeSelection.Size(), maxWorkCounter, nodeSelection);
            maxWorkCounter++;
        }
    }

    void SelectNodesMaxWorkProcHelper(const std::size_t &threshold, unsigned step, ContainerT &nodeSelection) {
        const unsigned numMaxWorkProc = activeSchedule->workDatastructures.stepMaxWorkProcessorCount[step];
        for (unsigned idx = 0; idx < numMaxWorkProc; idx++) {
            const unsigned proc = activeSchedule->workDatastructures.stepProcessorWork[step][idx].proc;
            const std::unordered_set<VertexIdxT<GraphT>> stepProcVert
                = activeSchedule->GetSetSchedule().stepProcessorVertices[step][proc];
            const size_t numInsert = std::min(threshold - nodeSelection.Size(), stepProcVert.size());
            auto endIt = stepProcVert.begin();
            std::advance(endIt, numInsert);
            std::for_each(stepProcVert.begin(), endIt, [&](const auto &val) { nodeSelection.Insert(val); });
        }
    }
};

}    // namespace osp
