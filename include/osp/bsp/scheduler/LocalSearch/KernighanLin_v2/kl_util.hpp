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
    KlActiveScheduleT *activeSchedule_;
    CostT maxWeight_;

    unsigned violationsThreshold_ = 0;
    CostT initialPenalty_ = 10.0;
    CostT penalty_ = 0;
    CostT reward_ = 0;

    void Initialize(KlActiveScheduleT &sched, const CostT maxComm, const CostT maxWork) {
        maxWeight_ = std::max(maxWork, maxComm * sched.getInstance().communicationCosts());
        activeSchedule_ = &sched;
        initialPenalty_ = static_cast<CostT>(std::sqrt(maxWeight_));
    }

    void InitRewardPenalty(double multiplier = 1.0) {
        multiplier = std::min(multiplier, 10.0);
        penalty_ = static_cast<CostT>(initialPenalty_ * multiplier);
        reward_ = static_cast<CostT>(maxWeight_ * multiplier);
    }
};

template <typename VertexType>
struct SetVertexLockManger {
    std::unordered_set<VertexType> lockedNodes_;

    void Initialize(size_t) {}

    void Lock(VertexType node) { lockedNodes_.insert(node); }

    void Unlock(VertexType node) { lockedNodes_.erase(node); }

    bool IsLocked(VertexType node) { return lockedNodes_.find(node) != lockedNodes_.end(); }

    void Clear() { lockedNodes_.clear(); }
};

template <typename VertexType>
struct VectorVertexLockManger {
    std::vector<bool> lockedNodes_;

    void Initialize(size_t numNodes) { lockedNodes_.resize(numNodes); }

    void Lock(VertexType node) { lockedNodes_[node] = true; }

    void Unlock(VertexType node) { lockedNodes_[node] = false; }

    bool IsLocked(VertexType node) { return lockedNodes_[node]; }

    void Clear() { lockedNodes_.assign(lockedNodes_.size(), false); }
};

template <typename GraphT, typename CostT, typename KlActiveScheduleT, unsigned windowSize>
struct AdaptiveAffinityTable {
    constexpr static unsigned windowRange_ = 2 * windowSize + 1;
    using VertexType = vertex_idx_t<Graph_t>;

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
        graph_ = &(sche.getInstance().getComputationalDag());

        lastIdx_ = 0;

        nodeIsSelected_.resize(graph_->NumVertices());
        selectedNodesIdx_.resize(graph_->NumVertices());
        selected_nodes.resize(initial_table_size);

        nodeIsSelected_.assign(nodeIsSelected_.size(), false);

        affinityTable_.resize(initialTableSize);
        const unsigned numProcs = sche.getInstance().numberOfProcessors();
        for (auto &table : affinityTable_) {
            table.resize(numProcs);
            for (auto &row : table) {
                row.resize(windowRange_);
            }
        }
    }

    inline std::vector<VertexType> &GetSelectedNodes() { return selected_nodes; }

    inline const std::vector<VertexType> &GetSelectedNodes() const { return selected_nodes; }

    inline size_t size() const { return lastIdx_ - gaps_.size(); }

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

            if (insert_location >= selected_nodes.size()) {
                const size_t oldSize = selected_nodes.size();
                const size_t newSize = std::min(oldSize * 2, static_cast<size_t>(graph_->NumVertices()));

                selected_nodes.resize(new_size);
                affinityTable_.resize(newSize);

                const unsigned numProcs = activeSchedule_->getInstance().numberOfProcessors();
                for (size_t i = oldSize; i < newSize; ++i) {
                    affinityTable_[i].resize(numProcs);
                    for (auto &row : affinityTable_[i]) {
                        row.resize(windowRange_);
                    }
                }
            }
            lastIdx_++;
        }

        nodeIsSelected_[node] = true;
        selectedNodesIdx_[node] = insertLocation;
        selected_nodes[insert_location] = node;

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
        selected_nodes.clear();
        gaps_.clear();
        lastIdx_ = 0;
    }

    void Trim() {
        while (!gaps_.empty() && lastIdx_ > 0) {
            size_t lastElementIdx = lastIdx_ - 1;

            // The last element could be a gap itself. If so, just shrink the size.
            // We don't need to touch the `gaps` vector, as it will be cleared.
            if (!node_is_selected[selected_nodes[last_element_idx]]) {
                lastIdx_--;
                continue;
            }

            size_t gapIdx = gaps_.back();
            gaps_.pop_back();

            // If the gap we picked is now at or after the end, we can ignore it.
            if (gapIdx >= lastIdx_) {
                continue;
            }

            VertexType nodeToMove = selected_nodes[last_element_idx];

            std::swap(affinityTable_[gapIdx], affinityTable_[lastElementIdx]);
            std::swap(selected_nodes[gap_idx], selected_nodes[last_element_idx]);
            selectedNodesIdx_[node_to_move] = gapIdx;

            lastIdx_--;
        }
        gaps_.clear();
    }
};

template <typename GraphT, typename CostT, typename KlActiveScheduleT, unsigned windowSize>
struct StaticAffinityTable {
    constexpr static unsigned windowRange_ = 2 * windowSize + 1;
    using VertexType = vertex_idx_t<Graph_t>;

  private:
    const KlActiveScheduleT *activeSchedule_;
    const GraphT *graph_;

    std::unordered_set<VertexType> selectedNodes_;

    std::vector<std::vector<std::vector<CostT>>> affinityTable_;

  public:
    void Initialize(const KlActiveScheduleT &sche, const std::size_t) {
        activeSchedule_ = &sche;
        graph_ = &(sche.getInstance().getComputationalDag());

        affinityTable_.resize(graph_->NumVertices());
        const unsigned numProcs = sche.getInstance().numberOfProcessors();
        for (auto &table : affinityTable_) {
            table.resize(numProcs);
            for (auto &row : table) {
                row.resize(windowRange_);
            }
        }
    }

    inline std::vector<VertexType> GetSelectedNodes() const { return {selected_nodes.begin(), selected_nodes.end()}; }

    inline size_t size() const { return selected_nodes.size(); }

    inline bool IsSelected(VertexType node) const { return selected_nodes.find(node) != selected_nodes.end(); }

    inline std::vector<std::vector<CostT>> &operator[](VertexType node) { return affinityTable_[node]; }

    inline std::vector<std::vector<CostT>> &At(VertexType node) { return affinityTable_[node]; }

    inline const std::vector<std::vector<CostT>> &At(VertexType node) const { return affinityTable_[node]; }

    inline std::vector<std::vector<CostT>> &GetAffinityTable(VertexType node) { return affinityTable_[node]; }

    bool Insert(VertexType node) {
        const auto pair = selected_nodes.insert(node);
        return pair.second;
    }

    void Remove(VertexType node) { selected_nodes.erase(node); }

    void ResetNodeSelection() { selected_nodes.clear(); }

    void Clear() {
        affinityTable_.clear();
        selected_nodes.clear();
    }

    void Trim() {}
};

template <typename GraphT, typename ContainerT, typename KlActiveScheduleT>
struct VertexSelectionStrategy {
    using EdgeType = edge_desc_t<Graph_t>;

    const KlActiveScheduleT *activeSchedule_;
    const GraphT *graph_;
    std::mt19937 *gen_;
    std::size_t selectionThreshold_ = 0;
    unsigned strategyCounter_ = 0;

    std::vector<vertex_idx_t<Graph_t>> permutation_;
    std::size_t permutationIdx_;

    unsigned maxWorkCounter_ = 0;

    inline void Initialize(const KlActiveScheduleT &sche, std::mt19937 &gen, const unsigned startStep, const unsigned endStep) {
        activeSchedule_ = &sche;
        graph_ = &(sche.getInstance().getComputationalDag());
        gen_ = &gen;

        permutation.reserve(graph->NumVertices() / active_schedule->num_steps() * (end_step - start_step));
    }

    inline void Setup(const unsigned startStep, const unsigned endStep) {
        maxWorkCounter_ = startStep;
        strategyCounter_ = 0;
        permutation.clear();

        const unsigned numProcs = activeSchedule_->getInstance().numberOfProcessors();
        for (unsigned step = startStep; step <= endStep; ++step) {
            const auto &processorVertices = activeSchedule_->getSetSchedule().step_processor_vertices[step];
            for (unsigned proc = 0; proc < numProcs; ++proc) {
                for (const auto node : processorVertices[proc]) {
                    permutation.push_back(node);
                }
            }
        }

        permutationIdx_ = 0;
        std::shuffle(permutation.begin(), permutation.end(), *gen);
    }

    void AddNeighboursToSelection(vertex_idx_t<Graph_t> node, ContainerT &nodes, const unsigned startStep, const unsigned endStep) {
        for (const auto parent : graph->parents(node)) {
            const unsigned parent_step = active_schedule->assigned_superstep(parent);
            if (parent_step >= start_step && parent_step <= end_step) {
                nodes.insert(parent);
            }
        }

        for (const auto child : graph->children(node)) {
            const unsigned child_step = active_schedule->assigned_superstep(child);
            if (child_step >= start_step && child_step <= end_step) {
                nodes.insert(child);
            }
        }
    }

    inline void SelectActiveNodes(ContainerT &nodeSelection, const unsigned startStep, const unsigned endStep) {
        if (strategyCounter_ < 3) {
            SelectNodesPermutationThreshold(selectionThreshold_, nodeSelection);
        } else if (strategyCounter_ == 4) {
            SelectNodesMaxWorkProc(selectionThreshold_, nodeSelection, startStep, endStep);
        }

        strategyCounter_++;
        strategyCounter_ %= 5;
    }

    void SelectNodesViolations(ContainerT &nodeSelection,
                               std::unordered_set<EdgeType> &currentViolations,
                               const unsigned startStep,
                               const unsigned endStep) {
        for (const auto &edge : current_violations) {
            const auto source_v = source(edge, *graph);
            const auto target_v = target(edge, *graph);

            const unsigned source_step = active_schedule->assigned_superstep(source_v);
            if (source_step >= start_step && source_step <= end_step) {
                node_selection.insert(source_v);
            }

            const unsigned target_step = active_schedule->assigned_superstep(target_v);
            if (target_step >= start_step && target_step <= end_step) {
                node_selection.insert(target_v);
            }
        }
    }

    void SelectNodesPermutationThreshold(const std::size_t &threshold, ContainerT &nodeSelection) {
        const size_t bound = std::min(threshold + permutation_idx, permutation.size());
        for (std::size_t i = permutationIdx_; i < bound; i++) {
            node_selection.insert(permutation[i]);
        }

        permutationIdx_ = bound;
        if (permutation_idx + threshold >= permutation.size()) {
            permutationIdx_ = 0;
            std::shuffle(permutation.begin(), permutation.end(), *gen);
        }
    }

    void SelectNodesMaxWorkProc(const std::size_t &threshold,
                                ContainerT &nodeSelection,
                                const unsigned startStep,
                                const unsigned endStep) {
        while (nodeSelection.size() < threshold) {
            if (maxWorkCounter_ > endStep) {
                maxWorkCounter_ = startStep;    // wrap around
                break;                          // stop after one full pass
            }

            SelectNodesMaxWorkProcHelper(threshold - nodeSelection.size(), maxWorkCounter_, nodeSelection);
            maxWorkCounter_++;
        }
    }

    void SelectNodesMaxWorkProcHelper(const std::size_t &threshold, unsigned step, ContainerT &nodeSelection) {
        const unsigned numMaxWorkProc = activeSchedule_->work_datastructures.step_max_work_processor_count[step];
        for (unsigned idx = 0; idx < numMaxWorkProc; idx++) {
            const unsigned proc = activeSchedule_->work_datastructures.step_processor_work_[step][idx].proc;
            const std::unordered_set<vertex_idx_t<Graph_t>> stepProcVert
                = activeSchedule_->getSetSchedule().step_processor_vertices[step][proc];
            const size_t numInsert = std::min(threshold - nodeSelection.size(), step_proc_vert.size());
            auto endIt = step_proc_vert.begin();
            std::advance(end_it, numInsert);
            std::for_each(step_proc_vert.begin(), end_it, [&](const auto &val) { nodeSelection.insert(val); });
        }
    }
};

}    // namespace osp
