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
#include "osp/bsp/model/util/SetSchedule.hpp"
#include "osp/bsp/model/util/VectorSchedule.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"

namespace osp {

/**
 * @brief A trait to check if a type is a memory constraint.
 *
 * This trait checks if a type has the required methods for a memory constraint.
 *
 */
template <typename T, typename = void>
struct IsLocalSearchMemoryConstraint : std::false_type {};

template <typename T>
struct IsLocalSearchMemoryConstraint<
    T,
    std::void_t<decltype(std::declval<T>().initialize(std::declval<SetSchedule<typename T::Graph_impl_t>>(),
                                                      std::declval<VectorSchedule<typename T::Graph_impl_t>>())),
                decltype(std::declval<T>().apply_move(std::declval<vertex_idx_t<typename T::Graph_impl_t>>(),
                                                      std::declval<unsigned>(),
                                                      std::declval<unsigned>(),
                                                      std::declval<unsigned>(),
                                                      std::declval<unsigned>())),
                decltype(std::declval<T>().compute_memory_datastructure(std::declval<unsigned>(), std::declval<unsigned>())),
                decltype(std::declval<T>().swap_steps(std::declval<unsigned>(), std::declval<unsigned>())),
                decltype(std::declval<T>().reset_superstep(std::declval<unsigned>())),
                decltype(std::declval<T>().override_superstep(
                    std::declval<unsigned>(), std::declval<unsigned>(), std::declval<unsigned>(), std::declval<unsigned>())),
                decltype(std::declval<T>().can_move(
                    std::declval<vertex_idx_t<typename T::Graph_impl_t>>(), std::declval<unsigned>(), std::declval<unsigned>())),
                decltype(std::declval<T>().clear()),
                decltype(T())>> : std::true_type {};

template <typename T>
inline constexpr bool isLocalSearchMemoryConstraintV = IsLocalSearchMemoryConstraint<T>::value;

/**
 * @brief The default memory constraint type, no memory constraints apply.
 *
 */
struct NoLocalSearchMemoryConstraint {
    using GraphImplT = void;
};

/**
 * @brief A memory constraint module for local memory constraints.
 *
 * @tparam Graph_t The graph type.
 */
template <typename GraphT>
struct LsLocalMemoryConstraint {
    using GraphImplT = GraphT;

    const SetSchedule<GraphT> *setSchedule_;
    const GraphT *graph_;

    std::vector<std::vector<v_memw_t<Graph_t>>> stepProcessorMemory_;

    LsLocalMemoryConstraint() : setSchedule_(nullptr), graph_(nullptr) {}

    inline void Initialize(const SetSchedule<GraphT> &setSchedule, const VectorSchedule<GraphT> &) {
        if (set_schedule_.getInstance().getArchitecture().getMemoryConstraintType() != MEMORY_CONSTRAINT_TYPE::LOCAL) {
            throw std::invalid_argument("Memory constraint type is not LOCAL");
        }

        setSchedule_ = &setSchedule;
        graph_ = &setSchedule_->getInstance().getComputationalDag();
        step_processor_memory = std::vector<std::vector<v_memw_t<Graph_t>>>(
            set_schedule->numberOfSupersteps(),
            std::vector<v_memw_t<Graph_t>>(set_schedule->getInstance().numberOfProcessors(), 0));
    }

    inline void ApplyMove(vertex_idx_t<Graph_t> vertex, unsigned fromProc, unsigned fromStep, unsigned toProc, unsigned toStep) {
        step_processor_memory[to_step][to_proc] += graph->VertexMemWeight(vertex);
        step_processor_memory[from_step][from_proc] -= graph->VertexMemWeight(vertex);
    }

    inline bool CanMove(vertex_idx_t<Graph_t> vertex, const unsigned proc, unsigned step) const {
        return step_processor_memory[step][proc] + graph->VertexMemWeight(vertex)
               <= set_schedule->getInstance().getArchitecture().memoryBound(proc);
    }

    void SwapSteps(const unsigned step1, const unsigned step2) {
        std::swap(step_processor_memory[step1], step_processor_memory[step2]);
    }

    void ComputeMemoryDatastructure(unsigned startStep, unsigned endStep) {
        for (unsigned step = startStep; step <= endStep; step++) {
            for (unsigned proc = 0; proc < setSchedule_->getInstance().numberOfProcessors(); proc++) {
                step_processor_memory[step][proc] = 0;

                for (const auto &node : setSchedule_->step_processor_vertices[step][proc]) {
                    step_processor_memory[step][proc] += graph->VertexMemWeight(node);
                }
            }
        }
    }

    inline void Clear() { step_processor_memory.clear(); }

    inline void ForwardMove(vertex_idx_t<Graph_t> vertex, unsigned, unsigned, unsigned toProc, unsigned toStep) {
        step_processor_memory[to_step][to_proc] += graph->VertexMemWeight(vertex);
        // step_processor_memory[from_step][from_proc] -= graph->VertexMemWeight(vertex);
    }

    inline void ResetSuperstep(unsigned step) {
        for (unsigned proc = 0; proc < setSchedule_->getInstance().getArchitecture().numberOfProcessors(); proc++) {
            step_processor_memory[step][proc] = 0;
        }
    }

    void OverrideSuperstep(unsigned step, unsigned proc, unsigned withStep, unsigned withProc) {
        step_processor_memory[step][proc] = step_processor_memory[with_step][with_proc];
    }

    bool SatisfiedMemoryConstraint() const {
        for (unsigned step = 0; step < setSchedule_->numberOfSupersteps(); step++) {
            for (unsigned proc = 0; proc < setSchedule_->getInstance().numberOfProcessors(); proc++) {
                if (step_processor_memory[step][proc] > set_schedule->getInstance().getArchitecture().memoryBound(proc)) {
                    return false;
                }
            }
        }
        return true;
    }
};

template <typename GraphT>
struct LsLocalIncEdgesMemoryConstraint {
    using GraphImplT = GraphT;

    const SetSchedule<GraphT> *setSchedule_;
    const VectorSchedule<GraphT> *vectorSchedule_;
    const GraphT *graph_;

    std::vector<std::vector<v_memw_t<Graph_t>>> stepProcessorMemory_;
    std::vector<std::vector<std::unordered_set<vertex_idx_t<Graph_t>>>> stepProcessorPred_;

    LsLocalIncEdgesMemoryConstraint() : setSchedule_(nullptr), vectorSchedule_(nullptr), graph_(nullptr) {}

    inline void Initialize(const SetSchedule<GraphT> &setSchedule, const VectorSchedule<GraphT> &vecSchedule) {
        if (set_schedule_.getInstance().getArchitecture().getMemoryConstraintType() != MEMORY_CONSTRAINT_TYPE::LOCAL_INC_EDGES) {
            throw std::invalid_argument("Memory constraint type is not LOCAL_INC_EDGES");
        }

        setSchedule_ = &setSchedule;
        vectorSchedule_ = &vecSchedule;
        graph_ = &setSchedule_->getInstance().getComputationalDag();
        step_processor_memory = std::vector<std::vector<v_memw_t<Graph_t>>>(
            set_schedule->numberOfSupersteps(),
            std::vector<v_memw_t<Graph_t>>(set_schedule->getInstance().numberOfProcessors(), 0));
        step_processor_pred = std::vector<std::vector<std::unordered_set<vertex_idx_t<Graph_t>>>>(
            set_schedule->numberOfSupersteps(),
            std::vector<std::unordered_set<vertex_idx_t<Graph_t>>>(set_schedule->getInstance().numberOfProcessors()));
    }

    inline void ApplyMove(vertex_idx_t<Graph_t> vertex, unsigned fromProc, unsigned fromStep, unsigned toProc, unsigned toStep) {
        step_processor_memory[to_step][to_proc] += graph->VertexCommWeight(vertex);
        step_processor_memory[from_step][from_proc] -= graph->VertexCommWeight(vertex);

        for (const auto &pred : graph->parents(vertex)) {
            if (vector_schedule->assignedSuperstep(pred) < to_step) {
                auto pair = step_processor_pred[to_step][to_proc].insert(pred);
                if (pair.second) {
                    step_processor_memory[to_step][to_proc] += graph->VertexCommWeight(pred);
                }
            }

            if (vector_schedule->assignedSuperstep(pred) < from_step) {
                bool remove = true;
                for (const auto &succ : graph->children(pred)) {
                    if (succ == vertex) {
                        continue;
                    }

                    if (vector_schedule->assignedProcessor(succ) == from_proc
                        && vector_schedule->assignedSuperstep(succ) == from_step) {
                        remove = false;
                        break;
                    }
                }

                if (remove) {
                    step_processor_memory[from_step][from_proc] -= graph->VertexCommWeight(pred);
                    step_processor_pred[from_step][from_proc].erase(pred);
                }
            }
        }

        if (toStep != fromStep) {
            for (const auto &succ : graph->children(vertex)) {
                if (to_step > from_step && vector_schedule->assignedSuperstep(succ) == to_step) {
                    if (step_processor_pred[vector_schedule->assignedSuperstep(succ)][vector_schedule->assignedProcessor(succ)].find(
                            vertex)
                        != step_processor_pred[vector_schedule->assignedSuperstep(succ)][vector_schedule->assignedProcessor(succ)]
                               .end()) {
                        step_processor_memory[vector_schedule->assignedSuperstep(succ)][vector_schedule->assignedProcessor(succ)]
                            -= graph->VertexCommWeight(vertex);

                        step_processor_pred[vector_schedule->assignedSuperstep(succ)][vector_schedule->assignedProcessor(succ)].erase(
                            vertex);
                    }
                }

                if (vector_schedule->assignedSuperstep(succ) > to_step) {
                    auto pair
                        = step_processor_pred[vector_schedule->assignedSuperstep(succ)][vector_schedule->assignedProcessor(succ)]
                              .insert(vertex);
                    if (pair.second) {
                        step_processor_memory[vector_schedule->assignedSuperstep(succ)][vector_schedule->assignedProcessor(succ)]
                            += graph->VertexCommWeight(vertex);
                    }
                }
            }
        }
    }

    void SwapSteps(const unsigned step1, const unsigned step2) {
        std::swap(step_processor_memory[step1], step_processor_memory[step2]);
        std::swap(step_processor_pred[step1], step_processor_pred[step2]);
    }

    void ComputeMemoryDatastructure(unsigned startStep, unsigned endStep) {
        for (unsigned step = startStep; step <= endStep; step++) {
            for (unsigned proc = 0; proc < setSchedule_->getInstance().numberOfProcessors(); proc++) {
                step_processor_memory[step][proc] = 0;
                step_processor_pred[step][proc].clear();

                for (const auto &node : setSchedule_->step_processor_vertices[step][proc]) {
                    step_processor_memory[step][proc] += graph->VertexCommWeight(node);

                    for (const auto &pred : graph_->parents(node)) {
                        if (vectorSchedule_->assignedSuperstep(pred) < step) {
                            auto pair = step_processor_pred[step][proc].insert(pred);
                            if (pair.second) {
                                step_processor_memory[step][proc] += graph->VertexCommWeight(pred);
                            }
                        }
                    }
                }
            }
        }
    }

    inline void Clear() {
        step_processor_memory.clear();
        step_processor_pred.clear();
    }

    inline void ResetSuperstep(unsigned step) {
        for (unsigned proc = 0; proc < setSchedule_->getInstance().getArchitecture().numberOfProcessors(); proc++) {
            step_processor_memory[step][proc] = 0;
            step_processor_pred[step][proc].clear();
        }
    }

    void OverrideSuperstep(unsigned step, unsigned proc, unsigned withStep, unsigned withProc) {
        step_processor_memory[step][proc] = step_processor_memory[with_step][with_proc];
        step_processor_pred[step][proc] = step_processor_pred[with_step][with_proc];
    }

    inline bool CanMove(vertex_idx_t<Graph_t> vertex, const unsigned proc, unsigned step) const {
        v_memw_t<Graph_t> incMemory = graph_->VertexCommWeight(vertex);
        for (const auto &pred : graph->parents(vertex)) {
            if (vector_schedule->assignedSuperstep(pred) < step) {
                if (step_processor_pred[step][proc].find(pred) == step_processor_pred[step][proc].end()) {
                    inc_memory += graph->VertexCommWeight(pred);
                }
            }
        }

        if (step > vectorSchedule_->assignedSuperstep(vertex)) {
            if (step_processor_pred[step][proc].find(vertex) != step_processor_pred[step][proc].end()) {
                incMemory -= graph_->VertexCommWeight(vertex);
            }
        }

        if (step >= vectorSchedule_->assignedSuperstep(vertex)) {
            return step_processor_memory[step][proc] + inc_memory
                   <= set_schedule->getInstance().getArchitecture().memoryBound(proc);
        }

        if (step_processor_memory[step][proc] + inc_memory > set_schedule->getInstance().getArchitecture().memoryBound(proc)) {
            return false;
        }

        for (const auto &succ : graph->children(vertex)) {
            const auto &succ_step = vector_schedule->assignedSuperstep(succ);
            const auto &succ_proc = vector_schedule->assignedProcessor(succ);

            if (succ_step == vector_schedule->assignedSuperstep(vertex)
                and succ_proc != vector_schedule->assignedProcessor(vertex)) {
                if (step_processor_memory[succ_step][succ_proc] + graph->VertexCommWeight(vertex)
                    > set_schedule->getInstance().getArchitecture().memoryBound(succ_proc)) {
                    return false;
                }
            }
        }

        return true;
    }
};

template <typename GraphT>
struct LsLocalSourcesIncEdgesMemoryConstraint {
    using GraphImplT = GraphT;

    const SetSchedule<GraphT> *setSchedule_;
    const VectorSchedule<GraphT> *vectorSchedule_;
    const GraphT *graph_;

    std::vector<std::vector<v_memw_t<Graph_t>>> stepProcessorMemory_;
    std::vector<std::vector<std::unordered_set<vertex_idx_t<Graph_t>>>> stepProcessorPred_;

    LsLocalSourcesIncEdgesMemoryConstraint() : setSchedule_(nullptr), vectorSchedule_(nullptr), graph_(nullptr) {}

    inline void SwapSteps(const unsigned step1, const unsigned step2) {
        std::swap(step_processor_memory[step1], step_processor_memory[step2]);
        std::swap(step_processor_pred[step1], step_processor_pred[step2]);
    }

    inline void Initialize(const SetSchedule<GraphT> &setSchedule, const VectorSchedule<GraphT> &vecSchedule) {
        if (set_schedule_.getInstance().getArchitecture().getMemoryConstraintType()
            != MEMORY_CONSTRAINT_TYPE::LOCAL_SOURCES_INC_EDGES) {
            throw std::invalid_argument("Memory constraint type is not LOCAL_SOURCES_INC_EDGES");
        }

        setSchedule_ = &setSchedule;
        vectorSchedule_ = &vecSchedule;
        graph_ = &setSchedule_->getInstance().getComputationalDag();
        step_processor_memory = std::vector<std::vector<v_memw_t<Graph_t>>>(
            set_schedule->numberOfSupersteps(),
            std::vector<v_memw_t<Graph_t>>(set_schedule->getInstance().numberOfProcessors(), 0));
        step_processor_pred = std::vector<std::vector<std::unordered_set<vertex_idx_t<Graph_t>>>>(
            set_schedule->numberOfSupersteps(),
            std::vector<std::unordered_set<vertex_idx_t<Graph_t>>>(set_schedule->getInstance().numberOfProcessors()));
    }

    inline void ApplyMove(vertex_idx_t<Graph_t> vertex, unsigned fromProc, unsigned fromStep, unsigned toProc, unsigned toStep) {
        if (IsSource(vertex, *graph_)) {
            step_processor_memory[to_step][to_proc] += graph->VertexMemWeight(vertex);
            step_processor_memory[from_step][from_proc] -= graph->VertexMemWeight(vertex);
        }

        for (const auto &pred : graph->parents(vertex)) {
            if (vector_schedule->assignedSuperstep(pred) < to_step) {
                auto pair = step_processor_pred[to_step][to_proc].insert(pred);
                if (pair.second) {
                    step_processor_memory[to_step][to_proc] += graph->VertexCommWeight(pred);
                }
            }

            if (vector_schedule->assignedSuperstep(pred) < from_step) {
                bool remove = true;
                for (const auto &succ : graph->children(pred)) {
                    if (succ == vertex) {
                        continue;
                    }

                    if (vector_schedule->assignedProcessor(succ) == from_proc
                        && vector_schedule->assignedSuperstep(succ) == from_step) {
                        remove = false;
                        break;
                    }
                }

                if (remove) {
                    step_processor_memory[from_step][from_proc] -= graph->VertexCommWeight(pred);
                    step_processor_pred[from_step][from_proc].erase(pred);
                }
            }
        }

        if (toStep != fromStep) {
            for (const auto &succ : graph->children(vertex)) {
                if (to_step > from_step && vector_schedule->assignedSuperstep(succ) == to_step) {
                    if (step_processor_pred[vector_schedule->assignedSuperstep(succ)][vector_schedule->assignedProcessor(succ)].find(
                            vertex)
                        != step_processor_pred[vector_schedule->assignedSuperstep(succ)][vector_schedule->assignedProcessor(succ)]
                               .end()) {
                        step_processor_memory[vector_schedule->assignedSuperstep(succ)][vector_schedule->assignedProcessor(succ)]
                            -= graph->VertexCommWeight(vertex);

                        step_processor_pred[vector_schedule->assignedSuperstep(succ)][vector_schedule->assignedProcessor(succ)].erase(
                            vertex);
                    }
                }

                if (vector_schedule->assignedSuperstep(succ) > to_step) {
                    auto pair
                        = step_processor_pred[vector_schedule->assignedSuperstep(succ)][vector_schedule->assignedProcessor(succ)]
                              .insert(vertex);
                    if (pair.second) {
                        step_processor_memory[vector_schedule->assignedSuperstep(succ)][vector_schedule->assignedProcessor(succ)]
                            += graph->VertexCommWeight(vertex);
                    }
                }
            }
        }
    }

    void ComputeMemoryDatastructure(unsigned startStep, unsigned endStep) {
        for (unsigned step = startStep; step <= endStep; step++) {
            for (unsigned proc = 0; proc < setSchedule_->getInstance().numberOfProcessors(); proc++) {
                step_processor_memory[step][proc] = 0;
                step_processor_pred[step][proc].clear();

                for (const auto &node : setSchedule_->step_processor_vertices[step][proc]) {
                    if (IsSource(node, *graph_)) {
                        step_processor_memory[step][proc] += graph->VertexMemWeight(node);
                    }

                    for (const auto &pred : graph_->parents(node)) {
                        if (vectorSchedule_->assignedSuperstep(pred) < step) {
                            auto pair = step_processor_pred[step][proc].insert(pred);
                            if (pair.second) {
                                step_processor_memory[step][proc] += graph->VertexCommWeight(pred);
                            }
                        }
                    }
                }
            }
        }
    }

    inline void Clear() {
        step_processor_memory.clear();
        step_processor_pred.clear();
    }

    inline void ResetSuperstep(unsigned step) {
        for (unsigned proc = 0; proc < setSchedule_->getInstance().getArchitecture().numberOfProcessors(); proc++) {
            step_processor_memory[step][proc] = 0;
            step_processor_pred[step][proc].clear();
        }
    }

    void OverrideSuperstep(unsigned step, unsigned proc, unsigned withStep, unsigned withProc) {
        step_processor_memory[step][proc] = step_processor_memory[with_step][with_proc];
        step_processor_pred[step][proc] = step_processor_pred[with_step][with_proc];
    }

    inline bool CanMove(vertex_idx_t<Graph_t> vertex, const unsigned proc, unsigned step) const {
        v_memw_t<Graph_t> incMemory = 0;

        if (IsSource(vertex, *graph_)) {
            incMemory += graph_->VertexMemWeight(vertex);
        }

        for (const auto &pred : graph->parents(vertex)) {
            if (vector_schedule->assignedSuperstep(pred) < step) {
                if (step_processor_pred[step][proc].find(pred) == step_processor_pred[step][proc].end()) {
                    inc_memory += graph->VertexCommWeight(pred);
                }
            }
        }

        if (vectorSchedule_->assignedSuperstep(vertex) < step) {
            if (step_processor_pred[step][proc].find(vertex) != step_processor_pred[step][proc].end()) {
                incMemory -= graph_->VertexCommWeight(vertex);
            }
        }

        if (vectorSchedule_->assignedSuperstep(vertex) <= step) {
            return step_processor_memory[step][proc] + inc_memory
                   <= set_schedule->getInstance().getArchitecture().memoryBound(proc);
        }

        if (step_processor_memory[step][proc] + inc_memory > set_schedule->getInstance().getArchitecture().memoryBound(proc)) {
            return false;
        }

        for (const auto &succ : graph->children(vertex)) {
            const auto &succ_step = vector_schedule->assignedSuperstep(succ);
            const auto &succ_proc = vector_schedule->assignedProcessor(succ);

            if (succ_step == vector_schedule->assignedSuperstep(vertex)) {
                if (vector_schedule->assignedProcessor(vertex) != succ_proc || (not IsSource(vertex, *graph))) {
                    if (step_processor_memory[succ_step][succ_proc] + graph->VertexCommWeight(vertex)
                        > set_schedule->getInstance().getArchitecture().memoryBound(succ_proc)) {
                        return false;
                    }

                } else {
                    if (IsSource(vertex, *graph)) {
                        if (step_processor_memory[succ_step][succ_proc] + graph->VertexCommWeight(vertex)
                                - graph->VertexMemWeight(vertex)
                            > set_schedule->getInstance().getArchitecture().memoryBound(succ_proc)) {
                            return false;
                        }
                    }
                }
            }
        }

        return true;
    }
};

}    // namespace osp
