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
struct is_local_search_memory_constraint : std::false_type {};

template <typename T>
struct is_local_search_memory_constraint<
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
inline constexpr bool is_local_search_memory_constraint_v = is_local_search_memory_constraint<T>::value;

/**
 * @brief The default memory constraint type, no memory constraints apply.
 *
 */
struct no_local_search_memory_constraint {
    using Graph_impl_t = void;
};

/**
 * @brief A memory constraint module for local memory constraints.
 *
 * @tparam Graph_t The graph type.
 */
template <typename Graph_t>
struct ls_local_memory_constraint {
    using Graph_impl_t = Graph_t;

    const SetSchedule<Graph_t> *set_schedule;
    const Graph_t *graph;

    std::vector<std::vector<v_memw_t<Graph_t>>> step_processor_memory;

    ls_local_memory_constraint() : set_schedule(nullptr), graph(nullptr) {}

    inline void initialize(const SetSchedule<Graph_t> &set_schedule_, const VectorSchedule<Graph_t> &) {
        if (set_schedule_.getInstance().getArchitecture().getMemoryConstraintType() != MEMORY_CONSTRAINT_TYPE::LOCAL) {
            throw std::invalid_argument("Memory constraint type is not LOCAL");
        }

        set_schedule = &set_schedule_;
        graph = &set_schedule->getInstance().getComputationalDag();
        step_processor_memory = std::vector<std::vector<v_memw_t<Graph_t>>>(
            set_schedule->numberOfSupersteps(),
            std::vector<v_memw_t<Graph_t>>(set_schedule->getInstance().numberOfProcessors(), 0));
    }

    inline void apply_move(vertex_idx_t<Graph_t> vertex, unsigned from_proc, unsigned from_step, unsigned to_proc, unsigned to_step) {
        step_processor_memory[to_step][to_proc] += graph->vertex_mem_weight(vertex);
        step_processor_memory[from_step][from_proc] -= graph->vertex_mem_weight(vertex);
    }

    inline bool can_move(vertex_idx_t<Graph_t> vertex, const unsigned proc, unsigned step) const {
        return step_processor_memory[step][proc] + graph->vertex_mem_weight(vertex)
               <= set_schedule->getInstance().getArchitecture().memoryBound(proc);
    }

    void swap_steps(const unsigned step1, const unsigned step2) {
        std::swap(step_processor_memory[step1], step_processor_memory[step2]);
    }

    void compute_memory_datastructure(unsigned start_step, unsigned end_step) {
        for (unsigned step = start_step; step <= end_step; step++) {
            for (unsigned proc = 0; proc < set_schedule->getInstance().numberOfProcessors(); proc++) {
                step_processor_memory[step][proc] = 0;

                for (const auto &node : set_schedule->step_processor_vertices[step][proc]) {
                    step_processor_memory[step][proc] += graph->vertex_mem_weight(node);
                }
            }
        }
    }

    inline void clear() { step_processor_memory.clear(); }

    inline void forward_move(vertex_idx_t<Graph_t> vertex, unsigned, unsigned, unsigned to_proc, unsigned to_step) {
        step_processor_memory[to_step][to_proc] += graph->vertex_mem_weight(vertex);
        // step_processor_memory[from_step][from_proc] -= graph->vertex_mem_weight(vertex);
    }

    inline void reset_superstep(unsigned step) {
        for (unsigned proc = 0; proc < set_schedule->getInstance().getArchitecture().numberOfProcessors(); proc++) {
            step_processor_memory[step][proc] = 0;
        }
    }

    void override_superstep(unsigned step, unsigned proc, unsigned with_step, unsigned with_proc) {
        step_processor_memory[step][proc] = step_processor_memory[with_step][with_proc];
    }

    bool satisfied_memory_constraint() const {
        for (unsigned step = 0; step < set_schedule->numberOfSupersteps(); step++) {
            for (unsigned proc = 0; proc < set_schedule->getInstance().numberOfProcessors(); proc++) {
                if (step_processor_memory[step][proc] > set_schedule->getInstance().getArchitecture().memoryBound(proc)) {
                    return false;
                }
            }
        }
        return true;
    }
};

template <typename Graph_t>
struct ls_local_inc_edges_memory_constraint {
    using Graph_impl_t = Graph_t;

    const SetSchedule<Graph_t> *set_schedule;
    const VectorSchedule<Graph_t> *vector_schedule;
    const Graph_t *graph;

    std::vector<std::vector<v_memw_t<Graph_t>>> step_processor_memory;
    std::vector<std::vector<std::unordered_set<vertex_idx_t<Graph_t>>>> step_processor_pred;

    ls_local_inc_edges_memory_constraint() : set_schedule(nullptr), vector_schedule(nullptr), graph(nullptr) {}

    inline void initialize(const SetSchedule<Graph_t> &set_schedule_, const VectorSchedule<Graph_t> &vec_schedule_) {
        if (set_schedule_.getInstance().getArchitecture().getMemoryConstraintType() != MEMORY_CONSTRAINT_TYPE::LOCAL_INC_EDGES) {
            throw std::invalid_argument("Memory constraint type is not LOCAL_INC_EDGES");
        }

        set_schedule = &set_schedule_;
        vector_schedule = &vec_schedule_;
        graph = &set_schedule->getInstance().getComputationalDag();
        step_processor_memory = std::vector<std::vector<v_memw_t<Graph_t>>>(
            set_schedule->numberOfSupersteps(),
            std::vector<v_memw_t<Graph_t>>(set_schedule->getInstance().numberOfProcessors(), 0));
        step_processor_pred = std::vector<std::vector<std::unordered_set<vertex_idx_t<Graph_t>>>>(
            set_schedule->numberOfSupersteps(),
            std::vector<std::unordered_set<vertex_idx_t<Graph_t>>>(set_schedule->getInstance().numberOfProcessors()));
    }

    inline void apply_move(vertex_idx_t<Graph_t> vertex, unsigned from_proc, unsigned from_step, unsigned to_proc, unsigned to_step) {
        step_processor_memory[to_step][to_proc] += graph->vertex_comm_weight(vertex);
        step_processor_memory[from_step][from_proc] -= graph->vertex_comm_weight(vertex);

        for (const auto &pred : graph->parents(vertex)) {
            if (vector_schedule->assignedSuperstep(pred) < to_step) {
                auto pair = step_processor_pred[to_step][to_proc].insert(pred);
                if (pair.second) { step_processor_memory[to_step][to_proc] += graph->vertex_comm_weight(pred); }
            }

            if (vector_schedule->assignedSuperstep(pred) < from_step) {
                bool remove = true;
                for (const auto &succ : graph->children(pred)) {
                    if (succ == vertex) { continue; }

                    if (vector_schedule->assignedProcessor(succ) == from_proc
                        && vector_schedule->assignedSuperstep(succ) == from_step) {
                        remove = false;
                        break;
                    }
                }

                if (remove) {
                    step_processor_memory[from_step][from_proc] -= graph->vertex_comm_weight(pred);
                    step_processor_pred[from_step][from_proc].erase(pred);
                }
            }
        }

        if (to_step != from_step) {
            for (const auto &succ : graph->children(vertex)) {
                if (to_step > from_step && vector_schedule->assignedSuperstep(succ) == to_step) {
                    if (step_processor_pred[vector_schedule->assignedSuperstep(succ)][vector_schedule->assignedProcessor(succ)].find(
                            vertex)
                        != step_processor_pred[vector_schedule->assignedSuperstep(succ)][vector_schedule->assignedProcessor(succ)]
                               .end()) {
                        step_processor_memory[vector_schedule->assignedSuperstep(succ)][vector_schedule->assignedProcessor(succ)]
                            -= graph->vertex_comm_weight(vertex);

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
                            += graph->vertex_comm_weight(vertex);
                    }
                }
            }
        }
    }

    void swap_steps(const unsigned step1, const unsigned step2) {
        std::swap(step_processor_memory[step1], step_processor_memory[step2]);
        std::swap(step_processor_pred[step1], step_processor_pred[step2]);
    }

    void compute_memory_datastructure(unsigned start_step, unsigned end_step) {
        for (unsigned step = start_step; step <= end_step; step++) {
            for (unsigned proc = 0; proc < set_schedule->getInstance().numberOfProcessors(); proc++) {
                step_processor_memory[step][proc] = 0;
                step_processor_pred[step][proc].clear();

                for (const auto &node : set_schedule->step_processor_vertices[step][proc]) {
                    step_processor_memory[step][proc] += graph->vertex_comm_weight(node);

                    for (const auto &pred : graph->parents(node)) {
                        if (vector_schedule->assignedSuperstep(pred) < step) {
                            auto pair = step_processor_pred[step][proc].insert(pred);
                            if (pair.second) { step_processor_memory[step][proc] += graph->vertex_comm_weight(pred); }
                        }
                    }
                }
            }
        }
    }

    inline void clear() {
        step_processor_memory.clear();
        step_processor_pred.clear();
    }

    inline void reset_superstep(unsigned step) {
        for (unsigned proc = 0; proc < set_schedule->getInstance().getArchitecture().numberOfProcessors(); proc++) {
            step_processor_memory[step][proc] = 0;
            step_processor_pred[step][proc].clear();
        }
    }

    void override_superstep(unsigned step, unsigned proc, unsigned with_step, unsigned with_proc) {
        step_processor_memory[step][proc] = step_processor_memory[with_step][with_proc];
        step_processor_pred[step][proc] = step_processor_pred[with_step][with_proc];
    }

    inline bool can_move(vertex_idx_t<Graph_t> vertex, const unsigned proc, unsigned step) const {
        v_memw_t<Graph_t> inc_memory = graph->vertex_comm_weight(vertex);
        for (const auto &pred : graph->parents(vertex)) {
            if (vector_schedule->assignedSuperstep(pred) < step) {
                if (step_processor_pred[step][proc].find(pred) == step_processor_pred[step][proc].end()) {
                    inc_memory += graph->vertex_comm_weight(pred);
                }
            }
        }

        if (step > vector_schedule->assignedSuperstep(vertex)) {
            if (step_processor_pred[step][proc].find(vertex) != step_processor_pred[step][proc].end()) {
                inc_memory -= graph->vertex_comm_weight(vertex);
            }
        }

        if (step >= vector_schedule->assignedSuperstep(vertex)) {
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
                if (step_processor_memory[succ_step][succ_proc] + graph->vertex_comm_weight(vertex)
                    > set_schedule->getInstance().getArchitecture().memoryBound(succ_proc)) {
                    return false;
                }
            }
        }

        return true;
    }
};

template <typename Graph_t>
struct ls_local_sources_inc_edges_memory_constraint {
    using Graph_impl_t = Graph_t;

    const SetSchedule<Graph_t> *set_schedule;
    const VectorSchedule<Graph_t> *vector_schedule;
    const Graph_t *graph;

    std::vector<std::vector<v_memw_t<Graph_t>>> step_processor_memory;
    std::vector<std::vector<std::unordered_set<vertex_idx_t<Graph_t>>>> step_processor_pred;

    ls_local_sources_inc_edges_memory_constraint() : set_schedule(nullptr), vector_schedule(nullptr), graph(nullptr) {}

    inline void swap_steps(const unsigned step1, const unsigned step2) {
        std::swap(step_processor_memory[step1], step_processor_memory[step2]);
        std::swap(step_processor_pred[step1], step_processor_pred[step2]);
    }

    inline void initialize(const SetSchedule<Graph_t> &set_schedule_, const VectorSchedule<Graph_t> &vec_schedule_) {
        if (set_schedule_.getInstance().getArchitecture().getMemoryConstraintType()
            != MEMORY_CONSTRAINT_TYPE::LOCAL_SOURCES_INC_EDGES) {
            throw std::invalid_argument("Memory constraint type is not LOCAL_SOURCES_INC_EDGES");
        }

        set_schedule = &set_schedule_;
        vector_schedule = &vec_schedule_;
        graph = &set_schedule->getInstance().getComputationalDag();
        step_processor_memory = std::vector<std::vector<v_memw_t<Graph_t>>>(
            set_schedule->numberOfSupersteps(),
            std::vector<v_memw_t<Graph_t>>(set_schedule->getInstance().numberOfProcessors(), 0));
        step_processor_pred = std::vector<std::vector<std::unordered_set<vertex_idx_t<Graph_t>>>>(
            set_schedule->numberOfSupersteps(),
            std::vector<std::unordered_set<vertex_idx_t<Graph_t>>>(set_schedule->getInstance().numberOfProcessors()));
    }

    inline void apply_move(vertex_idx_t<Graph_t> vertex, unsigned from_proc, unsigned from_step, unsigned to_proc, unsigned to_step) {
        if (is_source(vertex, *graph)) {
            step_processor_memory[to_step][to_proc] += graph->vertex_mem_weight(vertex);
            step_processor_memory[from_step][from_proc] -= graph->vertex_mem_weight(vertex);
        }

        for (const auto &pred : graph->parents(vertex)) {
            if (vector_schedule->assignedSuperstep(pred) < to_step) {
                auto pair = step_processor_pred[to_step][to_proc].insert(pred);
                if (pair.second) { step_processor_memory[to_step][to_proc] += graph->vertex_comm_weight(pred); }
            }

            if (vector_schedule->assignedSuperstep(pred) < from_step) {
                bool remove = true;
                for (const auto &succ : graph->children(pred)) {
                    if (succ == vertex) { continue; }

                    if (vector_schedule->assignedProcessor(succ) == from_proc
                        && vector_schedule->assignedSuperstep(succ) == from_step) {
                        remove = false;
                        break;
                    }
                }

                if (remove) {
                    step_processor_memory[from_step][from_proc] -= graph->vertex_comm_weight(pred);
                    step_processor_pred[from_step][from_proc].erase(pred);
                }
            }
        }

        if (to_step != from_step) {
            for (const auto &succ : graph->children(vertex)) {
                if (to_step > from_step && vector_schedule->assignedSuperstep(succ) == to_step) {
                    if (step_processor_pred[vector_schedule->assignedSuperstep(succ)][vector_schedule->assignedProcessor(succ)].find(
                            vertex)
                        != step_processor_pred[vector_schedule->assignedSuperstep(succ)][vector_schedule->assignedProcessor(succ)]
                               .end()) {
                        step_processor_memory[vector_schedule->assignedSuperstep(succ)][vector_schedule->assignedProcessor(succ)]
                            -= graph->vertex_comm_weight(vertex);

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
                            += graph->vertex_comm_weight(vertex);
                    }
                }
            }
        }
    }

    void compute_memory_datastructure(unsigned start_step, unsigned end_step) {
        for (unsigned step = start_step; step <= end_step; step++) {
            for (unsigned proc = 0; proc < set_schedule->getInstance().numberOfProcessors(); proc++) {
                step_processor_memory[step][proc] = 0;
                step_processor_pred[step][proc].clear();

                for (const auto &node : set_schedule->step_processor_vertices[step][proc]) {
                    if (is_source(node, *graph)) { step_processor_memory[step][proc] += graph->vertex_mem_weight(node); }

                    for (const auto &pred : graph->parents(node)) {
                        if (vector_schedule->assignedSuperstep(pred) < step) {
                            auto pair = step_processor_pred[step][proc].insert(pred);
                            if (pair.second) { step_processor_memory[step][proc] += graph->vertex_comm_weight(pred); }
                        }
                    }
                }
            }
        }
    }

    inline void clear() {
        step_processor_memory.clear();
        step_processor_pred.clear();
    }

    inline void reset_superstep(unsigned step) {
        for (unsigned proc = 0; proc < set_schedule->getInstance().getArchitecture().numberOfProcessors(); proc++) {
            step_processor_memory[step][proc] = 0;
            step_processor_pred[step][proc].clear();
        }
    }

    void override_superstep(unsigned step, unsigned proc, unsigned with_step, unsigned with_proc) {
        step_processor_memory[step][proc] = step_processor_memory[with_step][with_proc];
        step_processor_pred[step][proc] = step_processor_pred[with_step][with_proc];
    }

    inline bool can_move(vertex_idx_t<Graph_t> vertex, const unsigned proc, unsigned step) const {
        v_memw_t<Graph_t> inc_memory = 0;

        if (is_source(vertex, *graph)) { inc_memory += graph->vertex_mem_weight(vertex); }

        for (const auto &pred : graph->parents(vertex)) {
            if (vector_schedule->assignedSuperstep(pred) < step) {
                if (step_processor_pred[step][proc].find(pred) == step_processor_pred[step][proc].end()) {
                    inc_memory += graph->vertex_comm_weight(pred);
                }
            }
        }

        if (vector_schedule->assignedSuperstep(vertex) < step) {
            if (step_processor_pred[step][proc].find(vertex) != step_processor_pred[step][proc].end()) {
                inc_memory -= graph->vertex_comm_weight(vertex);
            }
        }

        if (vector_schedule->assignedSuperstep(vertex) <= step) {
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
                if (vector_schedule->assignedProcessor(vertex) != succ_proc || (not is_source(vertex, *graph))) {
                    if (step_processor_memory[succ_step][succ_proc] + graph->vertex_comm_weight(vertex)
                        > set_schedule->getInstance().getArchitecture().memoryBound(succ_proc)) {
                        return false;
                    }

                } else {
                    if (is_source(vertex, *graph)) {
                        if (step_processor_memory[succ_step][succ_proc] + graph->vertex_comm_weight(vertex)
                                - graph->vertex_mem_weight(vertex)
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
