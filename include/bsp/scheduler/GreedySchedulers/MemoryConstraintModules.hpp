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

#include "bsp/model/BspSchedule.hpp"
#include "graph_algorithms/directed_graph_util.hpp"

namespace osp {

/**
 * @brief A trait to check if a type is a memory constraint.
 *
 * This trait checks if a type has the required methods for a memory constraint.
 *
 */
template<typename T, typename = void>
struct is_memory_constraint : std::false_type {};

template<typename T>
struct is_memory_constraint<
    T, std::void_t<decltype(std::declval<T>().initialize(std::declval<BspInstance<typename T::Graph_impl_t>>())),
                   decltype(std::declval<T>().can_add(std::declval<vertex_idx_t<typename T::Graph_impl_t>>(),
                                                      std::declval<unsigned>())),
                   decltype(std::declval<T>().add(std::declval<vertex_idx_t<typename T::Graph_impl_t>>(),
                                                  std::declval<unsigned>())),
                   decltype(std::declval<T>().reset(std::declval<unsigned>())), decltype(T())>> : std::true_type {};

template<typename T>
inline constexpr bool is_memory_constraint_v = is_memory_constraint<T>::value;

/**
 * @brief The default memory constraint type, no memory constraints apply.
 *
 */
struct no_memory_constraint {
    using Graph_impl_t = void;
};

/**
 * @brief A memory constraint module for local memory constraints.
 *
 * @tparam Graph_t The graph type.
 */
template<typename Graph_t>
struct local_memory_constraint {

    using Graph_impl_t = Graph_t;

    const BspInstance<Graph_t> *instance;

    std::vector<v_memw_t<Graph_t>> current_proc_memory;

    local_memory_constraint() : instance(nullptr) {}

    inline void initialize(const BspInstance<Graph_t> &instance_) {
        instance = &instance_;
        current_proc_memory = std::vector<v_memw_t<Graph_t>>(instance->numberOfProcessors(), 0);

        if (instance->getArchitecture().getMemoryConstraintType() != LOCAL) {
            throw std::invalid_argument("Memory constraint type is not LOCAL");
        }
    }

    inline bool can_add(const vertex_idx_t<Graph_t> &v, const unsigned proc) const {
        return current_proc_memory[proc] + instance->getComputationalDag().vertex_mem_weight(v) <=
               instance->getArchitecture().memoryBound(proc);
    }

    inline void add(const vertex_idx_t<Graph_t> &v, const unsigned proc) {
        current_proc_memory[proc] += instance->getComputationalDag().vertex_mem_weight(v);
    }

    inline bool can_add(const unsigned proc, const v_memw_t<Graph_t> &custom_mem_weight,
                        const v_memw_t<Graph_t>&) const {
        return current_proc_memory[proc] + custom_mem_weight <= instance->getArchitecture().memoryBound(proc);
    }

    inline void add(const unsigned proc, const v_memw_t<Graph_t> &custom_mem_weight, const v_memw_t<Graph_t>&) {
        current_proc_memory[proc] += custom_mem_weight;
    }

    inline void reset(const unsigned proc) { current_proc_memory[proc] = 0; }
};

/**
 * @brief A memory constraint module for local memory constraints.
 *
 * @tparam Graph_t The graph type.
 */

/**
 * @brief A memory constraint module for persistent and transient memory constraints.
 *
 * @tparam Graph_t The graph type.
 */
template<typename Graph_t>
struct persistent_transient_memory_constraint {

    static_assert(
        std::is_convertible_v<v_commw_t<Graph_t>, v_memw_t<Graph_t>>,
        "persistent_transient_memory_constraint requires that memory and communication weights are convertible.");

    using Graph_impl_t = Graph_t;

    const BspInstance<Graph_t> *instance;

    std::vector<v_memw_t<Graph_t>> current_proc_persistent_memory;
    std::vector<v_commw_t<Graph_t>> current_proc_transient_memory;

    persistent_transient_memory_constraint() : instance(nullptr) {}

    inline void initialize(const BspInstance<Graph_t> &instance_) {
        instance = &instance_;

        current_proc_persistent_memory = std::vector<v_memw_t<Graph_t>>(instance->numberOfProcessors(), 0);
        current_proc_transient_memory = std::vector<v_commw_t<Graph_t>>(instance->numberOfProcessors(), 0);

        if (instance->getArchitecture().getMemoryConstraintType() != PERSISTENT_AND_TRANSIENT) {
            throw std::invalid_argument("Memory constraint type is not PERSISTENT_AND_TRANSIENT");
        }
    }

    inline bool can_add(const vertex_idx_t<Graph_t> &v, const unsigned proc) const {

        return (
            current_proc_persistent_memory[proc] + instance->getComputationalDag().vertex_mem_weight(v) +
                std::max(current_proc_transient_memory[proc], instance->getComputationalDag().vertex_comm_weight(v)) <=
            instance->getArchitecture().memoryBound(proc));
    }

    inline void add(const vertex_idx_t<Graph_t> &v, const unsigned proc) {

        current_proc_persistent_memory[proc] += instance->getComputationalDag().vertex_mem_weight(v);
        current_proc_transient_memory[proc] =
            std::max(current_proc_transient_memory[proc], instance->getComputationalDag().vertex_comm_weight(v));
    }

    inline bool can_add(const unsigned proc, const v_memw_t<Graph_t> &custom_mem_weight,
                        const v_commw_t<Graph_t> &custom_comm_weight) const {

        return (current_proc_persistent_memory[proc] + custom_mem_weight +
                    std::max(current_proc_transient_memory[proc], custom_comm_weight) <=
                instance->getArchitecture().memoryBound(proc));
    }

    inline void add(const unsigned proc, const v_memw_t<Graph_t> &custom_mem_weight,
                    const v_commw_t<Graph_t> &custom_comm_weight ) {

        current_proc_persistent_memory[proc] += custom_mem_weight;
        current_proc_transient_memory[proc] = std::max(current_proc_transient_memory[proc], custom_comm_weight);
    }

    inline void reset(const unsigned) {}
};

template<typename Graph_t>
struct global_memory_constraint {

    using Graph_impl_t = Graph_t;

    const BspInstance<Graph_t> *instance;

    std::vector<v_memw_t<Graph_t>> current_proc_memory;

    global_memory_constraint() : instance(nullptr) {}

    inline void initialize(const BspInstance<Graph_t> &instance_) {
        instance = &instance_;
        current_proc_memory = std::vector<v_memw_t<Graph_t>>(instance->numberOfProcessors(), 0);

        if (instance->getArchitecture().getMemoryConstraintType() != GLOBAL) {
            throw std::invalid_argument("Memory constraint type is not GLOBAL");
        }
    }

    inline bool can_add(const vertex_idx_t<Graph_t> &v, const unsigned proc) const {
        return current_proc_memory[proc] + instance->getComputationalDag().vertex_mem_weight(v) <=
               instance->getArchitecture().memoryBound(proc);
    }

    inline void add(const vertex_idx_t<Graph_t> &v, const unsigned proc) {
        current_proc_memory[proc] += instance->getComputationalDag().vertex_mem_weight(v);
    }

    inline bool can_add(const unsigned proc, const v_memw_t<Graph_t> &custom_mem_weight,
                        const v_commw_t<Graph_t> &) const {
        return current_proc_memory[proc] + custom_mem_weight <= instance->getArchitecture().memoryBound(proc);
    }

    inline void add(const unsigned proc, const v_memw_t<Graph_t> &custom_mem_weight, const v_commw_t<Graph_t> &) {
        current_proc_memory[proc] += custom_mem_weight;
    }

    inline void reset(const unsigned) {}
};

template<typename T, typename = void>
struct is_memory_constraint_schedule : std::false_type {};

template<typename T>
struct is_memory_constraint_schedule<
    T, std::void_t<decltype(std::declval<T>().initialize(std::declval<BspSchedule<typename T::Graph_impl_t>>(),
                                                         std::declval<unsigned>())),
                   decltype(std::declval<T>().can_add(std::declval<vertex_idx_t<typename T::Graph_impl_t>>(),
                                                      std::declval<unsigned>())),
                   decltype(std::declval<T>().add(std::declval<vertex_idx_t<typename T::Graph_impl_t>>(),
                                                  std::declval<unsigned>())),
                   decltype(std::declval<T>().reset(std::declval<unsigned>())), decltype(T())>> : std::true_type {};

template<typename T>
inline constexpr bool is_memory_constraint_schedule_v = is_memory_constraint_schedule<T>::value;

template<typename Graph_t>
struct local_in_out_memory_constraint {

    static_assert(std::is_convertible_v<v_commw_t<Graph_t>, v_memw_t<Graph_t>>,
                  "local_in_out_memory_constraint requires that memory and communication weights are convertible.");

    using Graph_impl_t = Graph_t;

    const BspInstance<Graph_t> *instance;
    const BspSchedule<Graph_t> *schedule;

    const unsigned *current_superstep = 0;

    std::vector<v_memw_t<Graph_t>> current_proc_memory;

    local_in_out_memory_constraint() : instance(nullptr), schedule(nullptr) {}

    inline void initialize(const BspSchedule<Graph_t> &schedule_, const unsigned &supstepIdx) {
        current_superstep = &supstepIdx;
        schedule = &schedule_;
        instance = &schedule->getInstance();
        current_proc_memory = std::vector<v_memw_t<Graph_t>>(instance->numberOfProcessors(), 0);

        if (instance->getArchitecture().getMemoryConstraintType() != LOCAL_IN_OUT) {
            throw std::invalid_argument("Memory constraint type is not LOCAL_IN_OUT");
        }
    }

    inline bool can_add(const vertex_idx_t<Graph_t> &v, const unsigned proc) const {

        v_memw_t<Graph_t> inc_memory = instance->getComputationalDag().vertex_mem_weight(v) +
                                       instance->getComputationalDag().vertex_comm_weight(v);

        for (const auto &pred : instance->getComputationalDag().parents(v)) {

            if (schedule->assignedProcessor(pred) == schedule->assignedProcessor(v) &&
                schedule->assignedSuperstep(pred) == *current_superstep) {
                inc_memory -= instance->getComputationalDag().vertex_comm_weight(pred);
            }
        }

        return current_proc_memory[proc] + inc_memory <= instance->getArchitecture().memoryBound(proc);
    }

    inline void add(const vertex_idx_t<Graph_t> &v, const unsigned proc) {

        current_proc_memory[proc] += instance->getComputationalDag().vertex_mem_weight(v) +
                                     instance->getComputationalDag().vertex_comm_weight(v);

        for (const auto &pred : instance->getComputationalDag().parents(v)) {

            if (schedule->assignedProcessor(pred) == schedule->assignedProcessor(v) &&
                schedule->assignedSuperstep(pred) == *current_superstep) {
                current_proc_memory[proc] -= instance->getComputationalDag().vertex_comm_weight(pred);
            }
        }
    }

    inline void reset(const unsigned proc) { current_proc_memory[proc] = 0; }
};

template<typename Graph_t>
struct local_inc_edges_memory_constraint {

    using Graph_impl_t = Graph_t;

    const BspInstance<Graph_t> *instance;
    const BspSchedule<Graph_t> *schedule;

    const unsigned *current_superstep = 0;

    std::vector<v_commw_t<Graph_t>> current_proc_memory;
    std::vector<std::unordered_set<vertex_idx_t<Graph_t>>> current_proc_predec;

    local_inc_edges_memory_constraint() : instance(nullptr), schedule(nullptr) {}

    inline void initialize(const BspSchedule<Graph_t> &schedule_, const unsigned &supstepIdx) {
        current_superstep = &supstepIdx;
        schedule = &schedule_;
        instance = &schedule->getInstance();

        current_proc_memory = std::vector<v_commw_t<Graph_t>>(instance->numberOfProcessors(), 0);
        current_proc_predec = std::vector<std::unordered_set<vertex_idx_t<Graph_t>>>(instance->numberOfProcessors());

        if (instance->getArchitecture().getMemoryConstraintType() != LOCAL_INC_EDGES) {
            throw std::invalid_argument("Memory constraint type is not LOCAL_INC_EDGES");
        }
    }

    inline bool can_add(const vertex_idx_t<Graph_t> &v, const unsigned proc) const {

        v_commw_t<Graph_t> inc_memory = instance->getComputationalDag().vertex_comm_weight(v);

        for (const auto &pred : instance->getComputationalDag().parents(v)) {

            if (schedule->assignedSuperstep(pred) != *current_superstep &&
                current_proc_predec[proc].find(pred) == current_proc_predec[proc].end()) {
                inc_memory += instance->getComputationalDag().vertex_comm_weight(pred);
            }
        }

        return current_proc_memory[proc] + inc_memory <= instance->getArchitecture().memoryBound(proc);
    }

    inline void add(const vertex_idx_t<Graph_t> &v, const unsigned proc) {

        current_proc_memory[proc] += instance->getComputationalDag().vertex_comm_weight(v);

        for (const auto &pred : instance->getComputationalDag().parents(v)) {

            if (schedule->assignedSuperstep(pred) != *current_superstep) {
                const auto pair = current_proc_predec[proc].insert(pred);
                if (pair.second) {
                    current_proc_memory[proc] += instance->getComputationalDag().vertex_comm_weight(pred);
                }
            }
        }
    }

    inline void reset(const unsigned proc) {
        current_proc_memory[proc] = 0;
        current_proc_predec[proc].clear();
    }
};

template<typename Graph_t>
struct local_inc_edges_2_memory_constraint {

    static_assert(
        std::is_convertible_v<v_commw_t<Graph_t>, v_memw_t<Graph_t>>,
        "local_inc_edges_2_memory_constraint requires that memory and communication weights are convertible.");

    using Graph_impl_t = Graph_t;

    const BspInstance<Graph_t> *instance;
    const BspSchedule<Graph_t> *schedule;

    const unsigned *current_superstep = 0;

    std::vector<v_memw_t<Graph_t>> current_proc_memory;
    std::vector<std::unordered_set<vertex_idx_t<Graph_t>>> current_proc_predec;

    local_inc_edges_2_memory_constraint() : instance(nullptr), schedule(nullptr) {}

    inline void initialize(const BspSchedule<Graph_t> &schedule_, const unsigned &supstepIdx) {
        current_superstep = &supstepIdx;
        schedule = &schedule_;
        instance = &schedule->getInstance();

        current_proc_memory = std::vector<v_memw_t<Graph_t>>(instance->numberOfProcessors(), 0);
        current_proc_predec = std::vector<std::unordered_set<vertex_idx_t<Graph_t>>>(instance->numberOfProcessors());

        if (instance->getArchitecture().getMemoryConstraintType() != LOCAL_SOURCES_INC_EDGES) {
            throw std::invalid_argument("Memory constraint type is not LOCAL_INC_EDGES_2");
        }
    }

    inline bool can_add(const vertex_idx_t<Graph_t> &v, const unsigned proc) const {

        v_memw_t<Graph_t> inc_memory = 0;

        if (is_source(v, instance->getComputationalDag())) {
            inc_memory += instance->getComputationalDag().vertex_mem_weight(v);
        }

        for (const auto &pred : instance->getComputationalDag().parents(v)) {

            if (schedule->assignedSuperstep(v) != *current_superstep &&
                current_proc_predec[proc].find(pred) == current_proc_predec[proc].end()) {
                inc_memory += instance->getComputationalDag().vertex_comm_weight(pred);
            }
        }

        return current_proc_memory[proc] + inc_memory <= instance->getArchitecture().memoryBound(proc);
    }

    inline void add(const vertex_idx_t<Graph_t> &v, const unsigned proc) {

        if (is_source(v, instance->getComputationalDag())) {
            current_proc_memory[proc] += instance->getComputationalDag().vertex_mem_weight(v);
        }

        for (const auto &pred : instance->getComputationalDag().parents(v)) {

            if (schedule->assignedSuperstep(pred) != *current_superstep) {
                const auto pair = current_proc_predec[proc].insert(pred);
                if (pair.second) {
                    current_proc_memory[proc] += instance->getComputationalDag().vertex_comm_weight(pred);
                }
            }
        }
    }

    inline void reset(const unsigned proc) {
        current_proc_memory[proc] = 0;
        current_proc_predec[proc].clear();
    }
};

} // namespace osp