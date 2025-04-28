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

#include "bsp/model/BspInstance.hpp"

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
struct no_memory_constraint {};

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

    inline void reset(const unsigned proc) { current_proc_memory[proc] = 0; }
};

/**
 * @brief A memory constraint module for persistent and transient memory constraints.
 *
 * @tparam Graph_t The graph type.
 */
template<typename Graph_t>
struct persistent_transient_memory_constraint {

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

    inline void reset(const unsigned) {}
};

} // namespace osp