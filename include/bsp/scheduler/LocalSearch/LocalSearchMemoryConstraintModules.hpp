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
struct is_local_search_memory_constraint : std::false_type {};

template<typename T>
struct is_local_search_memory_constraint<
    T, std::void_t<decltype(std::declval<T>().initialize(std::declval<BspInstance<typename T::Graph_impl_t>>())),
                   decltype(std::declval<T>().can_add(std::declval<vertex_idx_t<typename T::Graph_impl_t>>(),
                                                      std::declval<unsigned>())),
                   decltype(std::declval<T>().add(std::declval<vertex_idx_t<typename T::Graph_impl_t>>(),
                                                  std::declval<unsigned>())),
                   decltype(std::declval<T>().reset(std::declval<unsigned>())), decltype(T())>> : std::true_type {};

template<typename T>
inline constexpr bool is_local_search_memory_constraint = is_memory_constraint<T>::value;

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
template<typename Graph_t>
struct local_search_local_memory_constraint {

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

} // namespace osp