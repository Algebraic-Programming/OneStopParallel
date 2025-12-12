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
#include "osp/graph_algorithms/directed_graph_util.hpp"

namespace osp {

/**
 * @brief A trait to check if a type is a memory constraint.
 *
 * This trait checks if a type has the required methods for a memory constraint.
 *
 */
template <typename T, typename = void>
struct IsMemoryConstraint : std::false_type {};

template <typename T>
struct IsMemoryConstraint<
    T,
    std::void_t<decltype(std::declval<T>().initialize(std::declval<BspInstance<typename T::Graph_impl_t>>())),
                decltype(std::declval<T>().can_add(std::declval<VertexIdxT<typename T::Graph_impl_t>>(), std::declval<unsigned>())),
                decltype(std::declval<T>().add(std::declval<VertexIdxT<typename T::Graph_impl_t>>(), std::declval<unsigned>())),
                decltype(std::declval<T>().reset(std::declval<unsigned>())),
                decltype(T())>> : std::true_type {};

template <typename T>
inline constexpr bool isMemoryConstraintV = IsMemoryConstraint<T>::value;

/**
 * @brief The default memory constraint type, no memory constraints apply.
 *
 */
struct NoMemoryConstraint {
    using GraphImplT = void;
};

/**
 * @brief A memory constraint module for local memory constraints.
 *
 * @tparam Graph_t The graph type.
 */
template <typename GraphT>
struct LocalMemoryConstraint {
    using GraphImplT = GraphT;

    const BspInstance<GraphT> *instance_;

    std::vector<VMemwT<GraphT>> currentProcMemory_;

    LocalMemoryConstraint() : instance_(nullptr) {}

    inline void Initialize(const BspInstance<GraphT> &instance) {
        instance_ = &instance;
        current_proc_memory = std::vector<VMemwT<GraphT>>(instance->NumberOfProcessors(), 0);

        if (instance->GetArchitecture().GetMemoryConstraintType() != MemoryConstraintType::LOCAL) {
            throw std::invalid_argument("Memory constraint type is not LOCAL");
        }
    }

    inline bool CanAdd(const VertexIdxT<GraphT> &v, const unsigned proc) const {
        return current_proc_memory[proc] + instance->GetComputationalDag().VertexMemWeight(v)
               <= instance->GetArchitecture().memoryBound(proc);
    }

    inline void Add(const VertexIdxT<GraphT> &v, const unsigned proc) {
        current_proc_memory[proc] += instance->GetComputationalDag().VertexMemWeight(v);
    }

    inline bool CanAdd(const unsigned proc, const VMemwT<GraphT> &customMemWeight, const VMemwT<GraphT> &) const {
        return current_proc_memory[proc] + custom_mem_weight <= instance->GetArchitecture().memoryBound(proc);
    }

    inline void Add(const unsigned proc, const VMemwT<GraphT> &customMemWeight, const VMemwT<GraphT> &) {
        current_proc_memory[proc] += custom_mem_weight;
    }

    inline void Reset(const unsigned proc) { current_proc_memory[proc] = 0; }
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
template <typename GraphT>
struct PersistentTransientMemoryConstraint {
    static_assert(std::is_convertible_v<VCommwT<GraphT>, VMemwT<GraphT>>,
                  "persistent_transient_memory_constraint requires that memory and communication weights are convertible.");

    using GraphImplT = GraphT;

    const BspInstance<GraphT> *instance_;

    std::vector<VMemwT<GraphT>> currentProcPersistentMemory_;
    std::vector<VCommwT<GraphT>> currentProcTransientMemory_;

    PersistentTransientMemoryConstraint() : instance_(nullptr) {}

    inline void Initialize(const BspInstance<GraphT> &instance) {
        instance_ = &instance;

        current_proc_persistent_memory = std::vector<VMemwT<GraphT>>(instance->NumberOfProcessors(), 0);
        current_proc_transient_memory = std::vector<VCommwT<GraphT>>(instance->NumberOfProcessors(), 0);

        if (instance->GetArchitecture().GetMemoryConstraintType() != MemoryConstraintType::PERSISTENT_AND_TRANSIENT) {
            throw std::invalid_argument("Memory constraint type is not PERSISTENT_AND_TRANSIENT");
        }
    }

    inline bool CanAdd(const VertexIdxT<GraphT> &v, const unsigned proc) const {
        return (current_proc_persistent_memory[proc] + instance->GetComputationalDag().VertexMemWeight(v)
                    + std::max(current_proc_transient_memory[proc], instance->GetComputationalDag().VertexCommWeight(v))
                <= instance->GetArchitecture().memoryBound(proc));
    }

    inline void Add(const VertexIdxT<GraphT> &v, const unsigned proc) {
        current_proc_persistent_memory[proc] += instance->GetComputationalDag().VertexMemWeight(v);
        current_proc_transient_memory[proc]
            = std::max(current_proc_transient_memory[proc], instance->GetComputationalDag().VertexCommWeight(v));
    }

    inline bool CanAdd(const unsigned proc, const VMemwT<GraphT> &customMemWeight, const VCommwT<GraphT> &customCommWeight) const {
        return (current_proc_persistent_memory[proc] + custom_mem_weight
                    + std::max(current_proc_transient_memory[proc], custom_comm_weight)
                <= instance->GetArchitecture().memoryBound(proc));
    }

    inline void Add(const unsigned proc, const VMemwT<GraphT> &customMemWeight, const VCommwT<GraphT> &customCommWeight) {
        current_proc_persistent_memory[proc] += custom_mem_weight;
        current_proc_transient_memory[proc] = std::max(current_proc_transient_memory[proc], custom_comm_weight);
    }

    inline void Reset(const unsigned) {}
};

template <typename GraphT>
struct GlobalMemoryConstraint {
    using GraphImplT = GraphT;

    const BspInstance<GraphT> *instance_;

    std::vector<VMemwT<GraphT>> currentProcMemory_;

    GlobalMemoryConstraint() : instance_(nullptr) {}

    inline void Initialize(const BspInstance<GraphT> &instance) {
        instance_ = &instance;
        current_proc_memory = std::vector<VMemwT<GraphT>>(instance->NumberOfProcessors(), 0);

        if (instance->GetArchitecture().GetMemoryConstraintType() != MemoryConstraintType::GLOBAL) {
            throw std::invalid_argument("Memory constraint type is not GLOBAL");
        }
    }

    inline bool CanAdd(const VertexIdxT<GraphT> &v, const unsigned proc) const {
        return current_proc_memory[proc] + instance->GetComputationalDag().VertexMemWeight(v)
               <= instance->GetArchitecture().memoryBound(proc);
    }

    inline void Add(const VertexIdxT<GraphT> &v, const unsigned proc) {
        current_proc_memory[proc] += instance->GetComputationalDag().VertexMemWeight(v);
    }

    inline bool CanAdd(const unsigned proc, const VMemwT<GraphT> &customMemWeight, const VCommwT<GraphT> &) const {
        return current_proc_memory[proc] + custom_mem_weight <= instance->GetArchitecture().memoryBound(proc);
    }

    inline void Add(const unsigned proc, const VMemwT<GraphT> &customMemWeight, const VCommwT<GraphT> &) {
        current_proc_memory[proc] += custom_mem_weight;
    }

    inline void Reset(const unsigned) {}
};

template <typename T, typename = void>
struct IsMemoryConstraintSchedule : std::false_type {};

template <typename T>
struct IsMemoryConstraintSchedule<
    T,
    std::void_t<decltype(std::declval<T>().initialize(std::declval<BspSchedule<typename T::Graph_impl_t>>(), std::declval<unsigned>())),
                decltype(std::declval<T>().can_add(std::declval<VertexIdxT<typename T::Graph_impl_t>>(), std::declval<unsigned>())),
                decltype(std::declval<T>().add(std::declval<VertexIdxT<typename T::Graph_impl_t>>(), std::declval<unsigned>())),
                decltype(std::declval<T>().reset(std::declval<unsigned>())),
                decltype(T())>> : std::true_type {};

template <typename T>
inline constexpr bool isMemoryConstraintScheduleV = IsMemoryConstraintSchedule<T>::value;

template <typename GraphT>
struct LocalInOutMemoryConstraint {
    static_assert(std::is_convertible_v<VCommwT<GraphT>, VMemwT<GraphT>>,
                  "local_in_out_memory_constraint requires that memory and communication weights are convertible.");

    using GraphImplT = GraphT;

    const BspInstance<GraphT> *instance_;
    const BspSchedule<GraphT> *schedule_;

    const unsigned *currentSuperstep_ = 0;

    std::vector<VMemwT<GraphT>> currentProcMemory_;

    LocalInOutMemoryConstraint() : instance_(nullptr), schedule_(nullptr) {}

    inline void Initialize(const BspSchedule<GraphT> &schedule, const unsigned &supstepIdx) {
        currentSuperstep_ = &supstepIdx;
        schedule_ = &schedule;
        instance_ = &schedule_->GetInstance();
        current_proc_memory = std::vector<VMemwT<GraphT>>(instance->NumberOfProcessors(), 0);

        if (instance->GetArchitecture().GetMemoryConstraintType() != MemoryConstraintType::LOCAL_IN_OUT) {
            throw std::invalid_argument("Memory constraint type is not LOCAL_IN_OUT");
        }
    }

    inline bool CanAdd(const VertexIdxT<GraphT> &v, const unsigned proc) const {
        VMemwT<GraphT> incMemory
            = instance_->GetComputationalDag().VertexMemWeight(v) + instance_->GetComputationalDag().VertexCommWeight(v);

        for (const auto &pred : instance->GetComputationalDag().Parents(v)) {
            if (schedule->assignedProcessor(pred) == schedule->assignedProcessor(v)
                && schedule->assignedSuperstep(pred) == *current_superstep) {
                inc_memory -= instance->GetComputationalDag().VertexCommWeight(pred);
            }
        }

        return current_proc_memory[proc] + inc_memory <= instance->GetArchitecture().memoryBound(proc);
    }

    inline void Add(const VertexIdxT<GraphT> &v, const unsigned proc) {
        current_proc_memory[proc]
            += instance->GetComputationalDag().VertexMemWeight(v) + instance->GetComputationalDag().VertexCommWeight(v);

        for (const auto &pred : instance->GetComputationalDag().Parents(v)) {
            if (schedule->assignedProcessor(pred) == schedule->assignedProcessor(v)
                && schedule->assignedSuperstep(pred) == *current_superstep) {
                current_proc_memory[proc] -= instance->GetComputationalDag().VertexCommWeight(pred);
            }
        }
    }

    inline void Reset(const unsigned proc) { current_proc_memory[proc] = 0; }
};

template <typename GraphT>
struct LocalIncEdgesMemoryConstraint {
    using GraphImplT = GraphT;

    const BspInstance<GraphT> *instance_;
    const BspSchedule<GraphT> *schedule_;

    const unsigned *currentSuperstep_ = 0;

    std::vector<VCommwT<GraphT>> currentProcMemory_;
    std::vector<std::unordered_set<VertexIdxT<GraphT>>> currentProcPredec_;

    LocalIncEdgesMemoryConstraint() : instance_(nullptr), schedule_(nullptr) {}

    inline void Initialize(const BspSchedule<GraphT> &schedule, const unsigned &supstepIdx) {
        currentSuperstep_ = &supstepIdx;
        schedule_ = &schedule;
        instance_ = &schedule_->GetInstance();

        current_proc_memory = std::vector<VCommwT<GraphT>>(instance->NumberOfProcessors(), 0);
        current_proc_predec = std::vector<std::unordered_set<VertexIdxT<GraphT>>>(instance->NumberOfProcessors());

        if (instance->GetArchitecture().GetMemoryConstraintType() != MemoryConstraintType::LOCAL_INC_EDGES) {
            throw std::invalid_argument("Memory constraint type is not LOCAL_INC_EDGES");
        }
    }

    inline bool CanAdd(const VertexIdxT<GraphT> &v, const unsigned proc) const {
        VCommwT<GraphT> incMemory = instance_->GetComputationalDag().VertexCommWeight(v);

        for (const auto &pred : instance->GetComputationalDag().Parents(v)) {
            if (schedule->assignedSuperstep(pred) != *current_superstep
                && current_proc_predec[proc].find(pred) == current_proc_predec[proc].end()) {
                inc_memory += instance->GetComputationalDag().VertexCommWeight(pred);
            }
        }

        return current_proc_memory[proc] + inc_memory <= instance->GetArchitecture().memoryBound(proc);
    }

    inline void Add(const VertexIdxT<GraphT> &v, const unsigned proc) {
        current_proc_memory[proc] += instance->GetComputationalDag().VertexCommWeight(v);

        for (const auto &pred : instance->GetComputationalDag().Parents(v)) {
            if (schedule->assignedSuperstep(pred) != *current_superstep) {
                const auto pair = current_proc_predec[proc].insert(pred);
                if (pair.second) {
                    current_proc_memory[proc] += instance->GetComputationalDag().VertexCommWeight(pred);
                }
            }
        }
    }

    inline void Reset(const unsigned proc) {
        current_proc_memory[proc] = 0;
        current_proc_predec[proc].clear();
    }
};

template <typename GraphT>
struct LocalSourcesIncEdgesMemoryConstraint {
    static_assert(std::is_convertible_v<VCommwT<GraphT>, VMemwT<GraphT>>,
                  "local_sources_inc_edges_memory_constraint requires that memory and communication weights are convertible.");

    using GraphImplT = GraphT;

    const BspInstance<GraphT> *instance_;
    const BspSchedule<GraphT> *schedule_;

    const unsigned *currentSuperstep_ = 0;

    std::vector<VMemwT<GraphT>> currentProcMemory_;
    std::vector<std::unordered_set<VertexIdxT<GraphT>>> currentProcPredec_;

    LocalSourcesIncEdgesMemoryConstraint() : instance_(nullptr), schedule_(nullptr) {}

    inline void Initialize(const BspSchedule<GraphT> &schedule, const unsigned &supstepIdx) {
        currentSuperstep_ = &supstepIdx;
        schedule_ = &schedule;
        instance_ = &schedule_->GetInstance();

        current_proc_memory = std::vector<VMemwT<GraphT>>(instance->NumberOfProcessors(), 0);
        current_proc_predec = std::vector<std::unordered_set<VertexIdxT<GraphT>>>(instance->NumberOfProcessors());

        if (instance->GetArchitecture().GetMemoryConstraintType() != MemoryConstraintType::LOCAL_SOURCES_INC_EDGES) {
            throw std::invalid_argument("Memory constraint type is not LOCAL_INC_EDGES_2");
        }
    }

    inline bool CanAdd(const VertexIdxT<GraphT> &v, const unsigned proc) const {
        VMemwT<GraphT> incMemory = 0;

        if (IsSource(v, instance_->GetComputationalDag())) {
            incMemory += instance_->GetComputationalDag().VertexMemWeight(v);
        }

        for (const auto &pred : instance->GetComputationalDag().Parents(v)) {
            if (schedule->assignedSuperstep(v) != *current_superstep
                && current_proc_predec[proc].find(pred) == current_proc_predec[proc].end()) {
                inc_memory += instance->GetComputationalDag().VertexCommWeight(pred);
            }
        }

        return current_proc_memory[proc] + inc_memory <= instance->GetArchitecture().memoryBound(proc);
    }

    inline void Add(const VertexIdxT<GraphT> &v, const unsigned proc) {
        if (IsSource(v, instance_->GetComputationalDag())) {
            current_proc_memory[proc] += instance->GetComputationalDag().VertexMemWeight(v);
        }

        for (const auto &pred : instance->GetComputationalDag().Parents(v)) {
            if (schedule->assignedSuperstep(pred) != *current_superstep) {
                const auto pair = current_proc_predec[proc].insert(pred);
                if (pair.second) {
                    current_proc_memory[proc] += instance->GetComputationalDag().VertexCommWeight(pred);
                }
            }
        }
    }

    inline void Reset(const unsigned proc) {
        current_proc_memory[proc] = 0;
        current_proc_predec[proc].clear();
    }
};

}    // namespace osp
