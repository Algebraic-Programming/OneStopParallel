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
    std::void_t<decltype(std::declval<T>().Initialize(std::declval<BspInstance<typename T::GraphImplT>>())),
                decltype(std::declval<T>().CanAdd(std::declval<VertexIdxT<typename T::GraphImplT>>(), std::declval<unsigned>())),
                decltype(std::declval<T>().Add(std::declval<VertexIdxT<typename T::GraphImplT>>(), std::declval<unsigned>())),
                decltype(std::declval<T>().Reset(std::declval<unsigned>())),
                decltype(T())>> : std::true_type {};

template <typename T>
inline constexpr bool IsMemoryConstraintV = IsMemoryConstraint<T>::value;

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
        currentProcMemory_ = std::vector<VMemwT<GraphT>>(instance.NumberOfProcessors(), 0);

        if (instance.GetArchitecture().GetMemoryConstraintType() != MemoryConstraintType::LOCAL) {
            throw std::invalid_argument("Memory constraint type is not LOCAL");
        }
    }

    inline bool CanAdd(const VertexIdxT<GraphT> &v, const unsigned proc) const {
        return currentProcMemory_[proc] + instance_->GetComputationalDag().VertexMemWeight(v)
               <= instance_->GetArchitecture().MemoryBound(proc);
    }

    inline void Add(const VertexIdxT<GraphT> &v, const unsigned proc) {
        currentProcMemory_[proc] += instance_->GetComputationalDag().VertexMemWeight(v);
    }

    inline bool CanAdd(const unsigned proc, const VMemwT<GraphT> &customMemWeight, const VMemwT<GraphT> &) const {
        return currentProcMemory_[proc] + customMemWeight <= instance_->GetArchitecture().MemoryBound(proc);
    }

    inline void Add(const unsigned proc, const VMemwT<GraphT> &customMemWeight, const VMemwT<GraphT> &) {
        currentProcMemory_[proc] += customMemWeight;
    }

    inline void Reset(const unsigned proc) { currentProcMemory_[proc] = 0; }
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

        currentProcPersistentMemory_.assign(instance.NumberOfProcessors(), 0);
        currentProcTransientMemory_.assign(instance.NumberOfProcessors(), 0);

        if (instance.GetArchitecture().GetMemoryConstraintType() != MemoryConstraintType::PERSISTENT_AND_TRANSIENT) {
            throw std::invalid_argument("Memory constraint type is not PERSISTENT_AND_TRANSIENT");
        }
    }

    inline bool CanAdd(const VertexIdxT<GraphT> &v, const unsigned proc) const {
        return (currentProcPersistentMemory_[proc] + instance_->GetComputationalDag().VertexMemWeight(v)
                    + std::max(currentProcTransientMemory_[proc], instance_->GetComputationalDag().VertexCommWeight(v))
                <= instance_->GetArchitecture().MemoryBound(proc));
    }

    inline void Add(const VertexIdxT<GraphT> &v, const unsigned proc) {
        currentProcPersistentMemory_[proc] += instance_->GetComputationalDag().VertexMemWeight(v);
        currentProcTransientMemory_[proc]
            = std::max(currentProcTransientMemory_[proc], instance_->GetComputationalDag().VertexCommWeight(v));
    }

    inline bool CanAdd(const unsigned proc, const VMemwT<GraphT> &customMemWeight, const VCommwT<GraphT> &customCommWeight) const {
        return (currentProcPersistentMemory_[proc] + customMemWeight + std::max(currentProcTransientMemory_[proc], customCommWeight)
                <= instance_->GetArchitecture().MemoryBound(proc));
    }

    inline void Add(const unsigned proc, const VMemwT<GraphT> &customMemWeight, const VCommwT<GraphT> &customCommWeight) {
        currentProcPersistentMemory_[proc] += customMemWeight;
        currentProcTransientMemory_[proc] = std::max(currentProcTransientMemory_[proc], customCommWeight);
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
        currentProcMemory_ = std::vector<VMemwT<GraphT>>(instance.NumberOfProcessors(), 0);

        if (instance.GetArchitecture().GetMemoryConstraintType() != MemoryConstraintType::GLOBAL) {
            throw std::invalid_argument("Memory constraint type is not GLOBAL");
        }
    }

    inline bool CanAdd(const VertexIdxT<GraphT> &v, const unsigned proc) const {
        return currentProcMemory_[proc] + instance_->GetComputationalDag().VertexMemWeight(v)
               <= instance_->GetArchitecture().MemoryBound(proc);
    }

    inline void Add(const VertexIdxT<GraphT> &v, const unsigned proc) {
        currentProcMemory_[proc] += instance_->GetComputationalDag().VertexMemWeight(v);
    }

    inline bool CanAdd(const unsigned proc, const VMemwT<GraphT> &customMemWeight, const VCommwT<GraphT> &) const {
        return currentProcMemory_[proc] + customMemWeight <= instance_->GetArchitecture().MemoryBound(proc);
    }

    inline void Add(const unsigned proc, const VMemwT<GraphT> &customMemWeight, const VCommwT<GraphT> &) {
        currentProcMemory_[proc] += customMemWeight;
    }

    inline void Reset(const unsigned) {}
};

template <typename T, typename = void>
struct IsMemoryConstraintSchedule : std::false_type {};

template <typename T>
struct IsMemoryConstraintSchedule<
    T,
    std::void_t<decltype(std::declval<T>().Initialize(std::declval<BspSchedule<typename T::GraphImplT>>(), std::declval<unsigned>())),
                decltype(std::declval<T>().CanAdd(std::declval<VertexIdxT<typename T::GraphImplT>>(), std::declval<unsigned>())),
                decltype(std::declval<T>().Add(std::declval<VertexIdxT<typename T::GraphImplT>>(), std::declval<unsigned>())),
                decltype(std::declval<T>().Reset(std::declval<unsigned>())),
                decltype(T())>> : std::true_type {};

template <typename T>
inline constexpr bool IsMemoryConstraintScheduleV = IsMemoryConstraintSchedule<T>::value;

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
        currentProcMemory_.assign(instance_->NumberOfProcessors(), 0);

        if (instance_->GetArchitecture().GetMemoryConstraintType() != MemoryConstraintType::LOCAL_IN_OUT) {
            throw std::invalid_argument("Memory constraint type is not LOCAL_IN_OUT");
        }
    }

    inline bool CanAdd(const VertexIdxT<GraphT> &v, const unsigned proc) const {
        VMemwT<GraphT> incMemory
            = instance_->GetComputationalDag().VertexMemWeight(v) + instance_->GetComputationalDag().VertexCommWeight(v);

        for (const auto &pred : instance_->GetComputationalDag().Parents(v)) {
            if (schedule_->AssignedProcessor(pred) == schedule_->AssignedProcessor(v)
                && schedule_->AssignedSuperstep(pred) == *currentSuperstep_) {
                incMemory -= instance_->GetComputationalDag().VertexCommWeight(pred);
            }
        }

        return currentProcMemory_[proc] + incMemory <= instance_->GetArchitecture().MemoryBound(proc);
    }

    inline void Add(const VertexIdxT<GraphT> &v, const unsigned proc) {
        currentProcMemory_[proc]
            += instance_->GetComputationalDag().VertexMemWeight(v) + instance_->GetComputationalDag().VertexCommWeight(v);

        for (const auto &pred : instance_->GetComputationalDag().Parents(v)) {
            if (schedule_->AssignedProcessor(pred) == schedule_->AssignedProcessor(v)
                && schedule_->AssignedSuperstep(pred) == *currentSuperstep_) {
                currentProcMemory_[proc] -= instance_->GetComputationalDag().VertexCommWeight(pred);
            }
        }
    }

    inline void Reset(const unsigned proc) { currentProcMemory_[proc] = 0; }
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

        currentProcMemory_.assign(instance_->NumberOfProcessors(), 0);
        currentProcPredec_.assign(instance_->NumberOfProcessors(), std::unordered_set<VertexIdxT<GraphT>>());

        if (instance_->GetArchitecture().GetMemoryConstraintType() != MemoryConstraintType::LOCAL_INC_EDGES) {
            throw std::invalid_argument("Memory constraint type is not LOCAL_INC_EDGES");
        }
    }

    inline bool CanAdd(const VertexIdxT<GraphT> &v, const unsigned proc) const {
        VCommwT<GraphT> incMemory = instance_->GetComputationalDag().VertexCommWeight(v);

        for (const auto &pred : instance_->GetComputationalDag().Parents(v)) {
            if (schedule_->AssignedSuperstep(pred) != *currentSuperstep_
                && currentProcPredec_[proc].find(pred) == currentProcPredec_[proc].end()) {
                incMemory += instance_->GetComputationalDag().VertexCommWeight(pred);
            }
        }

        return currentProcMemory_[proc] + incMemory <= instance_->GetArchitecture().MemoryBound(proc);
    }

    inline void Add(const VertexIdxT<GraphT> &v, const unsigned proc) {
        currentProcMemory_[proc] += instance_->GetComputationalDag().VertexCommWeight(v);

        for (const auto &pred : instance_->GetComputationalDag().Parents(v)) {
            if (schedule_->AssignedSuperstep(pred) != *currentSuperstep_) {
                const auto pair = currentProcPredec_[proc].insert(pred);
                if (pair.second) {
                    currentProcMemory_[proc] += instance_->GetComputationalDag().VertexCommWeight(pred);
                }
            }
        }
    }

    inline void Reset(const unsigned proc) {
        currentProcMemory_[proc] = 0;
        currentProcPredec_[proc].clear();
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

        currentProcMemory_.assign(instance_->NumberOfProcessors(), 0);
        currentProcPredec_.assign(instance_->NumberOfProcessors(), std::unordered_set<VertexIdxT<GraphT>>());

        if (instance_->GetArchitecture().GetMemoryConstraintType() != MemoryConstraintType::LOCAL_SOURCES_INC_EDGES) {
            throw std::invalid_argument("Memory constraint type is not LOCAL_SOURCES_INC_EDGES");
        }
    }

    inline bool CanAdd(const VertexIdxT<GraphT> &v, const unsigned proc) const {
        VMemwT<GraphT> incMemory = 0;

        if (IsSource(v, instance_->GetComputationalDag())) {
            incMemory += instance_->GetComputationalDag().VertexMemWeight(v);
        }

        for (const auto &pred : instance_->GetComputationalDag().Parents(v)) {
            if (schedule_->AssignedSuperstep(pred) != *currentSuperstep_
                && currentProcPredec_[proc].find(pred) == currentProcPredec_[proc].end()) {
                incMemory += instance_->GetComputationalDag().VertexCommWeight(pred);
            }
        }

        return currentProcMemory_[proc] + incMemory <= instance_->GetArchitecture().MemoryBound(proc);
    }

    inline void Add(const VertexIdxT<GraphT> &v, const unsigned proc) {
        if (IsSource(v, instance_->GetComputationalDag())) {
            currentProcMemory_[proc] += instance_->GetComputationalDag().VertexMemWeight(v);
        }

        for (const auto &pred : instance_->GetComputationalDag().Parents(v)) {
            if (schedule_->AssignedSuperstep(pred) != *currentSuperstep_) {
                const auto pair = currentProcPredec_[proc].insert(pred);
                if (pair.second) {
                    currentProcMemory_[proc] += instance_->GetComputationalDag().VertexCommWeight(pred);
                }
            }
        }
    }

    inline void Reset(const unsigned proc) {
        currentProcMemory_[proc] = 0;
        currentProcPredec_[proc].clear();
    }
};

}    // namespace osp
