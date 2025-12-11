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

    const BspInstance<GraphT> *instance;

    std::vector<VMemwT<GraphT>> currentProcMemory;

    LocalMemoryConstraint() : instance(nullptr) {}

    inline void Initialize(const BspInstance<GraphT> &instance_) {
        instance = &instance_;
        currentProcMemory = std::vector<VMemwT<GraphT>>(instance->NumberOfProcessors(), 0);

        if (instance->GetArchitecture().GetMemoryConstraintType() != MEMORY_CONSTRAINT_TYPE::LOCAL) {
            throw std::invalid_argument("Memory constraint type is not LOCAL");
        }
    }

    inline bool CanAdd(const VertexIdxT<GraphT> &v, const unsigned proc) const {
        return currentProcMemory[proc] + instance->GetComputationalDag().VertexMemWeight(v)
               <= instance->GetArchitecture().MemoryBound(proc);
    }

    inline void Add(const VertexIdxT<GraphT> &v, const unsigned proc) {
        currentProcMemory[proc] += instance->GetComputationalDag().VertexMemWeight(v);
    }

    inline bool CanAdd(const unsigned proc, const VMemwT<GraphT> &customMemWeight, const VMemwT<GraphT> &) const {
        return currentProcMemory[proc] + customMemWeight <= instance->GetArchitecture().MemoryBound(proc);
    }

    inline void Add(const unsigned proc, const VMemwT<GraphT> &customMemWeight, const VMemwT<GraphT> &) {
        currentProcMemory[proc] += customMemWeight;
    }

    inline void Reset(const unsigned proc) { currentProcMemory[proc] = 0; }
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

    const BspInstance<GraphT> *instance;

    std::vector<VMemwT<GraphT>> currentProcPersistentMemory;
    std::vector<VCommwT<GraphT>> currentProcTransientMemory;

    PersistentTransientMemoryConstraint() : instance(nullptr) {}

    inline void Initialize(const BspInstance<GraphT> &instance_) {
        instance = &instance_;

        currentProcPersistentMemory = std::vector<VMemwT<GraphT>>(instance->NumberOfProcessors(), 0);
        currentProcTransientMemory = std::vector<VCommwT<GraphT>>(instance->NumberOfProcessors(), 0);

        if (instance->GetArchitecture().GetMemoryConstraintType() != MEMORY_CONSTRAINT_TYPE::PERSISTENT_AND_TRANSIENT) {
            throw std::invalid_argument("Memory constraint type is not PERSISTENT_AND_TRANSIENT");
        }
    }

    inline bool CanAdd(const VertexIdxT<GraphT> &v, const unsigned proc) const {
        return (currentProcPersistentMemory[proc] + instance->GetComputationalDag().VertexMemWeight(v)
                    + std::max(currentProcTransientMemory[proc], instance->GetComputationalDag().VertexCommWeight(v))
                <= instance->GetArchitecture().MemoryBound(proc));
    }

    inline void Add(const VertexIdxT<GraphT> &v, const unsigned proc) {
        currentProcPersistentMemory[proc] += instance->GetComputationalDag().VertexMemWeight(v);
        currentProcTransientMemory[proc]
            = std::max(currentProcTransientMemory[proc], instance->GetComputationalDag().VertexCommWeight(v));
    }

    inline bool CanAdd(const unsigned proc, const VMemwT<GraphT> &customMemWeight, const VCommwT<GraphT> &customCommWeight) const {
        return (currentProcPersistentMemory[proc] + customMemWeight + std::max(currentProcTransientMemory[proc], customCommWeight)
                <= instance->GetArchitecture().MemoryBound(proc));
    }

    inline void Add(const unsigned proc, const VMemwT<GraphT> &customMemWeight, const VCommwT<GraphT> &customCommWeight) {
        currentProcPersistentMemory[proc] += customMemWeight;
        currentProcTransientMemory[proc] = std::max(currentProcTransientMemory[proc], customCommWeight);
    }

    inline void Reset(const unsigned) {}
};

template <typename GraphT>
struct GlobalMemoryConstraint {
    using GraphImplT = GraphT;

    const BspInstance<GraphT> *instance;

    std::vector<VMemwT<GraphT>> currentProcMemory;

    GlobalMemoryConstraint() : instance(nullptr) {}

    inline void Initialize(const BspInstance<GraphT> &instance_) {
        instance = &instance_;
        currentProcMemory = std::vector<VMemwT<GraphT>>(instance->numberOfProcessors(), 0);

        if (instance->getArchitecture().getMemoryConstraintType() != MEMORY_CONSTRAINT_TYPE::GLOBAL) {
            throw std::invalid_argument("Memory constraint type is not GLOBAL");
        }
    }

    inline bool CanAdd(const VertexIdxT<GraphT> &v, const unsigned proc) const {
        return currentProcMemory[proc] + instance->getComputationalDag().VertexMemWeight(v)
               <= instance->getArchitecture().memoryBound(proc);
    }

    inline void Add(const VertexIdxT<GraphT> &v, const unsigned proc) {
        currentProcMemory[proc] += instance->getComputationalDag().VertexMemWeight(v);
    }

    inline bool CanAdd(const unsigned proc, const VMemwT<GraphT> &customMemWeight, const VCommwT<GraphT> &) const {
        return currentProcMemory[proc] + customMemWeight <= instance->getArchitecture().memoryBound(proc);
    }

    inline void Add(const unsigned proc, const VMemwT<GraphT> &customMemWeight, const VCommwT<GraphT> &) {
        currentProcMemory[proc] += customMemWeight;
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
inline constexpr bool isMemoryConstraintScheduleV = IsMemoryConstraintSchedule<T>::value;

template <typename GraphT>
struct LocalInOutMemoryConstraint {
    static_assert(std::is_convertible_v<VCommwT<GraphT>, VMemwT<GraphT>>,
                  "local_in_out_memory_constraint requires that memory and communication weights are convertible.");

    using GraphImplT = GraphT;

    const BspInstance<GraphT> *instance;
    const BspSchedule<GraphT> *schedule;

    const unsigned *currentSuperstep = 0;

    std::vector<VMemwT<GraphT>> currentProcMemory;

    LocalInOutMemoryConstraint() : instance(nullptr), schedule(nullptr) {}

    inline void Initialize(const BspSchedule<GraphT> &schedule_, const unsigned &supstepIdx) {
        currentSuperstep = &supstepIdx;
        schedule = &schedule_;
        instance = &schedule->GetInstance();
        currentProcMemory = std::vector<VMemwT<GraphT>>(instance->NumberOfProcessors(), 0);

        if (instance->GetArchitecture().GetMemoryConstraintType() != MEMORY_CONSTRAINT_TYPE::LOCAL_IN_OUT) {
            throw std::invalid_argument("Memory constraint type is not LOCAL_IN_OUT");
        }
    }

    inline bool CanAdd(const VertexIdxT<GraphT> &v, const unsigned proc) const {
        VMemwT<GraphT> incMemory
            = instance->GetComputationalDag().VertexMemWeight(v) + instance->GetComputationalDag().VertexCommWeight(v);

        for (const auto &pred : instance->GetComputationalDag().Parents(v)) {
            if (schedule->AssignedProcessor(pred) == schedule->AssignedProcessor(v)
                && schedule->AssignedSuperstep(pred) == *currentSuperstep) {
                incMemory -= instance->GetComputationalDag().VertexCommWeight(pred);
            }
        }

        return currentProcMemory[proc] + incMemory <= instance->GetArchitecture().MemoryBound(proc);
    }

    inline void Add(const VertexIdxT<GraphT> &v, const unsigned proc) {
        currentProcMemory[proc]
            += instance->GetComputationalDag().VertexMemWeight(v) + instance->GetComputationalDag().VertexCommWeight(v);

        for (const auto &pred : instance->GetComputationalDag().Parents(v)) {
            if (schedule->AssignedProcessor(pred) == schedule->AssignedProcessor(v)
                && schedule->AssignedSuperstep(pred) == *currentSuperstep) {
                currentProcMemory[proc] -= instance->GetComputationalDag().VertexCommWeight(pred);
            }
        }
    }

    inline void Reset(const unsigned proc) { currentProcMemory[proc] = 0; }
};

template <typename GraphT>
struct LocalIncEdgesMemoryConstraint {
    using GraphImplT = GraphT;

    const BspInstance<GraphT> *instance;
    const BspSchedule<GraphT> *schedule;

    const unsigned *currentSuperstep = 0;

    std::vector<VCommwT<GraphT>> currentProcMemory;
    std::vector<std::unordered_set<VertexIdxT<GraphT>>> currentProcPredec;

    LocalIncEdgesMemoryConstraint() : instance(nullptr), schedule(nullptr) {}

    inline void Initialize(const BspSchedule<GraphT> &schedule_, const unsigned &supstepIdx) {
        currentSuperstep = &supstepIdx;
        schedule = &schedule_;
        instance = &schedule->GetInstance();

        currentProcMemory = std::vector<VCommwT<GraphT>>(instance->NumberOfProcessors(), 0);
        currentProcPredec = std::vector<std::unordered_set<VertexIdxT<GraphT>>>(instance->NumberOfProcessors());

        if (instance->GetArchitecture().GetMemoryConstraintType() != MEMORY_CONSTRAINT_TYPE::LOCAL_INC_EDGES) {
            throw std::invalid_argument("Memory constraint type is not LOCAL_INC_EDGES");
        }
    }

    inline bool CanAdd(const VertexIdxT<GraphT> &v, const unsigned proc) const {
        VCommwT<GraphT> incMemory = instance->GetComputationalDag().VertexCommWeight(v);

        for (const auto &pred : instance->GetComputationalDag().Parents(v)) {
            if (schedule->AssignedSuperstep(pred) != *currentSuperstep
                && currentProcPredec[proc].find(pred) == currentProcPredec[proc].end()) {
                incMemory += instance->GetComputationalDag().VertexCommWeight(pred);
            }
        }

        return currentProcMemory[proc] + incMemory <= instance->GetArchitecture().MemoryBound(proc);
    }

    inline void Add(const VertexIdxT<GraphT> &v, const unsigned proc) {
        currentProcMemory[proc] += instance->GetComputationalDag().VertexCommWeight(v);

        for (const auto &pred : instance->GetComputationalDag().Parents(v)) {
            if (schedule->AssignedSuperstep(pred) != *currentSuperstep) {
                const auto pair = currentProcPredec[proc].insert(pred);
                if (pair.second) {
                    currentProcMemory[proc] += instance->GetComputationalDag().VertexCommWeight(pred);
                }
            }
        }
    }

    inline void Reset(const unsigned proc) {
        currentProcMemory[proc] = 0;
        currentProcPredec[proc].clear();
    }
};

template <typename GraphT>
struct LocalSourcesIncEdgesMemoryConstraint {
    static_assert(std::is_convertible_v<VCommwT<GraphT>, VMemwT<GraphT>>,
                  "local_sources_inc_edges_memory_constraint requires that memory and communication weights are convertible.");

    using GraphImplT = GraphT;

    const BspInstance<GraphT> *instance;
    const BspSchedule<GraphT> *schedule;

    const unsigned *currentSuperstep = 0;

    std::vector<VMemwT<GraphT>> currentProcMemory;
    std::vector<std::unordered_set<VertexIdxT<GraphT>>> currentProcPredec;

    LocalSourcesIncEdgesMemoryConstraint() : instance(nullptr), schedule(nullptr) {}

    inline void Initialize(const BspSchedule<GraphT> &schedule_, const unsigned &supstepIdx) {
        currentSuperstep = &supstepIdx;
        schedule = &schedule_;
        instance = &schedule->GetInstance();

        currentProcMemory = std::vector<VMemwT<GraphT>>(instance->NumberOfProcessors(), 0);
        currentProcPredec = std::vector<std::unordered_set<VertexIdxT<GraphT>>>(instance->NumberOfProcessors());

        if (instance->GetArchitecture().GetMemoryConstraintType() != MEMORY_CONSTRAINT_TYPE::LOCAL_SOURCES_INC_EDGES) {
            throw std::invalid_argument("Memory constraint type is not LOCAL_INC_EDGES_2");
        }
    }

    inline bool CanAdd(const VertexIdxT<GraphT> &v, const unsigned proc) const {
        VMemwT<GraphT> incMemory = 0;

        if (IsSource(v, instance->GetComputationalDag())) {
            incMemory += instance->GetComputationalDag().VertexMemWeight(v);
        }

        for (const auto &pred : instance->GetComputationalDag().Parents(v)) {
            if (schedule->AssignedSuperstep(v) != *currentSuperstep
                && currentProcPredec[proc].find(pred) == currentProcPredec[proc].end()) {
                incMemory += instance->GetComputationalDag().VertexCommWeight(pred);
            }
        }

        return currentProcMemory[proc] + incMemory <= instance->GetArchitecture().MemoryBound(proc);
    }

    inline void Add(const VertexIdxT<GraphT> &v, const unsigned proc) {
        if (IsSource(v, instance->GetComputationalDag())) {
            currentProcMemory[proc] += instance->GetComputationalDag().VertexMemWeight(v);
        }

        for (const auto &pred : instance->GetComputationalDag().Parents(v)) {
            if (schedule->AssignedSuperstep(pred) != *currentSuperstep) {
                const auto pair = currentProcPredec[proc].insert(pred);
                if (pair.second) {
                    currentProcMemory[proc] += instance->GetComputationalDag().VertexCommWeight(pred);
                }
            }
        }
    }

    inline void Reset(const unsigned proc) {
        currentProcMemory[proc] = 0;
        currentProcPredec[proc].clear();
    }
};

}    // namespace osp
