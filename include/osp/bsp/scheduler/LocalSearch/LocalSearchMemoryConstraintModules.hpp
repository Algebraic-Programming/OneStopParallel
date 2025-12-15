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
    std::void_t<decltype(std::declval<T>().Initialize(std::declval<SetSchedule<typename T::GraphImplT>>(),
                                                      std::declval<VectorSchedule<typename T::GraphImplT>>())),
                decltype(std::declval<T>().ApplyMove(std::declval<VertexIdxT<typename T::GraphImplT>>(),
                                                     std::declval<unsigned>(),
                                                     std::declval<unsigned>(),
                                                     std::declval<unsigned>(),
                                                     std::declval<unsigned>())),
                decltype(std::declval<T>().ComputeMemoryDatastructure(std::declval<unsigned>(), std::declval<unsigned>())),
                decltype(std::declval<T>().SwapSteps(std::declval<unsigned>(), std::declval<unsigned>())),
                decltype(std::declval<T>().ResetSuperstep(std::declval<unsigned>())),
                decltype(std::declval<T>().OverrideSuperstep(
                    std::declval<unsigned>(), std::declval<unsigned>(), std::declval<unsigned>(), std::declval<unsigned>())),
                decltype(std::declval<T>().CanMove(
                    std::declval<VertexIdxT<typename T::GraphImplT>>(), std::declval<unsigned>(), std::declval<unsigned>())),
                decltype(std::declval<T>().Clear()),
                decltype(T())>> : std::true_type {};

template <typename T>
inline constexpr bool IsLocalSearchMemoryConstraintV = IsLocalSearchMemoryConstraint<T>::value;

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

    std::vector<std::vector<VMemwT<GraphT>>> stepProcessorMemory_;

    LsLocalMemoryConstraint() : setSchedule_(nullptr), graph_(nullptr) {}

    inline void Initialize(const SetSchedule<GraphT> &setSchedule, const VectorSchedule<GraphT> &) {
        if (setSchedule.GetInstance().GetArchitecture().GetMemoryConstraintType() != MemoryConstraintType::LOCAL) {
            throw std::invalid_argument("Memory constraint type is not LOCAL");
        }

        setSchedule_ = &setSchedule;
        graph_ = &setSchedule_->GetInstance().GetComputationalDag();
        stepProcessorMemory_ = std::vector<std::vector<VMemwT<GraphT>>>(
            setSchedule_->NumberOfSupersteps(), std::vector<VMemwT<GraphT>>(setSchedule_->GetInstance().NumberOfProcessors(), 0));
    }

    inline void ApplyMove(VertexIdxT<GraphT> vertex, unsigned fromProc, unsigned fromStep, unsigned toProc, unsigned toStep) {
        stepProcessorMemory_[toStep][toProc] += graph_->VertexMemWeight(vertex);
        stepProcessorMemory_[fromStep][fromProc] -= graph_->VertexMemWeight(vertex);
    }

    inline bool CanMove(VertexIdxT<GraphT> vertex, const unsigned proc, unsigned step) const {
        return stepProcessorMemory_[step][proc] + graph_->VertexMemWeight(vertex)
               <= setSchedule_->GetInstance().GetArchitecture().MemoryBound(proc);
    }

    void SwapSteps(const unsigned step1, const unsigned step2) {
        std::swap(stepProcessorMemory_[step1], stepProcessorMemory_[step2]);
    }

    void ComputeMemoryDatastructure(unsigned startStep, unsigned endStep) {
        for (unsigned step = startStep; step <= endStep; step++) {
            for (unsigned proc = 0; proc < setSchedule_->GetInstance().NumberOfProcessors(); proc++) {
                stepProcessorMemory_[step][proc] = 0;

                for (const auto &node : setSchedule_->step_processor_vertices[step][proc]) {
                    stepProcessorMemory_[step][proc] += graph_->VertexMemWeight(node);
                }
            }
        }
    }

    inline void Clear() { stepProcessorMemory_.clear(); }

    inline void ForwardMove(VertexIdxT<GraphT> vertex, unsigned, unsigned, unsigned toProc, unsigned toStep) {
        stepProcessorMemory_[toStep][toProc] += graph_->VertexMemWeight(vertex);
    }

    inline void ResetSuperstep(unsigned step) {
        for (unsigned proc = 0; proc < setSchedule_->GetInstance().GetArchitecture().NumberOfProcessors(); proc++) {
            stepProcessorMemory_[step][proc] = 0;
        }
    }

    void OverrideSuperstep(unsigned step, unsigned proc, unsigned withStep, unsigned withProc) {
        stepProcessorMemory_[step][proc] = stepProcessorMemory_[withStep][withProc];
    }

    bool SatisfiedMemoryConstraint() const {
        for (unsigned step = 0; step < setSchedule_->NumberOfSupersteps(); step++) {
            for (unsigned proc = 0; proc < setSchedule_->GetInstance().NumberOfProcessors(); proc++) {
                if (stepProcessorMemory_[step][proc] > setSchedule_->GetInstance().GetArchitecture().MemoryBound(proc)) {
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

    std::vector<std::vector<VMemwT<GraphT>>> stepProcessorMemory_;
    std::vector<std::vector<std::unordered_set<VertexIdxT<GraphT>>>> stepProcessorPred_;

    LsLocalIncEdgesMemoryConstraint() : setSchedule_(nullptr), vectorSchedule_(nullptr), graph_(nullptr) {}

    inline void Initialize(const SetSchedule<GraphT> &setSchedule, const VectorSchedule<GraphT> &vecSchedule) {
        if (setSchedule.GetInstance().GetArchitecture().GetMemoryConstraintType() != MemoryConstraintType::LOCAL_INC_EDGES) {
            throw std::invalid_argument("Memory constraint type is not LOCAL_INC_EDGES");
        }

        setSchedule_ = &setSchedule;
        vectorSchedule_ = &vecSchedule;
        graph_ = &setSchedule_->GetInstance().GetComputationalDag();
        stepProcessorMemory_ = std::vector<std::vector<VMemwT<GraphT>>>(
            setSchedule_->NumberOfSupersteps(), std::vector<VMemwT<GraphT>>(setSchedule_->GetInstance().NumberOfProcessors(), 0));
        stepProcessorPred_ = std::vector<std::vector<std::unordered_set<VertexIdxT<GraphT>>>>(
            setSchedule_->NumberOfSupersteps(),
            std::vector<std::unordered_set<VertexIdxT<GraphT>>>(setSchedule_->GetInstance().NumberOfProcessors()));
    }

    inline void ApplyMove(VertexIdxT<GraphT> vertex, unsigned fromProc, unsigned fromStep, unsigned toProc, unsigned toStep) {
        stepProcessorMemory_[toStep][toProc] += graph_->VertexCommWeight(vertex);
        stepProcessorMemory_[fromStep][fromProc] -= graph_->VertexCommWeight(vertex);

        for (const auto &pred : graph_->Parents(vertex)) {
            if (vectorSchedule_->AssignedSuperstep(pred) < toStep) {
                auto pair = stepProcessorPred_[toStep][toProc].insert(pred);
                if (pair.second) {
                    stepProcessorMemory_[toStep][toProc] += graph_->VertexCommWeight(pred);
                }
            }

            if (vectorSchedule_->AssignedSuperstep(pred) < fromStep) {
                bool remove = true;
                for (const auto &succ : graph_->Children(pred)) {
                    if (succ == vertex) {
                        continue;
                    }

                    if (vectorSchedule_->AssignedProcessor(succ) == fromProc
                        && vectorSchedule_->AssignedSuperstep(succ) == fromStep) {
                        remove = false;
                        break;
                    }
                }

                if (remove) {
                    stepProcessorMemory_[fromStep][fromProc] -= graph_->VertexCommWeight(pred);
                    stepProcessorPred_[fromStep][fromProc].erase(pred);
                }
            }
        }

        if (toStep != fromStep) {
            for (const auto &succ : graph_->Children(vertex)) {
                if (toStep > fromStep && vectorSchedule_->AssignedSuperstep(succ) == toStep) {
                    if (stepProcessorPred_[vectorSchedule_->AssignedSuperstep(succ)][vectorSchedule_->AssignedProcessor(succ)].find(
                            vertex)
                        != stepProcessorPred_[vectorSchedule_->AssignedSuperstep(succ)][vectorSchedule_->AssignedProcessor(succ)]
                               .end()) {
                        stepProcessorMemory_[vectorSchedule_->AssignedSuperstep(succ)][vectorSchedule_->AssignedProcessor(succ)]
                            -= graph_->VertexCommWeight(vertex);

                        stepProcessorPred_[vectorSchedule_->AssignedSuperstep(succ)][vectorSchedule_->AssignedProcessor(succ)].erase(
                            vertex);
                    }
                }

                if (vectorSchedule_->AssignedSuperstep(succ) > toStep) {
                    auto pair
                        = stepProcessorPred_[vectorSchedule_->AssignedSuperstep(succ)][vectorSchedule_->AssignedProcessor(succ)]
                              .insert(vertex);
                    if (pair.second) {
                        stepProcessorMemory_[vectorSchedule_->AssignedSuperstep(succ)][vectorSchedule_->AssignedProcessor(succ)]
                            += graph_->VertexCommWeight(vertex);
                    }
                }
            }
        }
    }

    void SwapSteps(const unsigned step1, const unsigned step2) {
        std::swap(stepProcessorMemory_[step1], stepProcessorMemory_[step2]);
        std::swap(stepProcessorPred_[step1], stepProcessorPred_[step2]);
    }

    void ComputeMemoryDatastructure(unsigned startStep, unsigned endStep) {
        for (unsigned step = startStep; step <= endStep; step++) {
            for (unsigned proc = 0; proc < setSchedule_->GetInstance().NumberOfProcessors(); proc++) {
                stepProcessorMemory_[step][proc] = 0;
                stepProcessorPred_[step][proc].clear();

                for (const auto &node : setSchedule_->step_processor_vertices[step][proc]) {
                    stepProcessorMemory_[step][proc] += graph_->VertexCommWeight(node);

                    for (const auto &pred : graph_->Parents(node)) {
                        if (vectorSchedule_->AssignedSuperstep(pred) < step) {
                            auto pair = stepProcessorPred_[step][proc].insert(pred);
                            if (pair.second) {
                                stepProcessorMemory_[step][proc] += graph_->VertexCommWeight(pred);
                            }
                        }
                    }
                }
            }
        }
    }

    inline void Clear() {
        stepProcessorMemory_.clear();
        stepProcessorPred_.clear();
    }

    inline void ResetSuperstep(unsigned step) {
        for (unsigned proc = 0; proc < setSchedule_->GetInstance().GetArchitecture().NumberOfProcessors(); proc++) {
            stepProcessorMemory_[step][proc] = 0;
            stepProcessorPred_[step][proc].clear();
        }
    }

    void OverrideSuperstep(unsigned step, unsigned proc, unsigned withStep, unsigned withProc) {
        stepProcessorMemory_[step][proc] = stepProcessorMemory_[withStep][withProc];
        stepProcessorPred_[step][proc] = stepProcessorPred_[withStep][withProc];
    }

    inline bool CanMove(VertexIdxT<GraphT> vertex, const unsigned proc, unsigned step) const {
        VMemwT<GraphT> incMemory = graph_->VertexCommWeight(vertex);
        for (const auto &pred : graph_->Parents(vertex)) {
            if (vectorSchedule_->AssignedSuperstep(pred) < step) {
                if (stepProcessorPred_[step][proc].find(pred) == stepProcessorPred_[step][proc].end()) {
                    incMemory += graph_->VertexCommWeight(pred);
                }
            }
        }

        if (step > vectorSchedule_->AssignedSuperstep(vertex)) {
            if (stepProcessorPred_[step][proc].find(vertex) != stepProcessorPred_[step][proc].end()) {
                incMemory -= graph_->VertexCommWeight(vertex);
            }
        }

        if (step >= vectorSchedule_->AssignedSuperstep(vertex)) {
            return stepProcessorMemory_[step][proc] + incMemory <= setSchedule_->GetInstance().GetArchitecture().MemoryBound(proc);
        }

        if (stepProcessorMemory_[step][proc] + incMemory > setSchedule_->GetInstance().GetArchitecture().MemoryBound(proc)) {
            return false;
        }

        for (const auto &succ : graph_->Children(vertex)) {
            const auto &succStep = vectorSchedule_->AssignedSuperstep(succ);
            const auto &succProc = vectorSchedule_->AssignedProcessor(succ);

            if (succStep == vectorSchedule_->AssignedSuperstep(vertex) and succProc != vectorSchedule_->AssignedProcessor(vertex)) {
                if (stepProcessorMemory_[succStep][succProc] + graph_->VertexCommWeight(vertex)
                    > setSchedule_->GetInstance().GetArchitecture().MemoryBound(succProc)) {
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

    std::vector<std::vector<VMemwT<GraphT>>> stepProcessorMemory_;
    std::vector<std::vector<std::unordered_set<VertexIdxT<GraphT>>>> stepProcessorPred_;

    LsLocalSourcesIncEdgesMemoryConstraint() : setSchedule_(nullptr), vectorSchedule_(nullptr), graph_(nullptr) {}

    inline void SwapSteps(const unsigned step1, const unsigned step2) {
        std::swap(stepProcessorMemory_[step1], stepProcessorMemory_[step2]);
        std::swap(stepProcessorPred_[step1], stepProcessorPred_[step2]);
    }

    inline void Initialize(const SetSchedule<GraphT> &setSchedule, const VectorSchedule<GraphT> &vecSchedule) {
        if (setSchedule.GetInstance().GetArchitecture().GetMemoryConstraintType() != MemoryConstraintType::LOCAL_SOURCES_INC_EDGES) {
            throw std::invalid_argument("Memory constraint type is not LOCAL_SOURCES_INC_EDGES");
        }

        setSchedule_ = &setSchedule;
        vectorSchedule_ = &vecSchedule;
        graph_ = &setSchedule_->GetInstance().GetComputationalDag();
        stepProcessorMemory_ = std::vector<std::vector<VMemwT<GraphT>>>(
            setSchedule_->NumberOfSupersteps(), std::vector<VMemwT<GraphT>>(setSchedule_->GetInstance().NumberOfProcessors(), 0));
        stepProcessorPred_ = std::vector<std::vector<std::unordered_set<VertexIdxT<GraphT>>>>(
            setSchedule_->NumberOfSupersteps(),
            std::vector<std::unordered_set<VertexIdxT<GraphT>>>(setSchedule_->GetInstance().NumberOfProcessors()));
    }

    inline void ApplyMove(VertexIdxT<GraphT> vertex, unsigned fromProc, unsigned fromStep, unsigned toProc, unsigned toStep) {
        if (IsSource(vertex, *graph_)) {
            stepProcessorMemory_[toStep][toProc] += graph_->VertexMemWeight(vertex);
            stepProcessorMemory_[fromStep][fromProc] -= graph_->VertexMemWeight(vertex);
        }

        for (const auto &pred : graph_->Parents(vertex)) {
            if (vectorSchedule_->AssignedSuperstep(pred) < toStep) {
                auto pair = stepProcessorPred_[toStep][toProc].insert(pred);
                if (pair.second) {
                    stepProcessorMemory_[toStep][toProc] += graph_->VertexCommWeight(pred);
                }
            }

            if (vectorSchedule_->AssignedSuperstep(pred) < fromStep) {
                bool remove = true;
                for (const auto &succ : graph_->Children(pred)) {
                    if (succ == vertex) {
                        continue;
                    }

                    if (vectorSchedule_->AssignedProcessor(succ) == fromProc
                        && vectorSchedule_->AssignedSuperstep(succ) == fromStep) {
                        remove = false;
                        break;
                    }
                }

                if (remove) {
                    stepProcessorMemory_[fromStep][fromProc] -= graph_->VertexCommWeight(pred);
                    stepProcessorPred_[fromStep][fromProc].erase(pred);
                }
            }
        }

        if (toStep != fromStep) {
            for (const auto &succ : graph_->Children(vertex)) {
                if (toStep > fromStep && vectorSchedule_->AssignedSuperstep(succ) == toStep) {
                    if (stepProcessorPred_[vectorSchedule_->AssignedSuperstep(succ)][vectorSchedule_->AssignedProcessor(succ)].find(
                            vertex)
                        != stepProcessorPred_[vectorSchedule_->AssignedSuperstep(succ)][vectorSchedule_->AssignedProcessor(succ)]
                               .end()) {
                        stepProcessorMemory_[vectorSchedule_->AssignedSuperstep(succ)][vectorSchedule_->AssignedProcessor(succ)]
                            -= graph_->VertexCommWeight(vertex);

                        stepProcessorPred_[vectorSchedule_->AssignedSuperstep(succ)][vectorSchedule_->AssignedProcessor(succ)].erase(
                            vertex);
                    }
                }

                if (vectorSchedule_->AssignedSuperstep(succ) > toStep) {
                    auto pair
                        = stepProcessorPred_[vectorSchedule_->AssignedSuperstep(succ)][vectorSchedule_->AssignedProcessor(succ)]
                              .insert(vertex);
                    if (pair.second) {
                        stepProcessorMemory_[vectorSchedule_->AssignedSuperstep(succ)][vectorSchedule_->AssignedProcessor(succ)]
                            += graph_->VertexCommWeight(vertex);
                    }
                }
            }
        }
    }

    void ComputeMemoryDatastructure(unsigned startStep, unsigned endStep) {
        for (unsigned step = startStep; step <= endStep; step++) {
            for (unsigned proc = 0; proc < setSchedule_->GetInstance().NumberOfProcessors(); proc++) {
                stepProcessorMemory_[step][proc] = 0;
                stepProcessorPred_[step][proc].clear();

                for (const auto &node : setSchedule_->step_processor_vertices[step][proc]) {
                    if (IsSource(node, *graph_)) {
                        stepProcessorMemory_[step][proc] += graph_->VertexMemWeight(node);
                    }

                    for (const auto &pred : graph_->Parents(node)) {
                        if (vectorSchedule_->AssignedSuperstep(pred) < step) {
                            auto pair = stepProcessorPred_[step][proc].insert(pred);
                            if (pair.second) {
                                stepProcessorMemory_[step][proc] += graph_->VertexCommWeight(pred);
                            }
                        }
                    }
                }
            }
        }
    }

    inline void Clear() {
        stepProcessorMemory_.clear();
        stepProcessorPred_.clear();
    }

    inline void ResetSuperstep(unsigned step) {
        for (unsigned proc = 0; proc < setSchedule_->GetInstance().GetArchitecture().NumberOfProcessors(); proc++) {
            stepProcessorMemory_[step][proc] = 0;
            stepProcessorPred_[step][proc].clear();
        }
    }

    void OverrideSuperstep(unsigned step, unsigned proc, unsigned withStep, unsigned withProc) {
        stepProcessorMemory_[step][proc] = stepProcessorMemory_[withStep][withProc];
        stepProcessorPred_[step][proc] = stepProcessorPred_[withStep][withProc];
    }

    inline bool CanMove(VertexIdxT<GraphT> vertex, const unsigned proc, unsigned step) const {
        VMemwT<GraphT> incMemory = 0;

        if (IsSource(vertex, *graph_)) {
            incMemory += graph_->VertexMemWeight(vertex);
        }

        for (const auto &pred : graph_->Parents(vertex)) {
            if (vectorSchedule_->AssignedSuperstep(pred) < step) {
                if (stepProcessorPred_[step][proc].find(pred) == stepProcessorPred_[step][proc].end()) {
                    incMemory += graph_->VertexCommWeight(pred);
                }
            }
        }

        if (vectorSchedule_->AssignedSuperstep(vertex) < step) {
            if (stepProcessorPred_[step][proc].find(vertex) != stepProcessorPred_[step][proc].end()) {
                incMemory -= graph_->VertexCommWeight(vertex);
            }
        }

        if (vectorSchedule_->AssignedSuperstep(vertex) <= step) {
            return stepProcessorMemory_[step][proc] + incMemory <= setSchedule_->GetInstance().GetArchitecture().MemoryBound(proc);
        }

        if (stepProcessorMemory_[step][proc] + incMemory > setSchedule_->GetInstance().GetArchitecture().MemoryBound(proc)) {
            return false;
        }

        for (const auto &succ : graph_->Children(vertex)) {
            const auto &succStep = vectorSchedule_->AssignedSuperstep(succ);
            const auto &succProc = vectorSchedule_->AssignedProcessor(succ);

            if (succStep == vectorSchedule_->AssignedSuperstep(vertex)) {
                if (vectorSchedule_->AssignedProcessor(vertex) != succProc || (not IsSource(vertex, *graph_))) {
                    if (stepProcessorMemory_[succStep][succProc] + graph_->VertexCommWeight(vertex)
                        > setSchedule_->GetInstance().GetArchitecture().MemoryBound(succProc)) {
                        return false;
                    }

                } else {
                    if (IsSource(vertex, *graph_)) {
                        if (stepProcessorMemory_[succStep][succProc] + graph_->VertexCommWeight(vertex)
                                - graph_->VertexMemWeight(vertex)
                            > setSchedule_->GetInstance().GetArchitecture().MemoryBound(succProc)) {
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
