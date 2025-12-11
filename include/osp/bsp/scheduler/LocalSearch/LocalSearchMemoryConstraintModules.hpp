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
                decltype(std::declval<T>().apply_move(std::declval<VertexIdxT<typename T::Graph_impl_t>>(),
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
                    std::declval<VertexIdxT<typename T::Graph_impl_t>>(), std::declval<unsigned>(), std::declval<unsigned>())),
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

    const SetSchedule<GraphT> *setSchedule;
    const GraphT *graph;

    std::vector<std::vector<VMemwT<GraphT>>> stepProcessorMemory;

    LsLocalMemoryConstraint() : setSchedule(nullptr), graph(nullptr) {}

    inline void Initialize(const SetSchedule<GraphT> &setSched, const VectorSchedule<GraphT> &) {
        if (setSched.GetInstance().GetArchitecture().GetMemoryConstraintType() != MEMORY_CONSTRAINT_TYPE::LOCAL) {
            throw std::invalid_argument("Memory constraint type is not LOCAL");
        }

        setSchedule = &setSched;
        graph = &setSchedule->GetInstance().GetComputationalDag();
        stepProcessorMemory = std::vector<std::vector<VMemwT<GraphT>>>(
            setSchedule->NumberOfSupersteps(), std::vector<VMemwT<GraphT>>(setSchedule->GetInstance().NumberOfProcessors(), 0));
    }

    inline void ApplyMove(VertexIdxT<GraphT> vertex, unsigned fromProc, unsigned fromStep, unsigned toProc, unsigned toStep) {
        stepProcessorMemory[toStep][toProc] += graph->VertexMemWeight(vertex);
        stepProcessorMemory[fromStep][fromProc] -= graph->VertexMemWeight(vertex);
    }

    inline bool CanMove(VertexIdxT<GraphT> vertex, const unsigned proc, unsigned step) const {
        return stepProcessorMemory[step][proc] + graph->VertexMemWeight(vertex)
               <= setSchedule->GetInstance().GetArchitecture().MemoryBound(proc);
    }

    void SwapSteps(const unsigned step1, const unsigned step2) {
        std::swap(stepProcessorMemory[step1], stepProcessorMemory[step2]);
    }

    void ComputeMemoryDatastructure(unsigned startStep, unsigned endStep) {
        for (unsigned step = startStep; step <= endStep; step++) {
            for (unsigned proc = 0; proc < setSchedule->GetInstance().NumberOfProcessors(); proc++) {
                stepProcessorMemory[step][proc] = 0;

                for (const auto &node : setSchedule->stepProcessorVertices[step][proc]) {
                    stepProcessorMemory[step][proc] += graph->VertexMemWeight(node);
                }
            }
        }
    }

    inline void Clear() { stepProcessorMemory.clear(); }

    inline void ForwardMove(VertexIdxT<GraphT> vertex, unsigned, unsigned, unsigned toProc, unsigned toStep) {
        stepProcessorMemory[toStep][toProc] += graph->vertex_mem_weight(vertex);
        // step_processor_memory[from_step][from_proc] -= graph->vertex_mem_weight(vertex);
    }

    inline void ResetSuperstep(unsigned step) {
        for (unsigned proc = 0; proc < setSchedule->GetInstance().getArchitecture().numberOfProcessors(); proc++) {
            stepProcessorMemory[step][proc] = 0;
        }
    }

    void OverrideSuperstep(unsigned step, unsigned proc, unsigned withStep, unsigned withProc) {
        stepProcessorMemory[step][proc] = stepProcessorMemory[withStep][withProc];
    }

    bool SatisfiedMemoryConstraint() const {
        for (unsigned step = 0; step < setSchedule->numberOfSupersteps(); step++) {
            for (unsigned proc = 0; proc < setSchedule->GetInstance().numberOfProcessors(); proc++) {
                if (stepProcessorMemory[step][proc] > setSchedule->GetInstance().getArchitecture().memoryBound(proc)) {
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

    const SetSchedule<GraphT> *setSchedule;
    const VectorSchedule<GraphT> *vectorSchedule;
    const GraphT *graph;

    std::vector<std::vector<VMemwT<GraphT>>> stepProcessorMemory;
    std::vector<std::vector<std::unordered_set<VertexIdxT<GraphT>>>> stepProcessorPred;

    LsLocalIncEdgesMemoryConstraint() : setSchedule(nullptr), vectorSchedule(nullptr), graph(nullptr) {}

    inline void Initialize(const SetSchedule<GraphT> &setSched, const VectorSchedule<GraphT> &vecSchedule) {
        if (setSched.GetInstance().getArchitecture().getMemoryConstraintType() != MEMORY_CONSTRAINT_TYPE::LOCAL_INC_EDGES) {
            throw std::invalid_argument("Memory constraint type is not LOCAL_INC_EDGES");
        }

        setSchedule = &setSched;
        vectorSchedule = &vecSchedule;
        graph = &setSchedule->GetInstance().getComputationalDag();
        stepProcessorMemory = std::vector<std::vector<VMemwT<GraphT>>>(
            setSchedule->numberOfSupersteps(), std::vector<VMemwT<GraphT>>(setSchedule->GetInstance().numberOfProcessors(), 0));
        stepProcessorPred = std::vector<std::vector<std::unordered_set<VertexIdxT<GraphT>>>>(
            setSchedule->numberOfSupersteps(),
            std::vector<std::unordered_set<VertexIdxT<GraphT>>>(setSchedule->GetInstance().numberOfProcessors()));
    }

    inline void ApplyMove(VertexIdxT<GraphT> vertex, unsigned fromProc, unsigned fromStep, unsigned toProc, unsigned toStep) {
        stepProcessorMemory[toStep][toProc] += graph->vertex_comm_weight(vertex);
        stepProcessorMemory[fromStep][fromProc] -= graph->vertex_comm_weight(vertex);

        for (const auto &pred : graph->parents(vertex)) {
            if (vectorSchedule->assignedSuperstep(pred) < toStep) {
                auto pair = stepProcessorPred[toStep][toProc].insert(pred);
                if (pair.second) {
                    stepProcessorMemory[toStep][toProc] += graph->vertex_comm_weight(pred);
                }
            }

            if (vectorSchedule->assignedSuperstep(pred) < fromStep) {
                bool remove = true;
                for (const auto &succ : graph->children(pred)) {
                    if (succ == vertex) {
                        continue;
                    }

                    if (vectorSchedule->assignedProcessor(succ) == fromProc && vectorSchedule->assignedSuperstep(succ) == fromStep) {
                        remove = false;
                        break;
                    }
                }

                if (remove) {
                    stepProcessorMemory[fromStep][fromProc] -= graph->vertex_comm_weight(pred);
                    stepProcessorPred[fromStep][fromProc].erase(pred);
                }
            }
        }

        if (toStep != fromStep) {
            for (const auto &succ : graph->children(vertex)) {
                if (toStep > fromStep && vectorSchedule->assignedSuperstep(succ) == toStep) {
                    if (stepProcessorPred[vectorSchedule->assignedSuperstep(succ)][vectorSchedule->assignedProcessor(succ)].find(
                            vertex)
                        != stepProcessorPred[vectorSchedule->assignedSuperstep(succ)][vectorSchedule->assignedProcessor(succ)].end()) {
                        stepProcessorMemory[vectorSchedule->assignedSuperstep(succ)][vectorSchedule->assignedProcessor(succ)]
                            -= graph->vertex_comm_weight(vertex);

                        stepProcessorPred[vectorSchedule->assignedSuperstep(succ)][vectorSchedule->assignedProcessor(succ)].erase(
                            vertex);
                    }
                }

                if (vectorSchedule->assignedSuperstep(succ) > toStep) {
                    auto pair
                        = stepProcessorPred[vectorSchedule->assignedSuperstep(succ)][vectorSchedule->assignedProcessor(succ)].insert(
                            vertex);
                    if (pair.second) {
                        stepProcessorMemory[vectorSchedule->assignedSuperstep(succ)][vectorSchedule->assignedProcessor(succ)]
                            += graph->vertex_comm_weight(vertex);
                    }
                }
            }
        }
    }

    void SwapSteps(const unsigned step1, const unsigned step2) {
        std::swap(stepProcessorMemory[step1], stepProcessorMemory[step2]);
        std::swap(stepProcessorPred[step1], stepProcessorPred[step2]);
    }

    void ComputeMemoryDatastructure(unsigned startStep, unsigned endStep) {
        for (unsigned step = startStep; step <= endStep; step++) {
            for (unsigned proc = 0; proc < setSchedule->GetInstance().numberOfProcessors(); proc++) {
                stepProcessorMemory[step][proc] = 0;
                stepProcessorPred[step][proc].clear();

                for (const auto &node : setSchedule->step_processor_vertices[step][proc]) {
                    stepProcessorMemory[step][proc] += graph->vertex_comm_weight(node);

                    for (const auto &pred : graph->parents(node)) {
                        if (vectorSchedule->assignedSuperstep(pred) < step) {
                            auto pair = stepProcessorPred[step][proc].insert(pred);
                            if (pair.second) {
                                stepProcessorMemory[step][proc] += graph->vertex_comm_weight(pred);
                            }
                        }
                    }
                }
            }
        }
    }

    inline void Clear() {
        stepProcessorMemory.clear();
        stepProcessorPred.clear();
    }

    inline void ResetSuperstep(unsigned step) {
        for (unsigned proc = 0; proc < setSchedule->GetInstance().getArchitecture().numberOfProcessors(); proc++) {
            stepProcessorMemory[step][proc] = 0;
            stepProcessorPred[step][proc].clear();
        }
    }

    void OverrideSuperstep(unsigned step, unsigned proc, unsigned withStep, unsigned withProc) {
        stepProcessorMemory[step][proc] = stepProcessorMemory[withStep][withProc];
        stepProcessorPred[step][proc] = stepProcessorPred[withStep][withProc];
    }

    inline bool CanMove(VertexIdxT<GraphT> vertex, const unsigned proc, unsigned step) const {
        VMemwT<GraphT> incMemory = graph->vertex_comm_weight(vertex);
        for (const auto &pred : graph->parents(vertex)) {
            if (vectorSchedule->assignedSuperstep(pred) < step) {
                if (stepProcessorPred[step][proc].find(pred) == stepProcessorPred[step][proc].end()) {
                    incMemory += graph->vertex_comm_weight(pred);
                }
            }
        }

        if (step > vectorSchedule->assignedSuperstep(vertex)) {
            if (stepProcessorPred[step][proc].find(vertex) != stepProcessorPred[step][proc].end()) {
                incMemory -= graph->vertex_comm_weight(vertex);
            }
        }

        if (step >= vectorSchedule->assignedSuperstep(vertex)) {
            return stepProcessorMemory[step][proc] + incMemory <= setSchedule->GetInstance().getArchitecture().memoryBound(proc);
        }

        if (stepProcessorMemory[step][proc] + incMemory > setSchedule->GetInstance().getArchitecture().memoryBound(proc)) {
            return false;
        }

        for (const auto &succ : graph->children(vertex)) {
            const auto &succStep = vectorSchedule->assignedSuperstep(succ);
            const auto &succProc = vectorSchedule->assignedProcessor(succ);

            if (succStep == vectorSchedule->assignedSuperstep(vertex) and succProc != vectorSchedule->assignedProcessor(vertex)) {
                if (stepProcessorMemory[succStep][succProc] + graph->vertex_comm_weight(vertex)
                    > setSchedule->GetInstance().getArchitecture().memoryBound(succProc)) {
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

    const SetSchedule<GraphT> *setSchedule;
    const VectorSchedule<GraphT> *vectorSchedule;
    const GraphT *graph;

    std::vector<std::vector<VMemwT<GraphT>>> stepProcessorMemory;
    std::vector<std::vector<std::unordered_set<VertexIdxT<GraphT>>>> stepProcessorPred;

    LsLocalSourcesIncEdgesMemoryConstraint() : setSchedule(nullptr), vectorSchedule(nullptr), graph(nullptr) {}

    inline void SwapSteps(const unsigned step1, const unsigned step2) {
        std::swap(stepProcessorMemory[step1], stepProcessorMemory[step2]);
        std::swap(stepProcessorPred[step1], stepProcessorPred[step2]);
    }

    inline void Initialize(const SetSchedule<GraphT> &setSched, const VectorSchedule<GraphT> &vecSchedule) {
        if (setSched.GetInstance().getArchitecture().getMemoryConstraintType() != MEMORY_CONSTRAINT_TYPE::LOCAL_SOURCES_INC_EDGES) {
            throw std::invalid_argument("Memory constraint type is not LOCAL_SOURCES_INC_EDGES");
        }

        setSchedule = &setSched;
        vectorSchedule = &vecSchedule;
        graph = &setSchedule->GetInstance().getComputationalDag();
        stepProcessorMemory = std::vector<std::vector<VMemwT<GraphT>>>(
            setSchedule->numberOfSupersteps(), std::vector<VMemwT<GraphT>>(setSchedule->GetInstance().numberOfProcessors(), 0));
        stepProcessorPred = std::vector<std::vector<std::unordered_set<VertexIdxT<GraphT>>>>(
            setSchedule->numberOfSupersteps(),
            std::vector<std::unordered_set<VertexIdxT<GraphT>>>(setSchedule->GetInstance().numberOfProcessors()));
    }

    inline void ApplyMove(VertexIdxT<GraphT> vertex, unsigned fromProc, unsigned fromStep, unsigned toProc, unsigned toStep) {
        if (is_source(vertex, *graph)) {
            stepProcessorMemory[toStep][toProc] += graph->vertex_mem_weight(vertex);
            stepProcessorMemory[fromStep][fromProc] -= graph->vertex_mem_weight(vertex);
        }

        for (const auto &pred : graph->parents(vertex)) {
            if (vectorSchedule->assignedSuperstep(pred) < toStep) {
                auto pair = stepProcessorPred[toStep][toProc].insert(pred);
                if (pair.second) {
                    stepProcessorMemory[toStep][toProc] += graph->vertex_comm_weight(pred);
                }
            }

            if (vectorSchedule->assignedSuperstep(pred) < fromStep) {
                bool remove = true;
                for (const auto &succ : graph->children(pred)) {
                    if (succ == vertex) {
                        continue;
                    }

                    if (vectorSchedule->assignedProcessor(succ) == fromProc && vectorSchedule->assignedSuperstep(succ) == fromStep) {
                        remove = false;
                        break;
                    }
                }

                if (remove) {
                    stepProcessorMemory[fromStep][fromProc] -= graph->vertex_comm_weight(pred);
                    stepProcessorPred[fromStep][fromProc].erase(pred);
                }
            }
        }

        if (toStep != fromStep) {
            for (const auto &succ : graph->children(vertex)) {
                if (toStep > fromStep && vectorSchedule->assignedSuperstep(succ) == toStep) {
                    if (stepProcessorPred[vectorSchedule->assignedSuperstep(succ)][vectorSchedule->assignedProcessor(succ)].find(
                            vertex)
                        != stepProcessorPred[vectorSchedule->assignedSuperstep(succ)][vectorSchedule->assignedProcessor(succ)].end()) {
                        stepProcessorMemory[vectorSchedule->assignedSuperstep(succ)][vectorSchedule->assignedProcessor(succ)]
                            -= graph->vertex_comm_weight(vertex);

                        stepProcessorPred[vectorSchedule->assignedSuperstep(succ)][vectorSchedule->assignedProcessor(succ)].erase(
                            vertex);
                    }
                }

                if (vectorSchedule->assignedSuperstep(succ) > toStep) {
                    auto pair
                        = stepProcessorPred[vectorSchedule->assignedSuperstep(succ)][vectorSchedule->assignedProcessor(succ)].insert(
                            vertex);
                    if (pair.second) {
                        stepProcessorMemory[vectorSchedule->assignedSuperstep(succ)][vectorSchedule->assignedProcessor(succ)]
                            += graph->vertex_comm_weight(vertex);
                    }
                }
            }
        }
    }

    void ComputeMemoryDatastructure(unsigned startStep, unsigned endStep) {
        for (unsigned step = startStep; step <= endStep; step++) {
            for (unsigned proc = 0; proc < setSchedule->GetInstance().numberOfProcessors(); proc++) {
                stepProcessorMemory[step][proc] = 0;
                stepProcessorPred[step][proc].clear();

                for (const auto &node : setSchedule->step_processor_vertices[step][proc]) {
                    if (is_source(node, *graph)) {
                        stepProcessorMemory[step][proc] += graph->vertex_mem_weight(node);
                    }

                    for (const auto &pred : graph->parents(node)) {
                        if (vectorSchedule->assignedSuperstep(pred) < step) {
                            auto pair = stepProcessorPred[step][proc].insert(pred);
                            if (pair.second) {
                                stepProcessorMemory[step][proc] += graph->vertex_comm_weight(pred);
                            }
                        }
                    }
                }
            }
        }
    }

    inline void Clear() {
        stepProcessorMemory.clear();
        stepProcessorPred.clear();
    }

    inline void ResetSuperstep(unsigned step) {
        for (unsigned proc = 0; proc < setSchedule->GetInstance().getArchitecture().numberOfProcessors(); proc++) {
            stepProcessorMemory[step][proc] = 0;
            stepProcessorPred[step][proc].clear();
        }
    }

    void OverrideSuperstep(unsigned step, unsigned proc, unsigned withStep, unsigned withProc) {
        stepProcessorMemory[step][proc] = stepProcessorMemory[withStep][withProc];
        stepProcessorPred[step][proc] = stepProcessorPred[withStep][withProc];
    }

    inline bool CanMove(VertexIdxT<GraphT> vertex, const unsigned proc, unsigned step) const {
        VMemwT<GraphT> incMemory = 0;

        if (is_source(vertex, *graph)) {
            incMemory += graph->vertex_mem_weight(vertex);
        }

        for (const auto &pred : graph->parents(vertex)) {
            if (vectorSchedule->assignedSuperstep(pred) < step) {
                if (stepProcessorPred[step][proc].find(pred) == stepProcessorPred[step][proc].end()) {
                    incMemory += graph->vertex_comm_weight(pred);
                }
            }
        }

        if (vectorSchedule->assignedSuperstep(vertex) < step) {
            if (stepProcessorPred[step][proc].find(vertex) != stepProcessorPred[step][proc].end()) {
                incMemory -= graph->vertex_comm_weight(vertex);
            }
        }

        if (vectorSchedule->assignedSuperstep(vertex) <= step) {
            return stepProcessorMemory[step][proc] + incMemory <= setSchedule->GetInstance().getArchitecture().memoryBound(proc);
        }

        if (stepProcessorMemory[step][proc] + incMemory > setSchedule->GetInstance().getArchitecture().memoryBound(proc)) {
            return false;
        }

        for (const auto &succ : graph->children(vertex)) {
            const auto &succStep = vectorSchedule->assignedSuperstep(succ);
            const auto &succProc = vectorSchedule->assignedProcessor(succ);

            if (succStep == vectorSchedule->assignedSuperstep(vertex)) {
                if (vectorSchedule->assignedProcessor(vertex) != succProc || (not is_source(vertex, *graph))) {
                    if (stepProcessorMemory[succStep][succProc] + graph->vertex_comm_weight(vertex)
                        > setSchedule->GetInstance().getArchitecture().memoryBound(succProc)) {
                        return false;
                    }

                } else {
                    if (is_source(vertex, *graph)) {
                        if (stepProcessorMemory[succStep][succProc] + graph->vertex_comm_weight(vertex)
                                - graph->vertex_mem_weight(vertex)
                            > setSchedule->GetInstance().getArchitecture().memoryBound(succProc)) {
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
