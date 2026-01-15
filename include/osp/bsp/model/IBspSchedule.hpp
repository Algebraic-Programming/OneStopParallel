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

#include "BspInstance.hpp"

namespace osp {

/**
 * @class IBspSchedule
 * @brief Pure interface class to organize the interaction with a BSP schedule.
 *
 * A BSP schedule assigns nodes to processors and supersteps, is based on an instance, and has a number of supersteps.
 * It provides unified access for different data implementations.
 *
 * - The class `BspSchedule` implements the assignments as vectors.
 * - The class `SetBspSchedule` implements containers that contain all nodes assigned to a pair of processor and superstep.
 *
 * @tparam GraphT The type of the computational DAG, which must satisfy `is_computational_dag_v`.
 * @see BspInstance
 * @see BspSchedule
 * @see SetBspSchedule
 */
template <typename GraphT>
class IBspSchedule {
    using VertexIdx = VertexIdxT<GraphT>;

    static_assert(isComputationalDagV<GraphT>, "IBspSchedule can only be used with computational DAGs.");

  public:
    virtual ~IBspSchedule() = default;

    /**
     * @brief Get the BSP instance associated with this schedule.
     *
     * @return The BSP instance.
     */
    [[nodiscard]] virtual const BspInstance<GraphT> &GetInstance() const = 0;

    /**
     * @brief Set the assigned superstep for a node.
     *
     * @param node The node index.
     * @param superstep The assigned superstep.
     */
    virtual void SetAssignedSuperstep(VertexIdx node, unsigned int superstep) = 0;

    /**
     * @brief Set the assigned processor for a node.
     *
     * @param node The node index.
     * @param processor The assigned processor.
     */
    virtual void SetAssignedProcessor(VertexIdx node, unsigned int processor) = 0;

    /**
     * @brief Get the assigned superstep of a node.
     *
     * @param node The node index.
     * @return The assigned superstep of the node.
     *         If the node is not assigned to a superstep, this.NumberOfSupersteps() is returned.
     */
    [[nodiscard]] virtual unsigned AssignedSuperstep(VertexIdx node) const = 0;

    /**
     * @brief Get the assigned processor of a node.
     *
     * @param node The node index.
     * @return The assigned processor of the node.
     *         If the node is not assigned to a processor, this.GetInstance().NumberOfProcessors() is returned.
     */
    [[nodiscard]] virtual unsigned AssignedProcessor(VertexIdx node) const = 0;

    /**
     * @brief Get the number of supersteps in the schedule.
     *
     * @return The number of supersteps in the schedule.
     */
    [[nodiscard]] virtual unsigned NumberOfSupersteps() const = 0;
};

}    // namespace  osp
