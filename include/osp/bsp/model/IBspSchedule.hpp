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

/// @class IBspSchedule
/// @brief Interface for a BSP (Bulk Synchronous Parallel) schedule.
template <typename Graph_t>
class IBspSchedule {
    using vertex_idx = vertex_idx_t<Graph_t>;

  public:
    /// @brief Destructor.
    virtual ~IBspSchedule() = default;

    virtual const BspInstance<Graph_t> &getInstance() const = 0;

    /// @brief Set the assigned superstep for a node.
    /// @param node The node index.
    /// @param superstep The assigned superstep.
    virtual void setAssignedSuperstep(vertex_idx node, unsigned int superstep) = 0;

    /// @brief Set the assigned processor for a node.
    /// @param node The node index.
    /// @param processor The assigned processor.
    virtual void setAssignedProcessor(vertex_idx node, unsigned int processor) = 0;

    /// @brief Get the assigned superstep of a node.
    /// @param node The node index.
    /// @return The assigned superstep of the node.
    ///         If the node is not assigned to a superstep, this.numberOfSupersteps() is returned.
    virtual unsigned assignedSuperstep(vertex_idx node) const = 0;

    /// @brief Get the assigned processor of a node.
    /// @param node The node index.
    /// @return The assigned processor of the node.
    ///         If the node is not assigned to a processor, this.getInstance().numberOfProcessors() is returned.
    virtual unsigned assignedProcessor(vertex_idx node) const = 0;

    /// @brief Get the number of supersteps in the schedule.
    /// @return The number of supersteps in the schedule.
    virtual unsigned numberOfSupersteps() const = 0;
};

}    // namespace  osp
