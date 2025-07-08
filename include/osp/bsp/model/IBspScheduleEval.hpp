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
template<typename Graph_t>
class IBspScheduleEval {

    using vertex_idx = vertex_idx_t<Graph_t>;

  public:
    /// @brief Destructor.
    virtual ~IBspScheduleEval() = default;

    virtual v_workw_t<Graph_t> computeCosts() const = 0;
    virtual v_workw_t<Graph_t> computeWorkCosts() const = 0;
    virtual unsigned numberOfSupersteps() const = 0;
    virtual const BspInstance<Graph_t> &getInstance() const = 0;

};

} // namespace  osp