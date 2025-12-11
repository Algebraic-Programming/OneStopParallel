
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

// #define KL_DEBUG
// #define KL_DEBUG_1
// #define KL_DEBUG_COST_CHECK

#include "comm_cost_modules/kl_bsp_comm_cost.hpp"
#include "comm_cost_modules/kl_hyper_total_comm_cost.hpp"
#include "comm_cost_modules/kl_total_comm_cost.hpp"
#include "kl_improver.hpp"
#include "osp/bsp/scheduler/LocalSearch/LocalSearchMemoryConstraintModules.hpp"

namespace osp {

using double_cost_t = double;

template <typename Graph_t,
          typename MemoryConstraint_t = no_local_search_memory_constraint,
          unsigned window_size = 1,
          bool use_node_communication_costs_arg = true>
using kl_total_comm_improver
    = kl_improver<Graph_t,
                  kl_total_comm_cost_function<Graph_t, double_cost_t, MemoryConstraint_t, window_size, use_node_communication_costs_arg>,
                  MemoryConstraint_t,
                  window_size,
                  double_cost_t>;

template <typename Graph_t,
          typename MemoryConstraint_t = ls_local_memory_constraint<Graph_t>,
          unsigned window_size = 1,
          bool use_node_communication_costs_arg = true>
using kl_total_comm_improver_local_mem_constr
    = kl_improver<Graph_t,
                  kl_total_comm_cost_function<Graph_t, double_cost_t, MemoryConstraint_t, window_size, use_node_communication_costs_arg>,
                  MemoryConstraint_t,
                  window_size,
                  double_cost_t>;

template <typename Graph_t, typename MemoryConstraint_t = no_local_search_memory_constraint, unsigned window_size = 1>
using kl_total_lambda_comm_improver
    = kl_improver<Graph_t,
                  kl_hyper_total_comm_cost_function<Graph_t, double_cost_t, MemoryConstraint_t, window_size>,
                  MemoryConstraint_t,
                  window_size,
                  double_cost_t>;

template <typename Graph_t, typename MemoryConstraint_t = ls_local_memory_constraint<Graph_t>, unsigned window_size = 1>
using kl_total_lambda_comm_improver_local_mem_constr
    = kl_improver<Graph_t,
                  kl_hyper_total_comm_cost_function<Graph_t, double_cost_t, MemoryConstraint_t, window_size>,
                  MemoryConstraint_t,
                  window_size,
                  double_cost_t>;

template <typename Graph_t, typename MemoryConstraint_t = no_local_search_memory_constraint, unsigned window_size = 1>
using kl_bsp_comm_improver = kl_improver<Graph_t,
                                         kl_bsp_comm_cost_function<Graph_t, double_cost_t, MemoryConstraint_t, window_size>,
                                         MemoryConstraint_t,
                                         window_size,
                                         double_cost_t>;

template <typename Graph_t, typename MemoryConstraint_t = ls_local_memory_constraint<Graph_t>, unsigned window_size = 1>
using kl_bsp_comm_improver_local_mem_constr
    = kl_improver<Graph_t,
                  kl_bsp_comm_cost_function<Graph_t, double_cost_t, MemoryConstraint_t, window_size>,
                  MemoryConstraint_t,
                  window_size,
                  double_cost_t>;

}    // namespace osp
