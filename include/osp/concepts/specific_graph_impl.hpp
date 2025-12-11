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

/**
 * @file specific_graph_impl.hpp
 * @brief Type traits for specific graph implementations used in OneStopParallel.
 *
 * This file contains trait checks for specific graph implementations, such as
 * Compact Sparse Graphs (CSR-like structures), which may require specialized
 * handling or offer optimizations in certain algorithms.
 */

namespace osp {

/**
 * @brief Trait to check if a graph type is a `Compact_Sparse_Graph`.
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct is_Compact_Sparse_Graph : std::false_type {};

template <typename T>
inline constexpr bool is_Compact_Sparse_Graph_v = is_Compact_Sparse_Graph<T>::value;

/**
 * @brief Trait to check if a graph type is a `Compact_Sparse_Graph` that supports reordering.
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct is_Compact_Sparse_Graph_reorder : std::false_type {};

template <typename T>
inline constexpr bool is_Compact_Sparse_Graph_reorder_v = is_Compact_Sparse_Graph_reorder<T>::value;

}    // namespace osp
