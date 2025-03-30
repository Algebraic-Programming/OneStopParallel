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

#include <iterator>
#include <type_traits>

#include "directed_graph_concept.hpp"

namespace osp {

// constructable graph concept
template<typename T, typename = void>
struct is_constructable_graph : std::false_type {};

template<typename T>
struct is_constructable_graph<
    T, std::void_t<decltype(std::declval<T>().add_vertex(std::declval<int>(), std::declval<int>(), std::declval<int>(),
                                                         std::declval<unsigned>())),
                   decltype(std::declval<T>().add_edge(std::declval<vertex_idx>(), std::declval<vertex_idx>(),
                                                       std::declval<int>()))>> : std::true_type {};

template<typename T>
inline constexpr bool is_constructable_graph_v = is_constructable_graph<T>::value;

} // namespace osp