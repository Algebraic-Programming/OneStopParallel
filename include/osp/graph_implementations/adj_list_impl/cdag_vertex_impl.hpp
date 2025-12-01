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

namespace osp {

/**
 * @brief Implementation of a computational DAG vertex.
 *
 * This struct holds the properties of a vertex in a computational DAG, including its ID,
 * weights (work, communication, memory), and type.
 *
 * @tparam vertex_idx_t Type for vertex indices.
 * @tparam workw_t Type for work weights.
 * @tparam commw_t Type for communication weights.
 * @tparam memw_t Type for memory weights.
 * @tparam vertex_type_t Type for vertex types.
 */
template<typename vertex_idx_t, typename workw_t, typename commw_t, typename memw_t, typename vertex_type_t>
struct cdag_vertex_impl {

    using vertex_idx_type = vertex_idx_t;
    using work_weight_type = workw_t;
    using comm_weight_type = commw_t;
    using mem_weight_type = memw_t;
    using cdag_vertex_type_type = vertex_type_t;

    cdag_vertex_impl() = default;

    cdag_vertex_impl(const cdag_vertex_impl &other) = default;
    cdag_vertex_impl(cdag_vertex_impl &&other) noexcept = default;
    cdag_vertex_impl &operator=(const cdag_vertex_impl &other) = default;
    cdag_vertex_impl &operator=(cdag_vertex_impl &&other) noexcept = default;

    /**
     * @brief Constructs a vertex with specified properties.
     *
     * @param vertex_idx_ The unique identifier for the vertex.
     * @param work_w The computational work weight.
     * @param comm_w The communication weight.
     * @param mem_w The memory weight.
     * @param vertex_t The type of the vertex.
     */
    cdag_vertex_impl(vertex_idx_t vertex_idx_, workw_t work_w, commw_t comm_w, memw_t mem_w,
                     vertex_type_t vertex_t)
        : id(vertex_idx_), work_weight(work_w), comm_weight(comm_w), mem_weight(mem_w),
          vertex_type(vertex_t) {}

    vertex_idx_t id = 0;

    workw_t work_weight = 0;
    commw_t comm_weight = 0;
    memw_t mem_weight = 0;

    vertex_type_t vertex_type = 0;
};

/**
 * @brief A vertex implementation with integer weights. Indexed by size_t. Node types are unsigned.
 *
 * This struct implements a vertex with integer weights for work, communication, and memory.
 */
using cdag_vertex_impl_int = cdag_vertex_impl<size_t, int, int, int, unsigned>;

/**
 * @brief A vertex implementation with unsigned weights. Indexed by size_t. Node types are unsigned.
 *
 * This struct implements a vertex with unsigned weights for work, communication, and memory.
 */
using cdag_vertex_impl_unsigned = cdag_vertex_impl<size_t, unsigned, unsigned, unsigned, unsigned>;

} // namespace osp