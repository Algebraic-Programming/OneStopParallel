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

#include <cstddef>    // for std::size_t

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
template <typename VertexIdxT, typename WorkwT, typename CommwT, typename MemwT, typename VertexTypeT>
struct CdagVertexImpl {
    using VertexIdxType = VertexIdxT;
    using WorkWeightType = WorkwT;
    using CommWeightType = CommwT;
    using MemWeightType = MemwT;
    using CdagVertexTypeType = VertexTypeT;

    CdagVertexImpl() = default;

    CdagVertexImpl(const CdagVertexImpl &other) = default;
    CdagVertexImpl(CdagVertexImpl &&other) noexcept = default;
    CdagVertexImpl &operator=(const CdagVertexImpl &other) = default;
    CdagVertexImpl &operator=(CdagVertexImpl &&other) noexcept = default;

    /**
     * @brief Constructs a vertex with specified properties.
     *
     * @param vertex_idx_ The unique identifier for the vertex.
     * @param work_w The computational work weight.
     * @param comm_w The communication weight.
     * @param mem_w The memory weight.
     * @param vertex_t The type of the vertex.
     */
    CdagVertexImpl(VertexIdxT vertexIdx, WorkwT workW, CommwT commW, MemwT memW, VertexTypeT vertexT)
        : id_(vertexIdx), workWeight_(workW), commWeight_(commW), memWeight_(memW), vertexType_(vertexT) {}

    VertexIdxT id_ = 0;

    WorkwT workWeight_ = 0;
    CommwT commWeight_ = 0;
    MemwT memWeight_ = 0;

    VertexTypeT vertexType_ = 0;
};

/**
 * @brief A vertex implementation with integer weights. Indexed by std::size_t. Node types are unsigned.
 *
 * This struct implements a vertex with integer weights for work, communication, and memory.
 */
using CdagVertexImplInt = CdagVertexImpl<std::size_t, int, int, int, unsigned>;

/**
 * @brief A vertex implementation with unsigned weights. Indexed by std::size_t. Node types are unsigned.
 *
 * This struct implements a vertex with unsigned weights for work, communication, and memory.
 */
using CdagVertexImplUnsigned = CdagVertexImpl<std::size_t, unsigned, unsigned, unsigned, unsigned>;

}    // namespace osp
