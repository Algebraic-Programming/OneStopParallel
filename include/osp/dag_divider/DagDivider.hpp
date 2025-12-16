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
#include <vector>

#include "osp/concepts/directed_graph_concept.hpp"
#include "osp/concepts/graph_traits.hpp"

namespace osp {

/**
 * @class DagDivider
 * @brief Divides the wavefronts of a computational DAG into consecutive groups or sections.
 *
 */
template <typename GraphT>
class IDagDivider {
    static_assert(isDirectedGraphV<GraphT>, "Graph must be directed");

  public:
    virtual ~IDagDivider() = default;

    /**
     * @brief Divides the dag and returns the vertex maps.
     *
     * The returned vertex maps is a three-dimensional vector
     * - The first dimension represents the sections that the dag is divided into.
     * - The second dimension represents the connected components within each section.
     * - The third dimension lists the vertices within each connected component.
     *
     * @return const std::vector<std::vector<std::vector<unsigned>>>&
     *         A constant reference to the vertex maps.
     */
    virtual std::vector<std::vector<std::vector<VertexIdxT<GraphT>>>> Divide(const GraphT &dag) = 0;
};

}    // namespace osp
