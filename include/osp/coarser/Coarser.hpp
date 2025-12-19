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

#include <algorithm>
#include <set>
#include <vector>

#include "coarser_util.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/concepts/constructable_computational_dag_concept.hpp"
#include "osp/concepts/graph_traits.hpp"

namespace osp {

/**
 * @class Coarser
 * @brief Abstract base class for coarsening ComputationalDags.
 *
 */
template <typename GraphTIn, typename GraphTOut>
class Coarser {
    static_assert(isComputationalDagV<GraphTIn>, "GraphTIn must be a computational DAG");
    static_assert(isConstructableCdagV<GraphTOut> || isDirectConstructableCdagV<GraphTOut>,
                  "GraphTOut must be a (direct) constructable computational DAG");

    // probably too strict, need to be refined.
    // maybe add concept for when Gtaph_t2 is constructable/coarseable from GraphTIn
    static_assert(std::is_same_v<VWorkwT<GraphTIn>, VWorkwT<GraphTOut>>,
                  "GraphTIn and GraphTOut must have the same work weight type");
    static_assert(std::is_same_v<VMemwT<GraphTIn>, VMemwT<GraphTOut>>,
                  "GraphTIn and GraphTOut must have the same memory weight type");
    static_assert(std::is_same_v<VCommwT<GraphTIn>, VCommwT<GraphTOut>>,
                  "GraphTIn and GraphTOut must have the same communication weight type");

  public:
    /**
     * @brief Coarsens the input computational DAG into a simplified version.
     *
     * @param dag_in The input computational DAG to be coarsened. It is expected to be a valid graph structure.
     * @param coarsened_dag The output computational DAG after coarsening. It will be populated by this method.
     * @param vertex_contraction_map Output mapping from dag_in to coarsened_dag.
     * @return A status code indicating the success or failure of the coarsening operation.
     */
    virtual bool CoarsenDag(const GraphTIn &dagIn, GraphTOut &coarsenedDag, std::vector<VertexIdxT<GraphTOut>> &vertexContractionMap)
        = 0;

    /**
     * @brief Get the name of the coarsening algorithm.
     * @return A human-readable name of the coarsening algorithm, typically used for identification or logging purposes.
     */
    virtual std::string GetCoarserName() const = 0;

    /**
     * @brief Destructor for the Coarser class.
     */
    virtual ~Coarser() = default;
};

/**
 * @class Coarser
 * @brief Abstract base class for coarsening ComputationalDags.
 *
 */
template <typename GraphTIn, typename GraphTOut>
class CoarserGenContractionMap : public Coarser<GraphTIn, GraphTOut> {
  public:
    virtual std::vector<VertexIdxT<GraphTOut>> GenerateVertexContractionMap(const GraphTIn &dagIn) = 0;

    virtual bool CoarsenDag(const GraphTIn &dagIn,
                            GraphTOut &coarsenedDag,
                            std::vector<VertexIdxT<GraphTOut>> &vertexContractionMap) override {
        vertexContractionMap = dagIn.NumVertices() == 0 ? std::vector<VertexIdxT<GraphTOut>>()
                                                        : GenerateVertexContractionMap(dagIn);

        return coarser_util::ConstructCoarseDag(dagIn, coarsenedDag, vertexContractionMap);
    }

    /**
     * @brief Destructor for the CoarserGenContractionMap class.
     */
    virtual ~CoarserGenContractionMap() = default;
};

/**
 * @class Coarser
 * @brief Abstract base class for coarsening ComputationalDags.
 *
 */
template <typename GraphTIn, typename GraphTOut>
class CoarserGenExpansionMap : public Coarser<GraphTIn, GraphTOut> {
  public:
    virtual std::vector<std::vector<VertexIdxT<GraphTIn>>> GenerateVertexExpansionMap(const GraphTIn &dagIn) = 0;

    virtual bool CoarsenDag(const GraphTIn &dagIn,
                            GraphTOut &coarsenedDag,
                            std::vector<VertexIdxT<GraphTOut>> &vertexContractionMap) override {
        if (dagIn.NumVertices() == 0) {
            vertexContractionMap = std::vector<VertexIdxT<GraphTOut>>();
            return true;
        }

        std::vector<std::vector<VertexIdxT<GraphTIn>>> vertexExpansionMap = GenerateVertexExpansionMap(dagIn);
        assert(coarser_util::CheckValidExpansionMap<GraphTIn>(vertexExpansionMap));

        coarser_util::ReorderExpansionMap<GraphTIn>(dagIn, vertexExpansionMap);

        vertexContractionMap = coarser_util::InvertVertexExpansionMap<GraphTIn, GraphTOut>(vertexExpansionMap);

        return coarser_util::ConstructCoarseDag(dagIn, coarsenedDag, vertexContractionMap);
    }

    /**
     * @brief Get the name of the coarsening algorithm.
     * @return A human-readable name of the coarsening algorithm, typically used for identification or logging purposes.
     */
    virtual std::string GetCoarserName() const override = 0;

    /**
     * @brief Destructor for the CoarserGenExpansionMap class.
     */
    virtual ~CoarserGenExpansionMap() = default;
};

}    // namespace osp
