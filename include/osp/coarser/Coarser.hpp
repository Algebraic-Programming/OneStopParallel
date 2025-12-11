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
template <typename Graph_t_in, typename Graph_t_out>
class Coarser {
    static_assert(is_computational_dag_v<Graph_t_in>, "Graph_t_in must be a computational DAG");
    static_assert(is_constructable_cdag_v<Graph_t_out> || is_direct_constructable_cdag_v<Graph_t_out>,
                  "Graph_t_out must be a (direct) constructable computational DAG");

    // probably too strict, need to be refined.
    // maybe add concept for when Gtaph_t2 is constructable/coarseable from Graph_t_in
    static_assert(std::is_same_v<v_workw_t<Graph_t_in>, v_workw_t<Graph_t_out>>,
                  "Graph_t_in and Graph_t_out must have the same work weight type");
    static_assert(std::is_same_v<v_memw_t<Graph_t_in>, v_memw_t<Graph_t_out>>,
                  "Graph_t_in and Graph_t_out must have the same memory weight type");
    static_assert(std::is_same_v<v_commw_t<Graph_t_in>, v_commw_t<Graph_t_out>>,
                  "Graph_t_in and Graph_t_out must have the same communication weight type");

  public:
    /**
     * @brief Coarsens the input computational DAG into a simplified version.
     *
     * @param dag_in The input computational DAG to be coarsened. It is expected to be a valid graph structure.
     * @param coarsened_dag The output computational DAG after coarsening. It will be populated by this method.
     * @param vertex_contraction_map Output mapping from dag_in to coarsened_dag.
     * @return A status code indicating the success or failure of the coarsening operation.
     */
    virtual bool coarsenDag(const Graph_t_in &dag_in,
                            Graph_t_out &coarsened_dag,
                            std::vector<vertex_idx_t<Graph_t_out>> &vertex_contraction_map)
        = 0;

    /**
     * @brief Get the name of the coarsening algorithm.
     * @return A human-readable name of the coarsening algorithm, typically used for identification or logging purposes.
     */
    virtual std::string getCoarserName() const = 0;

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
template <typename Graph_t_in, typename Graph_t_out>
class CoarserGenContractionMap : public Coarser<Graph_t_in, Graph_t_out> {
  public:
    virtual std::vector<vertex_idx_t<Graph_t_out>> generate_vertex_contraction_map(const Graph_t_in &dag_in) = 0;

    virtual bool coarsenDag(const Graph_t_in &dag_in,
                            Graph_t_out &coarsened_dag,
                            std::vector<vertex_idx_t<Graph_t_out>> &vertex_contraction_map) override {
        vertex_contraction_map = dag_in.num_vertices() == 0 ? std::vector<vertex_idx_t<Graph_t_out>>()
                                                            : generate_vertex_contraction_map(dag_in);

        return coarser_util::construct_coarse_dag(dag_in, coarsened_dag, vertex_contraction_map);
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
template <typename Graph_t_in, typename Graph_t_out>
class CoarserGenExpansionMap : public Coarser<Graph_t_in, Graph_t_out> {
  public:
    virtual std::vector<std::vector<vertex_idx_t<Graph_t_in>>> generate_vertex_expansion_map(const Graph_t_in &dag_in) = 0;

    virtual bool coarsenDag(const Graph_t_in &dag_in,
                            Graph_t_out &coarsened_dag,
                            std::vector<vertex_idx_t<Graph_t_out>> &vertex_contraction_map) override {
        if (dag_in.num_vertices() == 0) {
            vertex_contraction_map = std::vector<vertex_idx_t<Graph_t_out>>();
            return true;
        }

        std::vector<std::vector<vertex_idx_t<Graph_t_in>>> vertex_expansion_map = generate_vertex_expansion_map(dag_in);
        assert(coarser_util::check_valid_expansion_map<Graph_t_in>(vertex_expansion_map));

        coarser_util::reorder_expansion_map<Graph_t_in>(dag_in, vertex_expansion_map);

        vertex_contraction_map = coarser_util::invert_vertex_expansion_map<Graph_t_in, Graph_t_out>(vertex_expansion_map);

        return coarser_util::construct_coarse_dag(dag_in, coarsened_dag, vertex_contraction_map);
    }

    /**
     * @brief Get the name of the coarsening algorithm.
     * @return A human-readable name of the coarsening algorithm, typically used for identification or logging purposes.
     */
    virtual std::string getCoarserName() const override = 0;

    /**
     * @brief Destructor for the CoarserGenExpansionMap class.
     */
    virtual ~CoarserGenExpansionMap() = default;
};

}    // namespace osp
