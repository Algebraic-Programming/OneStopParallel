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

#include "bsp/model/BspSchedule.hpp"
#include "concepts/computational_dag_concept.hpp"
#include "concepts/constructable_computational_dag_concept.hpp"
#include "concepts/graph_traits.hpp"
#include "graph_algorithms/directed_graph_edge_desc_util.hpp"


namespace osp {

/**
 * @class Coarser
 * @brief Abstract base class for coarsening ComputationalDags.
 *
 */
template<typename Graph_t_in, typename Graph_t_out>
class Coarser {

    static_assert(is_computational_dag_v<Graph_t_in>, "Graph_t_in must be a computational DAG");
    static_assert(is_constructable_cdag_v<Graph_t_out>, "Graph_t_out must be a constructable computational DAG");

    // probably too strict, need to be refined. 
    // maybe add concept for when Gtaph_t2 is constructable/coarseable from Graph_t_in
    static_assert(std::is_same_v<v_workw_t<Graph_t_in>, v_workw_t<Graph_t_out>>,
                  "Graph_t_in and Graph_t_out must have the same work weight type");
    static_assert(std::is_same_v<v_memw_t<Graph_t_in>, v_memw_t<Graph_t_out>>,
                  "Graph_t_in and Graph_t_out must have the same memory weight type");
    static_assert(std::is_same_v<v_commw_t<Graph_t_in>, v_commw_t<Graph_t_out>>,
                  "Graph_t_in and Graph_t_out must have the same communication weight type");

    

  public:

    enum class accumulationMethod { SUM, MAX };

    accumulationMethod v_work_weight_acc_method = accumulationMethod::SUM;
    accumulationMethod v_comm_weight_acc_method = accumulationMethod::SUM;
    accumulationMethod v_mem_weight_acc_method  = accumulationMethod::SUM;
    accumulationMethod e_comm_weight_acc_method = accumulationMethod::SUM;

    static bool check_valid_contraction_map(const std::vector<vertex_idx_t<Graph_t_out>> & vertex_contraction_map) {
        std::set<vertex_idx_t<Graph_t_out>> image(vertex_contraction_map.cbegin(), vertex_contraction_map.cend());
        const vertex_idx_t<Graph_t_out> image_size = static_cast<vertex_idx_t<Graph_t_out>>( image.size() );
        return std::all_of(image.cbegin(), image.cend(), [image_size](const vertex_idx_t<Graph_t_out>& vert) { return (vert >= static_cast<vertex_idx_t<Graph_t_out>>(0)) && (vert < image_size); } );
    }

    static std::vector<std::vector<vertex_idx_t<Graph_t_in>>> vertex_expansion_map(const std::vector<vertex_idx_t<Graph_t_out>> & vertex_contraction_map) {
        assert(check_valid_contraction_map(vertex_contraction_map));

        vertex_idx_t<Graph_t_out> max_vert = *std::max_element(vertex_contraction_map.cbegin(), vertex_contraction_map.cend());

        std::vector<std::vector<vertex_idx_t<Graph_t_in>>> expansion_map(max_vert + 1);

        for(std::size_t i = 0; i < vertex_contraction_map.size(); ++i) {
            expansion_map[vertex_contraction_map[i]].push_back(i);
        }

        return expansion_map;
    }

    virtual std::vector<vertex_idx_t<Graph_t_out>> generate_vertex_contraction_map(const Graph_t_in &dag_in) = 0;

    /**
     * @brief Coarsens the input computational DAG into a simplified version.
     *
     * @param dag_in The input computational DAG to be coarsened. It is expected to be a valid graph structure.
     * @param coarsened_dag The output computational DAG after coarsening. It will be populated by this method.
     * @param vertex_contraction_map Output mapping from dag_in to coarsened_dag. 
     * @return A status code indicating the success or failure of the coarsening operation.
     */
    virtual bool coarsenDag(const Graph_t_in &dag_in, Graph_t_out &coarsened_dag,
                            std::vector<vertex_idx_t<Graph_t_out>> &vertex_contraction_map) {

            vertex_contraction_map = generate_vertex_contraction_map(dag_in);
            assert(check_valid_contraction_map(vertex_contraction_map));
        
            // todo
            // if constexpr (has_quotient_graph_construction_method<Graph_t_in, Graph_t_out>) {
            //     return true;
            // }

            if constexpr (is_constructable_cdag_v<Graph_t_out>) {
                coarsened_dag = Graph_t_out();

                const vertex_idx_t<Graph_t_out> num_vert_quotient = (*std::max_element(vertex_contraction_map.cbegin(), vertex_contraction_map.cend())) + 1;

                for (vertex_idx_t<Graph_t_out> vert = 0; vert < num_vert_quotient; ++vert) {
                    coarsened_dag.add_vertex(0, 0, 0);
                }

                for (const vertex_idx_t<Graph_t_in> &vert : dag_in.vertices()) {
                    switch (v_work_weight_acc_method) {
                        case accumulationMethod::SUM : {
                            coarsened_dag.set_vertex_work_weight(vertex_contraction_map[vert],
                                                                coarsened_dag.vertex_work_weight(vertex_contraction_map[vert]) + dag_in.vertex_work_weight(vert));
                            break;
                        }
                        case accumulationMethod::MAX : {
                            coarsened_dag.set_vertex_work_weight(vertex_contraction_map[vert],
                                                                std::max(coarsened_dag.vertex_work_weight(vertex_contraction_map[vert]), dag_in.vertex_work_weight(vert)));
                            break;
                        }
                        default: {
                            coarsened_dag.set_vertex_work_weight(vertex_contraction_map[vert],
                                                                coarsened_dag.vertex_work_weight(vertex_contraction_map[vert]) + dag_in.vertex_work_weight(vert));
                            break;
                        }
                    }
                }

                for (const vertex_idx_t<Graph_t_in> &vert : dag_in.vertices()) {
                    switch (v_comm_weight_acc_method) {
                        case accumulationMethod::SUM : {
                            coarsened_dag.set_vertex_comm_weight(vertex_contraction_map[vert],
                                                                coarsened_dag.vertex_comm_weight(vertex_contraction_map[vert]) + dag_in.vertex_comm_weight(vert));
                            break;
                        }
                        case accumulationMethod::MAX : {
                            coarsened_dag.set_vertex_comm_weight(vertex_contraction_map[vert],
                                                                std::max(coarsened_dag.vertex_comm_weight(vertex_contraction_map[vert]), dag_in.vertex_comm_weight(vert)));
                            break;
                        }
                        default: {
                            coarsened_dag.set_vertex_comm_weight(vertex_contraction_map[vert],
                                                                coarsened_dag.vertex_comm_weight(vertex_contraction_map[vert]) + dag_in.vertex_comm_weight(vert));
                            break;
                        }
                    }
                }


                for (const vertex_idx_t<Graph_t_in> &vert : dag_in.vertices()) {
                    switch (v_mem_weight_acc_method) {
                        case accumulationMethod::SUM : {
                            coarsened_dag.set_vertex_mem_weight(vertex_contraction_map[vert],
                                                                coarsened_dag.vertex_mem_weight(vertex_contraction_map[vert]) + dag_in.vertex_mem_weight(vert));
                            break;
                        }
                        case accumulationMethod::MAX : {
                            coarsened_dag.set_vertex_mem_weight(vertex_contraction_map[vert],
                                                                std::max(coarsened_dag.vertex_mem_weight(vertex_contraction_map[vert]), dag_in.vertex_mem_weight(vert)));
                            break;
                        }
                        default: {
                            coarsened_dag.set_vertex_mem_weight(vertex_contraction_map[vert],
                                                                coarsened_dag.vertex_mem_weight(vertex_contraction_map[vert]) + dag_in.vertex_mem_weight(vert));
                            break;
                        }
                    }
                }

                if constexpr (has_typed_vertices_v<Graph_t_in> && is_constructable_cdag_typed_vertex_v<Graph_t_out>) {
                    static_assert(std::is_same_v<v_type_t<Graph_t_in>, v_type_t<Graph_t_out>>, "Vertex type types of in graph and out graph must be the same!" );

                    for (const vertex_idx_t<Graph_t_in> &vert : dag_in.vertices()) {
                        coarsened_dag.set_vertex_type(vertex_contraction_map[vert], dag_in.vertex_type(vert));
                    }
                    // assert(std::all_of(dag_in.vertices().begin(), dag_in.vertices().end(),
                    //         [&dag_in, &vertex_contraction_map, &coarsened_dag](const auto &vert){ return dag_in.vertex_type(vert) ==  coarsened_dag.vertex_type(vertex_contraction_map[vert]); })
                    //                 && "Contracted vertices must be of the same type");
                }

                for (const vertex_idx_t<Graph_t_in> &vert : dag_in.vertices()) {
                    for (const vertex_idx_t<Graph_t_in> &chld : dag_in.children(vert)) {
                        if (vertex_contraction_map[vert] == vertex_contraction_map[chld]) {
                            continue;
                        }

                        coarsened_dag.add_edge(vertex_contraction_map[vert], vertex_contraction_map[chld]);
                    }
                }

                if constexpr (has_edge_weights_v<Graph_t_in> && is_constructable_cdag_comm_edge_v<Graph_t_out>) {
                    static_assert(std::is_same_v<e_commw_t<Graph_t_in>, e_commw_t<Graph_t_out>>, "Edge weight type of in graph and out graph must be the same!" );

                    for (const vertex_idx_t<Graph_t_in> &vert : dag_in.vertices()) {
                        for (const vertex_idx_t<Graph_t_in> &chld : dag_in.children(vert)) {
                            if (vertex_contraction_map[vert] == vertex_contraction_map[chld]) {
                                continue;
                            }

                            edge_desc_t<Graph_t_in> ori_edge = edge_desc(vert, chld, dag_in).first;
                            edge_desc_t<Graph_t_out> contr_edge = edge_desc(vertex_contraction_map[vert], vertex_contraction_map[chld], coarsened_dag).first;

                            switch (e_comm_weight_acc_method) {
                                case accumulationMethod::SUM : {
                                    coarsened_dag.set_edge_comm_weight(contr_edge, coarsened_dag.edge_comm_weight(contr_edge) + dag_in.edge_comm_weight(ori_edge));
                                    break;
                                }
                                case accumulationMethod::MAX : {
                                    coarsened_dag.set_edge_comm_weight(contr_edge, std::max(coarsened_dag.edge_comm_weight(contr_edge), dag_in.edge_comm_weight(ori_edge)));
                                    break;
                                }
                                default: {
                                    coarsened_dag.set_edge_comm_weight(contr_edge, coarsened_dag.edge_comm_weight(contr_edge) + dag_in.edge_comm_weight(ori_edge));
                                    break;
                                }
                            }
                        }
                    }
                }

                return true;
            }
            
            return false;
        };

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



} // namespace osp