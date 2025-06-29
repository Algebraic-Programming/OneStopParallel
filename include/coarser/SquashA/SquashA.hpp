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

#include <set>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "auxiliary/datastructures/union_find.hpp"
#include "coarser/Coarser.hpp"
#include "graph_algorithms/directed_graph_coarsen_util.hpp"
#include "graph_algorithms/directed_graph_path_util.hpp"

namespace osp {

namespace SquashAParams {
enum class Mode { EDGE_WEIGHT, TRIANGLES };
struct Parameters {
    double geom_decay_num_nodes{17.0 / 16.0};
    double poisson_par{0.0};
    unsigned noise{0U};
    std::pair<unsigned, unsigned> edge_sort_ratio{3, 2};
    unsigned num_rep_without_node_decrease{4};
    double temperature_multiplier{1.125};
    unsigned number_of_temperature_increases{14};
    Mode mode{Mode::EDGE_WEIGHT};
    bool use_structured_poset{false};
    bool use_top_poset{true};
};
} // end namespace SquashAParams

template<typename Graph_t_in, typename Graph_t_out>
class SquashA : public CoarserGenExpansionMap<Graph_t_in, Graph_t_out> {
  private:
    SquashAParams::Parameters params;

    std::vector<int> generate_poset_in_map(const Graph_t_in &dag_in);

    template<typename T, typename CMP>
    std::vector<std::vector<vertex_idx_t<Graph_t_in>>>
    gen_exp_map_from_contractable_edges(const std::multiset<std::pair<edge_desc_t<Graph_t_in>, T>, CMP> &edge_weights,
                                        const std::vector<int> &poset_int_mapping, const Graph_t_in &dag_in) {
        static_assert(std::is_arithmetic_v<T>, "T must be of arithmetic type!");

        auto lower_third_it = edge_weights.begin();
        std::advance(lower_third_it, edge_weights.size() / 3);
        T lower_third_wt = std::max(lower_third_it->second, static_cast<T>(1)); // Could be 0

        Union_Find_Universe<vertex_idx_t<Graph_t_in>, vertex_idx_t<Graph_t_in>, v_workw_t<Graph_t_in>,
                            v_memw_t<Graph_t_in>>
            connected_components;
        for (const auto &vert : dag_in.vertices()) {
            connected_components.add_object(vert, dag_in.vertex_work_weight(vert), dag_in.vertex_mem_weight(vert));
        }

        std::vector<bool> merged_nodes(dag_in.num_vertices(), false);

        vertex_idx_t<Graph_t_in> num_nodes_decrease = 0;
        vertex_idx_t<Graph_t_in> num_nodes_aim =
            dag_in.num_vertices() - static_cast<vertex_idx_t<Graph_t_in>>(static_cast<double>(dag_in.num_vertices()) /
                                                                          params.geom_decay_num_nodes);

        double temperature = 1;
        unsigned temperature_increase_iteration = 0;
        while (num_nodes_decrease < num_nodes_aim &&
               temperature_increase_iteration <= params.number_of_temperature_increases) {
            for (const auto &wt_edge : edge_weights) {
                const auto &edge_d = wt_edge.first;
                const vertex_idx_t<Graph_t_in> edge_source = source(edge_d, dag_in);
                const vertex_idx_t<Graph_t_in> edge_target = target(edge_d, dag_in);

                // Previously merged
                if (merged_nodes[edge_source])
                    continue;
                if (merged_nodes[edge_target])
                    continue;

                // weight check
                if (connected_components.get_weight_of_component_by_name(edge_source) +
                        connected_components.get_weight_of_component_by_name(edge_target) >
                    static_cast<double>(lower_third_wt) * temperature)
                    continue;

                // no loops criteria check
                bool check_failed = false;
                // safety check - this should already be the case
                assert(abs(poset_int_mapping[edge_source] - poset_int_mapping[edge_target]) <= 1);
                // Checks over all affected edges
                // In edges first
                for (const auto &node : dag_in.parents(edge_source)) {
                    if (node == edge_target)
                        continue;
                    if (!merged_nodes[node])
                        continue;
                    if (poset_int_mapping[edge_source] >= poset_int_mapping[node] + 2)
                        continue;
                    check_failed = true;
                    break;
                }
                if (check_failed)
                    continue;
                // Out edges first
                for (const auto &node : dag_in.children(edge_source)) {
                    if (node == edge_target)
                        continue;
                    if (!merged_nodes[node])
                        continue;
                    if (poset_int_mapping[node] >= poset_int_mapping[edge_source] + 2)
                        continue;
                    check_failed = true;
                    break;
                }
                if (check_failed)
                    continue;
                // In edges second
                for (const auto &node : dag_in.parents(edge_target)) {
                    if (node == edge_source)
                        continue;
                    if (!merged_nodes[node])
                        continue;
                    if (poset_int_mapping[edge_target] >= poset_int_mapping[node] + 2)
                        continue;
                    check_failed = true;
                    break;
                }
                if (check_failed)
                    continue;
                // Out edges second
                for (const auto &node : dag_in.children(edge_target)) {
                    if (node == edge_source)
                        continue;
                    if (!merged_nodes[node])
                        continue;
                    if (poset_int_mapping[node] >= poset_int_mapping[edge_target] + 2)
                        continue;
                    check_failed = true;
                    break;
                }
                if (check_failed)
                    continue;

                // merging
                connected_components.join_by_name(edge_source, edge_target);
                merged_nodes[edge_source] = true;
                merged_nodes[edge_target] = true;
                num_nodes_decrease++;
            }

            temperature *= params.temperature_multiplier;
            temperature_increase_iteration++;
        }

        // Getting components to contract and adding graph contraction
        std::vector<std::vector<vertex_idx_t<Graph_t_in>>> partition_vec;

        vertex_idx_t<Graph_t_in> min_node_decrease =
            dag_in.num_vertices() - static_cast<vertex_idx_t<Graph_t_in>>(static_cast<double>(dag_in.num_vertices()) /
                                                                          std::pow(params.geom_decay_num_nodes, 0.25));
        if (num_nodes_decrease > 0 && num_nodes_decrease >= min_node_decrease) {
            partition_vec = connected_components.get_connected_components();

        } else {
            partition_vec.reserve(dag_in.num_vertices());
            for (const auto &vert : dag_in.vertices()) {
                std::vector<vertex_idx_t<Graph_t_in>> vect;
                vect.push_back(vert);
                partition_vec.emplace_back(vect);
            }
        }

        return partition_vec;
    }

  public:
    virtual std::vector<std::vector<vertex_idx_t<Graph_t_in>>>
    generate_vertex_expansion_map(const Graph_t_in &dag_in) override;

    SquashA(SquashAParams::Parameters params_ = SquashAParams::Parameters()) : params(params_) {};

    SquashA(const SquashA &) = default;
    SquashA(SquashA &&) = default;
    SquashA &operator=(const SquashA &) = default;
    SquashA &operator=(SquashA &&) = default;
    virtual ~SquashA() override = default;

    inline SquashAParams::Parameters &getParams() { return params; }
    inline void setParams(SquashAParams::Parameters params_) { params = params_; }

    std::string getCoarserName() const override { return "SquashA"; }
};

template<typename Graph_t_in, typename Graph_t_out>
std::vector<int> SquashA<Graph_t_in, Graph_t_out>::generate_poset_in_map(const Graph_t_in &dag_in) {
    std::vector<int> poset_int_mapping;
    if (!params.use_structured_poset) {
        poset_int_mapping = get_strict_poset_integer_map<Graph_t_in>(params.noise, params.poisson_par, dag_in);
    } else {
        if (params.use_top_poset) {
            poset_int_mapping = get_top_node_distance<Graph_t_in, int>(dag_in);
        } else {
            std::vector<int> bot_dist = get_bottom_node_distance<Graph_t_in, int>(dag_in);
            poset_int_mapping.resize(bot_dist.size());
            for (std::size_t i = 0; i < bot_dist.size(); i++) {
                poset_int_mapping[i] = -bot_dist[i];
            }
        }
    }
    return poset_int_mapping;
}

template<typename Graph_t_in, typename Graph_t_out>
std::vector<std::vector<vertex_idx_t<Graph_t_in>>>
SquashA<Graph_t_in, Graph_t_out>::generate_vertex_expansion_map(const Graph_t_in &dag_in) {
    static_assert(is_directed_graph_edge_desc_v<Graph_t_in>,
                  "Graph_t_in must satisfy the directed_graph_edge_desc concept");
    static_assert(is_computational_dag_edge_desc_v<Graph_t_in>,
                  "Graph_t_in must satisfy the is_computational_dag_edge_desc concept");
    // static_assert(has_hashable_edge_desc_v<Graph_t_in>, "Graph_t_in must have hashable edge descriptors");

    std::vector<int> poset_int_mapping = generate_poset_in_map(dag_in);

    if (params.mode == SquashAParams::Mode::EDGE_WEIGHT) {
        auto edge_w_cmp = [](const std::pair<edge_desc_t<Graph_t_in>, e_commw_t<Graph_t_in>> &lhs,
                             const std::pair<edge_desc_t<Graph_t_in>, e_commw_t<Graph_t_in>> &rhs) {
            return lhs.second < rhs.second;
        };
        std::multiset<std::pair<edge_desc_t<Graph_t_in>, e_commw_t<Graph_t_in>>, decltype(edge_w_cmp)> edge_weights(
            edge_w_cmp);
        {
            std::vector<edge_desc_t<Graph_t_in>> contractable_edges =
                get_contractable_edges_from_poset_int_map<Graph_t_in>(poset_int_mapping, dag_in);
            for (const auto &edge : contractable_edges) {

                if constexpr (has_edge_weights_v<Graph_t_in>) {
                    edge_weights.emplace(edge, dag_in.edge_comm_weight(edge));
                } else {
                    edge_weights.emplace(edge, dag_in.vertex_comm_weight(source(edge, dag_in)));
                }
            }
        }

        return gen_exp_map_from_contractable_edges<e_commw_t<Graph_t_in>, decltype(edge_w_cmp)>(
            edge_weights, poset_int_mapping, dag_in);

    } else if (params.mode == SquashAParams::Mode::TRIANGLES) {
        auto edge_w_cmp = [](const std::pair<edge_desc_t<Graph_t_in>, std::size_t> &lhs,
                             const std::pair<edge_desc_t<Graph_t_in>, std::size_t> &rhs) {
            return lhs.second < rhs.second;
        };
        std::multiset<std::pair<edge_desc_t<Graph_t_in>, std::size_t>, decltype(edge_w_cmp)> edge_weights(edge_w_cmp);
        {
            std::vector<edge_desc_t<Graph_t_in>> contractable_edges =
                get_contractable_edges_from_poset_int_map<Graph_t_in>(poset_int_mapping, dag_in);
            for (const auto &edge : contractable_edges) {
                std::size_t num_common_triangles =
                    num_common_parents(dag_in, source(edge, dag_in), target(edge, dag_in));
                num_common_triangles += num_common_children(dag_in, source(edge, dag_in), target(edge, dag_in));
                edge_weights.emplace(edge, num_common_triangles);
            }
        }

        return gen_exp_map_from_contractable_edges<std::size_t, decltype(edge_w_cmp)>(edge_weights, poset_int_mapping,
                                                                                      dag_in);

    } else {
        throw std::runtime_error("Edge sorting mode not recognised.");
    }

    return {};
}

} // end namespace osp