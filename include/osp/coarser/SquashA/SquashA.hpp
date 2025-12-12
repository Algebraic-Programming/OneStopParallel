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

#include "osp/auxiliary/datastructures/union_find.hpp"
#include "osp/coarser/Coarser.hpp"
#include "osp/graph_algorithms/directed_graph_coarsen_util.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"

namespace osp {

namespace squash_a_params {

enum class Mode { EDGE_WEIGHT, TRIANGLES };

struct Parameters {
    double geomDecayNumNodes_{17.0 / 16.0};
    double poissonPar_{0.0};
    unsigned noise_{0U};
    std::pair<unsigned, unsigned> edgeSortRatio_{3, 2};
    unsigned numRepWithoutNodeDecrease_{4};
    double temperatureMultiplier_{1.125};
    unsigned numberOfTemperatureIncreases_{14};
    Mode mode_{Mode::EDGE_WEIGHT};
    bool useStructuredPoset_{false};
    bool useTopPoset_{true};
};

}    // namespace squash_a_params

template <typename GraphTIn, typename GraphTOut>
class SquashA : public CoarserGenExpansionMap<GraphTIn, GraphTOut> {
  private:
    squash_a_params::Parameters params_;

    std::vector<int> GeneratePosetInMap(const GraphTIn &dagIn);

    template <typename T, typename CMP>
    std::vector<std::vector<vertex_idx_t<Graph_t_in>>> GenExpMapFromContractableEdges(
        const std::multiset<std::pair<edge_desc_t<Graph_t_in>, T>, CMP> &edgeWeights,
        const std::vector<int> &posetIntMapping,
        const GraphTIn &dagIn) {
        static_assert(std::is_arithmetic_v<T>, "T must be of arithmetic type!");

        auto lowerThirdIt = edge_weights.begin();
        std::advance(lower_third_it, edge_weights.size() / 3);
        T lowerThirdWt = std::max(lower_third_it->second, static_cast<T>(1));    // Could be 0

        Union_Find_Universe<vertex_idx_t<Graph_t_in>, vertex_idx_t<Graph_t_in>, VWorkwT<Graph_t_in>, v_memw_t<Graph_t_in>>
            connected_components;
        for (const auto &vert : dagIn.vertices()) {
            connected_components.add_object(vert, dag_in.VertexWorkWeight(vert), dag_in.VertexMemWeight(vert));
        }

        std::vector<bool> mergedNodes(dagIn.NumVertices(), false);

        vertex_idx_t<Graph_t_in> numNodesDecrease = 0;
        vertex_idx_t<Graph_t_in> numNodesAim
            = dagIn.NumVertices()
              - static_cast<vertex_idx_t<Graph_t_in>>(static_cast<double>(dagIn.NumVertices()) / params_.geomDecayNumNodes_);

        double temperature = 1;
        unsigned temperatureIncreaseIteration = 0;
        while (num_nodes_decrease < num_nodes_aim && temperatureIncreaseIteration <= params_.numberOfTemperatureIncreases_) {
            for (const auto &wt_edge : edge_weights) {
                const auto &edge_d = wt_edge.first;
                const vertex_idx_t<Graph_t_in> edge_source = Source(edge_d, dag_in);
                const vertex_idx_t<Graph_t_in> edge_target = Traget(edge_d, dag_in);

                // Previously merged
                if (merged_nodes[edge_source]) {
                    continue;
                }
                if (merged_nodes[edge_target]) {
                    continue;
                }

                // weight check
                if (connected_components.get_weight_of_component_by_name(edge_source)
                        + connected_components.get_weight_of_component_by_name(edge_target)
                    > static_cast<double>(lower_third_wt) * temperature) {
                    continue;
                }

                // no loops criteria check
                bool check_failed = false;
                // safety check - this should already be the case
                assert(abs(poset_int_mapping[edge_source] - poset_int_mapping[edge_target]) <= 1);
                // Checks over all affected edges
                // In edges first
                for (const auto &node : dag_in.Parents(edge_source)) {
                    if (node == edge_target) {
                        continue;
                    }
                    if (!merged_nodes[node]) {
                        continue;
                    }
                    if (poset_int_mapping[edge_source] >= poset_int_mapping[node] + 2) {
                        continue;
                    }
                    check_failed = true;
                    break;
                }
                if (check_failed) {
                    continue;
                }
                // Out edges first
                for (const auto &node : dag_in.Children(edge_source)) {
                    if (node == edge_target) {
                        continue;
                    }
                    if (!merged_nodes[node]) {
                        continue;
                    }
                    if (poset_int_mapping[node] >= poset_int_mapping[edge_source] + 2) {
                        continue;
                    }
                    check_failed = true;
                    break;
                }
                if (check_failed) {
                    continue;
                }
                // In edges second
                for (const auto &node : dag_in.Parents(edge_target)) {
                    if (node == edge_source) {
                        continue;
                    }
                    if (!merged_nodes[node]) {
                        continue;
                    }
                    if (poset_int_mapping[edge_target] >= poset_int_mapping[node] + 2) {
                        continue;
                    }
                    check_failed = true;
                    break;
                }
                if (check_failed) {
                    continue;
                }
                // Out edges second
                for (const auto &node : dag_in.Children(edge_target)) {
                    if (node == edge_source) {
                        continue;
                    }
                    if (!merged_nodes[node]) {
                        continue;
                    }
                    if (poset_int_mapping[node] >= poset_int_mapping[edge_target] + 2) {
                        continue;
                    }
                    check_failed = true;
                    break;
                }
                if (check_failed) {
                    continue;
                }

                // merging
                connected_components.join_by_name(edge_source, edge_target);
                merged_nodes[edge_source] = true;
                merged_nodes[edge_target] = true;
                num_nodes_decrease++;
            }

            temperature *= params_.temperatureMultiplier_;
            temperatureIncreaseIteration++;
        }

        // Getting components to contract and adding graph contraction
        std::vector<std::vector<vertex_idx_t<Graph_t_in>>> partitionVec;

        vertex_idx_t<Graph_t_in> minNodeDecrease
            = dagIn.NumVertices()
              - static_cast<vertex_idx_t<Graph_t_in>>(static_cast<double>(dagIn.NumVertices())
                                                      / std::pow(params_.geomDecayNumNodes_, 0.25));
        if (numNodesDecrease > 0 && num_nodes_decrease >= min_node_decrease) {
            partition_vec = connected_components.get_connected_components();

        } else {
            partitionVec.reserve(dagIn.NumVertices());
            for (const auto &vert : dagIn.vertices()) {
                std::vector<vertex_idx_t<Graph_t_in>> vect;
                vect.push_back(vert);
                partitionVec.emplace_back(vect);
            }
        }

        return partition_vec;
    }

  public:
    virtual std::vector<std::vector<vertex_idx_t<Graph_t_in>>> generate_vertex_expansion_map(const GraphTIn &dagIn) override;

    SquashA(squash_a_params::Parameters params = squash_a_params::Parameters()) : params_(params) {};

    SquashA(const SquashA &) = default;
    SquashA(SquashA &&) = default;
    SquashA &operator=(const SquashA &) = default;
    SquashA &operator=(SquashA &&) = default;
    virtual ~SquashA() override = default;

    inline squash_a_params::Parameters &GetParams() { return params_; }

    inline void SetParams(squash_a_params::Parameters params) { params_ = params; }

    std::string getCoarserName() const override { return "SquashA"; }
};

template <typename GraphTIn, typename GraphTOut>
std::vector<int> SquashA<GraphTIn, GraphTOut>::GeneratePosetInMap(const GraphTIn &dagIn) {
    std::vector<int> posetIntMapping;
    if (!params_.useStructuredPoset_) {
        posetIntMapping = get_strict_poset_integer_map<GraphTIn>(params_.noise_, params_.poissonPar_, dagIn);
    } else {
        if (params_.useTopPoset_) {
            posetIntMapping = get_top_node_distance<GraphTIn, int>(dagIn);
        } else {
            std::vector<int> botDist = get_bottom_node_distance<GraphTIn, int>(dagIn);
            posetIntMapping.resize(botDist.size());
            for (std::size_t i = 0; i < botDist.size(); i++) {
                posetIntMapping[i] = -botDist[i];
            }
        }
    }
    return posetIntMapping;
}

template <typename GraphTIn, typename GraphTOut>
std::vector<std::vector<vertex_idx_t<Graph_t_in>>> SquashA<GraphTIn, GraphTOut>::GenerateVertexExpansionMap(const GraphTIn &dagIn) {
    static_assert(IsDirectedGraphEdgeDescV<Graph_t_in>, "Graph_t_in must satisfy the directed_graph_edge_desc concept");
    static_assert(IsComputationalDagEdgeDescV<Graph_t_in>, "Graph_t_in must satisfy the is_computational_dag_edge_desc concept");
    // static_assert(has_hashable_edge_desc_v<Graph_t_in>, "Graph_t_in must have hashable edge descriptors");

    std::vector<int> posetIntMapping = GeneratePosetInMap(dagIn);

    if constexpr (HasEdgeWeightsV<Graph_t_in>) {
        if (params_.mode_ == squash_a_params::Mode::EDGE_WEIGHT) {
            auto edgeWCmp
                = [](const std::pair<edge_desc_t<Graph_t_in>, e_commw_t<Graph_t_in>> &lhs,
                     const std::pair<edge_desc_t<Graph_t_in>, e_commw_t<Graph_t_in>> &rhs) { return lhs.second < rhs.second; };
            std::multiset<std::pair<edge_desc_t<Graph_t_in>, e_commw_t<Graph_t_in>>, decltype(edge_w_cmp)> edge_weights(edge_w_cmp);
            {
                std::vector<edge_desc_t<Graph_t_in>> contractableEdges
                    = get_contractable_edges_from_poset_int_map<GraphTIn>(posetIntMapping, dagIn);
                for (const auto &edge : contractable_edges) {
                    if constexpr (HasEdgeWeightsV<Graph_t_in>) {
                        edge_weights.emplace(edge, dag_in.EdgeCommWeight(edge));
                    } else {
                        edge_weights.emplace(edge, dag_in.VertexCommWeight(Source(edge, dag_in)));
                    }
                }
            }

            return gen_exp_map_from_contractable_edges<e_commw_t<Graph_t_in>, decltype(edge_w_cmp)>(
                edge_weights, poset_int_mapping, dag_in);
        }
    }
    if (params_.mode_ == squash_a_params::Mode::TRIANGLES) {
        auto edgeWCmp = [](const std::pair<edge_desc_t<Graph_t_in>, std::size_t> &lhs,
                           const std::pair<edge_desc_t<Graph_t_in>, std::size_t> &rhs) { return lhs.second < rhs.second; };
        std::multiset<std::pair<edge_desc_t<Graph_t_in>, std::size_t>, decltype(edge_w_cmp)> edgeWeights(edgeWCmp);
        {
            std::vector<edge_desc_t<Graph_t_in>> contractableEdges
                = get_contractable_edges_from_poset_int_map<GraphTIn>(posetIntMapping, dagIn);
            for (const auto &edge : contractable_edges) {
                std::size_t num_common_triangles = num_common_parents(dag_in, Source(edge, dag_in), Traget(edge, dag_in));
                num_common_triangles += num_common_children(dag_in, Source(edge, dag_in), Traget(edge, dag_in));
                edge_weights.emplace(edge, num_common_triangles);
            }
        }

        return gen_exp_map_from_contractable_edges<std::size_t, decltype(edgeWCmp)>(edge_weights, posetIntMapping, dagIn);

    } else {
        throw std::runtime_error("Edge sorting mode not recognised.");
    }

    return {};
}

}    // end namespace osp
