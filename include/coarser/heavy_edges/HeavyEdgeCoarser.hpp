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

#include "auxiliary/datastructures/union_find.hpp"
#include "coarser/Coarser_gen_exp_map.hpp"
#include "graph_implementations/boost_graphs/boost_graph.hpp"

namespace osp {

/**
 * @class Coarser
 * @brief Abstract base class for coarsening ComputationalDags.
 *
 */
template<typename Graph_t_in, typename Graph_t_out>
class HeavyEdgeCoarser : public CoarserGenExpansionMap<Graph_t_in, Graph_t_out> {

    static_assert(is_computational_dag_edge_desc_v<Graph_t_in>,
                  "HeavyEdgePreProcess can only be used with computational DAGs with edge weights.");

    using VertexType = vertex_idx_t<Graph_t_in>;
    using EdgeType = edge_desc_t<Graph_t_in>;

    using vertex_type_t_or_default =
        std::conditional_t<is_computational_dag_typed_vertices_v<Graph_t_in>, v_type_t<Graph_t_in>, unsigned>;

    struct mutable_vertex_labeled {

        mutable_vertex_labeled() : work_weight(0), comm_weight(0), mem_weight(0), vertex_type(0) {}

        v_workw_t<Graph_t_in> work_weight;
        v_commw_t<Graph_t_in> comm_weight;
        v_memw_t<Graph_t_in> mem_weight;

        vertex_type_t_or_default vertex_type;

        std::vector<VertexType> merged_labels;
    };

    struct mutable_edge {
        mutable_edge() : comm_weight(0) {}
        e_commw_t<Graph_t_in> comm_weight;
    };

    using mutable_graph_t =
        boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, mutable_vertex_labeled, mutable_edge>;

    using mutable_vertex = typename boost::graph_traits<mutable_graph_t>::vertex_descriptor;

    bool mutable_has_path(mutable_vertex src, mutable_vertex dest, mutable_graph_t &g) {

        std::unordered_set<mutable_vertex> visited;
        visited.emplace(src);

        std::queue<mutable_vertex> next;
        next.push(src);

        while (!next.empty()) {
            auto v = next.front();
            next.pop();

            for (const auto &child : boost::extensions::make_source_iterator_range(boost::adjacent_vertices(v, g))) {

                if (child == dest) {
                    return true;
                }

                if (visited.find(child) == visited.end()) {
                    visited.emplace(child);
                    next.push(child);
                }
            }
        }

        return false;
    }

    const double heavy_is_x_times_median;
    const double min_percent_components_retained;
    const double bound_component_weight_percent;

  public:
    HeavyEdgeCoarser(double heavy_is_x_times_median_ = 2.0, double min_percent_components_retained_ = 0.25,
                     double bound_component_weight_percent_ = 0.25)
        : heavy_is_x_times_median(heavy_is_x_times_median_),
          min_percent_components_retained(min_percent_components_retained_),
          bound_component_weight_percent(bound_component_weight_percent_) {}

    /**
     * @brief Destructor for the Coarser class.
     */
    virtual ~HeavyEdgeCoarser() = default;

    /**
     * @brief Get the name of the coarsening algorithm.
     * @return The name of the coarsening algorithm.
     */
    virtual std::string getCoarserName() const override { return "HeavyEdgeCoarser"; }

    virtual bool coarsenDag(const Graph_t_in &dag_in, Graph_t_out &coarsened_dag,
                            std::vector<vertex_idx_t<Graph_t_out>> &vertex_contraction_map) override {

        // Making edge comunications list
        std::vector<e_commw_t<Graph_t_in>> edge_communications;
        edge_communications.reserve(graph.num_edges());
        for (const auto &edge : graph.edges()) {
            edge_communications.emplace_back(graph.edge_comm_weight(edge));
        }

        // Computing the median and setting it to at least one
        e_commw_t<Graph_t> median_edge_weight = 1;
        if (not edge_communications.empty()) {

            auto median_it = edge_communications.begin();
            std::advance(median_it, edge_communications.size() / 2);
            std::nth_element(edge_communications.begin(), median_it, edge_communications.end());
            median_edge_weight =
                std::max(edge_communications[edge_communications.size() / 2], static_cast<e_commw_t<Graph_t>>(1));
        }

        e_commw_t<Graph_t> minimal_edge_weight =
            static_cast<e_commw_t<Graph_t>>(heavy_is_x_times_median * median_edge_weight);

        // Computing max component size
        std::unordered_map<VertexType, std::vector<VertexType>> map;
        v_workw_t<Graph_t> max_component_size = 0;
        for (const VertexType &vert : graph.vertices()) {
            max_component_size += graph.vertex_work_weight(vert);
            map[vert] = std::vector<VertexType>{vert};
        }

        max_component_size = static_cast<v_workw_t<Graph_t>>(max_component_size * bound_component_weight_percent);

        mutable_graph_t mutable_graph(dag_in.num_vertices());

        for (const VertexType &vert : dag_in.vertices()) {
            mutable_graph[vert].work_weight = dag_in.vertex_work_weight(vert);
            mutable_graph[vert].comm_weight = dag_in.vertex_comm_weight(vert);
            mutable_graph[vert].mem_weight = dag_in.vertex_mem_weight(vert);

            if constexpr (is_computational_dag_typed_vertices_v<Graph_t_in>) {
                mutable_graph[vert].vertex_type = dag_in.vertex_type(vert);
            } else {
                mutable_graph[vert].vertex_type = 0;
            }

            mutable_graph[vert].merged_labels.push_back(vert);
        }

        for (const EdgeType &edge : dag_in.edges()) {
            const auto [edge_labeled, valid] =
                boost::add_edge(source(edge, dag_in), target(edge, dag_in), mutable_graph);
            mutable_graph[edge_labeled].comm_weight = dag_in.edge_comm_weight(edge);
        }

        while (boost::num_vertices(mutable_graph) > min_percent_components_retained * dag_in.num_vertices()) {

            mutable_edge contract_edge;
            e_commw_t<Graph_t> max_weight = 0;
            for (const auto e : boost::extensions::make_source_iterator_range(boost::edges(mutable_graph))) {

                if (mutable_graph[e].comm_weight > max_weight &&
                    mutable_graph[e.m_source].vertex_type == mutable_graph[e.m_target].vertex_type &&
                    mutable_graph[e.m_source].work_weight + mutable_graph[e.m_target].work_weight <
                        max_component_size) {
                    max_weight = mutable_graph[e].comm_weight;
                    contract_edge = e;
                }
            }

            if (max_weight < minimal_edge_weight) {
                break;
            }

            for (const auto &out_edge : boost::extensions::make_source_iterator_range(
                     boost::out_edges(contract_edge.m_target, mutable_graph))) {

                const auto pair = boost::edge(contract_edge.m_source, out_edge.m_target, mutable_graph);
                if (pair.second) {
                    mutable_graph[pair.first].communicationWeight += mutable_graph[out_edge].communicationWeight;
                } else {
                    const auto [new_edge, valid] =
                        boost::add_edge(contract_edge.m_source, out_edge.m_target, mutable_graph);
                    assert(valid);
                    mutable_graph[new_edge].communicationWeight = mutable_graph[out_edge].communicationWeight;
                }
            }

            // add in_edges of contract_edge.m_target to contract_edge.m_source
            for (const auto &in_edge : boost::extensions::make_source_iterator_range(
                     boost::in_edges(contract_edge.m_target, mutable_graph))) {

                // skip contract_edge
                if (in_edge == contract_edge) {
                    continue;
                }

                const auto pair = boost::edge(in_edge.m_source, contract_edge.m_source, mutable_graph);

                if (pair.second) { // edge already exists
                    mutable_graph[pair.first].communicationWeight += mutable_graph[in_edge].communicationWeight;
                } else {

                    if (mutable_has_path(contract_edge.m_source, in_edge.m_source, mutable_graph)) { // merge is closing cycle

                        const auto other_pair = boost::edge(contract_edge.m_source, in_edge.m_source, mutable_graph);

                        if (other_pair.second) {
                            mutable_graph[other_pair.first].communicationWeight +=
                                mutable_graph[in_edge].communicationWeight;
                        } else {

                            const auto [new_edge, valid] =
                                boost::add_edge(contract_edge.m_source, in_edge.m_source, mutable_graph);
                            assert(valid);
                            mutable_graph[new_edge].communicationWeight = mutable_graph[in_edge].communicationWeight;
                        }

                        // add zero weight edges to repair precedence constraints
                        for (const auto &out_edge : boost::extensions::make_source_iterator_range(
                                 boost::out_edges(contract_edge.m_target, mutable_graph))) {

                            const auto another_pair = boost::edge(in_edge.m_source, out_edge.m_target, mutable_graph);
                            if (not another_pair.second) {
                                const auto [new_edge, valid] =
                                    boost::add_edge(in_edge.m_source, out_edge.m_target, mutable_graph);
                                assert(valid);
                                mutable_graph[new_edge].communicationWeight = 0;
                                // dag_out.addEdge(in_edge.m_source, out_edge.m_target, 0);
                            }
                        }

                    } else {

                        const auto [new_edge, valid] =
                            boost::add_edge(in_edge.m_source, contract_edge.m_source, mutable_graph);
                        assert(valid);
                        mutable_graph[new_edge].communicationWeight = mutable_graph[in_edge].communicationWeight;
                    }
                }
            }

            mutable_graph[contract_edge.m_source].workWeight += mutable_graph[contract_edge.m_target].workWeight;
            mutable_graph[contract_edge.m_source].memoryWeight += mutable_graph[contract_edge.m_target].memoryWeight;
            mutable_graph[contract_edge.m_source].communicationWeight +=
                mutable_graph[contract_edge.m_target].communicationWeight;

            std::move(mutable_graph[contract_edge.m_target].merged_labels.begin(),
                      mutable_graph[contract_edge.m_target].merged_labels.end(),
                      std::back_inserter(mutable_graph[contract_edge.m_source].merged_labels));

            while (boost::in_degree(contract_edge.m_target, mutable_graph) > 0) {
                const auto in_edge = *boost::in_edges(contract_edge.m_target, mutable_graph).first;
                boost::remove_edge(in_edge, mutable_graph);
            }

            while (boost::out_degree(contract_edge.m_target, mutable_graph) > 0) {
                const auto out_edge = *boost::out_edges(contract_edge.m_target, mutable_graph).first;
                boost::remove_edge(out_edge, mutable_graph);
            }

            boost::remove_vertex(contract_edge.m_target, mutable_graph);
        }

        coarsened_dag = Graph_t_out(mutable_graph.num_vertices());

        vertex_contraction_map = std::vector<vertex_idx_t<Graph_t_out>>(dag_in.num_vertices());

        for (const VertexType &vert : coarsened_dag.vertices()) {

            coarsened_dag.set_vertex_work_weight(vert, mutable_graph[vert].work_weight);
            coarsened_dag.set_vertex_comm_weight(vert, mutable_graph[vert].comm_weight);
            coarsened_dag.set_vertex_mem_weight(vert, mutable_graph[vert].mem_weight);

            if constexpr (is_computational_dag_typed_vertices_v<Graph_t_out>) {
                coarsened_dag.set_vertex_type(vert, mutable_graph[vert].vertex_type);
            }

            for (const auto &orig_vert : mutable_graph[vert].merged_labels) {
                vertex_contraction_map[orig_vert] = vert;
            }
        }

        for (const auto &edge : boost::extensions::make_source_iterator_range(boost::edges(mutable_graph))) {

            dag_out.add_edge(edge.m_source, edge.m_target, mutable_graph[edge].comm_weight);
        }

        return true;
    }
};

} // namespace osp
