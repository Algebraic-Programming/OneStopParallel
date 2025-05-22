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
#include "coarser/Coarser.hpp"

namespace osp {

/**
 * @class Coarser
 * @brief Abstract base class for coarsening ComputationalDags.
 *
 */
template<typename Graph_t_in, typename Graph_t_out>
class HeavyEdgeCoarser : public Coarser<Graph_t_in, Graph_t_out> {

    const float heavy_is_x_times_median;
    const float min_percent_components_retained;
    const float bound_component_weight_percent;

  public:
    HeavyEdgeCoarser(float heavy_is_x_times_median_ = 2.0, float min_percent_components_retained_ = 0.25,
                     float bound_component_weight_percent_ = 0.25)
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

    virtual std::vector<vertex_idx_t<Graph_t_out>> generate_vertex_contraction_map(const Graph_t_in &dag_in) {

        static_assert(is_computational_dag_edge_desc_v<Graph_t_in>,
                      "HeavyEdgePreProcess can only be used with computational DAGs with edge weights.");

        using VertexType = vertex_idx_t<Graph_t_in>;
        using EdgeType = edge_desc_t<Graph_t_in>;

        // Initialising the union find structure
        union_find_universe_t<Graph_t_in> uf_structure;
        for (const VertexType &vert : graph.vertices()) {
            uf_structure.add_object(vert, graph.vertex_work_weight(vert));
        }

        // Making edge comunications list
        std::vector<e_commw_t<Graph_t_in>> edge_communications;
        edge_communications.reserve(graph.num_edges());
        for (const auto &edge : graph.edges()) {
            edge_communications.emplace_back(graph.edge_comm_weight(edge));
        }

        // Computing the median and setting it to at least one
        e_commw_t<Graph_t_in> median_edge_weight = 1;
        if (not edge_communications.empty()) {

            auto median_it = edge_communications.begin();
            std::advance(median_it, edge_communications.size() / 2);
            std::nth_element(edge_communications.begin(), median_it, edge_communications.end());
            median_edge_weight =
                std::max(edge_communications[edge_communications.size() / 2], static_cast<e_commw_t<Graph_t_in>>(1));
        }

        // Making edge list
        e_commw_t<Graph_t_in> minimal_edge_weight =
            static_cast<e_commw_t<Graph_t_in>>(heavy_is_x_times_median * median_edge_weight);
        std::vector<EdgeType> edge_list;
        edge_list.reserve(graph.num_edges());
        for (const auto &edge : graph.edges()) {
            if (graph.edge_comm_weight(edge) > minimal_edge_weight) {
                edge_list.emplace_back(edge);
            }
        }

        // Sorting edge list
        std::sort(edge_list.begin(), edge_list.end(), [graph](const EdgeType &left, const EdgeType &right) {
            return graph.edge_comm_weight(left) > graph.edge_comm_weight(right);
        });

        // Computing max component size
        v_workw_t<Graph_t_in> max_component_size = 0;
        for (const VertexType &vert : graph.vertices()) {
            max_component_size += graph.vertex_work_weight(vert);
        }

        max_component_size = static_cast<v_workw_t<Graph_t_in>>(max_component_size * bound_component_weight_percent);

        // Joining heavy edges
        for (const EdgeType &edge : edge_list) {
            if (static_cast<double>(uf_structure.get_number_of_connected_components()) - 1.0 <
                min_percent_components_retained * static_cast<double>(graph.num_vertices()))
                break;

            v_workw_t<Graph_t_in> weight_comp_a = uf_structure.get_weight_of_component_by_name(source(edge, graph));
            v_workw_t<Graph_t_in> weight_comp_b = uf_structure.get_weight_of_component_by_name(target(edge, graph));
            if (weight_comp_a + weight_comp_b > max_component_size)
                continue;

            uf_structure.join_by_name(edge.m_source, edge.m_target);
        }

        return uf_structure.get_connected_components();
    }

    
    virtual bool coarsenDag(const Graph_t_in &dag_in, Graph_t_out &coarsened_dag,
                            std::vector<vertex_idx_t<Graph_t_out>> &vertex_contraction_map) override {

        assert(dag_out.numberOfVertices() == 0);
        assert(map.empty());

        // Making edge comunications list
        std::vector<int> edge_communications(dag_in.numberOfEdges());

        for (const auto &edge : dag_in.edges()) {
            edge_communications.emplace_back(dag_in[edge].communicationWeight);
        }

        // Computing the median and setting it to at least one
        int median_edge_weight;
        if (edge_communications.size() == 0) {
            median_edge_weight = 0;
        } else {
            auto median_it = edge_communications.begin() + edge_communications.size() / 2;
            std::nth_element(edge_communications.begin(), median_it, edge_communications.end());
            median_edge_weight = edge_communications[edge_communications.size() / 2];
        }
        median_edge_weight = std::max(median_edge_weight, 1);

        // Making edge list
        float minimal_edge_weight = heavy_is_x_times_median * median_edge_weight;

        // std::vector<EdgeType> edge_list(dag_in.numberOfEdges());

        // for (const auto &edge : dag_in.edges()) {
        //     if (dag_in[edge].communicationWeight > minimal_edge_weight) {
        //         edge_list.emplace_back(edge);
        //     }
        // }

        // // Sorting edge list
        // std::sort(edge_list.begin(), edge_list.end(), [dag_in](const EdgeType &left, const EdgeType &right) {
        //     return dag_in[left].communicationWeight > dag_in[right].communicationWeight;
        // });

        // Computing max component size
        unsigned max_component_size = 0;
        for (const VertexType &vert : dag_in.vertices()) {
            max_component_size += dag_in.nodeWorkWeight(vert);
        }
        max_component_size *= bound_component_weight_percent;

        GraphType_labeled graph_labeled(dag_in.numberOfVertices());

        unsigned label = 0;
        for (const VertexType &vert : dag_in.vertices()) {
            graph_labeled[vert] = dag_in[vert];
            graph_labeled[vert].label = label;
            graph_labeled[vert].merged_labels.push_back(label);
            label++;
        }

        for (const EdgeType &edge : dag_in.edges()) {
            const auto [edge_labeled, valid] = boost::add_edge(edge.m_source, edge.m_target, graph_labeled);
            if (not valid) {
                throw std::invalid_argument("Adding Edge was not sucessful");
            }
            graph_labeled[edge_labeled] = dag_in[edge];
        }

        while (boost::num_vertices(graph_labeled) > min_percent_components_retained * dag_in.numberOfVertices()) {

            EdgeType edge;
            int max_weight = 0;
            for (const auto e : boost::extensions::make_source_iterator_range(boost::edges(graph_labeled))) {

                if (graph_labeled[e].communicationWeight > max_weight &&
                    graph_labeled[e.m_source].workWeight + graph_labeled[e.m_target].workWeight < max_component_size) {
                    max_weight = graph_labeled[e].communicationWeight;
                    edge = e;
                }
            }

            if (max_weight < minimal_edge_weight) {
                break;
            }

            for (const auto &out_edge :
                 boost::extensions::make_source_iterator_range(boost::out_edges(edge.m_target, graph_labeled))) {
                // for (const auto &out_edge : dag_out.out_edges(edge.m_target)) {

                const auto pair = boost::edge(edge.m_source, out_edge.m_target, graph_labeled);
                if (pair.second) {
                    graph_labeled[pair.first].communicationWeight += graph_labeled[out_edge].communicationWeight;
                } else {
                    const auto [new_edge, valid] = boost::add_edge(edge.m_source, out_edge.m_target, graph_labeled);
                    assert(valid);
                    graph_labeled[new_edge].communicationWeight = graph_labeled[out_edge].communicationWeight;
                }
            }

            // add in_edges of edge.m_target to edge.m_source
            for (const auto &in_edge :
                 boost::extensions::make_source_iterator_range(boost::in_edges(edge.m_target, graph_labeled))) {

                // skip edge
                if (in_edge == edge) {
                    continue;
                }

                const auto pair = boost::edge(in_edge.m_source, edge.m_source, graph_labeled);

                if (pair.second) { // edge already exists
                    graph_labeled[pair.first].communicationWeight += graph_labeled[in_edge].communicationWeight;
                } else {

                    if (has_path(edge.m_source, in_edge.m_source, graph_labeled)) { // merge is closing cycle

                        const auto other_pair = boost::edge(edge.m_source, in_edge.m_source, graph_labeled);

                        if (other_pair.second) {
                            graph_labeled[other_pair.first].communicationWeight +=
                                graph_labeled[in_edge].communicationWeight;
                        } else {

                            const auto [new_edge, valid] =
                                boost::add_edge(edge.m_source, in_edge.m_source, graph_labeled);
                            assert(valid);
                            graph_labeled[new_edge].communicationWeight = graph_labeled[in_edge].communicationWeight;
                        }

                        // add zero weight edges to repair precedence constraints
                        for (const auto &out_edge : boost::extensions::make_source_iterator_range(
                                 boost::out_edges(edge.m_target, graph_labeled))) {

                            const auto another_pair = boost::edge(in_edge.m_source, out_edge.m_target, graph_labeled);
                            if (not another_pair.second) {
                                const auto [new_edge, valid] =
                                    boost::add_edge(in_edge.m_source, out_edge.m_target, graph_labeled);
                                assert(valid);
                                graph_labeled[new_edge].communicationWeight = 0;
                                // dag_out.addEdge(in_edge.m_source, out_edge.m_target, 0);
                            }
                        }

                    } else {

                        const auto [new_edge, valid] = boost::add_edge(in_edge.m_source, edge.m_source, graph_labeled);
                        assert(valid);
                        graph_labeled[new_edge].communicationWeight = graph_labeled[in_edge].communicationWeight;
                    }
                }
            }

            graph_labeled[edge.m_source].workWeight += graph_labeled[edge.m_target].workWeight;
            graph_labeled[edge.m_source].memoryWeight += graph_labeled[edge.m_target].memoryWeight;
            graph_labeled[edge.m_source].communicationWeight += graph_labeled[edge.m_target].communicationWeight;

            std::move(graph_labeled[edge.m_target].merged_labels.begin(),
                      graph_labeled[edge.m_target].merged_labels.end(),
                      std::back_inserter(graph_labeled[edge.m_source].merged_labels));

            while (boost::in_degree(edge.m_target, graph_labeled) > 0) {
                const auto in_edge = *boost::in_edges(edge.m_target, graph_labeled).first;
                boost::remove_edge(in_edge, graph_labeled);
            }

            while (boost::out_degree(edge.m_target, graph_labeled) > 0) {
                const auto out_edge = *boost::out_edges(edge.m_target, graph_labeled).first;
                boost::remove_edge(out_edge, graph_labeled);
            }

            boost::remove_vertex(edge.m_target, graph_labeled);
        }

        dag_out = ComputationalDag(boost::num_vertices(graph_labeled));
        map = std::vector<std::vector<VertexType>>(dag_out.numberOfVertices());

        unsigned idx = 0;
        for (const VertexType &vert : dag_out.vertices()) {
            dag_out[vert].workWeight = graph_labeled[vert].workWeight;
            dag_out[vert].communicationWeight = graph_labeled[vert].communicationWeight;
            dag_out[vert].memoryWeight = graph_labeled[vert].memoryWeight;
            map[idx] = graph_labeled[vert].merged_labels;
            idx++;
        }

        for (const auto &edge : boost::extensions::make_source_iterator_range(boost::edges(graph_labeled))) {

            dag_out.addEdge(edge.m_source, edge.m_target, graph_labeled[edge].communicationWeight);
        }

        return RETURN_STATUS::SUCCESS;
    }
}

}; // namespace osp
