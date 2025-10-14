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
#include <stdexcept>
#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/graph_algorithms/directed_graph_edge_desc_util.hpp"

namespace osp {

template<typename index_type = size_t, typename workw_type = int, typename memw_type = int, typename commw_type = int>
class Hypergraph {

  public:

    Hypergraph() = default;

    Hypergraph(index_type num_vertices_, index_type num_hyperedges_)
        : Num_vertices(num_vertices_), Num_hyperedges(num_hyperedges_), vertex_work_weights(num_vertices_, 1),
        vertex_memory_weights(num_vertices_, 1), hyperedge_weights(num_hyperedges_, 1),
        incident_hyperedges_to_vertex(num_vertices_), vertices_in_hyperedge(num_hyperedges_){}

    Hypergraph(const Hypergraph<index_type, workw_type, memw_type, commw_type> &other) = default;
    Hypergraph &operator=(const Hypergraph<index_type, workw_type, memw_type, commw_type> &other) = default;

    virtual ~Hypergraph() = default;

    inline index_type num_vertices() const { return Num_vertices; }
    inline index_type num_hyperedges() const { return Num_hyperedges; }
    inline index_type num_pins() const { return Num_pins; }
    inline workw_type get_vertex_work_weight(index_type node) const { return vertex_work_weights[node]; }
    inline memw_type get_vertex_memory_weight(index_type node) const { return vertex_memory_weights[node]; }
    inline commw_type get_hyperedge_weight(index_type hyperedge) const { return hyperedge_weights[hyperedge]; }

    void add_pin(index_type vertex_idx, index_type hyperedge_idx);
    void add_vertex(workw_type work_weight = 1, memw_type memory_weight = 1);
    void add_empty_hyperedge(commw_type weight = 1);
    void add_hyperedge(const std::vector<index_type>& pins, commw_type weight = 1);
    void set_vertex_work_weight(index_type vertex_idx, workw_type weight);
    void set_vertex_memory_weight(index_type vertex_idx, memw_type weight);
    void set_hyperedge_weight(index_type hyperedge_idx, commw_type weight);

    workw_type compute_total_vertex_work_weight() const;
    memw_type compute_total_vertex_memory_weight() const;

    void clear();
    void reset(index_type num_vertices_, index_type num_hyperedges_);

    inline const std::vector<index_type> &get_incident_hyperedges(index_type vertex) const { return incident_hyperedges_to_vertex[vertex]; }
    inline const std::vector<index_type> &get_vertices_in_hyperedge(index_type hyperedge) const { return vertices_in_hyperedge[hyperedge]; }

    template<typename Graph_t>
    void convert_from_cdag_as_dag(const Graph_t& dag);

    template<typename Graph_t>
    void convert_from_cdag_as_hyperdag(const Graph_t& dag);

  private:
    index_type Num_vertices = 0, Num_hyperedges = 0, Num_pins = 0;

    std::vector<workw_type> vertex_work_weights;
    std::vector<memw_type> vertex_memory_weights;
    std::vector<commw_type> hyperedge_weights;

    std::vector<std::vector<index_type>> incident_hyperedges_to_vertex;
    std::vector<std::vector<index_type>> vertices_in_hyperedge;
};

template<typename index_type, typename workw_type, typename memw_type, typename commw_type>
void Hypergraph<index_type, workw_type, memw_type, commw_type>::add_pin(index_type vertex_idx, index_type hyperedge_idx)
{
    if(vertex_idx >= Num_vertices)
    {
        throw std::invalid_argument("Invalid Argument while adding pin: vertex index out of range.");
    }
    else if(hyperedge_idx >= Num_hyperedges)
    {
        throw std::invalid_argument("Invalid Argument while adding pin: hyperedge index out of range.");
    }
    else{    
        incident_hyperedges_to_vertex[vertex_idx].push_back(hyperedge_idx);
        vertices_in_hyperedge[hyperedge_idx].push_back(vertex_idx);
        ++Num_pins;
    }
}

template<typename index_type, typename workw_type, typename memw_type, typename commw_type>
void Hypergraph<index_type, workw_type, memw_type, commw_type>::add_vertex(workw_type work_weight, memw_type memory_weight)
{
    vertex_work_weights.push_back(work_weight);
    vertex_memory_weights.push_back(memory_weight);
    incident_hyperedges_to_vertex.emplace_back();
    ++Num_vertices;
}

template<typename index_type, typename workw_type, typename memw_type, typename commw_type>
void Hypergraph<index_type, workw_type, memw_type, commw_type>::add_empty_hyperedge(commw_type weight)
{
    vertices_in_hyperedge.emplace_back();
    hyperedge_weights.push_back(weight);
    ++Num_hyperedges;
}

template<typename index_type, typename workw_type, typename memw_type, typename commw_type>
void Hypergraph<index_type, workw_type, memw_type, commw_type>::add_hyperedge(const std::vector<index_type>& pins, commw_type weight)
{
    vertices_in_hyperedge.emplace_back(pins);
    hyperedge_weights.push_back(weight);
    for(index_type vertex : pins)
        incident_hyperedges_to_vertex[vertex].push_back(Num_hyperedges);
    ++Num_hyperedges;
    Num_pins += static_cast<index_type>(pins.size());
}

template<typename index_type, typename workw_type, typename memw_type, typename commw_type>
void Hypergraph<index_type, workw_type, memw_type, commw_type>::set_vertex_work_weight(index_type vertex_idx, workw_type weight)
{
    if(vertex_idx >= Num_vertices)
        throw std::invalid_argument("Invalid Argument while setting vertex weight: vertex index out of range.");
    else   
        vertex_work_weights[vertex_idx] = weight;
}

template<typename index_type, typename workw_type, typename memw_type, typename commw_type>
void Hypergraph<index_type, workw_type, memw_type, commw_type>::set_vertex_memory_weight(index_type vertex_idx, memw_type weight)
{
    if(vertex_idx >= Num_vertices)
        throw std::invalid_argument("Invalid Argument while setting vertex weight: vertex index out of range.");
    else   
        vertex_memory_weights[vertex_idx] = weight;
}

template<typename index_type, typename workw_type, typename memw_type, typename commw_type>
void Hypergraph<index_type, workw_type, memw_type, commw_type>::set_hyperedge_weight(index_type hyperedge_idx, commw_type weight)
{
    if(hyperedge_idx >= Num_hyperedges)
        throw std::invalid_argument("Invalid Argument while setting hyperedge weight: hyepredge index out of range.");
    else   
        hyperedge_weights[hyperedge_idx] = weight;
}

template<typename index_type, typename workw_type, typename memw_type, typename commw_type>
workw_type Hypergraph<index_type, workw_type, memw_type, commw_type>::compute_total_vertex_work_weight() const
{
    workw_type total = 0;
    for(index_type node = 0; node < Num_vertices; ++node)
        total += vertex_work_weights[node];
    return total;
}

template<typename index_type, typename workw_type, typename memw_type, typename commw_type>
memw_type Hypergraph<index_type, workw_type, memw_type, commw_type>::compute_total_vertex_memory_weight() const
{
    memw_type total = 0;
    for(index_type node = 0; node < Num_vertices; ++node)
        total += vertex_memory_weights[node];
    return total;
}

template<typename index_type, typename workw_type, typename memw_type, typename commw_type>
void Hypergraph<index_type, workw_type, memw_type, commw_type>::clear()
{
    Num_vertices = 0;
    Num_hyperedges = 0;
    Num_pins = 0;

    vertex_work_weights.clear();
    vertex_memory_weights.clear();
    hyperedge_weights.clear();
    incident_hyperedges_to_vertex.clear();
    vertices_in_hyperedge.clear();
}

template<typename index_type, typename workw_type, typename memw_type, typename commw_type>
void Hypergraph<index_type, workw_type, memw_type, commw_type>::reset(index_type num_vertices_, index_type num_hyperedges_)
{
    clear();

    Num_vertices = num_vertices_;
    Num_hyperedges = num_hyperedges_;

    vertex_work_weights.resize(num_vertices_, 1);
    vertex_memory_weights.resize(num_vertices_, 1);
    hyperedge_weights.resize(num_hyperedges_, 1);
    incident_hyperedges_to_vertex.resize(num_vertices_);
    vertices_in_hyperedge.resize(num_hyperedges_);
}

template<typename index_type, typename workw_type, typename memw_type, typename commw_type>
template<typename Graph_t>
void Hypergraph<index_type, workw_type, memw_type, commw_type>::convert_from_cdag_as_dag(const Graph_t& dag)
{
    static_assert(std::is_same_v<vertex_idx_t<Graph_t>, index_type>, "Index type mismatch, cannot convert DAG to hypergraph.");
    static_assert(std::is_same_v<v_workw_t<Graph_t>, workw_type>, "Work weight type mismatch, cannot convert DAG to hypergraph.");
    static_assert(std::is_same_v<v_memw_t<Graph_t>, memw_type>, "Memory weight type mismatch, cannot convert DAG to hypergraph.");
    static_assert(!has_edge_weights_v<Graph_t> || std::is_same_v<e_commw_t<Graph_t>, commw_type>, "Communication weight type mismatch, cannot convert DAG to hypergraph.");

    reset(dag.num_vertices(), 0);
    for(const auto &node : dag.vertices())
    {
        set_vertex_work_weight(node, dag.vertex_work_weight(node));
        set_vertex_memory_weight(node, dag.vertex_mem_weight(node));
        for (const auto &child : dag.children(node))
            if constexpr(has_edge_weights_v<Graph_t>)
                add_hyperedge({node, child}, dag.edge_comm_weight(edge_desc(node, child, dag).first));
            else 
                add_hyperedge({node, child});
    }
}

template<typename index_type, typename workw_type, typename memw_type, typename commw_type>
template<typename Graph_t>
void Hypergraph<index_type, workw_type, memw_type, commw_type>::convert_from_cdag_as_hyperdag(const Graph_t& dag)
{
    static_assert(std::is_same_v<vertex_idx_t<Graph_t>, index_type>, "Index type mismatch, cannot convert DAG to hypergraph.");
    static_assert(std::is_same_v<v_workw_t<Graph_t>, workw_type>, "Work weight type mismatch, cannot convert DAG to hypergraph.");
    static_assert(std::is_same_v<v_memw_t<Graph_t>, memw_type>, "Memory weight type mismatch, cannot convert DAG to hypergraph.");
    static_assert(std::is_same_v<v_commw_t<Graph_t>, commw_type>, "Communication weight type mismatch, cannot convert DAG to hypergraph.");

    reset(dag.num_vertices(), 0);
    for(const auto &node : dag.vertices())
    {
        set_vertex_work_weight(node, dag.vertex_work_weight(node));
        set_vertex_memory_weight(node, dag.vertex_mem_weight(node));
        if(dag.out_degree(node) == 0)
            continue;
        std::vector<index_type> new_hyperedge({node});
        for (const auto &child : dag.children(node))
            new_hyperedge.push_back(child);
        add_hyperedge(new_hyperedge, dag.vertex_comm_weight(node));
    }
}

} // namespace osp