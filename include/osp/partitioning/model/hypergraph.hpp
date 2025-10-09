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

namespace osp {

class hypergraph {

  public:

    hypergraph() = default;

    hypergraph(unsigned num_vertices_, unsigned num_hyperedges_)
        : Num_vertices(num_vertices_), Num_hyperedges(num_hyperedges_), vertex_weights(num_vertices_, 0),
        incident_hyperedges_to_vertex(num_vertices_), vertices_in_hyperedge(num_hyperedges_){}

    hypergraph(const hypergraph &other) = default;
    hypergraph &operator=(const hypergraph &other) = default;

    virtual ~hypergraph() = default;

    inline unsigned num_vertices() const { return Num_vertices; }
    inline unsigned num_hyperedges() const { return Num_hyperedges; }
    inline unsigned num_pins() const { return Num_pins; }

    void add_pin(unsigned vertex_idx, unsigned hyperedge_idx);
    void add_vertex(unsigned weight = 0);
    void add_empty_hyperedge();
    void add_hyperedge(const std::vector<unsigned>& pins);
    void set_vertex_weight(unsigned vertex_idx, int weight);

    void clear();
    void reset(unsigned num_vertices_, unsigned num_hyperedges_);

    inline const std::vector<unsigned> &get_incident_hyperedges(unsigned vertex) const { return incident_hyperedges_to_vertex[vertex]; }
    inline const std::vector<unsigned> &get_vertices_in_hyperedge(unsigned hyperedge) const { return vertices_in_hyperedge[hyperedge]; }

    template<typename Graph_t>
    void convert_from_cdag_as_dag(const Graph_t& dag);

    template<typename Graph_t>
    void convert_from_cdag_as_hyperdag(const Graph_t& dag);

    void read_spmv_from_matrixmarket();

  private:
    unsigned Num_vertices = 0, Num_hyperedges = 0, Num_pins = 0;

    std::vector<int> vertex_weights;

    std::vector<std::vector<unsigned>> incident_hyperedges_to_vertex;
    std::vector<std::vector<unsigned>> vertices_in_hyperedge;
};

void hypergraph::add_pin(unsigned vertex_idx, unsigned hyperedge_idx)
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

void hypergraph::add_vertex(unsigned weight = 0)
{
    vertex_weights.push_back(weight);
    incident_hyperedges_to_vertex.emplace_back();
    ++Num_vertices;
}

void hypergraph::add_empty_hyperedge()
{
    vertices_in_hyperedge.emplace_back();
    ++Num_hyperedges;
}

void hypergraph::add_hyperedge(const std::vector<unsigned>& pins)
{
    vertices_in_hyperedge.emplace_back(pins);
    for(unsigned vertex : pins)
        incident_hyperedges_to_vertex[vertex].push_back(Num_hyperedges);
    ++Num_hyperedges;
    Num_pins += pins.size();
}

void hypergraph::set_vertex_weight(unsigned vertex_idx, int weight)
{
    if(vertex_idx >= Num_vertices)
        throw std::invalid_argument("Invalid Argument while setting vertex weight: vertex index out of range.");
    else   
        vertex_weights[vertex_idx] = weight;
}

void hypergraph::clear()
{
    Num_vertices = 0;
    Num_hyperedges = 0;
    Num_pins = 0;

    vertex_weights.clear();
    incident_hyperedges_to_vertex.clear();
    vertices_in_hyperedge.clear();
}

void hypergraph::reset(unsigned num_vertices_, unsigned num_hyperedges_)
{
    clear();

    Num_vertices = num_vertices_;
    Num_hyperedges = num_hyperedges_;

    vertex_weights.resize(num_vertices_, 0);
    incident_hyperedges_to_vertex.resize(num_vertices_);
    vertices_in_hyperedge.resize(num_hyperedges_);
}

template<typename Graph_t>
void convert_from_cdag_as_dag(const Graph_t& dag)
{
    reset(dag.num_vertices(), dag.num_edges());
    for(const auto &node : dag.vertices())
    {
        set_vertex_weight(node, dag.vertex_work_weight(node));
        for (const auto &child : dag.children(node))
            add_hyperedge({node, child});
    }
}

template<typename Graph_t>
void convert_from_cdag_as_hyperdag(const Graph_t& dag)
{
    unsigned nr_of_non_sinks = 0;
    for(const auto &node : dag.vertices())
        if(dag.out_degree(node) > 0)
            ++ nr_of_non_sinks;
    
    reset(dag.num_vertices(), nr_of_non_sinks);
    for(const auto &node : dag.vertices())
    {
        set_vertex_weight(node, dag.vertex_work_weight(node));
        if(dag.out_degree(node) == 0)
            continue;
        std::vector<unsigned> new_hyperedge({node});
        for (const auto &child : dag.children(node))
            new_hyperedge.push_back(child);
        add_hyperedge(new_hyperedge);
    }
}

} // namespace osp