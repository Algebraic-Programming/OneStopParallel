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

class Hypergraph {

  public:

    Hypergraph() = default;

    Hypergraph(unsigned num_vertices_, unsigned num_hyperedges_)
        : Num_vertices(num_vertices_), Num_hyperedges(num_hyperedges_), vertex_work_weights(num_vertices_, 1),
        vertex_memory_weights(num_vertices_, 1), hyperedge_weights(num_hyperedges_, 1),
        incident_hyperedges_to_vertex(num_vertices_), vertices_in_hyperedge(num_hyperedges_){}

    Hypergraph(const Hypergraph &other) = default;
    Hypergraph &operator=(const Hypergraph &other) = default;

    virtual ~Hypergraph() = default;

    inline unsigned num_vertices() const { return Num_vertices; }
    inline unsigned num_hyperedges() const { return Num_hyperedges; }
    inline unsigned num_pins() const { return Num_pins; }
    inline int get_vertex_work_weight(unsigned node) const { return vertex_work_weights[node]; }
    inline int get_vertex_memory_weight(unsigned node) const { return vertex_memory_weights[node]; }
    inline int get_hyperedge_weight(unsigned hyperedge) const { return hyperedge_weights[hyperedge]; }

    void add_pin(unsigned vertex_idx, unsigned hyperedge_idx);
    void add_vertex(int work_weight = 1, int memory_weight = 1);
    void add_empty_hyperedge(int weight = 1);
    void add_hyperedge(const std::vector<unsigned>& pins, int weight = 1);
    void set_vertex_work_weight(unsigned vertex_idx, int weight);
    void set_vertex_memory_weight(unsigned vertex_idx, int weight);
    void set_hyperedge_weight(unsigned hyperedge_idx, int weight);

    int compute_total_vertex_work_weight() const;
    int compute_total_vertex_memory_weight() const;

    void clear();
    void reset(unsigned num_vertices_, unsigned num_hyperedges_);

    inline const std::vector<unsigned> &get_incident_hyperedges(unsigned vertex) const { return incident_hyperedges_to_vertex[vertex]; }
    inline const std::vector<unsigned> &get_vertices_in_hyperedge(unsigned hyperedge) const { return vertices_in_hyperedge[hyperedge]; }

    template<typename Graph_t>
    void convert_from_cdag_as_dag(const Graph_t& dag);

    template<typename Graph_t>
    void convert_from_cdag_as_hyperdag(const Graph_t& dag);

  private:
    unsigned Num_vertices = 0, Num_hyperedges = 0, Num_pins = 0;

    std::vector<int> vertex_work_weights;
    std::vector<int> vertex_memory_weights;
    std::vector<int> hyperedge_weights;

    std::vector<std::vector<unsigned>> incident_hyperedges_to_vertex;
    std::vector<std::vector<unsigned>> vertices_in_hyperedge;
};

void Hypergraph::add_pin(unsigned vertex_idx, unsigned hyperedge_idx)
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

void Hypergraph::add_vertex(int work_weight, int memory_weight)
{
    vertex_work_weights.push_back(work_weight);
    vertex_memory_weights.push_back(memory_weight);
    incident_hyperedges_to_vertex.emplace_back();
    ++Num_vertices;
}

void Hypergraph::add_empty_hyperedge(int weight)
{
    vertices_in_hyperedge.emplace_back();
    hyperedge_weights.push_back(weight);
    ++Num_hyperedges;
}

void Hypergraph::add_hyperedge(const std::vector<unsigned>& pins, int weight)
{
    vertices_in_hyperedge.emplace_back(pins);
    hyperedge_weights.push_back(weight);
    for(unsigned vertex : pins)
        incident_hyperedges_to_vertex[vertex].push_back(Num_hyperedges);
    ++Num_hyperedges;
    Num_pins += static_cast<unsigned>(pins.size());
}

void Hypergraph::set_vertex_work_weight(unsigned vertex_idx, int weight)
{
    if(vertex_idx >= Num_vertices)
        throw std::invalid_argument("Invalid Argument while setting vertex weight: vertex index out of range.");
    else   
        vertex_work_weights[vertex_idx] = weight;
}

void Hypergraph::set_vertex_memory_weight(unsigned vertex_idx, int weight)
{
    if(vertex_idx >= Num_vertices)
        throw std::invalid_argument("Invalid Argument while setting vertex weight: vertex index out of range.");
    else   
        vertex_memory_weights[vertex_idx] = weight;
}

void Hypergraph::set_hyperedge_weight(unsigned hyperedge_idx, int weight)
{
    if(hyperedge_idx >= Num_hyperedges)
        throw std::invalid_argument("Invalid Argument while setting hyperedge weight: hyepredge index out of range.");
    else   
        hyperedge_weights[hyperedge_idx] = weight;
}

int Hypergraph::compute_total_vertex_work_weight() const
{
    int total = 0;
    for(unsigned node = 0; node < Num_vertices; ++node)
        total += vertex_work_weights[node];
    return total;
}

int Hypergraph::compute_total_vertex_memory_weight() const
{
    int total = 0;
    for(unsigned node = 0; node < Num_vertices; ++node)
        total += vertex_memory_weights[node];
    return total;
}

void Hypergraph::clear()
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

void Hypergraph::reset(unsigned num_vertices_, unsigned num_hyperedges_)
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

template<typename Graph_t>
void Hypergraph::convert_from_cdag_as_dag(const Graph_t& dag)
{
    reset(static_cast<unsigned>(dag.num_vertices()), 0);
    for(const auto &node : dag.vertices())
    {
        set_vertex_work_weight(static_cast<unsigned>(node), static_cast<int>(dag.vertex_work_weight(node)));
        set_vertex_memory_weight(static_cast<unsigned>(node), static_cast<int>(dag.vertex_mem_weight(node)));
        for (const auto &child : dag.children(node))
            add_hyperedge({static_cast<unsigned>(node), static_cast<unsigned>(child)}); // TODO add edge weights if present
    }
}

template<typename Graph_t>
void Hypergraph::convert_from_cdag_as_hyperdag(const Graph_t& dag)
{
    reset(static_cast<unsigned>(dag.num_vertices()), 0);
    for(const auto &node : dag.vertices())
    {
        set_vertex_work_weight(static_cast<unsigned>(node), static_cast<int>(dag.vertex_work_weight(node)));
        set_vertex_memory_weight(static_cast<unsigned>(node), static_cast<int>(dag.vertex_mem_weight(node)));
        if(dag.out_degree(node) == 0)
            continue;
        std::vector<unsigned> new_hyperedge({static_cast<unsigned>(node)});
        for (const auto &child : dag.children(node))
            new_hyperedge.push_back(static_cast<unsigned>(child));
        add_hyperedge(new_hyperedge, static_cast<int>(dag.vertex_comm_weight(node)));
    }
}

} // namespace osp