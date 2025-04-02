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
#include <functional>
#include <iostream>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

#include "auxiliary/misc.hpp"
#include "concepts/computational_dag_concept.hpp"
#include "source_iterator_range.hpp"


struct boost_vertex {

    boost_vertex() : workWeight(0), communicationWeight(0), memoryWeight(0), nodeType(0) {}
    boost_vertex(int workWeight_, int communicationWeight_, int memoryWeight_, unsigned nodeType_ = 0)
        : workWeight(workWeight_), communicationWeight(communicationWeight_), memoryWeight(memoryWeight_),
          nodeType(nodeType_) {}

    int workWeight;
    int communicationWeight;
    int memoryWeight;
    unsigned nodeType;
};

struct boost_edge {
    boost_edge() : communicationWeight(0) {}
    boost_edge(int communicationWeight_) : communicationWeight(communicationWeight_) {}

    int communicationWeight;
};

using boost_graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, boost_vertex, boost_edge>;
using boost_vertex_type = boost::graph_traits<boost_graph>::vertex_descriptor;
using boost_edge_desc = boost::graph_traits<boost_graph>::edge_descriptor;

struct boost_edge_type_hash {

    std::size_t operator()(const boost_edge_desc &p) const {

        auto h1 = std::hash<boost_vertex_type>{}(p.m_source);
        osp::hash_combine(h1, p.m_target);
        osp::hash_combine(h1, p.m_eproperty);

        return h1;
    }
};

/**
 * @class ComputationalDag
 * @brief Represents a computational directed acyclic graph (DAG).
 *
 * The ComputationalDag class is used to represent a computational DAG, which consists of vertices and edges.
 * Each vertex represents a computational task, and each edge represents a communication dependency between tasks.
 * The class provides various methods to manipulate and analyze the DAG, such as adding vertices and edges,
 * calculating the longest path, and retrieving topological order of vertices.
 */

class boost_graph_adapter {
  public:
    // graph_traits specialization
    using directed_edge_descriptor = boost_edge_desc;
    using vertex_idx = boost_vertex_type;

    // cdag_traits specialization
    using vertex_work_weight_type = int;
    using vertex_comm_weight_type = int;
    using vertex_mem_weight_type = int;
    using vertex_type_type = unsigned;
    using edge_comm_weight_type = int;

    boost_graph_adapter(const std::vector<std::vector<vertex_idx>> &out_, const std::vector<int> &workW_,
                        const std::vector<int> &commW_,
                        const std::unordered_map<std::pair<int, int>, int, osp::pair_hash> &comm_edge_W)
        : number_of_vertex_types(0) {
        graph.m_vertices.reserve(out_.size());

        assert(out_.size() == workW_.size());
        assert(out_.size() == commW_.size());

        for (size_t i = 0; i < out_.size(); ++i) {
            add_vertex(workW_[i], commW_[i]);
        }
        for (size_t i = 0; i < out_.size(); ++i) {

            for (const auto &j : out_[i]) {
                assert(comm_edge_W.find(std::make_pair(i, j)) != comm_edge_W.cend());
                add_edge(i, j, comm_edge_W.at(std::make_pair(i, j)));
            }
        }
        updateNumberOfVertexTypes();
    }

    boost_graph_adapter(const std::vector<std::vector<vertex_idx>> &out_, const std::vector<int> &workW_,
                        const std::vector<int> &commW_)
        : number_of_vertex_types(0) {
        graph.m_vertices.reserve(out_.size());

        assert(out_.size() == workW_.size());
        assert(out_.size() == commW_.size());

        for (size_t i = 0; i < out_.size(); ++i) {
            add_vertex(workW_[i], commW_[i]);
        }
        for (size_t i = 0; i < out_.size(); ++i) {

            for (const auto &j : out_[i]) {
                add_edge(i, j);
            }
        }
        updateNumberOfVertexTypes();
    }

    boost_graph_adapter(const std::vector<std::vector<vertex_idx>> &out_, const std::vector<int> &workW_,
                        const std::vector<int> &commW_, const std::vector<unsigned> &nodeType_)
        : number_of_vertex_types(0) {
        graph.m_vertices.reserve(out_.size());

        assert(out_.size() == workW_.size());
        assert(out_.size() == commW_.size());
        assert(out_.size() == nodeType_.size());

        for (size_t i = 0; i < out_.size(); ++i) {
            add_vertex(workW_[i], commW_[i], 0, nodeType_[i]);
        }
        for (size_t i = 0; i < out_.size(); ++i) {

            for (const auto &j : out_[i]) {
                add_edge(i, j);
            }
        }
        updateNumberOfVertexTypes();
    }

    /**
     * @brief Default constructor for the ComputationalDag class.
     */
    explicit boost_graph_adapter() : graph(0), number_of_vertex_types(0) {}
    explicit boost_graph_adapter(unsigned number_of_nodes) : graph(number_of_nodes), number_of_vertex_types(0) {
        updateNumberOfVertexTypes();
    }

    inline const boost_graph &get_boost_graph() const { return graph; }
    inline boost_graph &get_boost_graph() { return graph; }

    inline size_t num_vertices() const { return boost::num_vertices(graph); }
    inline size_t num_edges() const { return boost::num_edges(graph); }

    void updateNumberOfVertexTypes();
    inline unsigned num_vertex_types() const { return number_of_vertex_types; };

    auto vertices() const { return boost::make_iterator_range(boost::vertices(graph)); }
    auto vertices() { return boost::make_iterator_range(boost::vertices(graph)); }

    // template<typename T>
    // void debug() const {
    //     static_assert(sizeof(T *) == 0);
    // }

    auto parents(const vertex_idx &v) const {
        // auto ciao = boost::extensions::make_source_iterator_range(boost::inv_adjacent_vertices(v, graph));

        // debug<typename std::iterator_traits<decltype(std::begin(ciao))>::value_type>();

        // debug<typename decltype(ciao.begin())::value_type>();
        return boost::extensions::make_source_iterator_range(boost::inv_adjacent_vertices(v, graph));
    }

    auto parents(const vertex_idx &v) {
        return boost::extensions::make_source_iterator_range(boost::inv_adjacent_vertices(v, graph));
    }

    auto children(const vertex_idx &v) const {
        return boost::extensions::make_source_iterator_range(boost::adjacent_vertices(v, graph));
    }

    auto children(const vertex_idx &v) {
        return boost::extensions::make_source_iterator_range(boost::adjacent_vertices(v, graph));
    }

    auto edges() const { return boost::extensions::make_source_iterator_range(boost::edges(graph)); }

    auto edges() { return boost::extensions::make_source_iterator_range(boost::edges(graph)); }

    auto in_edges(const vertex_idx &v) const {
        return boost::extensions::make_source_iterator_range(boost::in_edges(v, graph));
    }

    auto in_edges(const vertex_idx &v) {
        return boost::extensions::make_source_iterator_range(boost::in_edges(v, graph));
    }

    auto out_edges(const vertex_idx &v) const {
        return boost::extensions::make_source_iterator_range(boost::out_edges(v, graph));
    }

    auto out_edges(const vertex_idx &v) {
        return boost::extensions::make_source_iterator_range(boost::out_edges(v, graph));
    }

    vertex_idx source(const boost_edge_desc &e) const { return boost::source(e, graph); }
    vertex_idx target(const boost_edge_desc &e) const { return boost::target(e, graph); }

    inline size_t out_degree(const vertex_idx &v) const { return boost::out_degree(v, graph); }
    inline size_t in_degree(const vertex_idx &v) const { return boost::in_degree(v, graph); }

    int vertex_work_weight(const vertex_idx &v) const { return graph[v].workWeight; }
    int vertex_comm_weight(const vertex_idx &v) const { return graph[v].communicationWeight; }
    int vertex_mem_weight(const vertex_idx &v) const { return graph[v].memoryWeight; }
    unsigned vertex_type(const vertex_idx &v) const { return graph[v].nodeType; }

    int edge_comm_weight(const directed_edge_descriptor &e) const { return graph[e].communicationWeight; }

    void set_vertex_memory_weight(const vertex_idx &v, const int memory_weight) {
        graph[v].memoryWeight = memory_weight;
    }
    void set_vertex_work_weight(const vertex_idx &v, const int work_weight) { graph[v].workWeight = work_weight; }
    void set_vertex_type(const vertex_idx &v, const unsigned node_type) {
        graph[v].nodeType = node_type;
        number_of_vertex_types = std::max(number_of_vertex_types, node_type + 1);
    }

    void set_vertex_comm_weight(const vertex_idx &v, const int comm_weight) {
        graph[v].communicationWeight = comm_weight;
    }
    void set_edge_comm_weight(const directed_edge_descriptor &e, const int comm_weight) {
        graph[e].communicationWeight = comm_weight;
    }

    vertex_idx add_vertex(const int work_weight, const int comm_weight, const int memory_weight = 0,
                          const unsigned node_type = 0) {
        number_of_vertex_types = std::max(number_of_vertex_types, node_type + 1);
        return boost::add_vertex(boost_vertex{work_weight, comm_weight, memory_weight, node_type}, graph);
    }

    std::pair<boost::detail::edge_desc_impl<boost::bidirectional_tag, std::size_t>, bool>
    add_edge(const vertex_idx &src, const vertex_idx &tar, int comm_weight = DEFAULT_EDGE_COMM_WEIGHT);

  private:
    boost_graph graph;

    unsigned number_of_vertex_types;

    static constexpr int DEFAULT_EDGE_COMM_WEIGHT = 1;
};

static_assert(osp::is_directed_graph_edge_desc_v<boost_graph_adapter>,
              "boost_graph_adapter does not satisfy the computational_dag concept");

static_assert(osp::is_computation_dag_typed_vertices_edge_desc_v<boost_graph_adapter>,
              "boost_graph_adapter must satisfy the computation_dag_typed_vertices_edge_desc concept");

