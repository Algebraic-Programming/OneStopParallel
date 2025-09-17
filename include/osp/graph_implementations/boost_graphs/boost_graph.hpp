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

#include "osp/auxiliary/hash_util.hpp"
#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/concepts/directed_graph_edge_desc_concept.hpp"
#include "osp/concepts/constructable_computational_dag_concept.hpp"
#include "osp/graph_algorithms/computational_dag_construction_util.hpp"
#include "source_iterator_range.hpp"

template<typename vertex_workw_t, typename vertex_commw_t, typename vertex_memw_t, typename vertex_type_t>
struct boost_vertex {

    boost_vertex() : workWeight(0), communicationWeight(0), memoryWeight(0), nodeType(0) {}
    boost_vertex(vertex_workw_t workWeight_, vertex_commw_t communicationWeight_, vertex_memw_t memoryWeight_,
                 vertex_type_t nodeType_ = 0)
        : workWeight(workWeight_), communicationWeight(communicationWeight_), memoryWeight(memoryWeight_),
          nodeType(nodeType_) {}

    vertex_workw_t workWeight;
    vertex_commw_t communicationWeight;
    vertex_memw_t memoryWeight;
    vertex_type_t nodeType;
};

using boost_vertex_def_int = boost_vertex<int, int, int, unsigned>;
using boost_vertex_def_uint = boost_vertex<unsigned, unsigned, unsigned, unsigned>;

template<typename edge_commw_t>
struct boost_edge {
    boost_edge() : communicationWeight(0) {}
    boost_edge(edge_commw_t communicationWeight_) : communicationWeight(communicationWeight_) {}

    edge_commw_t communicationWeight;
};

using boost_edge_def_int = boost_edge<int>;
using boost_edge_def_uint = boost_edge<unsigned>;

template<typename vertex_workw_t, typename vertex_commw_t, typename vertex_memw_t, typename vertex_type_t,
         typename edge_commw_t>
using boost_graph_impl =
    boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
                          boost_vertex<vertex_workw_t, vertex_commw_t, vertex_memw_t, vertex_type_t>,
                          boost_edge<edge_commw_t>>;

using boost_edge_desc = typename boost::graph_traits<
    boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS>>::edge_descriptor;

template<>
struct std::hash<boost_edge_desc> {
    std::size_t operator()(const boost_edge_desc &p) const noexcept {
        auto h1 = std::hash<std::size_t>{}(p.m_source);
        osp::hash_combine(h1, p.m_target);

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

template<typename vertex_workw_t, typename vertex_commw_t, typename vertex_memw_t, typename vertex_type_t,
         typename edge_commw_t>
class boost_graph {

    using boost_graph_impl_t =
        boost_graph_impl<vertex_workw_t, vertex_commw_t, vertex_memw_t, vertex_type_t, edge_commw_t>;

  public:
    // graph_traits specialization
    using directed_edge_descriptor = typename boost::graph_traits<boost_graph_impl_t>::edge_descriptor;
    using vertex_idx = typename boost::graph_traits<boost_graph_impl_t>::vertex_descriptor;

    // cdag_traits specialization
    using vertex_work_weight_type = vertex_workw_t;
    using vertex_comm_weight_type = vertex_commw_t;
    using vertex_mem_weight_type = vertex_memw_t;
    using vertex_type_type = vertex_type_t;
    using edge_comm_weight_type = edge_commw_t;

    boost_graph(
        const std::vector<std::vector<vertex_idx>> &out_, const std::vector<vertex_work_weight_type> &workW_,
        const std::vector<vertex_comm_weight_type> &commW_,
        const std::unordered_map<std::pair<vertex_idx, vertex_idx>, edge_comm_weight_type, osp::pair_hash> &comm_edge_W)
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

    boost_graph(const std::vector<std::vector<vertex_idx>> &out_, const std::vector<vertex_work_weight_type> &workW_,
                const std::vector<vertex_comm_weight_type> &commW_)
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

    boost_graph(const std::vector<std::vector<vertex_idx>> &out_, const std::vector<vertex_work_weight_type> &workW_,
                const std::vector<vertex_comm_weight_type> &commW_, const std::vector<vertex_type_type> &nodeType_)
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
    explicit boost_graph() : graph(0), number_of_vertex_types(0) {}
    boost_graph(vertex_idx number_of_nodes) : graph(number_of_nodes), number_of_vertex_types(0) {}
    boost_graph(unsigned number_of_nodes)
        : graph(static_cast<vertex_idx>(number_of_nodes)), number_of_vertex_types(0) {}

    boost_graph(const boost_graph &other) = default;

    boost_graph &operator=(const boost_graph &other) = default;

    boost_graph(boost_graph &&other) : number_of_vertex_types(other.number_of_vertex_types) {
        std::swap(this->graph, other.graph);
        other.number_of_vertex_types = 0;
    }

    boost_graph &operator=(boost_graph &&other) {
        if (this != &other) {
            std::swap(graph, other.graph);
            number_of_vertex_types = other.number_of_vertex_types;
            other.number_of_vertex_types = 0;
            other.graph.clear();
        }
        return *this;
    }

    virtual ~boost_graph() = default;

    template<typename Graph_t>
    boost_graph(const Graph_t &other) : number_of_vertex_types(0) {

        static_assert(osp::is_computational_dag_v<Graph_t>, "Graph_t must satisfy the is_computation_dag concept");

        graph.m_vertices.reserve(other.num_vertices());

        osp::construct_computational_dag(other, *this);
    };

    inline const boost_graph_impl_t &get_boost_graph() const { return graph; }
    inline boost_graph_impl_t &get_boost_graph() { return graph; }

    inline size_t num_vertices() const { return boost::num_vertices(graph); }
    inline size_t num_edges() const { return boost::num_edges(graph); }

    void updateNumberOfVertexTypes() {

        number_of_vertex_types = 0;
        for (const auto &v : vertices()) {
            if (vertex_type(v) >= number_of_vertex_types) {
                number_of_vertex_types = vertex_type(v) + 1;
            }
        }
    }

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

    vertex_idx source(const directed_edge_descriptor &e) const { return boost::source(e, graph); }
    vertex_idx target(const directed_edge_descriptor &e) const { return boost::target(e, graph); }

    inline size_t out_degree(const vertex_idx &v) const { return boost::out_degree(v, graph); }
    inline size_t in_degree(const vertex_idx &v) const { return boost::in_degree(v, graph); }

    vertex_work_weight_type vertex_work_weight(const vertex_idx &v) const { return graph[v].workWeight; }
    vertex_comm_weight_type vertex_comm_weight(const vertex_idx &v) const { return graph[v].communicationWeight; }
    vertex_mem_weight_type vertex_mem_weight(const vertex_idx &v) const { return graph[v].memoryWeight; }
    vertex_type_type vertex_type(const vertex_idx &v) const { return graph[v].nodeType; }

    edge_comm_weight_type edge_comm_weight(const directed_edge_descriptor &e) const {
        return graph[e].communicationWeight;
    }

    void set_vertex_mem_weight(const vertex_idx &v, const vertex_mem_weight_type memory_weight) {
        graph[v].memoryWeight = memory_weight;
    }
    void set_vertex_work_weight(const vertex_idx &v, const vertex_work_weight_type work_weight) {
        graph[v].workWeight = work_weight;
    }
    void set_vertex_type(const vertex_idx &v, const vertex_type_type node_type) {
        graph[v].nodeType = node_type;
        number_of_vertex_types = std::max(number_of_vertex_types, node_type + 1);
    }

    void set_vertex_comm_weight(const vertex_idx &v, const vertex_comm_weight_type comm_weight) {
        graph[v].communicationWeight = comm_weight;
    }
    void set_edge_comm_weight(const directed_edge_descriptor &e, const edge_comm_weight_type comm_weight) {
        graph[e].communicationWeight = comm_weight;
    }

    vertex_idx add_vertex(const vertex_work_weight_type work_weight, const vertex_comm_weight_type comm_weight,
                          const vertex_mem_weight_type memory_weight = 0, const vertex_type_type node_type = 0) {
        number_of_vertex_types = std::max(number_of_vertex_types, node_type + 1);
        return boost::add_vertex(boost_vertex{work_weight, comm_weight, memory_weight, node_type}, graph);
    }

    std::pair<boost::detail::edge_desc_impl<boost::bidirectional_tag, std::size_t>, bool>
    add_edge(const vertex_idx &src, const vertex_idx &tar, edge_commw_t comm_weight = DEFAULT_EDGE_COMM_WEIGHT) {

        const auto pair = boost::add_edge(src, tar, {comm_weight}, graph);

        number_of_vertex_types = std::max(number_of_vertex_types, 1u); // in case adding edges adds vertices
        return pair;
    }

    void remove_edge(const directed_edge_descriptor &e) { boost::remove_edge(e, graph); }

    void remove_vertex(const vertex_idx &v) {
        boost::remove_vertex(v, graph);
        updateNumberOfVertexTypes();
    }

    void clear_vertex(const vertex_idx &v) { boost::clear_vertex(v, graph); }

  private:
    boost_graph_impl_t graph;

    vertex_type_type number_of_vertex_types;

    static constexpr edge_comm_weight_type DEFAULT_EDGE_COMM_WEIGHT = 1;
};

template<typename vertex_workw_t, typename vertex_commw_t, typename vertex_memw_t, typename vertex_type_t, typename edge_commw_t>
inline auto edges(const boost_graph<vertex_workw_t, vertex_commw_t, vertex_memw_t, vertex_type_t, edge_commw_t> &graph) {
    return graph.edges();
}

template<typename vertex_workw_t, typename vertex_commw_t, typename vertex_memw_t, typename vertex_type_t, typename edge_commw_t>
inline auto out_edges(osp::vertex_idx_t<boost_graph<vertex_workw_t, vertex_commw_t, vertex_memw_t, vertex_type_t, edge_commw_t>> v,
                      const boost_graph<vertex_workw_t, vertex_commw_t, vertex_memw_t, vertex_type_t, edge_commw_t> &graph) {
    return graph.out_edges(v);
}

template<typename vertex_workw_t, typename vertex_commw_t, typename vertex_memw_t, typename vertex_type_t, typename edge_commw_t>
inline auto in_edges(osp::vertex_idx_t<boost_graph<vertex_workw_t, vertex_commw_t, vertex_memw_t, vertex_type_t, edge_commw_t>> v,
                     const boost_graph<vertex_workw_t, vertex_commw_t, vertex_memw_t, vertex_type_t, edge_commw_t> &graph) {
    return graph.in_edges(v);
}

template<typename vertex_workw_t, typename vertex_commw_t, typename vertex_memw_t, typename vertex_type_t, typename edge_commw_t>
inline osp::vertex_idx_t<boost_graph<vertex_workw_t, vertex_commw_t, vertex_memw_t, vertex_type_t, edge_commw_t>> source(const osp::edge_desc_t<boost_graph<vertex_workw_t, vertex_commw_t, vertex_memw_t, vertex_type_t, edge_commw_t>> &edge, const boost_graph<vertex_workw_t, vertex_commw_t, vertex_memw_t, vertex_type_t, edge_commw_t> &graph) {
    return graph.source(edge);
}

template<typename vertex_workw_t, typename vertex_commw_t, typename vertex_memw_t, typename vertex_type_t, typename edge_commw_t>
inline osp::vertex_idx_t<boost_graph<vertex_workw_t, vertex_commw_t, vertex_memw_t, vertex_type_t, edge_commw_t>> target(const osp::edge_desc_t<boost_graph<vertex_workw_t, vertex_commw_t, vertex_memw_t, vertex_type_t, edge_commw_t>> &edge, const boost_graph<vertex_workw_t, vertex_commw_t, vertex_memw_t, vertex_type_t, edge_commw_t> &graph) {
    return graph.target(edge);
}

using boost_graph_int_t = boost_graph<int, int, int, unsigned, int>;
using boost_graph_uint_t = boost_graph<unsigned, unsigned, unsigned, unsigned, unsigned>;


static_assert(osp::is_directed_graph_edge_desc_v<boost_graph_int_t>,
              "boost_graph_adapter does not satisfy the directed_graph_edge_desc concept");

static_assert(osp::is_computational_dag_typed_vertices_edge_desc_v<boost_graph_int_t>,
              "boost_graph_adapter must satisfy the computational_dag_typed_vertices_edge_desc concept");

static_assert(osp::is_constructable_cdag_vertex_v<boost_graph_int_t>,
              "boost_graph_adapter must satisfy the is_constructable_cdag_vertex concept");

static_assert(osp::is_constructable_cdag_typed_vertex_v<boost_graph_int_t>,
              "boost_graph_adapter must satisfy the is_constructable_cdag_typed_vertex concept");

static_assert(osp::is_constructable_cdag_edge_v<boost_graph_int_t>,
              "boost_graph_adapter must satisfy the is_constructable_cdag_edge concept");

static_assert(osp::is_constructable_cdag_comm_edge_v<boost_graph_int_t>,
              "boost_graph_adapter must satisfy the is_constructable_cdag_comm_edge concept");