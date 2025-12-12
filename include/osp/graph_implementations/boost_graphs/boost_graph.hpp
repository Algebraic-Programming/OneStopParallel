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
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <functional>
#include <iostream>

#include "osp/auxiliary/hash_util.hpp"
#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/concepts/constructable_computational_dag_concept.hpp"
#include "osp/concepts/directed_graph_edge_desc_concept.hpp"
#include "osp/graph_algorithms/computational_dag_construction_util.hpp"
#include "source_iterator_range.hpp"

template <typename VertexWorkwT, typename VertexCommwT, typename VertexMemwT, typename VertexTypeT>
struct BoostVertex {
    BoostVertex() : workWeight_(0), communicationWeight_(0), memoryWeight_(0), nodeType_(0) {}

    BoostVertex(VertexWorkwT workWeight, VertexCommwT communicationWeight, VertexMemwT memoryWeight, VertexTypeT nodeType = 0)
        : workWeight_(workWeight), communicationWeight_(communicationWeight), memoryWeight_(memoryWeight), nodeType_(nodeType) {}

    VertexWorkwT workWeight_;
    VertexCommwT communicationWeight_;
    VertexMemwT memoryWeight_;
    VertexTypeT nodeType_;
};

using BoostVertexDefInt = BoostVertex<int, int, int, unsigned>;
using BoostVertexDefUint = BoostVertex<unsigned, unsigned, unsigned, unsigned>;

template <typename EdgeCommwT>
struct BoostEdge {
    BoostEdge() : communicationWeight_(0) {}

    BoostEdge(EdgeCommwT communicationWeight) : communicationWeight_(communicationWeight) {}

    EdgeCommwT communicationWeight_;
};

using BoostEdgeDefInt = BoostEdge<int>;
using BoostEdgeDefUint = BoostEdge<unsigned>;

template <typename VertexWorkwT, typename VertexCommwT, typename VertexMemwT, typename VertexTypeT, typename EdgeCommwT>
using BoostGraphImpl = boost::adjacency_list<boost::vecS,
                                             boost::vecS,
                                             boost::bidirectionalS,
                                             BoostVertex<VertexWorkwT, VertexCommwT, VertexMemwT, VertexTypeT>,
                                             BoostEdge<EdgeCommwT>>;

using BoostEdgeDesc =
    typename boost::graph_traits<boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS>>::edge_descriptor;

template <>
struct std::hash<BoostEdgeDesc> {
    std::size_t operator()(const BoostEdgeDesc &p) const noexcept {
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

template <typename VertexWorkwT, typename VertexCommwT, typename VertexMemwT, typename VertexTypeT, typename EdgeCommwT>
class BoostGraph {
    using BoostGraphImplT = BoostGraphImpl<VertexWorkwT, VertexCommwT, VertexMemwT, VertexTypeT, EdgeCommwT>;

  public:
    // graph_traits specialization
    using DirectedEdgeDescriptor = typename boost::graph_traits<BoostGraphImplT>::edge_descriptor;
    using VertexIdx = typename boost::graph_traits<BoostGraphImplT>::vertex_descriptor;

    // cdag_traits specialization
    using VertexWorkWeightType = VertexWorkwT;
    using VertexCommWeightType = VertexCommwT;
    using VertexMemWeightType = VertexMemwT;
    using VertexTypeType = VertexTypeT;
    using EdgeCommWeightType = EdgeCommwT;

    BoostGraph(const std::vector<std::vector<VertexIdx>> &out,
               const std::vector<VertexWorkWeightType> &workW,
               const std::vector<VertexCommWeightType> &commW,
               const std::unordered_map<std::pair<vertex_idx, vertex_idx>, edge_comm_weight_type, osp::pair_hash> &commEdgeW)
        : numberOfVertexTypes_(0) {
        graph_.m_vertices.reserve(out.size());

        assert(out.size() == workW.size());
        assert(out.size() == commW.size());

        for (size_t i = 0; i < out.size(); ++i) {
            AddVertex(workW[i], commW[i]);
        }
        for (size_t i = 0; i < out.size(); ++i) {
            for (const auto &j : out[i]) {
                assert(comm_edge_W.find(std::make_pair(i, j)) != comm_edge_W.cend());
                AddEdge(i, j, comm_edge_W.at(std::make_pair(i, j)));
            }
        }
        UpdateNumberOfVertexTypes();
    }

    BoostGraph(const std::vector<std::vector<VertexIdx>> &out,
               const std::vector<VertexWorkWeightType> &workW,
               const std::vector<VertexCommWeightType> &commW)
        : numberOfVertexTypes_(0) {
        graph_.m_vertices.reserve(out.size());

        assert(out.size() == workW.size());
        assert(out.size() == commW.size());

        for (size_t i = 0; i < out.size(); ++i) {
            AddVertex(workW[i], commW[i]);
        }
        for (size_t i = 0; i < out.size(); ++i) {
            for (const auto &j : out[i]) {
                AddEdge(i, j);
            }
        }
        UpdateNumberOfVertexTypes();
    }

    BoostGraph(const std::vector<std::vector<VertexIdx>> &out,
               const std::vector<VertexWorkWeightType> &workW,
               const std::vector<VertexCommWeightType> &commW,
               const std::vector<VertexTypeType> &nodeType)
        : numberOfVertexTypes_(0) {
        graph_.m_vertices.reserve(out.size());

        assert(out.size() == workW.size());
        assert(out.size() == commW.size());
        assert(out.size() == nodeType.size());

        for (size_t i = 0; i < out.size(); ++i) {
            AddVertex(workW[i], commW[i], 0, nodeType[i]);
        }
        for (size_t i = 0; i < out.size(); ++i) {
            for (const auto &j : out[i]) {
                AddEdge(i, j);
            }
        }
        UpdateNumberOfVertexTypes();
    }

    /**
     * @brief Default constructor for the ComputationalDag class.
     */
    explicit BoostGraph() : graph_(0), numberOfVertexTypes_(0) {}

    BoostGraph(VertexIdx numberOfNodes) : graph_(numberOfNodes), numberOfVertexTypes_(0) {}

    BoostGraph(unsigned numberOfNodes) : graph_(static_cast<VertexIdx>(numberOfNodes)), numberOfVertexTypes_(0) {}

    BoostGraph(const BoostGraph &other) = default;

    BoostGraph &operator=(const BoostGraph &other) = default;

    BoostGraph(BoostGraph &&other) : numberOfVertexTypes_(other.numberOfVertexTypes_) {
        std::swap(this->graph_, other.graph_);
        other.numberOfVertexTypes_ = 0;
    }

    BoostGraph &operator=(BoostGraph &&other) {
        if (this != &other) {
            std::swap(graph_, other.graph_);
            numberOfVertexTypes_ = other.numberOfVertexTypes_;
            other.numberOfVertexTypes_ = 0;
            other.graph_.clear();
        }
        return *this;
    }

    virtual ~BoostGraph() = default;

    template <typename GraphT>
    BoostGraph(const GraphT &other) : numberOfVertexTypes_(0) {
        static_assert(osp::IsComputationalDagV<Graph_t>, "Graph_t must satisfy the is_computation_dag concept");

        graph_.m_vertices.reserve(other.num_vertices());

        osp::constructComputationalDag(other, *this);
    }

    inline const BoostGraphImplT &GetBoostGraph() const { return graph_; }

    inline BoostGraphImplT &GetBoostGraph() { return graph_; }

    inline size_t NumVertices() const { return boost::num_vertices(graph_); }

    inline size_t NumEdges() const { return boost::num_edges(graph_); }

    void UpdateNumberOfVertexTypes() {
        numberOfVertexTypes_ = 0;
        for (const auto &v : vertices()) {
            if (VertexType(v) >= numberOfVertexTypes_) {
                numberOfVertexTypes_ = VertexType(v) + 1;
            }
        }
    }

    inline unsigned NumVertexTypes() const { return numberOfVertexTypes_; };

    auto Vertices() const { return boost::make_iterator_range(boost::vertices(graph_)); }

    auto Vertices() { return boost::make_iterator_range(boost::vertices(graph_)); }

    // template<typename T>
    // void debug() const {
    //     static_assert(sizeof(T *) == 0);
    // }

    auto Parents(const VertexIdx &v) const {
        // auto ciao = boost::extensions::make_source_iterator_range(boost::inv_adjacent_vertices(v, graph));

        // debug<typename std::iterator_traits<decltype(std::begin(ciao))>::value_type>();

        // debug<typename decltype(ciao.begin())::value_type>();
        return boost::extensions::make_source_iterator_range(boost::inv_adjacent_vertices(v, graph_));
    }

    auto Parents(const VertexIdx &v) {
        return boost::extensions::make_source_iterator_range(boost::inv_adjacent_vertices(v, graph_));
    }

    auto Children(const VertexIdx &v) const {
        return boost::extensions::make_source_iterator_range(boost::adjacent_vertices(v, graph_));
    }

    auto Children(const VertexIdx &v) {
        return boost::extensions::make_source_iterator_range(boost::adjacent_vertices(v, graph_));
    }

    auto Edges() const { return boost::extensions::make_source_iterator_range(boost::edges(graph_)); }

    auto Edges() { return boost::extensions::make_source_iterator_range(boost::edges(graph_)); }

    auto InEdges(const VertexIdx &v) const { return boost::extensions::make_source_iterator_range(boost::in_edges(v, graph_)); }

    auto InEdges(const VertexIdx &v) { return boost::extensions::make_source_iterator_range(boost::in_edges(v, graph_)); }

    auto OutEdges(const VertexIdx &v) const { return boost::extensions::make_source_iterator_range(boost::out_edges(v, graph_)); }

    auto OutEdges(const VertexIdx &v) { return boost::extensions::make_source_iterator_range(boost::out_edges(v, graph_)); }

    VertexIdx Source(const DirectedEdgeDescriptor &e) const { return boost::source(e, graph_); }

    VertexIdx Target(const DirectedEdgeDescriptor &e) const { return boost::target(e, graph_); }

    inline size_t OutDegree(const VertexIdx &v) const { return boost::out_degree(v, graph_); }

    inline size_t InDegree(const VertexIdx &v) const { return boost::in_degree(v, graph_); }

    VertexWorkWeightType VertexWorkWeight(const VertexIdx &v) const { return graph_[v].workWeight; }

    VertexCommWeightType VertexCommWeight(const VertexIdx &v) const { return graph_[v].communicationWeight; }

    VertexMemWeightType VertexMemWeight(const VertexIdx &v) const { return graph_[v].memoryWeight; }

    VertexTypeType VertexType(const VertexIdx &v) const { return graph_[v].nodeType; }

    EdgeCommWeightType EdgeCommWeight(const DirectedEdgeDescriptor &e) const { return graph_[e].communicationWeight; }

    void SetVertexMemWeight(const VertexIdx &v, const VertexMemWeightType memoryWeight) { graph_[v].memoryWeight = memoryWeight; }

    void SetVertexWorkWeight(const VertexIdx &v, const VertexWorkWeightType workWeight) { graph_[v].workWeight = workWeight; }

    void SetVertexType(const VertexIdx &v, const VertexTypeType nodeType) {
        graph_[v].nodeType = nodeType;
        numberOfVertexTypes_ = std::max(numberOfVertexTypes_, nodeType + 1);
    }

    void SetVertexCommWeight(const VertexIdx &v, const VertexCommWeightType commWeight) {
        graph_[v].communicationWeight = commWeight;
    }

    void SetEdgeCommWeight(const DirectedEdgeDescriptor &e, const EdgeCommWeightType commWeight) {
        graph_[e].communicationWeight = commWeight;
    }

    VertexIdx AddVertex(const VertexWorkWeightType workWeight,
                        const VertexCommWeightType commWeight,
                        const VertexMemWeightType memoryWeight = 0,
                        const VertexTypeType nodeType = 0) {
        numberOfVertexTypes_ = std::max(numberOfVertexTypes_, nodeType + 1);
        return boost::add_vertex(boost_vertex{workWeight, commWeight, memoryWeight, nodeType}, graph_);
    }

    std::pair<boost::detail::edge_desc_impl<boost::bidirectional_tag, std::size_t>, bool> AddEdge(
        const VertexIdx &src, const VertexIdx &tar, EdgeCommwT commWeight = defaultEdgeCommWeight_) {
        const auto pair = boost::add_edge(src, tar, {commWeight}, graph_);

        numberOfVertexTypes_ = std::max(numberOfVertexTypes_, 1u);    // in case adding edges adds vertices
        return pair;
    }

    void RemoveEdge(const DirectedEdgeDescriptor &e) { boost::remove_edge(e, graph_); }

    void RemoveVertex(const VertexIdx &v) {
        boost::remove_vertex(v, graph_);
        UpdateNumberOfVertexTypes();
    }

    void ClearVertex(const VertexIdx &v) { boost::clear_vertex(v, graph_); }

  private:
    BoostGraphImplT graph_;

    VertexTypeType numberOfVertexTypes_;

    static constexpr EdgeCommWeightType defaultEdgeCommWeight_ = 1;
};

template <typename VertexWorkwT, typename VertexCommwT, typename VertexMemwT, typename VertexTypeT, typename EdgeCommwT>
inline auto Edges(const BoostGraph<VertexWorkwT, VertexCommwT, VertexMemwT, VertexTypeT, EdgeCommwT> &graph) {
    return graph.edges();
}

template <typename VertexWorkwT, typename VertexCommwT, typename VertexMemwT, typename VertexTypeT, typename EdgeCommwT>
inline auto outEdges(osp::vertex_idx_t<boost_graph<vertex_workw_t, vertex_commw_t, vertex_memw_t, vertex_type_t, edge_commw_t>> v,
                     const boost_graph<vertex_workw_t, vertex_commw_t, vertex_memw_t, vertex_type_t, edge_commw_t> &graph) {
    return graph.out_edges(v);
}

template <typename VertexWorkwT, typename VertexCommwT, typename VertexMemwT, typename VertexTypeT, typename EdgeCommwT>
inline auto inEdges(osp::vertex_idx_t<boost_graph<vertex_workw_t, vertex_commw_t, vertex_memw_t, vertex_type_t, edge_commw_t>> v,
                    const boost_graph<vertex_workw_t, vertex_commw_t, vertex_memw_t, vertex_type_t, edge_commw_t> &graph) {
    return graph.in_edges(v);
}

template <typename VertexWorkwT, typename VertexCommwT, typename VertexMemwT, typename VertexTypeT, typename EdgeCommwT>
inline osp::vertex_idx_t<boost_graph<vertex_workw_t, vertex_commw_t, vertex_memw_t, vertex_type_t, edge_commw_t>> Source(
    const osp::edge_desc_t<boost_graph<vertex_workw_t, vertex_commw_t, vertex_memw_t, vertex_type_t, edge_commw_t>> &edge,
    const BoostGraph<VertexWorkwT, VertexCommwT, VertexMemwT, VertexTypeT, EdgeCommwT> &graph) {
    return graph.source(edge);
}

template <typename VertexWorkwT, typename VertexCommwT, typename VertexMemwT, typename VertexTypeT, typename EdgeCommwT>
inline osp::vertex_idx_t<boost_graph<vertex_workw_t, vertex_commw_t, vertex_memw_t, vertex_type_t, edge_commw_t>> Target(
    const osp::edge_desc_t<boost_graph<vertex_workw_t, vertex_commw_t, vertex_memw_t, vertex_type_t, edge_commw_t>> &edge,
    const BoostGraph<VertexWorkwT, VertexCommwT, VertexMemwT, VertexTypeT, EdgeCommwT> &graph) {
    return graph.target(edge);
}

using BoostGraphIntT = BoostGraph<int, int, int, unsigned, int>;
using BoostGraphUintT = BoostGraph<unsigned, unsigned, unsigned, unsigned, unsigned>;

static_assert(osp::IsDirectedGraphEdgeDescV<boost_graph_int_t>,
              "boost_graph_adapter does not satisfy the directed_graph_edge_desc concept");

static_assert(osp::IsComputationalDagTypedVerticesEdgeDescV<boost_graph_int_t>,
              "boost_graph_adapter must satisfy the computational_dag_typed_vertices_edge_desc concept");

static_assert(osp::is_constructable_cdag_vertex_v<boost_graph_int_t>,
              "boost_graph_adapter must satisfy the is_constructable_cdag_vertex concept");

static_assert(osp::is_constructable_cdag_typed_vertex_v<boost_graph_int_t>,
              "boost_graph_adapter must satisfy the is_constructable_cdag_typed_vertex concept");

static_assert(osp::is_constructable_cdag_edge_v<boost_graph_int_t>,
              "boost_graph_adapter must satisfy the is_constructable_cdag_edge concept");

static_assert(osp::is_constructable_cdag_comm_edge_v<boost_graph_int_t>,
              "boost_graph_adapter must satisfy the is_constructable_cdag_comm_edge concept");
