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

#include "boost_extensions/inv_breadth_first_search.hpp"
#include "boost_extensions/source_iterator_range.hpp"
#include <numeric>
#include <queue>
#include <vector>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/graph_traits.hpp>

#include "auxiliary/Balanced_Coin_Flips.hpp"
#include "auxiliary/auxiliary.hpp"

struct Vertex {

    Vertex() : workWeight(0), communicationWeight(0), memoryWeight(0) {}
    Vertex(int workWeight_, int communicationWeight_, int memoryWeight_)
        : workWeight(workWeight_), communicationWeight(communicationWeight_), memoryWeight(memoryWeight_) {}

    int workWeight;
    int communicationWeight;
    int memoryWeight;
};

struct Edge {
    Edge() : communicationWeight(0) {}
    Edge(int communicationWeight_) : communicationWeight(communicationWeight_) {}
    
    int communicationWeight;

};

using GraphType = boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, Vertex, Edge>;
using VertexType = boost::graph_traits<GraphType>::vertex_descriptor;
using EdgeType = boost::graph_traits<GraphType>::edge_descriptor;

struct EdgeType_hash {

    std::size_t operator()(const EdgeType &p) const {

        auto h1 = std::hash<VertexType>{}(p.m_source);
        hash_combine(h1, p.m_target);
        hash_combine(h1, p.m_eproperty);

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
class ComputationalDag {
    static constexpr int DEFAULT_EDGE_COMM_WEIGHT = 1;

  private:
    GraphType graph;

  public:
    ComputationalDag(const std::vector<std::vector<int>> &out_, const std::vector<int> &workW_,
                     const std::vector<int> &commW_,
                     const std::unordered_map<std::pair<int, int>, int, pair_hash> &comm_edge_W) {
        graph.m_vertices.reserve(out_.size());

        assert(out_.size() == workW_.size());
        assert(out_.size() == commW_.size());

        for (size_t i = 0; i < out_.size(); ++i) {
            addVertex(workW_[i], commW_[i]);
        }
        for (size_t i = 0; i < out_.size(); ++i) {
            const auto &v_idx = boost::vertex(i, graph);
            for (const auto &j : out_[i]) {
                assert(comm_edge_W.find(std::make_pair(i, j)) != comm_edge_W.cend());
                addEdge(v_idx, boost::vertex(j, graph), comm_edge_W.at(std::make_pair(i, j)));
            }
        }
    }


    ComputationalDag(const std::vector<std::vector<int>> &out_, const std::vector<int> &workW_,
                     const std::vector<int> &commW_) {
        graph.m_vertices.reserve(out_.size());

        assert(out_.size() == workW_.size());
        assert(out_.size() == commW_.size());

        for (size_t i = 0; i < out_.size(); ++i) {
            addVertex(workW_[i], commW_[i]);
        }
        for (size_t i = 0; i < out_.size(); ++i) {
            const auto &v_idx = boost::vertex(i, graph);
            for (const auto &j : out_[i]) {
                addEdge(v_idx, boost::vertex(j, graph));
            }
        }
    }

    /**
     * @brief Default constructor for the ComputationalDag class.
     */
    explicit ComputationalDag() : graph(0) {}

    unsigned int numberOfVertices() const { return boost::num_vertices(graph); }
    unsigned int numberOfEdges() const { return boost::num_edges(graph); }

    const GraphType &getGraph() const { return graph; }
    GraphType &getGraph() { return graph; }

    std::vector<VertexType> sourceVertices() const;
    std::vector<VertexType> sinkVertices() const;

    enum TOP_SORT_ORDER { AS_IT_COMES, MAX_CHILDREN, RANDOM };
    std::vector<VertexType> GetTopOrder(const TOP_SORT_ORDER q_order = AS_IT_COMES) const;

    std::vector<VertexType> dfs_topoOrder() const;
    std::vector<VertexType> dfs_reverse_topoOrder() const;
    std::vector<VertexType> GetFilteredTopOrder(const std::vector<bool> &valid) const;

    auto vertices() const { return boost::make_iterator_range(boost::vertices(graph)); }

    auto vertices() { return boost::make_iterator_range(boost::vertices(graph)); }

    auto parents(const VertexType &v) const {
        return boost::extensions::make_source_iterator_range(boost::inv_adjacent_vertices(v, graph));
    }

    auto parents(const VertexType &v) {
        return boost::extensions::make_source_iterator_range(boost::inv_adjacent_vertices(v, graph));
    }

    std::vector<VertexType> successors(const VertexType &v) const;

    std::vector<VertexType> ancestors(const VertexType &v) const;

    auto children(const VertexType &v) const {
        return boost::extensions::make_source_iterator_range(boost::adjacent_vertices(v, graph));
    }

    auto children(const VertexType &v) {
        return boost::extensions::make_source_iterator_range(boost::adjacent_vertices(v, graph));
    }

    auto edges() const { return boost::extensions::make_source_iterator_range(boost::edges(graph)); }

    auto edges() { return boost::extensions::make_source_iterator_range(boost::edges(graph)); }

    auto in_edges(const VertexType &v) const {
        return boost::extensions::make_source_iterator_range(boost::in_edges(v, graph));
    }

    auto in_edges(const VertexType &v) {
        return boost::extensions::make_source_iterator_range(boost::in_edges(v, graph));
    }

    auto out_edges(const VertexType &v) const {
        return boost::extensions::make_source_iterator_range(boost::out_edges(v, graph));
    }

    auto out_edges(const VertexType &v) {
        return boost::extensions::make_source_iterator_range(boost::out_edges(v, graph));
    }

    inline unsigned numberOfChildren(const VertexType &v) const { return boost::out_degree(v, graph); }
    inline unsigned numberOfParents(const VertexType &v) const { return boost::in_degree(v, graph); }

    inline bool isSink(const VertexType &v) const { return boost::out_degree(v, graph) == 0; }
    inline bool isSource(const VertexType &v) const { return boost::in_degree(v, graph) == 0; }

    VertexType source(const EdgeType &e) const { return boost::source(e, graph); }
    VertexType target(const EdgeType &e) const { return boost::target(e, graph); }

    Vertex operator[](const VertexType &v) const { return graph[v]; }
    Vertex &operator[](const VertexType &v) { return graph[v]; }

    Edge operator[](const EdgeType &e) const { return graph[e]; }
    Edge &operator[](const EdgeType &e) { return graph[e]; }

    int nodeWorkWeight(const VertexType &v) const { return (*this)[v].workWeight; }
    int nodeCommunicationWeight(const VertexType &v) const { return (*this)[v].communicationWeight; }
    int nodeMemoryWeight(const VertexType &v) const { return (*this)[v].memoryWeight; }
  
    int edgeCommunicationWeight(const EdgeType &e) const { return (*this)[e].communicationWeight; }

    template<typename VertexIterator>
    int sumOfVerticesWorkWeights(VertexIterator begin, VertexIterator end) const {
        return std::accumulate(begin, end, 0,
                               [this](const auto sum, const VertexType &v) { return sum + nodeWorkWeight(v); });
    }

    int sumOfVerticesWorkWeights(const std::initializer_list<VertexType> vertices_) const {
        return sumOfVerticesWorkWeights(vertices_.begin(), vertices_.end());
    }

    template<typename VertexIterator>
    int sumOfVerticesCommunicationWeights(VertexIterator begin, VertexIterator end) const {
        return std::accumulate(
            begin, end, 0, [this](const auto sum, const VertexType &v) { return sum + nodeCommunicationWeight(v); });
    }

    int sumOfVerticesCommunicationWeights(const std::initializer_list<VertexType> vertices_) const {
        return sumOfVerticesCommunicationWeights(vertices_.begin(), vertices_.end());
    }

    template<typename EdgeIterator>
    int sumOfEdgesCommunicationWeights(EdgeIterator begin, EdgeIterator end) const {
        return std::accumulate(begin, end, 0,
                               [this](const auto sum, const EdgeType &e) { return sum + edgeCommunicationWeight(e); });
    }

    int sumOfEdgesCommunicationWeights(const std::initializer_list<EdgeType> edges_) const {
        return sumOfEdgesCommunicationWeights(edges_.begin(), edges_.end());
    }

    void setNodeWorkWeight(const VertexType &v, const int work_weight) { graph[v].workWeight = work_weight; }

    void setNodeCommunicationWeight(const VertexType &v, const int comm_weight) {
        graph[v].communicationWeight = comm_weight;
    }

    void setEdgeCommunicationWeight(const EdgeType &e, const int comm_weight) {
        graph[e].communicationWeight = comm_weight;
    }

    VertexType addVertex(const int work_weight, const int comm_weight, const int memory_weight = 0) {
        return boost::add_vertex(Vertex{work_weight, comm_weight, memory_weight}, graph);
    }

    EdgeType addEdge(const VertexType &src, const VertexType &tar, int memory_weight = DEFAULT_EDGE_COMM_WEIGHT);

    void printGraph(std::ostream &os = std::cout) const;

    // computes bottom node distance
    std::vector<unsigned> get_bottom_node_distance() const;

    // computes top node distance
    std::vector<unsigned> get_top_node_distance() const;

    // Generates f:V \to Z such that (v,w) \in E \Rightarrow f(v) < f(w)
    // TODO: possibly make it less random? by either biasing a coinflip or make high chance to repeat the last
    std::vector<int> get_strict_poset_integer_map(unsigned const noise = 0, double const poisson_param = 0) const;

    // calculates number of edges in the longest path
    // returns 0 if no vertices
    size_t longestPath(const std::set<VertexType> &vertices) const;

    std::vector<VertexType> longestChain() const;

    /**
     * @brief Computes contracted graph along the partition without self-loops.
     *
     * @param partition parts of size one can but need not be included
     * @return std::pair<ComputationalDag, std::vector<unsigned>> Returns computational dag
     * and vertex mapping from old to new vertices
     */
    std::pair<ComputationalDag, std::unordered_map<VertexType, VertexType>>
    contracted_graph_without_loops(const std::vector<std::unordered_set<VertexType>> &partition) const;

    /**
     * @brief Computed Map returns true if and only if for the edge (u,v) there exists a path of length two from u to v
     * 
     * @return std::map<EdgeType, bool> 
     */
    std::map<EdgeType, bool> long_edges_in_triangles() const;

    /**
     * @brief Computes the average degree of the graph
     * 
     * @return double 
     */
    double average_degree() const;


};
