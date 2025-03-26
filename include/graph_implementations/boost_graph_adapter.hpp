// /*
// Copyright 2024 Huawei Technologies Co., Ltd.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// @author Toni Boehnlein, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner   
// */

// #pragma once

// #include "boost_extensions/inv_breadth_first_search.hpp"
// #include "boost_extensions/source_iterator_range.hpp"
// #include <numeric>
// #include <queue>
// #include <vector>
// #include <omp.h>

// #include <boost/graph/adjacency_list.hpp>
// #include <boost/graph/copy.hpp>
// #include <boost/graph/graph_traits.hpp>



// struct boost_vertex {

//     boost_vertex() : workWeight(0), communicationWeight(0), memoryWeight(0), nodeType(0) {}
//     boost_vertex(int workWeight_, int communicationWeight_, int memoryWeight_, unsigned nodeType_ = 0) : workWeight(workWeight_), communicationWeight(communicationWeight_), memoryWeight(memoryWeight_), nodeType(nodeType_) {}

//     int workWeight;
//     int communicationWeight;
//     int memoryWeight;
//     unsigned nodeType;

// };

// struct boost_edge {
//     boost_edge() : communicationWeight(0) {}
//     boost_edge(int communicationWeight_) : communicationWeight(communicationWeight_) {}
    
//     int communicationWeight;

// };

// using boost_graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, boost_vertex, boost_edge>;
// using boost_vertex_type = boost::graph_traits<GraphType>::vertex_descriptor;
// using boost_edge_type = boost::graph_traits<GraphType>::edge_descriptor;

// struct bost_edge_hash {

//     std::size_t operator()(const boost_edge &p) const {

//         auto h1 = std::hash<boost_vertex>{}(p.m_source);
//         hash_combine(h1, p.m_target);
//         hash_combine(h1, p.m_eproperty);

//         return h1;
//     }
// };

// /**
//  * @class ComputationalDag
//  * @brief Represents a computational directed acyclic graph (DAG).
//  *
//  * The ComputationalDag class is used to represent a computational DAG, which consists of vertices and edges.
//  * Each vertex represents a computational task, and each edge represents a communication dependency between tasks.
//  * The class provides various methods to manipulate and analyze the DAG, such as adding vertices and edges,
//  * calculating the longest path, and retrieving topological order of vertices.
//  */



// class boost_graph_adapter {
//   private:
//     boost_graph graph;

//     unsigned number_of_vertex_types;

//   public:

//   boost_graph_adapter(const std::vector<std::vector<int>> &out_, const std::vector<int> &workW_,
//                      const std::vector<int> &commW_,
//                      const std::unordered_map<std::pair<int, int>, int, pair_hash> &comm_edge_W) : number_of_vertex_types(0) {
//         graph.m_vertices.reserve(out_.size());

//         assert(out_.size() == workW_.size());
//         assert(out_.size() == commW_.size());

//         for (size_t i = 0; i < out_.size(); ++i) {
//             addVertex(workW_[i], commW_[i]);
//         }
//         for (size_t i = 0; i < out_.size(); ++i) {
//             const auto &v_idx = boost::vertex(i, graph);
//             for (const auto &j : out_[i]) {
//                 assert(comm_edge_W.find(std::make_pair(i, j)) != comm_edge_W.cend());
//                 addEdge(v_idx, boost::vertex(j, graph), comm_edge_W.at(std::make_pair(i, j)));
//             }
//         }
//         updateNumberOfNodeTypes();
//     }


//     boost_graph_adapter(const std::vector<std::vector<int>> &out_, const std::vector<int> &workW_,
//                      const std::vector<int> &commW_) : number_of_vertex_types(0) {
//         graph.m_vertices.reserve(out_.size());

//         assert(out_.size() == workW_.size());
//         assert(out_.size() == commW_.size());

//         for (size_t i = 0; i < out_.size(); ++i) {
//             addVertex(workW_[i], commW_[i]);
//         }
//         for (size_t i = 0; i < out_.size(); ++i) {
//             const auto &v_idx = boost::vertex(i, graph);
//             for (const auto &j : out_[i]) {
//                 addEdge(v_idx, boost::vertex(j, graph));
//             }
//         }
//         updateNumberOfNodeTypes();
//     }

//     boost_graph_adapter(const std::vector<std::vector<int>> &out_, const std::vector<int> &workW_,
//                      const std::vector<int> &commW_, const std::vector<unsigned> &nodeType_) : number_of_vertex_types(0) {
//         graph.m_vertices.reserve(out_.size());

//         assert(out_.size() == workW_.size());
//         assert(out_.size() == commW_.size());
//         assert(out_.size() == nodeType_.size());

//         for (size_t i = 0; i < out_.size(); ++i) {
//             addVertex(workW_[i], commW_[i], 0, nodeType_[i]);
//         }
//         for (size_t i = 0; i < out_.size(); ++i) {
//             const auto &v_idx = boost::vertex(i, graph);
//             for (const auto &j : out_[i]) {
//                 addEdge(v_idx, boost::vertex(j, graph));
//             }
//         }
//         updateNumberOfNodeTypes();
//     }

//     /**
//      * @brief Default constructor for the ComputationalDag class.
//      */
//     explicit boost_graph_adapter() : graph(0), number_of_vertex_types(0) {}
//     explicit boost_graph_adapter(unsigned number_of_nodes) : graph(number_of_nodes), number_of_vertex_types(0) { updateNumberOfNodeTypes(); }

//     const boost_graph &get_graph() const { return graph; }
//     boost_graph &get_graph() { return graph; }

//     size_t num_vertices() const { return boost::num_vertices(graph); }
//     size_t num_edges() const { return boost::num_edges(graph); }

//     auto vertices() const { return boost::make_iterator_range(boost::vertices(graph)); }
//     auto vertices() { return boost::make_iterator_range(boost::vertices(graph)); }

//     auto parents(const vertex_idx &v) const {
//         return boost::extensions::make_source_iterator_range(boost::inv_adjacent_vertices(v, graph));
//     }

//     auto parents(const vertex_idx &v) {
//         return boost::extensions::make_source_iterator_range(boost::inv_adjacent_vertices(v, graph));
//     }

//     auto children(const vertex_idx &v) const {
//         return boost::extensions::make_source_iterator_range(boost::adjacent_vertices(v, graph));
//     }

//     auto children(const vertex_idx &v) {
//         return boost::extensions::make_source_iterator_range(boost::adjacent_vertices(v, graph));
//     }

//     auto edges() const { return boost::extensions::make_source_iterator_range(boost::edges(graph)); }

//     auto edges() { return boost::extensions::make_source_iterator_range(boost::edges(graph)); }

//     auto in_edges(const vertex_idx &v) const {
//         return boost::extensions::make_source_iterator_range(boost::in_edges(v, graph));
//     }

//     auto in_edges(const vertex_idx &v) {
//         return boost::extensions::make_source_iterator_range(boost::in_edges(v, graph));
//     }

//     auto out_edges(const vertex_idx &v) const {
//         return boost::extensions::make_source_iterator_range(boost::out_edges(v, graph));
//     }

//     auto out_edges(const vertex_idx &v) {
//         return boost::extensions::make_source_iterator_range(boost::out_edges(v, graph));
//     }

//     inline unsigned out_degree(const vertex_idx &v) const { return boost::out_degree(v, graph); }
//     inline unsigned in_degree(const vertex_idx &v) const { return boost::in_degree(v, graph); }

//     int vertex_work_weight(const vertex_idx &v) const { return (*this)[v].workWeight; }
//     int vertex_comm_weight(const vertex_idx &v) const { return (*this)[v].communicationWeight; }
//     int vertex_mem_weight(const vertex_idx &v) const { return (*this)[v].memoryWeight; }
//     unsigned vertex_type(const vertex_idx &v) const { return (*this)[v].nodeType; }
  
//     int edge_com_weight(const EdgeType &e) const { return (*this)[e].communicationWeight; }

//     void setNodeMemoryWeight(const VertexType &v, const int memory_weight) { graph[v].memoryWeight = memory_weight; }
//     void setNodeWorkWeight(const VertexType &v, const int work_weight) { graph[v].workWeight = work_weight; }
//     void setNodeType(const VertexType &v, const unsigned node_type) {
//         graph[v].nodeType = node_type;
//         number_of_vertex_types = std::max(number_of_vertex_types, node_type + 1);
//     }

//     void setNodeCommunicationWeight(const VertexType &v, const int comm_weight) {
//         graph[v].communicationWeight = comm_weight;
//     }
//     void setEdgeCommunicationWeight(const EdgeType &e, const int comm_weight) {
//         graph[e].communicationWeight = comm_weight;
//     }

//     VertexType addVertex(const int work_weight, const int comm_weight, const int memory_weight = 0, const unsigned node_type = 0) {
//         number_of_vertex_types = std::max(number_of_vertex_types, node_type + 1);
//         return boost::add_vertex(Vertex{work_weight, comm_weight, memory_weight, node_type}, graph);
//     }

//     EdgeType addEdge(const VertexType &src, const VertexType &tar, int memory_weight = DEFAULT_EDGE_COMM_WEIGHT);

//     EdgeType addEdge(const VertexType &src, const VertexType &tar, double val, int memory_weight = DEFAULT_EDGE_COMM_WEIGHT);

//     void printGraph(std::ostream &os = std::cout) const;

// };
