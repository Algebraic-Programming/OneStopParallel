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

#include "model/ComputationalDag.hpp"

#include <iostream>
#include <stdexcept>

#include "boost_extensions/inv_breadth_first_search.hpp"
#include <boost/graph/topological_sort.hpp>

#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/reverse_graph.hpp>

#include <vector>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/named_function_params.hpp>

namespace bfs_visitors {
struct bfs_visitor_successors : boost::default_bfs_visitor {
    std::vector<VertexType> &visited;

    explicit bfs_visitor_successors(std::vector<VertexType> &visited_) : visited(visited_) {}

    template<typename V, typename G>
    void discover_vertex(const V &v, const G &) const {
        visited.emplace_back(v);
    }
};

struct bfs_visitor_ancestors : boost::default_bfs_visitor {
    std::vector<VertexType> &visited;

    explicit bfs_visitor_ancestors(std::vector<VertexType> &visited_) : visited(visited_) {}

    template<typename V, typename G>
    void discover_vertex(const V &v, const G &) const {
        visited.emplace_back(v);
    }

    template<typename E, typename G>
    void examine_edge(const E &e, const G &g) const {
        visited.emplace_back(boost::source(e, g));
    }
};

} // namespace bfs_visitors

void ComputationalDag::printGraph(std::ostream &os) const {
    for (const auto &v : vertices()) {
        os << "[" << v << "]{work=" << nodeWorkWeight(v) << ", comm=" << nodeCommunicationWeight(v) << ", type=" << nodeType(v) << "}\n";
        for (const auto &e : out_edges(v)) {
            const auto &vc = target(e);
            os << "  +--(comm=" << edgeCommunicationWeight(e) << ")--> [" << vc << "]{work=" << nodeWorkWeight(vc)
               << ", comm=" << nodeCommunicationWeight(vc) << ", type=" << nodeType(vc) << "}\n";
        }
    }
}

EdgeType ComputationalDag::addEdge(const VertexType &src, const VertexType &tar, const int memory_weight) {
    const auto [edge, valid] = boost::add_edge(src, tar, {memory_weight}, graph);
    if (not valid) {
        throw std::invalid_argument("Adding Edge was not sucessful");
    }
    assert(graph[edge].communicationWeight == memory_weight);
    number_of_vertex_types = std::max(number_of_vertex_types, 1u); // in case adding edges adds vertices
    return edge;
}

EdgeType ComputationalDag::addEdge(const VertexType &src, const VertexType &tar, double val, const int memory_weight) {
    const auto [edge, valid] = boost::add_edge(src, tar, {memory_weight, val}, graph);
    if (not valid) {
        throw std::invalid_argument("Adding Edge was not sucessful");
    }
    assert(graph[edge].communicationWeight == memory_weight);
    number_of_vertex_types = std::max(number_of_vertex_types, 1u); // in case adding edges adds vertices
    return edge;
}

std::vector<VertexType> ComputationalDag::sourceVertices() const {
    std::vector<VertexType> vec;

    for (auto vp = boost::vertices(graph); vp.first != vp.second; ++vp.first) {
        if (boost::in_degree(*vp.first, graph) == 0) {
            vec.push_back(*vp.first);
        }
    }

    return vec;
}

std::vector<VertexType> ComputationalDag::sinkVertices() const {
    std::vector<VertexType> vec;

    for (auto vp = boost::vertices(graph); vp.first != vp.second; ++vp.first) {
        if (boost::out_degree(*vp.first, graph) == 0) {
            vec.push_back(*vp.first);
        }
    }

    return vec;
}

std::vector<VertexType> ComputationalDag::dfs_topoOrder() const {

    std::vector<VertexType> vec = dfs_reverse_topoOrder();
    std::reverse(vec.begin(), vec.end());

    return vec;
}

std::vector<VertexType> ComputationalDag::dfs_reverse_topoOrder() const {
    std::vector<VertexType> vec;

    std::back_insert_iterator result(vec);
    typedef boost::topo_sort_visitor<decltype(result)> TopoVisitor;
    boost::depth_first_search(graph, boost::visitor(TopoVisitor(result)));

    return vec;
}

std::vector<VertexType> ComputationalDag::successors(const VertexType &v) const {
    std::vector<VertexType> visited_vertices;
    const bfs_visitors::bfs_visitor_successors visitor(visited_vertices);
    boost::breadth_first_search(graph, v, boost::visitor(visitor));
    return visited_vertices;
}

/**
 * TODO: Replace by a simple bfs_visitor using discover_vertex function
 */
std::vector<VertexType> ComputationalDag::GetTopOrder(const TOP_SORT_ORDER q_order) const {
    std::vector<VertexType> predecessors_count(numberOfVertices(), 0);
    std::vector<VertexType> TopOrder;

    if (q_order == AS_IT_COMES) {
        std::queue<VertexType> next;

        // Find source nodes
        for (const VertexType &i : sourceVertices())
            next.push(i);

        // Execute BFS
        while (!next.empty()) {
            const VertexType node = next.front();
            next.pop();
            TopOrder.push_back(node);

            for (const VertexType &current : children(node)) {
                ++predecessors_count[current];
                if (predecessors_count[current] == parents(current).size())
                    next.push(current);
            }
        }
    }

    if (q_order == MAX_CHILDREN) {
        const auto q_cmp = [](const std::pair<VertexType, size_t> &left, const std::pair<VertexType, size_t> &right) {
            return (left.second < right.second) || ((left.second < right.second) && (left.first < right.first));
        };
        std::priority_queue<std::pair<VertexType, size_t>, std::vector<std::pair<VertexType, size_t>>, decltype(q_cmp)>
            next(q_cmp);

        // Find source nodes
        for (const VertexType &i : sourceVertices())
            next.emplace(i, children(i).size());

        // Execute BFS
        while (!next.empty()) {
            const auto [node, n_chldrn] = next.top();
            next.pop();
            TopOrder.push_back(node);

            for (const VertexType &current : children(node)) {
                ++predecessors_count[current];
                if (predecessors_count[current] == parents(current).size())
                    next.emplace(current, children(current).size());
            }
        }
    }

    if (q_order == RANDOM) {
        std::vector<VertexType> next;

        // Find source nodes
        for (const VertexType &i : sourceVertices())
            next.push_back(i);

        // Execute BFS
        while (!next.empty()) {
            auto node_it = next.begin();
            std::advance(node_it, randInt(next.size()));
            const VertexType node = *node_it;
            next.erase(node_it);
            TopOrder.push_back(node);

            for (const VertexType &current : children(node)) {
                ++predecessors_count[current];
                if (predecessors_count[current] == parents(current).size())
                    next.push_back(current);
            }
        }
    }

    if (TopOrder.size() != numberOfVertices())
        throw std::runtime_error("Error during topological ordering: TopOrder.size() != numberOfVertices() [" +
                                 std::to_string(TopOrder.size()) + " != " + std::to_string(numberOfVertices()) + "]");

    return TopOrder;
};

std::vector<VertexType> ComputationalDag::ancestors(const VertexType &v) const {
    std::vector<VertexType> visited_vertices;
    const bfs_visitors::bfs_visitor_ancestors visitor(visited_vertices);
    boost::extensions::inv_breadth_first_search(graph, v, visitor);
    return visited_vertices;
}

size_t ComputationalDag::longestPath(const std::set<VertexType> &vertices) const {
    std::queue<VertexType> bfs_queue;
    std::map<VertexType, size_t> distances, in_degrees, visit_counter;

    // Find source nodes
    for (const VertexType &node : vertices) {
        unsigned indeg = 0;
        for (const VertexType &parent : parents(node))
            if (vertices.count(parent) == 1)
                ++indeg;

        if (indeg == 0) {
            bfs_queue.push(node);
            distances[node] = 0;
        }
        in_degrees[node] = indeg;
        visit_counter[node] = 0;
    }

    // Execute BFS
    while (!bfs_queue.empty()) {
        const VertexType current = bfs_queue.front();
        bfs_queue.pop();

        for (const VertexType &child : children(current)) {
            if (vertices.count(child) == 0)
                continue;

            ++visit_counter[child];
            if (visit_counter[child] == in_degrees[child]) {
                bfs_queue.push(child);
                distances[child] = distances[current] + 1;
            }
        }
    }

    return std::accumulate(vertices.cbegin(), vertices.cend(), 0,
                           [&](const size_t mx, const VertexType &node) { return std::max(mx, distances[node]); });
}

size_t ComputationalDag::longestPath() const {
    size_t max_edgecount = 0;
    std::queue<VertexType> bfs_queue;
    std::vector<VertexType> distances(numberOfVertices()), in_degrees(numberOfVertices()),
        visit_counter(numberOfVertices());

    // Find source nodes
    for (VertexType node = 0; node < numberOfVertices(); node++) {
        unsigned indeg = 0;
        for (const VertexType &parent : parents(node))
            ++indeg;

        if (indeg == 0) {
            bfs_queue.push(node);
            distances[node] = 0;
        }
        in_degrees[node] = indeg;
        visit_counter[node] = 0;
    }

    // Execute BFS
    while (!bfs_queue.empty()) {
        const VertexType current = bfs_queue.front();
        bfs_queue.pop();

        for (const VertexType &child : children(current)) {

            ++visit_counter[child];
            if (visit_counter[child] == in_degrees[child]) {
                bfs_queue.push(child);
                distances[child] = distances[current] + 1;
                max_edgecount = std::max(max_edgecount, distances[child]);
            }
        }
    }

    return max_edgecount;
}

std::vector<VertexType> ComputationalDag::longestChain() const {

    std::vector<VertexType> chain;
    if (numberOfVertices() == 0) {
        return chain;
    }

    std::vector<unsigned> top_length(numberOfVertices(), 0);
    unsigned running_longest_chain = 0;

    VertexType end_longest_chain = 0;

    // calculating lenght of longest path
    for (const VertexType &node : GetTopOrder()) {

        unsigned max_temp = 0;
        for (const auto &parent : parents(node)) {
            max_temp = std::max(max_temp, top_length[parent]);
        }

        top_length[node] = max_temp + 1;
        if (top_length[node] > running_longest_chain) {
            end_longest_chain = node;
            running_longest_chain = top_length[node];
        }
    }

    // reconstructing longest path
    chain.push_back(end_longest_chain);
    while (numberOfParents(end_longest_chain) != 0) {

        for (const VertexType &in_node : parents(end_longest_chain)) {
            if (top_length[in_node] != top_length[end_longest_chain] - 1) {
                continue;
            }

            end_longest_chain = in_node;
            chain.push_back(end_longest_chain);
            break;
        }
    }

    std::reverse(chain.begin(), chain.end());
    return chain;
}


int ComputationalDag::critical_path_weight() const {

    if (numberOfVertices() == 0) {
        return 0;
    }

    std::vector<int> top_length(numberOfVertices(), 0);
    int critical_path_weight = 0;

    // calculating lenght of longest path
    for (const VertexType &node : GetTopOrder()) {

        int max_temp = 0;
        for (const auto &parent : parents(node)) {
            max_temp = std::max(max_temp, top_length[parent]);
        }

        top_length[node] = max_temp + nodeWorkWeight(node);

        if (top_length[node] > critical_path_weight) {
            
            critical_path_weight = top_length[node];
        }
    }

    return critical_path_weight;
}

bool ComputationalDag::has_path(VertexType src, VertexType dest) const {

    std::unordered_set<VertexType> visited;
    visited.emplace(src);

    std::queue<VertexType> next;
    next.push(src);

    while (!next.empty()) {
        VertexType v = next.front();
        next.pop();

        for (const VertexType &child : children(v)) {

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

std::pair<ComputationalDag, std::unordered_map<VertexType, VertexType>>
ComputationalDag::contracted_graph_without_loops(const std::vector<std::unordered_set<VertexType>> &partition) const {
    ComputationalDag contr_graph;
    std::unordered_map<VertexType, VertexType> contraction_map;
    contraction_map.reserve(numberOfVertices());

    // Allocating vertices
    std::unordered_map<VertexType, bool> allocated_into_new_graph;
    allocated_into_new_graph.reserve(numberOfVertices());
    // TODO should this be allocated_into_new_graph.reserve(numberOfVertices(), false);
    for (auto node : vertices()) {
        allocated_into_new_graph[node] = false;
    }

    for (auto &part : partition) {
        int part_ww = 0;
        int part_cw = 0;
        for (const VertexType &node : part) {
            part_ww += nodeWorkWeight(node);
            part_cw += nodeCommunicationWeight(node);
        }

        VertexType part_node = contr_graph.addVertex(part_ww, part_cw);

        for (const VertexType &node : part) {
            contraction_map[node] = part_node;
            allocated_into_new_graph[node] = true;
        }
    }
    for (auto node : vertices()) {
        if (allocated_into_new_graph[node])
            continue;
        contraction_map[node] = contr_graph.addVertex(nodeWorkWeight(node), nodeCommunicationWeight(node));
        allocated_into_new_graph[node] = true;
    }
    assert(std::all_of(allocated_into_new_graph.begin(), allocated_into_new_graph.end(),
                       [](const auto &val) { return val.second; }));

    // Allocating edges
    std::unordered_map<std::pair<VertexType, VertexType>, int, pair_hash> part_to_part_commw;
    for (const EdgeType &edge : edges()) {
        if (contraction_map[edge.m_source] == contraction_map[edge.m_target])
            continue;
        std::pair<VertexType, VertexType> edge_pair =
            std::make_pair(contraction_map[edge.m_source], contraction_map[edge.m_target]);
        if (part_to_part_commw.find(edge_pair) == part_to_part_commw.cend()) {
            part_to_part_commw[edge_pair] = edgeCommunicationWeight(edge);
        } else {
            part_to_part_commw[edge_pair] += edgeCommunicationWeight(edge);
        }
    }
    for (auto &[edge_new, cw] : part_to_part_commw) {
        contr_graph.addEdge(edge_new.first, edge_new.second, cw);
    }

    return std::make_pair(contr_graph, contraction_map);
}

std::vector<VertexType> ComputationalDag::GetFilteredTopOrder(const std::vector<bool> &valid) const {
    std::vector<VertexType> filteredOrder;
    for (const auto &node : GetTopOrder())
        if (valid[node])
            filteredOrder.push_back(node);

    return filteredOrder;
}

std::vector<int> ComputationalDag::get_strict_poset_integer_map(unsigned const noise,
                                                                double const poisson_param) const {

    std::vector<VertexType> top_order = GetTopOrder();

    Repeat_Chance repeater_coin;

    std::unordered_map<EdgeType, bool, EdgeType_hash> up_or_down;

    for (const auto &edge : edges()) {
        bool val = repeater_coin.get_flip();
        up_or_down.emplace(edge, val);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::poisson_distribution<> poisson_gen(poisson_param);

    std::vector<unsigned> top_distance = get_top_node_distance();
    std::vector<unsigned> bot_distance = get_bottom_node_distance();
    std::vector<int> new_top(numberOfVertices(), 0);
    std::vector<int> new_bot(numberOfVertices(), 0);

    unsigned max_path = 0;
    // for (unsigned i = 0; i<n ; i++) {
    for (const auto &vertex : vertices()) {
        max_path = std::max(max_path, top_distance[vertex]);
    }

    for (auto &source : sourceVertices()) {
        new_top[source] = randInt(max_path - bot_distance[source] + 1 + 2 * noise) - noise;
    }

    for (auto &sink : sinkVertices()) {
        new_bot[sink] = randInt(max_path - top_distance[sink] + 1 + 2 * noise) - noise;
    }

    for (const auto &vertex : top_order) {
        // for (int i = 0; i< top_order.size(); i++) {

        if (isSource(vertex))
            continue;
        // if (In[top_order[i]].empty()) continue;

        int max_temp = INT_MIN;

        for (const auto &edge : in_edges(vertex)) {

            int temp = new_top[source(edge)];
            // int temp = new_top[edge.m_source];
            if (up_or_down.at(edge)) {
                temp += 1 + poisson_gen(gen);
            }
            max_temp = std::max(max_temp, temp);
        }
        new_top[vertex] = max_temp;
    }

    for (std::reverse_iterator iter = top_order.crbegin(); iter != top_order.crend(); ++iter) {

        if (isSink(*iter))
            continue;

        int max_temp = INT_MIN;

        for (const auto &edge : out_edges(*iter)) {
            int temp = new_bot[target(edge)];
            if (!up_or_down.at(edge)) {
                temp += 1 + poisson_gen(gen);
            }
            max_temp = std::max(max_temp, temp);
        }
        new_bot[*iter] = max_temp;
    }

    std::vector<int> output(numberOfVertices());
    for (unsigned i = 0; i < numberOfVertices(); i++) {
        output[i] = new_top[i] - new_bot[i];
    }
    return output;
}

// computes bottom node distance
std::vector<unsigned> ComputationalDag::get_bottom_node_distance() const {

    std::vector<unsigned> bottom_distance(numberOfVertices(), 0);

    for (const auto &vertex : dfs_reverse_topoOrder()) {
        // for (int i = top_order.size() - 1; i >= 0; i--) {

        unsigned max_temp = 0;
        for (const auto &j : children(vertex)) {
            // for (auto &j : Out[vertex]) {
            max_temp = std::max(max_temp, bottom_distance[j]);
        }
        bottom_distance[vertex] = ++max_temp;
    }
    return bottom_distance;
}

// computes top node distance
std::vector<unsigned> ComputationalDag::get_top_node_distance() const {

    std::vector<unsigned> top_distance(numberOfVertices(), 0);
    // std::vector<int> top_order = GetTopOrder();
    // for (int i = 0; i < top_order.size(); i++) {
    for (const auto &vertex : GetTopOrder()) {
        unsigned max_temp = 0;
        for (const auto &j : parents(vertex)) {
            // for (auto &j : In[top_order[i]]) {
            max_temp = std::max(max_temp, top_distance[j]);
        }
        top_distance[vertex] = ++max_temp;
    }
    return top_distance;
}

std::unordered_set<EdgeType, EdgeType_hash> ComputationalDag::long_edges_in_triangles() const {

    std::unordered_set<EdgeType, EdgeType_hash> deleted_edges;

    for (const auto &vertex : graph.vertex_set()) {

        std::unordered_set<VertexType> children_set;

        for (const auto &v : children(vertex)) {
            children_set.emplace(v);
        }

        for (const auto &edge : boost::extensions::make_source_iterator_range(boost::out_edges(vertex, graph))) {

            const auto &child = boost::target(edge, graph);

            for (const auto &parent :
                 boost::extensions::make_source_iterator_range(boost::inv_adjacent_vertices(child, graph))) {

                // const auto &pair = boost::edge(vertex, parent, graph);

                if (children_set.find(parent) != children_set.cend()) {
                    deleted_edges.emplace(edge);
                    break;
                }
            }
        }
    }

    return deleted_edges;
}

std::unordered_set<EdgeType, EdgeType_hash> ComputationalDag::long_edges_in_triangles_parallel() const {

    std::unordered_set<EdgeType, EdgeType_hash> deleted_edges;
    std::vector<std::vector<EdgeType>> deleted_edges_thread(omp_get_max_threads());

    // std::cout << "Computing longest edges in triangle with number of threads: " << omp_get_max_threads() <<
    // std::endl;

#pragma omp parallel for schedule(dynamic, 4)
    for (const auto &vertex : graph.vertex_set()) {

        const unsigned int proc = omp_get_thread_num();
        std::unordered_set<VertexType> children_set;

        for (const auto &v : children(vertex)) {
            children_set.emplace(v);
        }

        for (const auto &edge : boost::extensions::make_source_iterator_range(boost::out_edges(vertex, graph))) {

            const auto &child = boost::target(edge, graph);

            for (const auto &parent :
                 boost::extensions::make_source_iterator_range(boost::inv_adjacent_vertices(child, graph))) {

                // const auto &pair = boost::edge(vertex, parent, graph);

                if (children_set.find(parent) != children_set.cend()) {

                    deleted_edges_thread[proc].emplace_back(edge);
                    break;
                }
            }
        }
    }

    size_t sum_sizes = 0;
    for (unsigned i = 0; i < deleted_edges_thread.size(); i++) {
        sum_sizes += deleted_edges_thread[i].size();
    }
    deleted_edges.reserve(sum_sizes);

    for (unsigned i = 0; i < deleted_edges_thread.size(); i++) {
        for (const auto &edge : deleted_edges_thread[i]) {
            deleted_edges.emplace(edge);
        }
    }

    return deleted_edges;
}

double ComputationalDag::average_degree() const {

    return static_cast<double>(numberOfEdges()) / static_cast<double>(numberOfVertices());
}

int ComputationalDag::get_max_memory_weight() const {
    int max_memory_weight = 0;

    for (const auto &node : vertices()) {
        max_memory_weight = std::max(max_memory_weight, nodeMemoryWeight(node));
    }
    return max_memory_weight;
}

int ComputationalDag::get_max_memory_weight(unsigned nodeType_) const {
    int max_memory_weight = 0;

    for (const auto &node : vertices()) {
        if (nodeType(node) == nodeType_) {
            max_memory_weight = std::max(max_memory_weight, nodeMemoryWeight(node));
        }
    }
    return max_memory_weight;
}

void ComputationalDag::updateNumberOfNodeTypes() {
    number_of_vertex_types = 0;
    for (unsigned node = 0; node < numberOfVertices(); node++) {
        if(nodeType(node) >= number_of_vertex_types) {
            number_of_vertex_types = nodeType(node) + 1;
        }
    }
}