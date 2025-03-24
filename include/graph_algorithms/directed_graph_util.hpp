#pragma once

#include <queue>
#include <unordered_set>
#include <vector>

#include "concepts/directed_graph_concept.hpp"

namespace osp {

// Function to get source vertices
template<typename Graph_t>
std::vector<vertex_idx> source_vertices(Graph_t &graph) {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    std::vector<vertex_idx> vec;

    for (const vertex_idx v_idx : graph.vertices()) {
        if (graph.in_degree(v_idx) == 0) {
            vec.push_back(v_idx);
        }
    }

    return vec;
}

// Function to get sink vertices
template<typename Graph_t>
std::vector<vertex_idx> sink_vertices(Graph_t &graph) {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    std::vector<vertex_idx> vec;

    for (const vertex_idx v_idx : graph.vertices()) {
        if (graph.out_degree(v_idx) == 0) {
            vec.push_back(v_idx);
        }
    }

    return vec;
}

template<typename Graph_t>
bool has_path(const vertex_idx src, const vertex_idx dest, const Graph_t &graph) {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    std::unordered_set<vertex_idx> visited;
    visited.emplace(src);

    std::queue<vertex_idx> next;
    next.push(src);

    while (!next.empty()) {
        vertex_idx v = next.front();
        next.pop();

        for (const vertex_idx &child : graph.children(v)) {

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

template<typename Graph_t>
class source_vertices_iterator {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    using iterator_category = std::input_iterator_tag;
    using value_type = vertex_idx;

    vertex_idx current_idx;

    Graph_t &graph;

    source_vertices_iterator(vertex_idx idx, Graph_t &graph_) : current_idx(idx), graph(graph_) {};

  public:
    source_vertices_iterator(Graph_t &graph_) : current_idx(graph_.num_vertices()), graph(graph_) {

        for (const vertex_idx v_idx : graph.vertices()) {
            if (graph.in_degree(v_idx) == 0) {
                current_idx = v_idx;
                break;
            }
        }
    };

    ~source_vertices_iterator() = default;

    const value_type &operator*() const { return current_idx; }
    value_type *operator->() { return &current_idx; }

    // Prefix increment
    source_vertices_iterator &operator++() {

        vertex_idx v_idx = current_idx + 1;
        while (v_idx < graph.num_vertices()) {
            if (graph.in_degree(v_idx) == 0) {
                current_idx = v_idx;
                break;
            }
            v_idx++;
        }

        if (graph.num_vertices() == v_idx) {
            current_idx = graph.num_vertices();
        }

        return *this;
    }

    // Postfix increment
    source_vertices_iterator operator++(int) {
        source_vertices_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    source_vertices_iterator begin() { return source_vertices_iterator(graph); }
    source_vertices_iterator end() { return source_vertices_iterator(graph.num_vertices(), graph); }

    friend bool operator==(const source_vertices_iterator &one, const source_vertices_iterator &other) {
        return one.current_idx == other.current_idx;
    };
    friend bool operator!=(const source_vertices_iterator &one, const source_vertices_iterator &other) {
        return one.current_idx != other.current_idx;
    };
};

} // namespace osp

// // Function to get sink vertices
// template<typename Graph_t, typename = std::enable_if_t<is_undirected_graph_v<Graph_t>>>
// std::vector<vertex_idx> sink_vertices(const Graph_t& graph) {
//     std::vector<vertex_idx> vec;

//     // some additiona stuff for undirected

//     for (const vertex_idx v_idx : graph.vertices()) {
//         if (graph.out_degree(v_idx) == 0) {
//             vec.push_back(v_idx);
//         }
//     }

//     return vec;
// }
