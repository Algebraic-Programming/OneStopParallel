#pragma once

#include <iterator>
#include <vector>

#include "concepts/directed_graph_concept.hpp"

namespace osp {

template<typename Graph_t>
class edge_range {

    const Graph_t &graph;

    struct edge_iterator {

        vertex_idx current_vertex;
        edge_idx current_edge_idx;
        std::vector<directed_edge_descriptor>::const_iterator current_edge;

        const Graph_t &graph;

      public:
        using iterator_category = std::input_iterator_tag;
        using value_type = directed_edge_descriptor;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type *;
        using reference = const value_type &;

        edge_iterator(const Graph_t &graph_) : current_vertex(0u), current_edge_idx(0u), graph(graph_) {

            while (current_vertex != graph.num_vertices()) {
                if (graph.out_edges(current_vertex).begin() != graph.out_edges(current_vertex).end()) {
                    current_edge = graph.out_edges(current_vertex).begin();
                    break;
                }
                current_vertex++;
            }
        }

        edge_iterator(edge_idx current_edge_idx_, const Graph_t &graph_) : current_vertex(0u), current_edge_idx(current_edge_idx_), graph(graph_) {

            if (current_edge_idx < graph.num_edges()) {

                edge_idx tmp = 0u;

                if (tmp < current_edge_idx) {

                    while (current_vertex != graph.num_vertices()) {

                        current_edge = graph.out_edges(current_vertex).begin();

                        while (current_edge != graph.out_edges(current_vertex).end()) {

                            if (tmp == current_edge_idx) {
                                break;
                            }

                            current_edge++;
                            tmp++;
                        }

                        current_vertex++;
                    }
                }

            } else {
                current_edge_idx = graph.num_edges();
                current_vertex = graph.num_vertices();
            }
        }

        const value_type &operator*() const { return *current_edge; }
        value_type *operator->() { return current_edge.operator->(); }

        // Prefix increment
        edge_iterator &operator++() {

            current_edge++;
            current_edge_idx++;

            if (current_edge == graph.out_edges(current_vertex).end()) {

                current_vertex++;

                while (current_vertex != graph.num_vertices()) {

                    if (graph.out_edges(current_vertex).begin() != graph.out_edges(current_vertex).end()) {
                        current_edge = graph.out_edges(current_vertex).begin();
                        break;
                    }

                    current_vertex++;
                }
            }

            return *this;
        }

        // Postfix increment
        edge_iterator operator++(int) {
            edge_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        friend bool operator==(const edge_iterator &one, const edge_iterator &other) {
            return one.current_edge_idx == other.current_edge_idx;
        };
        friend bool operator!=(const edge_iterator &one, const edge_iterator &other) {
            return one.current_edge_idx != other.current_edge_idx;
        };
    };

  public:
    edge_range(const Graph_t &graph_) : graph(graph_) {}

    auto begin() const { return edge_iterator(graph); }

    auto end() const { return edge_iterator(graph.num_edges(), graph); }
};


class edge_source_view {

    const std::vector<directed_edge_descriptor> &edges;

    struct source_iterator {

        std::vector<directed_edge_descriptor>::const_iterator current_edge;

      public:
        using iterator_category = std::input_iterator_tag;
        using value_type = vertex_idx;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type *;
        using reference = const value_type &;

        source_iterator(const std::vector<directed_edge_descriptor>::const_iterator &current_edge_) : current_edge(current_edge_) {} 

        const value_type &operator*() const { return (*current_edge).source; }
       
        // Prefix increment
        source_iterator &operator++() {
            current_edge++;
            return *this;
        }

        // Postfix increment
        source_iterator operator++(int) {
            source_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        friend bool operator==(const source_iterator &one, const source_iterator &other) {
            return one.current_edge == other.current_edge;
        }
        friend bool operator!=(const source_iterator &one, const source_iterator &other) {
            return one.current_edge != other.current_edge;
        };
    };

  public:
    edge_source_view(const std::vector<directed_edge_descriptor> &edges_) : edges(edges_) {}

    auto begin() const { return source_iterator(edges.begin()); }

    auto end() const { return source_iterator(edges.end()); }
};


class edge_target_view {

    const std::vector<directed_edge_descriptor> &edges;

    struct target_iterator {

        std::vector<directed_edge_descriptor>::const_iterator current_edge;

      public:
        using iterator_category = std::input_iterator_tag;
        using value_type = vertex_idx;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type *;
        using reference = const value_type &;

        target_iterator(const std::vector<directed_edge_descriptor>::const_iterator &current_edge_) : current_edge(current_edge_) {} 

        const value_type &operator*() const { return (*current_edge).target; }
       
        // Prefix increment
        target_iterator &operator++() {
            current_edge++;
            return *this;
        }

        // Postfix increment
        target_iterator operator++(int) {
            target_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        friend bool operator==(const target_iterator &one, const target_iterator &other) {
            return one.current_edge == other.current_edge;
        }
        
        friend bool operator!=(const target_iterator &one, const target_iterator &other) {
            return one.current_edge != other.current_edge;
        }
    };

  public:
    edge_target_view(const std::vector<directed_edge_descriptor> &edges_) : edges(edges_) {}

    auto begin() const { return target_iterator(edges.begin()); }

    auto end() const { return target_iterator(edges.end()); }
};

} // namespace osp