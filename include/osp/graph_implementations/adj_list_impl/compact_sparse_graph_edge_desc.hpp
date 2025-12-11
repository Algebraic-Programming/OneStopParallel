/*
Copyright 2025 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Christos Matzoros, Pal Andras Papp, Raphael S. Steiner
*/
#pragma once

#include "osp/concepts/directed_graph_edge_desc_concept.hpp"
#include "osp/graph_implementations/adj_list_impl/compact_sparse_graph.hpp"

namespace osp {

template <bool keep_vertex_order,
          bool use_work_weights = false,
          bool use_comm_weights = false,
          bool use_mem_weights = false,
          bool use_edge_comm_weights = false,
          bool use_vert_types = false,
          typename vert_t = std::size_t,
          typename edge_t = std::size_t,
          typename work_weight_type = unsigned,
          typename comm_weight_type = unsigned,
          typename mem_weight_type = unsigned,
          typename e_comm_weight_type = unsigned,
          typename vertex_type_template_type = unsigned>
class Compact_Sparse_Graph_EdgeDesc : public Compact_Sparse_Graph<keep_vertex_order,
                                                                  use_work_weights,
                                                                  use_comm_weights,
                                                                  use_mem_weights,
                                                                  use_vert_types,
                                                                  vert_t,
                                                                  edge_t,
                                                                  work_weight_type,
                                                                  comm_weight_type,
                                                                  mem_weight_type,
                                                                  vertex_type_template_type> {
  private:
    using ThisT = Compact_Sparse_Graph_EdgeDesc<keep_vertex_order,
                                                use_work_weights,
                                                use_comm_weights,
                                                use_mem_weights,
                                                use_edge_comm_weights,
                                                use_vert_types,
                                                vert_t,
                                                edge_t,
                                                work_weight_type,
                                                comm_weight_type,
                                                mem_weight_type,
                                                e_comm_weight_type,
                                                vertex_type_template_type>;
    using BaseT = Compact_Sparse_Graph<keep_vertex_order,
                                       use_work_weights,
                                       use_comm_weights,
                                       use_mem_weights,
                                       use_vert_types,
                                       vert_t,
                                       edge_t,
                                       work_weight_type,
                                       comm_weight_type,
                                       mem_weight_type,
                                       vertex_type_template_type>;

  public:
    using vertex_idx = typename BaseT::vertex_idx;

    using vertex_work_weight_type = typename BaseT::vertex_work_weight_type;
    using vertex_comm_weight_type = typename BaseT::vertex_comm_weight_type;
    using vertex_mem_weight_type = typename BaseT::vertex_mem_weight_type;
    using vertex_type_type = typename BaseT::vertex_type_type;

    using directed_edge_descriptor = edge_t;
    using edge_comm_weight_type = e_comm_weight_type;

  protected:
    std::vector<edge_comm_weight_type> edge_comm_weights;

    class In_Edges_range {
      private:
        const vertex_idx tgt_vert;
        const typename BaseT::Compact_Parent_Edges::Parent_range par_range;
        const typename BaseT::Compact_Children_Edges &csc_out_edges;

        class In_Edges_iterator {
          public:
            using iterator_category = std::bidirectional_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = edge_t;
            using pointer = vertex_idx *;
            using reference = edge_t &;

          private:
            const vertex_idx target_vert;
            const typename BaseT::Compact_Children_Edges &csc_out_edges;

            typename std::vector<vertex_idx>::const_iterator current;

          public:
            In_Edges_iterator(const vertex_idx &target_vert_,
                              const typename BaseT::Compact_Children_Edges &csc_out_edges_,
                              const typename std::vector<vertex_idx>::const_iterator start_)
                : target_vert(target_vert_), csc_out_edges(csc_out_edges_), current(start_) {};
            In_Edges_iterator(const In_Edges_iterator &other)
                : target_vert(other.target_vert), csc_out_edges(other.csc_out_edges), current(other.current) {};

            In_Edges_iterator &operator=(const In_Edges_iterator &other) {
                if (this != &other) {
                    target_vert = other.target_vert;
                    csc_out_edges = other.csc_out_edges;
                    current = other.current;
                }
                return *this;
            };

            inline value_type operator*() const {
                const vertex_idx src_vert = *current;
                typename BaseT::Compact_Children_Edges::Children_range range = csc_out_edges.children(src_vert);

                assert(std::binary_search(range.cbegin(), range.cend(), target_vert));
                auto it = std::lower_bound(range.cbegin(), range.cend(), target_vert);

                edge_t diff = static_cast<edge_t>(std::distance(range.cbegin(), it));
                edge_t edge_desc_val = csc_out_edges.children_indx_begin(src_vert) + diff;

                return edge_desc_val;
            };

            inline In_Edges_iterator &operator++() {
                ++current;
                return *this;
            };

            inline In_Edges_iterator operator++(int) {
                In_Edges_iterator temp = *this;
                ++(*this);
                return temp;
            };

            inline In_Edges_iterator &operator--() {
                --current;
                return *this;
            };

            inline In_Edges_iterator operator--(int) {
                In_Edges_iterator temp = *this;
                --(*this);
                return temp;
            };

            inline bool operator==(const In_Edges_iterator &other) const { return current == other.current; };

            inline bool operator!=(const In_Edges_iterator &other) const { return !(*this == other); };

            inline bool operator<=(const In_Edges_iterator &other) const { return current <= other.current; };

            inline bool operator<(const In_Edges_iterator &other) const { return (*this <= other) && (*this != other); };

            inline bool operator>=(const In_Edges_iterator &other) const { return (!(*this <= other)) || (*this == other); };

            inline bool operator>(const In_Edges_iterator &other) const { return !(*this <= other); };
        };

      public:
        In_Edges_range() = default;
        In_Edges_range(const In_Edges_range &other) = default;
        In_Edges_range(In_Edges_range &&other) = default;
        In_Edges_range &operator=(const In_Edges_range &other) = default;
        In_Edges_range &operator=(In_Edges_range &&other) = default;
        virtual ~In_Edges_range() = default;

        In_Edges_range(const vertex_idx &tgt_vert_, const ThisT &graph, const typename BaseT::Compact_Children_Edges &csc_out_edges_)
            : tgt_vert(tgt_vert_), par_range(graph.parents(tgt_vert_)), csc_out_edges(csc_out_edges_) {};

        inline auto cbegin() const { return In_Edges_iterator(tgt_vert, csc_out_edges, par_range.cbegin()); };

        inline auto cend() const { return In_Edges_iterator(tgt_vert, csc_out_edges, par_range.cend()); };

        inline auto begin() const { return cbegin(); };

        inline auto end() const { return cend(); };
    };

  public:
    Compact_Sparse_Graph_EdgeDesc() = default;
    Compact_Sparse_Graph_EdgeDesc(const Compact_Sparse_Graph_EdgeDesc &other) = default;
    Compact_Sparse_Graph_EdgeDesc(Compact_Sparse_Graph_EdgeDesc &&other) = default;
    Compact_Sparse_Graph_EdgeDesc &operator=(const Compact_Sparse_Graph_EdgeDesc &other) = default;
    Compact_Sparse_Graph_EdgeDesc &operator=(Compact_Sparse_Graph_EdgeDesc &&other) = default;
    virtual ~Compact_Sparse_Graph_EdgeDesc() = default;

    template <typename edge_list_type>
    Compact_Sparse_Graph_EdgeDesc(vertex_idx num_vertices_, const edge_list_type &edges) : BaseT(num_vertices_, edges) {
        if constexpr (use_edge_comm_weights) {
            edge_comm_weights = std::vector<edge_comm_weight_type>(BaseT::num_edges(), static_cast<edge_comm_weight_type>(0));
        }
    }

    template <typename edge_list_type>
    Compact_Sparse_Graph_EdgeDesc(vertex_idx num_vertices_,
                                  const edge_list_type &edges,
                                  const std::vector<vertex_work_weight_type> &ww)
        : BaseT(num_vertices_, edges, ww) {
        if constexpr (use_edge_comm_weights) {
            edge_comm_weights = std::vector<edge_comm_weight_type>(BaseT::num_edges(), static_cast<edge_comm_weight_type>(0));
        }
    }

    template <typename edge_list_type>
    Compact_Sparse_Graph_EdgeDesc(vertex_idx num_vertices_,
                                  const edge_list_type &edges,
                                  const std::vector<vertex_work_weight_type> &&ww)
        : BaseT(num_vertices_, edges, std::move(ww)) {
        if constexpr (use_edge_comm_weights) {
            edge_comm_weights = std::vector<edge_comm_weight_type>(BaseT::num_edges(), static_cast<edge_comm_weight_type>(0));
        }
    }

    template <typename edge_list_type>
    Compact_Sparse_Graph_EdgeDesc(vertex_idx num_vertices_,
                                  const edge_list_type &edges,
                                  std::vector<vertex_work_weight_type> &ww,
                                  std::vector<vertex_comm_weight_type> &cw)
        : BaseT(num_vertices_, edges, ww, cw) {
        if constexpr (use_edge_comm_weights) {
            edge_comm_weights = std::vector<edge_comm_weight_type>(BaseT::num_edges(), static_cast<edge_comm_weight_type>(0));
        }
    }

    template <typename edge_list_type>
    Compact_Sparse_Graph_EdgeDesc(vertex_idx num_vertices_,
                                  const edge_list_type &edges,
                                  std::vector<vertex_work_weight_type> &&ww,
                                  std::vector<vertex_comm_weight_type> &&cw)
        : BaseT(num_vertices_, edges, std::move(ww), std::move(cw)) {
        if constexpr (use_edge_comm_weights) {
            edge_comm_weights = std::vector<edge_comm_weight_type>(BaseT::num_edges(), static_cast<edge_comm_weight_type>(0));
        }
    }

    template <typename edge_list_type>
    Compact_Sparse_Graph_EdgeDesc(vertex_idx num_vertices_,
                                  const edge_list_type &edges,
                                  const std::vector<vertex_work_weight_type> &ww,
                                  const std::vector<vertex_comm_weight_type> &cw,
                                  const std::vector<vertex_mem_weight_type> &mw)
        : BaseT(num_vertices_, edges, ww, cw, mw) {
        if constexpr (use_edge_comm_weights) {
            edge_comm_weights = std::vector<edge_comm_weight_type>(BaseT::num_edges(), static_cast<edge_comm_weight_type>(0));
        }
    }

    template <typename edge_list_type>
    Compact_Sparse_Graph_EdgeDesc(vertex_idx num_vertices_,
                                  const edge_list_type &edges,
                                  const std::vector<vertex_work_weight_type> &&ww,
                                  const std::vector<vertex_comm_weight_type> &&cw,
                                  const std::vector<vertex_mem_weight_type> &&mw)
        : BaseT(num_vertices_, edges, std::move(ww), std::move(cw), std::move(mw)) {
        if constexpr (use_edge_comm_weights) {
            edge_comm_weights = std::vector<edge_comm_weight_type>(BaseT::num_edges(), static_cast<edge_comm_weight_type>(0));
        }
    }

    template <typename edge_list_type>
    Compact_Sparse_Graph_EdgeDesc(vertex_idx num_vertices_,
                                  const edge_list_type &edges,
                                  const std::vector<vertex_work_weight_type> &ww,
                                  const std::vector<vertex_comm_weight_type> &cw,
                                  const std::vector<vertex_mem_weight_type> &mw,
                                  const std::vector<vertex_type_type> &vt)
        : BaseT(num_vertices_, edges, ww, cw, mw, vt) {
        if constexpr (use_edge_comm_weights) {
            edge_comm_weights = std::vector<edge_comm_weight_type>(BaseT::num_edges(), static_cast<edge_comm_weight_type>(0));
        }
    }

    template <typename edge_list_type>
    Compact_Sparse_Graph_EdgeDesc(vertex_idx num_vertices_,
                                  const edge_list_type &edges,
                                  const std::vector<vertex_work_weight_type> &&ww,
                                  const std::vector<vertex_comm_weight_type> &&cw,
                                  const std::vector<vertex_mem_weight_type> &&mw,
                                  const std::vector<vertex_type_type> &&vt)
        : BaseT(num_vertices_, edges, std::move(ww), std::move(cw), std::move(mw), std::move(vt)) {
        if constexpr (use_edge_comm_weights) {
            edge_comm_weights = std::vector<edge_comm_weight_type>(BaseT::num_edges(), static_cast<edge_comm_weight_type>(0));
        }
    }

    template <typename Graph_type>
    Compact_Sparse_Graph_EdgeDesc(const Graph_type &graph) : BaseT(graph) {
        if constexpr (use_edge_comm_weights) {
            edge_comm_weights = std::vector<edge_comm_weight_type>(BaseT::num_edges(), static_cast<edge_comm_weight_type>(0));
        }

        if constexpr (has_edge_weights_v<Graph_type> && use_edge_comm_weights) {
            for (const auto &edge : edges(graph)) {
                const auto src = source(edge, graph);
                const auto tgt = target(edge, graph);
                set_edge_comm_weight(src, tgt, graph.edge_comm_weight(edge));
            }
        }
    }

    inline auto edges() const { return integral_range<directed_edge_descriptor>(BaseT::number_of_edges); };

    inline directed_edge_descriptor edge(const vertex_idx &src, const vertex_idx &tgt) const {
        typename BaseT::Compact_Children_Edges::Children_range range = BaseT::csc_out_edges.children(src);

        assert(std::binary_search(range.cbegin(), range.cend(), tgt));
        auto it = std::lower_bound(range.cbegin(), range.cend(), tgt);

        directed_edge_descriptor diff = static_cast<directed_edge_descriptor>(std::distance(range.cbegin(), it));
        directed_edge_descriptor edge_desc_val = BaseT::csc_out_edges.children_indx_begin(src) + diff;

        return edge_desc_val;
    };

    inline vertex_idx source(const directed_edge_descriptor &edge) const { return BaseT::csc_out_edges.source(edge); };

    inline vertex_idx target(const directed_edge_descriptor &edge) const { return BaseT::csc_out_edges.target(edge); };

    inline auto out_edges(const vertex_idx &vert) const {
        return integral_range<directed_edge_descriptor>(BaseT::csc_out_edges.children_indx_begin(vert),
                                                        BaseT::csc_out_edges.children_indx_begin(vert + 1));
    };

    inline auto in_edges(const vertex_idx &vert) const { return In_Edges_range(vert, *this, BaseT::csc_out_edges); };

    template <typename RetT = edge_comm_weight_type>
    inline std::enable_if_t<use_edge_comm_weights, RetT> edge_comm_weight(const directed_edge_descriptor &edge) const {
        return edge_comm_weights[edge];
    }

    template <typename RetT = edge_comm_weight_type>
    inline std::enable_if_t<not use_edge_comm_weights, RetT> edge_comm_weight(const directed_edge_descriptor &edge) const {
        return static_cast<RetT>(1);
    }

    template <typename RetT = void>
    inline std::enable_if_t<use_edge_comm_weights, RetT> set_edge_comm_weight(const vertex_idx &src,
                                                                              const vertex_idx &tgt,
                                                                              const edge_comm_weight_type e_comm_weight) {
        if constexpr (keep_vertex_order) {
            edge_comm_weights[edge(src, tgt)] = e_comm_weight;
        } else {
            const vertex_idx internal_src = BaseT::vertex_permutation_from_original_to_internal[src];
            const vertex_idx internal_tgt = BaseT::vertex_permutation_from_original_to_internal[tgt];
            edge_comm_weights[edge(internal_src, internal_tgt)] = e_comm_weight;
        }
    }

    template <typename RetT = void>
    inline std::enable_if_t<not use_edge_comm_weights, RetT> set_edge_comm_weight(const vertex_idx &src,
                                                                                  const vertex_idx &tgt,
                                                                                  const edge_comm_weight_type e_comm_weight) {
        static_assert(use_edge_comm_weights, "To set edge communication weight, graph type must allow edge communication weights.");
    }
};

template <bool keep_vertex_order,
          bool use_work_weights,
          bool use_comm_weights,
          bool use_mem_weights,
          bool use_edge_comm_weights,
          bool use_vert_types,
          typename vert_t,
          typename edge_t,
          typename work_weight_type,
          typename comm_weight_type,
          typename mem_weight_type,
          typename e_comm_weight_type,
          typename vertex_type_template_type>
inline auto edges(const Compact_Sparse_Graph_EdgeDesc<keep_vertex_order,
                                                      use_work_weights,
                                                      use_comm_weights,
                                                      use_mem_weights,
                                                      use_edge_comm_weights,
                                                      use_vert_types,
                                                      vert_t,
                                                      edge_t,
                                                      work_weight_type,
                                                      comm_weight_type,
                                                      mem_weight_type,
                                                      e_comm_weight_type,
                                                      vertex_type_template_type> &graph) {
    return graph.edges();
}

template <bool keep_vertex_order,
          bool use_work_weights,
          bool use_comm_weights,
          bool use_mem_weights,
          bool use_edge_comm_weights,
          bool use_vert_types,
          typename vert_t,
          typename edge_t,
          typename work_weight_type,
          typename comm_weight_type,
          typename mem_weight_type,
          typename e_comm_weight_type,
          typename vertex_type_template_type>
inline auto out_edges(vertex_idx_t<Compact_Sparse_Graph_EdgeDesc<keep_vertex_order,
                                                                 use_work_weights,
                                                                 use_comm_weights,
                                                                 use_mem_weights,
                                                                 use_edge_comm_weights,
                                                                 use_vert_types,
                                                                 vert_t,
                                                                 edge_t,
                                                                 work_weight_type,
                                                                 comm_weight_type,
                                                                 mem_weight_type,
                                                                 e_comm_weight_type,
                                                                 vertex_type_template_type>> v,
                      const Compact_Sparse_Graph_EdgeDesc<keep_vertex_order,
                                                          use_work_weights,
                                                          use_comm_weights,
                                                          use_mem_weights,
                                                          use_edge_comm_weights,
                                                          use_vert_types,
                                                          vert_t,
                                                          edge_t,
                                                          work_weight_type,
                                                          comm_weight_type,
                                                          mem_weight_type,
                                                          e_comm_weight_type,
                                                          vertex_type_template_type> &graph) {
    return graph.out_edges(v);
}

template <bool keep_vertex_order,
          bool use_work_weights,
          bool use_comm_weights,
          bool use_mem_weights,
          bool use_edge_comm_weights,
          bool use_vert_types,
          typename vert_t,
          typename edge_t,
          typename work_weight_type,
          typename comm_weight_type,
          typename mem_weight_type,
          typename e_comm_weight_type,
          typename vertex_type_template_type>
inline auto in_edges(vertex_idx_t<Compact_Sparse_Graph_EdgeDesc<keep_vertex_order,
                                                                use_work_weights,
                                                                use_comm_weights,
                                                                use_mem_weights,
                                                                use_edge_comm_weights,
                                                                use_vert_types,
                                                                vert_t,
                                                                edge_t,
                                                                work_weight_type,
                                                                comm_weight_type,
                                                                mem_weight_type,
                                                                e_comm_weight_type,
                                                                vertex_type_template_type>> v,
                     const Compact_Sparse_Graph_EdgeDesc<keep_vertex_order,
                                                         use_work_weights,
                                                         use_comm_weights,
                                                         use_mem_weights,
                                                         use_edge_comm_weights,
                                                         use_vert_types,
                                                         vert_t,
                                                         edge_t,
                                                         work_weight_type,
                                                         comm_weight_type,
                                                         mem_weight_type,
                                                         e_comm_weight_type,
                                                         vertex_type_template_type> &graph) {
    return graph.in_edges(v);
}

template <bool keep_vertex_order,
          bool use_work_weights,
          bool use_comm_weights,
          bool use_mem_weights,
          bool use_edge_comm_weights,
          bool use_vert_types,
          typename vert_t,
          typename edge_t,
          typename work_weight_type,
          typename comm_weight_type,
          typename mem_weight_type,
          typename e_comm_weight_type,
          typename vertex_type_template_type>
inline vertex_idx_t<Compact_Sparse_Graph_EdgeDesc<keep_vertex_order,
                                                  use_work_weights,
                                                  use_comm_weights,
                                                  use_mem_weights,
                                                  use_edge_comm_weights,
                                                  use_vert_types,
                                                  vert_t,
                                                  edge_t,
                                                  work_weight_type,
                                                  comm_weight_type,
                                                  mem_weight_type,
                                                  e_comm_weight_type,
                                                  vertex_type_template_type>>
source(const edge_desc_t<Compact_Sparse_Graph_EdgeDesc<keep_vertex_order,
                                                       use_work_weights,
                                                       use_comm_weights,
                                                       use_mem_weights,
                                                       use_edge_comm_weights,
                                                       use_vert_types,
                                                       vert_t,
                                                       edge_t,
                                                       work_weight_type,
                                                       comm_weight_type,
                                                       mem_weight_type,
                                                       e_comm_weight_type,
                                                       vertex_type_template_type>> &edge,
       const Compact_Sparse_Graph_EdgeDesc<keep_vertex_order,
                                           use_work_weights,
                                           use_comm_weights,
                                           use_mem_weights,
                                           use_edge_comm_weights,
                                           use_vert_types,
                                           vert_t,
                                           edge_t,
                                           work_weight_type,
                                           comm_weight_type,
                                           mem_weight_type,
                                           e_comm_weight_type,
                                           vertex_type_template_type> &graph) {
    return graph.source(edge);
}

template <bool keep_vertex_order,
          bool use_work_weights,
          bool use_comm_weights,
          bool use_mem_weights,
          bool use_edge_comm_weights,
          bool use_vert_types,
          typename vert_t,
          typename edge_t,
          typename work_weight_type,
          typename comm_weight_type,
          typename mem_weight_type,
          typename e_comm_weight_type,
          typename vertex_type_template_type>
inline vertex_idx_t<Compact_Sparse_Graph_EdgeDesc<keep_vertex_order,
                                                  use_work_weights,
                                                  use_comm_weights,
                                                  use_mem_weights,
                                                  use_edge_comm_weights,
                                                  use_vert_types,
                                                  vert_t,
                                                  edge_t,
                                                  work_weight_type,
                                                  comm_weight_type,
                                                  mem_weight_type,
                                                  e_comm_weight_type,
                                                  vertex_type_template_type>>
target(const edge_desc_t<Compact_Sparse_Graph_EdgeDesc<keep_vertex_order,
                                                       use_work_weights,
                                                       use_comm_weights,
                                                       use_mem_weights,
                                                       use_edge_comm_weights,
                                                       use_vert_types,
                                                       vert_t,
                                                       edge_t,
                                                       work_weight_type,
                                                       comm_weight_type,
                                                       mem_weight_type,
                                                       e_comm_weight_type,
                                                       vertex_type_template_type>> &edge,
       const Compact_Sparse_Graph_EdgeDesc<keep_vertex_order,
                                           use_work_weights,
                                           use_comm_weights,
                                           use_mem_weights,
                                           use_edge_comm_weights,
                                           use_vert_types,
                                           vert_t,
                                           edge_t,
                                           work_weight_type,
                                           comm_weight_type,
                                           mem_weight_type,
                                           e_comm_weight_type,
                                           vertex_type_template_type> &graph) {
    return graph.target(edge);
}

template <bool keep_vertex_order,
          bool use_work_weights,
          bool use_comm_weights,
          bool use_mem_weights,
          bool use_edge_comm_weights,
          bool use_vert_types,
          typename vert_t,
          typename edge_t,
          typename work_weight_type,
          typename comm_weight_type,
          typename mem_weight_type,
          typename e_comm_weight_type,
          typename vertex_type_template_type>
struct is_Compact_Sparse_Graph<Compact_Sparse_Graph_EdgeDesc<keep_vertex_order,
                                                             use_work_weights,
                                                             use_comm_weights,
                                                             use_mem_weights,
                                                             use_edge_comm_weights,
                                                             use_vert_types,
                                                             vert_t,
                                                             edge_t,
                                                             work_weight_type,
                                                             comm_weight_type,
                                                             mem_weight_type,
                                                             e_comm_weight_type,
                                                             vertex_type_template_type>,
                               void> : std::true_type {};

template <bool use_work_weights,
          bool use_comm_weights,
          bool use_mem_weights,
          bool use_edge_comm_weights,
          bool use_vert_types,
          typename vert_t,
          typename edge_t,
          typename work_weight_type,
          typename comm_weight_type,
          typename mem_weight_type,
          typename e_comm_weight_type,
          typename vertex_type_template_type>
struct is_Compact_Sparse_Graph_reorder<Compact_Sparse_Graph_EdgeDesc<false,
                                                                     use_work_weights,
                                                                     use_comm_weights,
                                                                     use_mem_weights,
                                                                     use_edge_comm_weights,
                                                                     use_vert_types,
                                                                     vert_t,
                                                                     edge_t,
                                                                     work_weight_type,
                                                                     comm_weight_type,
                                                                     mem_weight_type,
                                                                     e_comm_weight_type,
                                                                     vertex_type_template_type>,
                                       void> : std::true_type {};

static_assert(is_Compact_Sparse_Graph_v<Compact_Sparse_Graph_EdgeDesc<true>>);
static_assert(is_Compact_Sparse_Graph_v<Compact_Sparse_Graph_EdgeDesc<false>>);
static_assert(!is_Compact_Sparse_Graph_reorder_v<Compact_Sparse_Graph_EdgeDesc<true>>);
static_assert(is_Compact_Sparse_Graph_reorder_v<Compact_Sparse_Graph_EdgeDesc<false>>);

static_assert(has_vertex_weights_v<Compact_Sparse_Graph_EdgeDesc<true, true>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the has_vertex_weights concept");

static_assert(has_vertex_weights_v<Compact_Sparse_Graph_EdgeDesc<false, true>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the has_vertex_weights concept");

static_assert(is_directed_graph_v<Compact_Sparse_Graph_EdgeDesc<false, false, false, false, false>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the directed_graph concept");

static_assert(is_directed_graph_v<Compact_Sparse_Graph_EdgeDesc<false, true, true, true, true>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the directed_graph concept");

static_assert(is_directed_graph_v<Compact_Sparse_Graph_EdgeDesc<true, false, false, false, false>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the directed_graph concept");

static_assert(is_directed_graph_v<Compact_Sparse_Graph_EdgeDesc<true, true, true, true, true>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the directed_graph concept");

static_assert(is_computational_dag_v<Compact_Sparse_Graph_EdgeDesc<false, true, true, true, false>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the is_computation_dag concept");

static_assert(is_computational_dag_v<Compact_Sparse_Graph_EdgeDesc<true, true, true, true, false>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the is_computation_dag concept");

static_assert(is_computational_dag_typed_vertices_v<Compact_Sparse_Graph_EdgeDesc<false, true, true, true, true, true>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the is_computation_dag with types concept");

static_assert(is_computational_dag_typed_vertices_v<Compact_Sparse_Graph_EdgeDesc<true, true, true, true, true, true>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the is_computation_dag with types concept");

static_assert(is_directed_graph_edge_desc_v<Compact_Sparse_Graph_EdgeDesc<true>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the directed graph edge descriptor concept.");

static_assert(is_directed_graph_edge_desc_v<Compact_Sparse_Graph_EdgeDesc<false>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the directed graph edge descriptor concept.");

static_assert(
    is_computational_dag_typed_vertices_edge_desc_v<Compact_Sparse_Graph_EdgeDesc<false, true, true, true, true, true>>,
    "Compact_Sparse_Graph_EdgeDesc must satisfy the is_computational_dag_typed_vertices_edge_desc_v with types concept");

static_assert(
    is_computational_dag_typed_vertices_edge_desc_v<Compact_Sparse_Graph_EdgeDesc<true, true, true, true, true, true>>,
    "Compact_Sparse_Graph_EdgeDesc must satisfy the is_computational_dag_typed_vertices_edge_desc_v with types concept");

static_assert(has_edge_weights_v<Compact_Sparse_Graph_EdgeDesc<false, true, true, true, true, true>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the has_edge_weights concept");

static_assert(has_edge_weights_v<Compact_Sparse_Graph_EdgeDesc<true, true, true, true, true, true>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the has_edge_weights concept");

static_assert(has_hashable_edge_desc_v<Compact_Sparse_Graph_EdgeDesc<true, true>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the has_hashable_edge_desc concept");

static_assert(has_hashable_edge_desc_v<Compact_Sparse_Graph_EdgeDesc<false, true>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the has_hashable_edge_desc concept");

using CSGE
    = Compact_Sparse_Graph_EdgeDesc<false, true, true, true, true, true, std::size_t, std::size_t, unsigned, unsigned, unsigned, unsigned, unsigned>;

}    // namespace osp
