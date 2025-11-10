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

#include "osp/auxiliary/hash_util.hpp"
#include "computational_dag_vector_impl.hpp"
#include "edge_iterator.hpp"
#include "osp/graph_algorithms/computational_dag_construction_util.hpp"
#include <vector>

// #include "container_iterator_adaptor.hpp"

namespace osp {

struct directed_edge_descriptor_impl {

    std::size_t idx;

    std::size_t source;
    std::size_t target;

    directed_edge_descriptor_impl() : idx(0), source(0), target(0) {}
    directed_edge_descriptor_impl(const directed_edge_descriptor_impl &other) = default;
    directed_edge_descriptor_impl(directed_edge_descriptor_impl &&other) = default;
    directed_edge_descriptor_impl &operator=(const directed_edge_descriptor_impl &other) = default;
    directed_edge_descriptor_impl &operator=(directed_edge_descriptor_impl &&other) = default;
    directed_edge_descriptor_impl(std::size_t source, std::size_t target, std::size_t idx)
        : idx(idx), source(source), target(target) {}
    ~directed_edge_descriptor_impl() = default;

    bool operator==(const directed_edge_descriptor_impl &other) const {
        return idx == other.idx && source == other.source && target == other.target;
    }

    bool operator!=(const directed_edge_descriptor_impl &other) const { return !(*this == other); }
};

template<typename edge_comm_weight_t>
struct cdag_edge_impl {

    using cdag_edge_comm_weight_type = edge_comm_weight_t;

    cdag_edge_impl(edge_comm_weight_t comm_weight = 1) : comm_weight(comm_weight) {}
    edge_comm_weight_t comm_weight;
};

using cdag_edge_impl_int = cdag_edge_impl<int>;
using cdag_edge_impl_unsigned = cdag_edge_impl<unsigned>;

template<typename v_impl, typename e_impl>
class computational_dag_edge_idx_vector_impl {
  public:
    // graph_traits specialization
    using vertex_idx = std::size_t;
    using directed_edge_descriptor = directed_edge_descriptor_impl;

    using out_edges_iterator_t = std::vector<directed_edge_descriptor>::const_iterator;
    using in_edges_iterator_t = std::vector<directed_edge_descriptor>::const_iterator;

    // cdag_traits specialization
    using vertex_work_weight_type = typename v_impl::work_weight_type;
    using vertex_comm_weight_type = typename v_impl::comm_weight_type;
    using vertex_mem_weight_type = typename v_impl::mem_weight_type;
    using vertex_type_type = typename v_impl::cdag_vertex_type_type;
    using edge_comm_weight_type = typename e_impl::cdag_edge_comm_weight_type;

  private:
    using ThisT = computational_dag_edge_idx_vector_impl<v_impl, e_impl>;

    std::vector<v_impl> vertices_;
    std::vector<e_impl> edges_;

    unsigned num_vertex_types_ = 0;

    std::vector<std::vector<directed_edge_descriptor>> out_edges_;
    std::vector<std::vector<directed_edge_descriptor>> in_edges_;

    // struct cdag_edge_source_view {
    //     using value_type = vertex_idx;

    //     vertex_idx operator()(directed_edge_descriptor &p) const { return p.source; }
    // };

    // struct cdag_edge_target_view {
    //     vertex_idx &operator()(directed_edge_descriptor &p) const { return p.target; }
    //     const vertex_idx &operator()(directed_edge_descriptor const &p) const { return p.target; }
    // };

    // using edge_adapter_source_t = ContainerAdaptor<cdag_edge_source_view, const
    // std::vector<directed_edge_descriptor>>; using edge_adapter_target_t = ContainerAdaptor<cdag_edge_target_view,
    //  const std::vector<directed_edge_descriptor>>;

  public:
    computational_dag_edge_idx_vector_impl() = default;

    computational_dag_edge_idx_vector_impl(vertex_idx num_vertices)
        : vertices_(num_vertices), out_edges_(num_vertices), in_edges_(num_vertices) {
        for (vertex_idx i = 0; i < num_vertices; ++i) {
            vertices_[i].id = i;
        }
    }

    computational_dag_edge_idx_vector_impl(const computational_dag_edge_idx_vector_impl &other) = default;

    template<typename Graph_t>
    computational_dag_edge_idx_vector_impl(const Graph_t &other) {

        static_assert(is_computational_dag_v<Graph_t>, "Graph_t must satisfy the is_computation_dag concept");

        construct_computational_dag(other, *this);
    };

    computational_dag_edge_idx_vector_impl &operator=(const computational_dag_edge_idx_vector_impl &other) = default;

    computational_dag_edge_idx_vector_impl(computational_dag_edge_idx_vector_impl &&other)
        : vertices_(std::move(other.vertices_)), edges_(std::move(other.edges_)),
          num_vertex_types_(other.num_vertex_types_), out_edges_(std::move(other.out_edges_)),
          in_edges_(std::move(other.in_edges_)) {

        other.num_vertex_types_ = 0;
    }

    computational_dag_edge_idx_vector_impl &operator=(computational_dag_edge_idx_vector_impl &&other) {
        if (this != &other) {
            vertices_ = std::move(other.vertices_);
            edges_ = std::move(other.edges_);
            out_edges_ = std::move(other.out_edges_);
            in_edges_ = std::move(other.in_edges_);
            num_vertex_types_ = other.num_vertex_types_;
            other.num_vertex_types_ = 0;
        }
        return *this;
    }

    virtual ~computational_dag_edge_idx_vector_impl() = default;

    inline vertex_idx num_edges() const { return edges_.size(); }
    inline vertex_idx num_vertices() const { return vertices_.size(); }

    inline auto edges() const { return edge_range_vector_impl<ThisT>(*this); }

    inline auto parents(vertex_idx v) const { return edge_source_range(in_edges_[v], *this); }
    inline auto children(vertex_idx v) const { return edge_target_range(out_edges_[v], *this); }

    inline auto vertices() const { return vertex_range<vertex_idx>(static_cast<vertex_idx>(vertices_.size())); }

    inline const std::vector<directed_edge_descriptor> &in_edges(vertex_idx v) const { return in_edges_[v]; }
    inline const std::vector<directed_edge_descriptor> &out_edges(vertex_idx v) const { return out_edges_[v]; }

    inline vertex_idx in_degree(vertex_idx v) const { return in_edges_[v].size(); }
    inline vertex_idx out_degree(vertex_idx v) const { return out_edges_[v].size(); }

    inline edge_comm_weight_type edge_comm_weight(directed_edge_descriptor e) const {
        return edges_[e.idx].comm_weight;
    }

    inline vertex_work_weight_type vertex_work_weight(vertex_idx v) const { return vertices_[v].work_weight; }
    inline vertex_comm_weight_type vertex_comm_weight(vertex_idx v) const { return vertices_[v].comm_weight; }
    inline vertex_mem_weight_type vertex_mem_weight(vertex_idx v) const { return vertices_[v].mem_weight; }

    inline unsigned num_vertex_types() const { return num_vertex_types_; }
    inline vertex_type_type vertex_type(vertex_idx v) const { return vertices_[v].vertex_type; }

    inline vertex_idx source(const directed_edge_descriptor &e) const { return e.source; }
    inline vertex_idx target(const directed_edge_descriptor &e) const { return e.target; }

    vertex_idx add_vertex(vertex_work_weight_type work_weight, vertex_comm_weight_type comm_weight,
                          vertex_mem_weight_type mem_weight, vertex_type_type vertex_type = 0) {

        vertices_.emplace_back(vertices_.size(), work_weight, comm_weight, mem_weight, vertex_type);

        out_edges_.push_back({});
        in_edges_.push_back({});

        num_vertex_types_ = std::max(num_vertex_types_, vertex_type + 1);

        return vertices_.back().id;
    }

    std::pair<directed_edge_descriptor, bool> add_edge(vertex_idx source, vertex_idx target,
                                                       edge_comm_weight_type comm_weight = 1) {

        if (source == target) {
            return {directed_edge_descriptor{}, false};
        }

        if (source >= vertices_.size() || target >= vertices_.size()) {
            return {directed_edge_descriptor{}, false};
        }

        for (const auto edge : out_edges_[source]) {
            if (edge.target == target) {
                return {directed_edge_descriptor{}, false};
            }
        }

        out_edges_[source].emplace_back(source, target, edges_.size());
        in_edges_[target].emplace_back(source, target, edges_.size());

        edges_.emplace_back(comm_weight);

        return {out_edges_[source].back(), true};
    }

    inline void set_vertex_work_weight(vertex_idx v, vertex_work_weight_type work_weight) {
        vertices_[v].work_weight = work_weight;
    }
    inline void set_vertex_comm_weight(vertex_idx v, vertex_comm_weight_type comm_weight) {
        vertices_[v].comm_weight = comm_weight;
    }
    inline void set_vertex_mem_weight(vertex_idx v, vertex_mem_weight_type mem_weight) {
        vertices_[v].mem_weight = mem_weight;
    }
    inline void set_vertex_type(vertex_idx v, vertex_type_type vertex_type) {
        vertices_[v].vertex_type = vertex_type;
        num_vertex_types_ = std::max(num_vertex_types_, vertex_type + 1);
    }

    inline void set_edge_comm_weight(directed_edge_descriptor e, edge_comm_weight_type comm_weight) {
        edges_[e.idx].comm_weight = comm_weight;
    }

    inline const v_impl &get_vertex_impl(vertex_idx v) const { return vertices_[v]; }
    inline const e_impl &get_edge_impl(directed_edge_descriptor e) const { return edges_[e.idx]; }
};

template<typename v_impl, typename e_impl>
inline auto edges(const computational_dag_edge_idx_vector_impl<v_impl, e_impl> &graph) {
    return graph.edges();
}

template<typename v_impl, typename e_impl>
inline auto out_edges(vertex_idx_t<computational_dag_edge_idx_vector_impl<v_impl, e_impl>> v,
                      const computational_dag_edge_idx_vector_impl<v_impl, e_impl> &graph) {
    return graph.out_edges(v);
}

template<typename v_impl, typename e_impl>
inline auto in_edges(vertex_idx_t<computational_dag_edge_idx_vector_impl<v_impl, e_impl>> v,
                     const computational_dag_edge_idx_vector_impl<v_impl, e_impl> &graph) {
    return graph.in_edges(v);
}


// default implementation to get the source of an edge
template<typename v_impl, typename e_impl>
inline vertex_idx_t<computational_dag_edge_idx_vector_impl<v_impl, e_impl>> source(const edge_desc_t<computational_dag_edge_idx_vector_impl<v_impl, e_impl>> &edge, const computational_dag_edge_idx_vector_impl<v_impl, e_impl> &graph) {
    return graph.source(edge);
}

// default implementation to get the target of an edge
template<typename v_impl, typename e_impl>
inline vertex_idx_t<computational_dag_edge_idx_vector_impl<v_impl, e_impl>> target(const edge_desc_t<computational_dag_edge_idx_vector_impl<v_impl, e_impl>> &edge, const computational_dag_edge_idx_vector_impl<v_impl, e_impl> &graph) {
    return graph.target(edge);
}


// default template specialization
using computational_dag_edge_idx_vector_impl_def_t =
    computational_dag_edge_idx_vector_impl<cdag_vertex_impl_unsigned, cdag_edge_impl_unsigned>;

using computational_dag_edge_idx_vector_impl_def_int_t =
    computational_dag_edge_idx_vector_impl<cdag_vertex_impl_int, cdag_edge_impl_int>;


static_assert(is_directed_graph_edge_desc_v<computational_dag_edge_idx_vector_impl_def_t>,
              "computational_dag_edge_idx_vector_impl must satisfy the directed_graph_edge_desc concept");

static_assert(
    is_computational_dag_typed_vertices_edge_desc_v<computational_dag_edge_idx_vector_impl_def_t>,
    "computational_dag_edge_idx_vector_impl must satisfy the computation_dag_typed_vertices_edge_desc concept");

} // namespace osp

template<>
struct std::hash<osp::directed_edge_descriptor_impl> {
    std::size_t operator()(const osp::directed_edge_descriptor_impl &p) const noexcept {
        auto h1 = std::hash<std::size_t>{}(p.source);
        osp::hash_combine(h1, p.target);

        return h1;
    }
};