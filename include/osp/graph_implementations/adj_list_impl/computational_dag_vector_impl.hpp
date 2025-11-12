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

#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/concepts/directed_graph_edge_desc_concept.hpp"
#include "osp/graph_algorithms/computational_dag_construction_util.hpp"
#include "osp/graph_implementations/vertex_iterator.hpp"
#include <vector>

namespace osp {

template<typename vertex_idx_t, typename workw_t, typename commw_t, typename memw_t, typename vertex_type_t>
struct cdag_vertex_impl {

    using vertex_idx_type = vertex_idx_t;
    using work_weight_type = workw_t;
    using comm_weight_type = commw_t;
    using mem_weight_type = memw_t;
    using cdag_vertex_type_type = vertex_type_t;

    cdag_vertex_impl() : id(0), work_weight(0), comm_weight(0), mem_weight(0), vertex_type(0) {}

    cdag_vertex_impl(const cdag_vertex_impl &other) = default;
    cdag_vertex_impl(cdag_vertex_impl &&other) = default;
    cdag_vertex_impl &operator=(const cdag_vertex_impl &other) = default;
    cdag_vertex_impl &operator=(cdag_vertex_impl &&other) = default;

    cdag_vertex_impl(vertex_idx_t vertex_idx_, workw_t work_w, commw_t comm_w, memw_t mem_w,
                     vertex_type_t vertex_t)
        : id(vertex_idx_), work_weight(work_w), comm_weight(comm_w), mem_weight(mem_w),
          vertex_type(vertex_t) {}

    vertex_idx_t id;

    workw_t work_weight;
    commw_t comm_weight;
    memw_t mem_weight;

    vertex_type_t vertex_type;
};

using cdag_vertex_impl_int = cdag_vertex_impl<size_t, int, int, int, unsigned>;
using cdag_vertex_impl_unsigned = cdag_vertex_impl<size_t, unsigned, unsigned, unsigned, unsigned>;

template<typename v_impl>
class computational_dag_vector_impl {
  public:
    using vertex_idx = typename v_impl::vertex_idx_type;

    using vertex_work_weight_type = typename v_impl::work_weight_type;
    using vertex_comm_weight_type = typename v_impl::comm_weight_type;
    using vertex_mem_weight_type = typename v_impl::mem_weight_type;
    using vertex_type_type = typename v_impl::cdag_vertex_type_type;

    computational_dag_vector_impl() = default;

    computational_dag_vector_impl(vertex_idx num_vertices)
        : vertices_(num_vertices), out_neigbors(num_vertices), in_neigbors(num_vertices), num_edges_(0),
          num_vertex_types_(0) {

        for (vertex_idx i = 0; i < num_vertices; ++i) {
            vertices_[i].id = i;
        }
    }

    computational_dag_vector_impl(const computational_dag_vector_impl &other) = default;
    computational_dag_vector_impl &operator=(const computational_dag_vector_impl &other) = default;

    template<typename Graph_t>
    computational_dag_vector_impl(const Graph_t &other) {

        static_assert(is_computational_dag_v<Graph_t>, "Graph_t must satisfy the is_computation_dag concept");

        construct_computational_dag(other, *this);
    }

    computational_dag_vector_impl(computational_dag_vector_impl &&other)
        : vertices_(std::move(other.vertices_)), out_neigbors(std::move(other.out_neigbors)),
          in_neigbors(std::move(other.in_neigbors)), num_edges_(other.num_edges_),
          num_vertex_types_(other.num_vertex_types_) {

        other.num_edges_ = 0;
        other.num_vertex_types_ = 0;
    };

    computational_dag_vector_impl &operator=(computational_dag_vector_impl &&other) {
        if (this != &other) {
            vertices_ = std::move(other.vertices_);
            out_neigbors = std::move(other.out_neigbors);
            in_neigbors = std::move(other.in_neigbors);
            num_edges_ = other.num_edges_;
            num_vertex_types_ = other.num_vertex_types_;

            other.num_edges_ = 0;
            other.num_vertex_types_ = 0;
        }
        return *this;
    }

    virtual ~computational_dag_vector_impl() = default;

    inline auto vertices() const { return vertex_range<vertex_idx>(static_cast<vertex_idx>(vertices_.size())); }

    inline vertex_idx num_vertices() const { return static_cast<vertex_idx>(vertices_.size()); }

    inline vertex_idx num_edges() const { return num_edges_; }

    inline const std::vector<vertex_idx> &parents(const vertex_idx v) const { return in_neigbors[v]; }

    inline const std::vector<vertex_idx> &children(const vertex_idx v) const { return out_neigbors[v]; }

    inline vertex_idx in_degree(const vertex_idx v) const { return static_cast<vertex_idx>(in_neigbors[v].size()); }

    inline vertex_idx out_degree(const vertex_idx v) const { return static_cast<vertex_idx>(out_neigbors[v].size()); }

    inline vertex_work_weight_type vertex_work_weight(const vertex_idx v) const { return vertices_[v].work_weight; }

    inline vertex_comm_weight_type vertex_comm_weight(const vertex_idx v) const { return vertices_[v].comm_weight; }

    inline vertex_mem_weight_type vertex_mem_weight(const vertex_idx v) const { return vertices_[v].mem_weight; }

    inline vertex_type_type vertex_type(const vertex_idx v) const { return vertices_[v].vertex_type; }

    inline vertex_type_type num_vertex_types() const { return num_vertex_types_; }

    inline const v_impl &get_vertex_impl(const vertex_idx v) const { return vertices_[v]; }

    vertex_idx add_vertex(vertex_work_weight_type work_weight, vertex_comm_weight_type comm_weight,
                          vertex_mem_weight_type mem_weight, vertex_type_type vertex_type = 0) {

        vertices_.emplace_back(vertices_.size(), work_weight, comm_weight, mem_weight, vertex_type);
        out_neigbors.push_back({});
        in_neigbors.push_back({});

        num_vertex_types_ = std::max(num_vertex_types_, vertex_type + 1);

        return vertices_.back().id;
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

    bool add_edge(vertex_idx source, vertex_idx target) {

        if (source >= vertices_.size() || target >= vertices_.size() || source == target)
            return false;

        for (const vertex_idx v_idx : out_neigbors[source]) {
            if (v_idx == target) {
                return false;
            }
        }

        out_neigbors[source].push_back(target);
        in_neigbors[target].push_back(source);
        num_edges_++;

        return true;
    }

  private:
    std::vector<v_impl> vertices_;

    std::vector<std::vector<vertex_idx>> out_neigbors;
    std::vector<std::vector<vertex_idx>> in_neigbors;

    vertex_idx num_edges_ = 0;
    unsigned num_vertex_types_ = 0;
};

// default template parameters
using computational_dag_vector_impl_def_t = computational_dag_vector_impl<cdag_vertex_impl_unsigned>;
using computational_dag_vector_impl_def_int_t = computational_dag_vector_impl<cdag_vertex_impl_int>;


static_assert(is_directed_graph_edge_desc_v<computational_dag_vector_impl<cdag_vertex_impl_unsigned>>,
              "computational_dag_vector_impl must satisfy the directed_graph_edge_desc concept");

static_assert(has_vertex_weights_v<computational_dag_vector_impl<cdag_vertex_impl_unsigned>>,
              "computational_dag_vector_impl must satisfy the has_vertex_weights concept");

static_assert(is_directed_graph_v<computational_dag_vector_impl<cdag_vertex_impl_unsigned>>,
              "computational_dag_vector_impl must satisfy the directed_graph concept");

static_assert(is_computational_dag_typed_vertices_v<computational_dag_vector_impl<cdag_vertex_impl_unsigned>>,
              "computational_dag_vector_impl must satisfy the is_computation_dag concept");

} // namespace osp