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

#include "computational_dag_vector_impl.hpp"
#include "concepts/computational_dag_concept.hpp"
#include "graph_algorithms/computational_dag_construction_util.hpp"
#include "graph_implementations/container_iterator_adaptor.hpp"
#include "graph_implementations/vertex_iterator.hpp"
#include <vector>

namespace osp {

template<typename from_t, typename to_t>
class vector_cast_view {

    using iter = typename std::vector<from_t>::const_iterator;
    const std::vector<from_t> &vec;
    struct cast_iterator {
        iter current_edge;

      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = to_t;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type *;
        using reference = const value_type &;

        cast_iterator() = default;
        cast_iterator(const cast_iterator &other) : current_edge(other.current_edge) {}

        cast_iterator &operator=(const cast_iterator &other) {
            if (this != &other) {
                current_edge = other.current_edge;
            }
            return *this;
        }

        cast_iterator(iter current_edge_) : current_edge(current_edge_) {}

        value_type operator*() const { return static_cast<to_t>(*current_edge); }

        // Prefix increment
        cast_iterator &operator++() {
            current_edge++;
            return *this;
        }

        // Postfix increment
        cast_iterator operator++(int) {
            cast_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        inline bool operator==(const cast_iterator &other) const { return current_edge == other.current_edge; }
        inline bool operator!=(const cast_iterator &other) const { return current_edge != other.current_edge; }
    };

  public:
    vector_cast_view(const std::vector<from_t> &vec_) : vec(vec_) {}

    auto begin() const { return cast_iterator(vec.begin()); }

    auto end() const { return cast_iterator(vec.end()); }

    auto size() const { return vec.size(); }
};

template<typename v_impl, typename index_t>
class dag_vector_adapter {

  public:
    using vertex_idx = typename v_impl::vertex_idx_type;

    using vertex_work_weight_type = typename v_impl::work_weight_type;
    using vertex_comm_weight_type = typename v_impl::comm_weight_type;
    using vertex_mem_weight_type = typename v_impl::mem_weight_type;
    using vertex_type_type = typename v_impl::cdag_vertex_type_type;
    using edge_comm_weight_type = typename v_impl::comm_weight_type;

    dag_vector_adapter(std::vector<std::vector<index_t>> &out_neigbors_,
                       std::vector<std::vector<index_t>> &in_neigbors_)
        : vertices_(out_neigbors_.size()), out_neigbors(out_neigbors_), in_neigbors(in_neigbors_), num_edges_(0),
          num_vertex_types_(1) {

        for (vertex_idx i = 0; i < out_neigbors_.size(); ++i) {
            vertices_[i].id = i;
            num_edges_ += out_neigbors_[i].size();
        }
    }

    dag_vector_adapter(const dag_vector_adapter &other) = default;
    dag_vector_adapter &operator=(const dag_vector_adapter &other) = default;

    virtual ~dag_vector_adapter() = default;

    inline auto vertices() const { return vertex_range<vertex_idx>(static_cast<vertex_idx>(vertices_.size())); }

    inline vertex_idx num_vertices() const { return static_cast<vertex_idx>(vertices_.size()); }

    inline std::size_t num_edges() const { return num_edges_; }

    inline auto parents(const vertex_idx v) const { return vector_cast_view<index_t, vertex_idx>(in_neigbors[v]); }

    inline auto children(const vertex_idx v) const { return vector_cast_view<index_t, vertex_idx>(out_neigbors[v]); }

    inline std::size_t in_degree(const vertex_idx v) const { return in_neigbors[v].size(); }

    inline std::size_t out_degree(const vertex_idx v) const { return out_neigbors[v].size(); }

    inline vertex_work_weight_type vertex_work_weight(const vertex_idx v) const { return vertices_[v].work_weight; }

    inline vertex_comm_weight_type vertex_comm_weight(const vertex_idx v) const { return vertices_[v].comm_weight; }

    inline vertex_mem_weight_type vertex_mem_weight(const vertex_idx v) const { return vertices_[v].mem_weight; }

    inline vertex_type_type vertex_type(const vertex_idx v) const { return vertices_[v].vertex_type; }

    inline vertex_type_type num_vertex_types() const { return num_vertex_types_; }

    inline const v_impl &get_vertex_impl(const vertex_idx v) const { return vertices_[v]; }

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

  private:
    std::vector<v_impl> vertices_;

    std::vector<std::vector<index_t>> &out_neigbors;
    std::vector<std::vector<index_t>> &in_neigbors;

    std::size_t num_edges_ = 0;
    unsigned num_vertex_types_ = 0;
};

static_assert(has_vertex_weights_v<dag_vector_adapter<cdag_vertex_impl_unsigned, int>>,
              "computational_dag_vector_impl must satisfy the has_vertex_weights concept");

static_assert(is_directed_graph_v<dag_vector_adapter<cdag_vertex_impl_unsigned, int>>,
              "computational_dag_vector_impl must satisfy the directed_graph concept");

static_assert(is_computational_dag_typed_vertices_v<dag_vector_adapter<cdag_vertex_impl_unsigned, int>>,
              "computational_dag_vector_impl must satisfy the is_computation_dag concept");

} // namespace osp