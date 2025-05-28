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
#include "graph_implementations/vertex_iterator.hpp"
#include <vector>

namespace osp {

template<typename v_impl>
class dag_vector_adapter {
  public:
    using vertex_idx = typename v_impl::vertex_idx_type;

    using vertex_work_weight_type = typename v_impl::work_weight_type;
    using vertex_comm_weight_type = typename v_impl::comm_weight_type;
    using vertex_mem_weight_type = typename v_impl::mem_weight_type;
    using vertex_type_type = typename v_impl::cdag_vertex_type_type;
    using edge_comm_weight_type = typename v_impl::comm_weight_type;

    dag_vector_adapter(const std::vector<std::vector<vertex_idx>> &out_neigbors_,
                       const std::vector<std::vector<vertex_idx>> &in_neigbors_)
        : vertices_(out_neigbors_.size()), out_neigbors(out_neigbors_), in_neigbors(in_neigbors_), num_edges_(0),
          num_vertex_types_(1) {

        for (size_t i = 0; i < out_neigbors_.size(); ++i) {
            vertices_[i].id = static_cast<vertex_idx>(i);
            num_edges_ += out_neigbors_[i].size();
        }
    }

    dag_vector_adapter(const dag_vector_adapter &other) = default;
    dag_vector_adapter &operator=(const dag_vector_adapter &other) = default;

    virtual ~dag_vector_adapter() = default;

    inline auto vertices() const { return vertex_range<vertex_idx>(static_cast<vertex_idx>(vertices_.size())); }

    inline std::size_t num_vertices() const { return vertices_.size(); }

    inline std::size_t num_edges() const { return num_edges_; }

    inline const std::vector<vertex_idx> &parents(const vertex_idx v) const { return in_neigbors.at(static_cast<std::size_t>(v)); }

    inline const std::vector<vertex_idx> &children(const vertex_idx v) const { return out_neigbors.at(static_cast<std::size_t>(v)); }

    inline std::size_t in_degree(const vertex_idx v) const { return in_neigbors[static_cast<std::size_t>(v)].size(); }

    inline std::size_t out_degree(const vertex_idx v) const { return out_neigbors[static_cast<std::size_t>(v)].size(); }

    inline vertex_work_weight_type vertex_work_weight(const vertex_idx v) const { return vertices_[static_cast<std::size_t>(v)].work_weight; }

    inline vertex_comm_weight_type vertex_comm_weight(const vertex_idx v) const { return vertices_[static_cast<std::size_t>(v)].comm_weight; }

    inline vertex_mem_weight_type vertex_mem_weight(const vertex_idx v) const { return vertices_[static_cast<std::size_t>(v)].mem_weight; }

    inline vertex_type_type vertex_type(const vertex_idx v) const { return vertices_[static_cast<std::size_t>(v)].vertex_type; }

    inline vertex_type_type num_vertex_types() const { return num_vertex_types_; }

    inline const v_impl &get_vertex_impl(const vertex_idx v) const { return vertices_[static_cast<std::size_t>(v)]; }

    inline void set_vertex_work_weight(vertex_idx v, vertex_work_weight_type work_weight) {
        vertices_[static_cast<std::size_t>(v)].work_weight = work_weight;
    }

    inline void set_vertex_comm_weight(vertex_idx v, vertex_comm_weight_type comm_weight) {
        vertices_[static_cast<std::size_t>(v)].comm_weight = comm_weight;
    }

    inline void set_vertex_mem_weight(vertex_idx v, vertex_mem_weight_type mem_weight) {
        vertices_[static_cast<std::size_t>(v)].mem_weight = mem_weight;
    }

    inline void set_vertex_type(vertex_idx v, vertex_type_type vertex_type) {
        vertices_[static_cast<std::size_t>(v)].vertex_type = vertex_type;
        num_vertex_types_ = std::max(num_vertex_types_, vertex_type + 1);
    }
  

  private:
    std::vector<v_impl> vertices_;

    const std::vector<std::vector<vertex_idx>> &out_neigbors;
    const std::vector<std::vector<vertex_idx>> &in_neigbors;

    std::size_t num_edges_ = 0;
    unsigned num_vertex_types_ = 0;
};

static_assert(has_vertex_weights_v<dag_vector_adapter<cdag_vertex_impl_unsigned>>,
              "computational_dag_vector_impl must satisfy the has_vertex_weights concept");

static_assert(is_directed_graph_v<dag_vector_adapter<cdag_vertex_impl_unsigned>>,
              "computational_dag_vector_impl must satisfy the directed_graph concept");

static_assert(is_computational_dag_typed_vertices_v<dag_vector_adapter<cdag_vertex_impl_unsigned>>,
              "computational_dag_vector_impl must satisfy the is_computation_dag concept");

} // namespace osp