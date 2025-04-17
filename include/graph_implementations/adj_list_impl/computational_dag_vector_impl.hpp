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

#include "concepts/computational_dag_concept.hpp"
#include "graph_implementations/vertex_iterator.hpp"
#include <vector>

namespace osp {

struct cdag_vertex_impl {

    cdag_vertex_impl(std::size_t vertex_idx_, int work_weight, int comm_weight, int mem_weight, unsigned vertex_type)
        : id(vertex_idx_), work_weight(work_weight), comm_weight(comm_weight), mem_weight(mem_weight),
          vertex_type(vertex_type) {}

    std::size_t id;

    int work_weight;
    int comm_weight;
    int mem_weight;

    unsigned vertex_type;
};

template<typename v_impl>
class computational_dag_vector_impl {
  public:
    static_assert(std::is_base_of<cdag_vertex_impl, v_impl>::value, "v_impl must be derived from cdag_vertex_impl");

    using vertex_idx = std::size_t;

    using vertex_work_weight_type = int;
    using vertex_comm_weight_type = int;
    using vertex_mem_weight_type = int;
    using vertex_type_type = unsigned;
    using edge_comm_weight_type = int;

    computational_dag_vector_impl() = default;
    computational_dag_vector_impl(const computational_dag_vector_impl &other) = default;
    computational_dag_vector_impl(computational_dag_vector_impl &&other) = default;
    computational_dag_vector_impl &operator=(const computational_dag_vector_impl &other) = default;
    computational_dag_vector_impl &operator=(computational_dag_vector_impl &&other) = default;
    virtual ~computational_dag_vector_impl() = default;

    inline auto vertices() const { return vertex_range<vertex_idx>(vertices_.size()); }

    inline std::size_t num_vertices() const { return vertices_.size(); }

    inline std::size_t num_edges() const { return num_edges_; }

    inline const std::vector<vertex_idx> &parents(const vertex_idx v) const { return in_neigbors[v]; }

    inline const std::vector<vertex_idx> &children(const vertex_idx v) const { return out_neigbors[v]; }

    inline std::size_t in_degree(const vertex_idx v) const { return in_neigbors[v].size(); }

    inline std::size_t out_degree(const vertex_idx v) const { return out_neigbors[v].size(); }

    inline int vertex_work_weight(const vertex_idx v) const { return vertices_[v].work_weight; }

    inline int vertex_comm_weight(const vertex_idx v) const { return vertices_[v].comm_weight; }

    inline int vertex_mem_weight(const vertex_idx v) const { return vertices_[v].mem_weight; }

    inline vertex_type_type vertex_type(const vertex_idx v) const { return vertices_[v].vertex_type; }

    inline vertex_type_type num_vertex_types() const { return num_vertex_types_; }

    inline const v_impl &get_vertex_impl(const vertex_idx v) const { return vertices_[v]; }

    vertex_idx add_vertex(int work_weight, int comm_weight, int mem_weight, vertex_type_type vertex_type = 0) {

        vertices_.emplace_back(vertices_.size(), work_weight, comm_weight, mem_weight, vertex_type);
        out_neigbors.push_back({});
        in_neigbors.push_back({});

        num_vertex_types_ = std::max(num_vertex_types_, vertex_type + 1);

        return vertices_.back().id;
    }

    inline void set_vertex_work_weight(vertex_idx v, int work_weight) { vertices_[v].work_weight = work_weight; }

    inline void set_vertex_comm_weight(vertex_idx v, int comm_weight) { vertices_[v].comm_weight = comm_weight; }

    inline void set_vertex_mem_weight(vertex_idx v, int mem_weight) { vertices_[v].mem_weight = mem_weight; }

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

    std::size_t num_edges_ = 0;
    unsigned num_vertex_types_ = 0;
};


// default template parameters
using computational_dag_vector_impl_def_t = computational_dag_vector_impl<cdag_vertex_impl>;

// TODO delete this
template<typename v_impl>
std::vector<vertex_idx_t<computational_dag_vector_impl<v_impl>>>
source_vertices(const computational_dag_vector_impl<v_impl> &graph) {

    std::cout << "calling custom source_vertices" << std::endl;

    std::vector<vertex_idx_t<computational_dag_vector_impl<v_impl>>> vec;
    for (const auto &v_idx : graph.vertices()) {
        if (graph.in_degree(v_idx) == 0) {
            vec.push_back(v_idx);
        }
    }
    return vec;
}

static_assert(has_vertex_weights_v<computational_dag_vector_impl<cdag_vertex_impl>>,
  "computational_dag_vector_impl must satisfy the has_vertex_weights concept");

static_assert(is_directed_graph_v<computational_dag_vector_impl<cdag_vertex_impl>>,
              "computational_dag_vector_impl must satisfy the directed_graph concept");

static_assert(is_computational_dag_typed_vertices_v<computational_dag_vector_impl<cdag_vertex_impl>>,
              "computational_dag_vector_impl must satisfy the is_computation_dag concept");

} // namespace osp