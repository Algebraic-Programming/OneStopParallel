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

#include "cdag_vertex_impl.hpp"
#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/graph_implementations/integral_range.hpp"
#include "vector_cast_view.hpp"
#include <vector>

namespace osp {

/**
 * @brief Adapter to view a pair of adjacency lists (out-neighbors and in-neighbors) as a computational DAG.
 *
 * This class adapts raw adjacency lists (vectors of vectors) into a graph interface compatible with
 * the OSP computational DAG concepts. It stores pointers to the external adjacency lists, so the
 * lifetime of these lists must exceed the lifetime of this adapter.
 *
 * This class satisfies the following concepts:
 * - `is_computational_dag_typed_vertices`
 * - `is_directed_graph`
 * - `has_vertex_weights`
 * - `is_directed_graph_edge_desc`
 *
 * @tparam v_impl The vertex implementation type. This type must satisfy the following requirements:
 * - It must define the following member types:
 *   - `vertex_idx_type`: The type used for vertex indices (e.g., `size_t`).
 *   - `work_weight_type`: The type used for computational work weights.
 *   - `comm_weight_type`: The type used for communication weights.
 *   - `mem_weight_type`: The type used for memory weights.
 *   - `cdag_vertex_type_type`: The type used for vertex types.
 * - It must have the following public data members:
 *   - `id`: Of type `vertex_idx_type`.
 *   - `work_weight`: Of type `work_weight_type`.
 *   - `comm_weight`: Of type `comm_weight_type`.
 *   - `mem_weight`: Of type `mem_weight_type`.
 *   - `vertex_type`: Of type `cdag_vertex_type_type`.
 * - It must be constructible with the signature:
 *   `v_impl(vertex_idx_type id, work_weight_type work_weight, comm_weight_type comm_weight, mem_weight_type mem_weight, cdag_vertex_type_type vertex_type)`
 *
 * @tparam index_t The type used for vertex indices in the adjacency lists.
 */
template<typename v_impl, typename index_t>
class dag_vector_adapter {

  public:
    using vertex_idx = typename v_impl::vertex_idx_type;

    using vertex_work_weight_type = typename v_impl::work_weight_type;
    using vertex_comm_weight_type = typename v_impl::comm_weight_type;
    using vertex_mem_weight_type = typename v_impl::mem_weight_type;
    using vertex_type_type = typename v_impl::cdag_vertex_type_type;

    dag_vector_adapter() = default;

    /**
     * @brief Constructs a dag_vector_adapter from adjacency lists.
     *
     * @param out_neigbors_ Vector of vectors representing out-neighbors for each vertex.
     * @param in_neigbors_ Vector of vectors representing in-neighbors for each vertex.
     *
     * @warning The adapter stores pointers to these vectors. They must remain valid for the lifetime of the adapter.
     */
    dag_vector_adapter(const std::vector<std::vector<index_t>> &out_neigbors_,
                       const std::vector<std::vector<index_t>> &in_neigbors_) : vertices_(out_neigbors_.size()), out_neigbors(&out_neigbors_), in_neigbors(&in_neigbors_), num_edges_(0), num_vertex_types_(1) {
        for (vertex_idx i = 0; i < static_cast<vertex_idx>(out_neigbors_.size()); ++i) {
            vertices_[i].id = i;
            num_edges_ += out_neigbors_[i].size();
        }
    }

    dag_vector_adapter(const dag_vector_adapter &other) = default;
    dag_vector_adapter &operator=(const dag_vector_adapter &other) = default;

    dag_vector_adapter(dag_vector_adapter &&other) noexcept = default;
    dag_vector_adapter &operator=(dag_vector_adapter &&other) noexcept = default;

    virtual ~dag_vector_adapter() = default;

    /**
     * @brief Re-initializes the adapter with new adjacency lists.
     *
     * @param in_neigbors_ New in-neighbors adjacency list.
     * @param out_neigbors_ New out-neighbors adjacency list.
     */
    void set_in_out_neighbors(const std::vector<std::vector<index_t>> &in_neigbors_, const std::vector<std::vector<index_t>> &out_neigbors_) {
        out_neigbors = &out_neigbors_;
        in_neigbors = &in_neigbors_;

        vertices_.resize(out_neigbors->size());

        num_edges_ = 0;
        for (vertex_idx i = 0; i < static_cast<vertex_idx>(out_neigbors->size()); ++i) {
            vertices_[i].id = i;
            num_edges_ += out_neigbors_[i].size();
        }

        num_vertex_types_ = 1;
    }

    /**
     * @brief Returns a range of all vertex indices.
     */
    [[nodiscard]] auto vertices() const { return integral_range<vertex_idx>(static_cast<vertex_idx>(vertices_.size())); }

    /**
     * @brief Returns the total number of vertices.
     */
    [[nodiscard]] vertex_idx num_vertices() const { return static_cast<vertex_idx>(vertices_.size()); }

    /**
     * @brief Returns the total number of edges.
     */
    [[nodiscard]] vertex_idx num_edges() const { return static_cast<vertex_idx>(num_edges_); }

    /**
     * @brief Returns a view of the parents (in-neighbors) of a vertex. Does not perform bounds checking.
     * @param v The vertex index.
     */
    [[nodiscard]] auto parents(const vertex_idx v) const { return vector_cast_view<index_t, vertex_idx>(in_neigbors_[v]); }

    /**
     * @brief Returns a view of the children (out-neighbors) of a vertex. Does not perform bounds checking.
     * @param v The vertex index.
     */
    [[nodiscard]] auto children(const vertex_idx v) const { return vector_cast_view<index_t, vertex_idx>(out_neigbors_[v]); }

    /**
     * @brief Returns the in-degree of a vertex. Does not perform bounds checking.
     * @param v The vertex index.
     */
    [[nodiscard]] vertex_idx in_degree(const vertex_idx v) const { return static_cast<vertex_idx>(in_neigbors_[v].size()); }

    /**
     * @brief Returns the out-degree of a vertex. Does not perform bounds checking.
     * @param v The vertex index.
     */
    [[nodiscard]] vertex_idx out_degree(const vertex_idx v) const { return static_cast<vertex_idx>(out_neigbors_[v].size()); }

    [[nodiscard]] vertex_work_weight_type vertex_work_weight(const vertex_idx v) const { return vertices_[v].work_weight; }

    [[nodiscard]] vertex_comm_weight_type vertex_comm_weight(const vertex_idx v) const { return vertices_[v].comm_weight; }

    [[nodiscard]] vertex_mem_weight_type vertex_mem_weight(const vertex_idx v) const { return vertices_[v].mem_weight; }

    [[nodiscard]] vertex_type_type vertex_type(const vertex_idx v) const { return vertices_[v].vertex_type; }

    [[nodiscard]] vertex_type_type num_vertex_types() const { return num_vertex_types_; }

    [[nodiscard]] const v_impl &get_vertex_impl(const vertex_idx v) const { return vertices_[v]; }

    void set_vertex_work_weight(const vertex_idx v, const vertex_work_weight_type work_weight) {
        vertices_.at(v).work_weight = work_weight;
    }

    void set_vertex_comm_weight(const vertex_idx v, const vertex_comm_weight_type comm_weight) {
        vertices_.at(v).comm_weight = comm_weight;
    }

    void set_vertex_mem_weight(const vertex_idx v, const vertex_mem_weight_type mem_weight) {
        vertices_.at(v).mem_weight = mem_weight;
    }

    void set_vertex_type(const vertex_idx v, const vertex_type_type vertex_type) {
        vertices_.at(v).vertex_type = vertex_type;
        num_vertex_types_ = std::max(num_vertex_types_, vertex_type + 1);
    }

  private:
    std::vector<v_impl> vertices_;

    const std::vector<std::vector<index_t>> *out_neigbors;
    const std::vector<std::vector<index_t>> *in_neigbors;

    std::size_t num_edges_ = 0;
    unsigned num_vertex_types_ = 0;
};

static_assert(is_directed_graph_edge_desc_v<dag_vector_adapter<cdag_vertex_impl_unsigned, int>>,
              "dag_vector_adapter must satisfy the directed_graph_edge_desc concept");

static_assert(has_vertex_weights_v<dag_vector_adapter<cdag_vertex_impl_unsigned, int>>,
              "dag_vector_adapter must satisfy the has_vertex_weights concept");

static_assert(is_directed_graph_v<dag_vector_adapter<cdag_vertex_impl_unsigned, int>>,
              "dag_vector_adapter must satisfy the directed_graph concept");

static_assert(is_computational_dag_typed_vertices_v<dag_vector_adapter<cdag_vertex_impl_unsigned, int>>,
              "dag_vector_adapter must satisfy the is_computation_dag concept");

} // namespace osp