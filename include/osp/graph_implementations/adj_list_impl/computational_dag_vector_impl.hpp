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
#include "osp/concepts/directed_graph_edge_desc_concept.hpp"
#include "osp/graph_algorithms/computational_dag_construction_util.hpp"
#include "osp/graph_implementations/integral_range.hpp"
#include <vector>

#include <algorithm>

namespace osp {

/**
 * @brief A vector-based implementation of a computational DAG.
 *
 * This class implements a computational DAG using adjacency lists stored in two std::vectors.
 * It manages the storage of vertices and edges, and provides an interface to query and modify the graph. 
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
 * @see cdag_vertex_impl for a reference implementation of the vertex type.
 */
template<typename v_impl>
class computational_dag_vector_impl {
  public:
    using vertex_idx = typename v_impl::vertex_idx_type;

    using vertex_work_weight_type = typename v_impl::work_weight_type;
    using vertex_comm_weight_type = typename v_impl::comm_weight_type;
    using vertex_mem_weight_type = typename v_impl::mem_weight_type;
    using vertex_type_type = typename v_impl::cdag_vertex_type_type;

    computational_dag_vector_impl() = default;

    /**
     * @brief Constructs a graph with a specified number of vertices.
     *
     * @param num_vertices The number of vertices to initialize.
     */
    explicit computational_dag_vector_impl(const vertex_idx num_vertices)
        : vertices_(num_vertices), out_neigbors(num_vertices), in_neigbors(num_vertices), num_edges_(0),
          num_vertex_types_(0) {

        for (vertex_idx i = 0; i < num_vertices; ++i) {
            vertices_.at(i).id = i;
        }
    }

    computational_dag_vector_impl(const computational_dag_vector_impl &other) = default;
    computational_dag_vector_impl &operator=(const computational_dag_vector_impl &other) = default;

    /**
     * @brief Constructs a graph from another graph type.
     *
     * This constructor initializes the graph by copying the structure and properties from another graph `other`.
     * The source graph `Graph_t` must satisfy the `is_computational_dag` concept.
     *
     * @tparam Graph_t The type of the source graph. Must satisfy `is_computational_dag_v`.
     * @param other The source graph to copy from.
     */
    template<typename Graph_t>
    explicit computational_dag_vector_impl(const Graph_t &other) {

        static_assert(is_computational_dag_v<Graph_t>, "Graph_t must satisfy the is_computation_dag concept");

        construct_computational_dag(other, *this);
    }

    computational_dag_vector_impl(computational_dag_vector_impl &&other) noexcept
        : vertices_(std::move(other.vertices_)), out_neigbors(std::move(other.out_neigbors)),
          in_neigbors(std::move(other.in_neigbors)), num_edges_(other.num_edges_),
          num_vertex_types_(other.num_vertex_types_) {

        other.num_edges_ = 0;
        other.num_vertex_types_ = 0;
    };

    computational_dag_vector_impl &operator=(computational_dag_vector_impl &&other) noexcept {
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

    /**
     * @brief Returns a range of all vertex indices.
     */
    [[nodiscard]] auto vertices() const { return integral_range<vertex_idx>(static_cast<vertex_idx>(vertices_.size())); }

    /**
     * @brief Returns the total number of vertices.
     */
    [[nodiscard]] vertex_idx num_vertices() const { return static_cast<vertex_idx>(vertices_.size()); }

    /**
     * @brief Checks if the graph is empty (no vertices).
     */
    [[nodiscard]] bool empty() const { return vertices_.empty(); }

    /**
     * @brief Returns the total number of edges.
     */
    [[nodiscard]] vertex_idx num_edges() const { return num_edges_; }

    /**
     * @brief Returns the parents (in-neighbors) of a vertex.
     * @param v The vertex index.
     */
    [[nodiscard]] const std::vector<vertex_idx> &parents(const vertex_idx v) const { return in_neigbors.at(v); }

    /**
     * @brief Returns the children (out-neighbors) of a vertex.
     * @param v The vertex index.
     */
    [[nodiscard]] const std::vector<vertex_idx> &children(const vertex_idx v) const { return out_neigbors.at(v); }

    /**
     * @brief Returns the in-degree of a vertex.
     * @param v The vertex index.
     */
    [[nodiscard]] vertex_idx in_degree(const vertex_idx v) const { return static_cast<vertex_idx>(in_neigbors.at(v).size()); }

    /**
     * @brief Returns the out-degree of a vertex.
     * @param v The vertex index.
     */
    [[nodiscard]] vertex_idx out_degree(const vertex_idx v) const { return static_cast<vertex_idx>(out_neigbors.at(v).size()); }

    [[nodiscard]] vertex_work_weight_type vertex_work_weight(const vertex_idx v) const { return vertices_.at(v).work_weight; }

    [[nodiscard]] vertex_comm_weight_type vertex_comm_weight(const vertex_idx v) const { return vertices_.at(v).comm_weight; }

    [[nodiscard]] vertex_mem_weight_type vertex_mem_weight(const vertex_idx v) const { return vertices_.at(v).mem_weight; }

    [[nodiscard]] vertex_type_type vertex_type(const vertex_idx v) const { return vertices_.at(v).vertex_type; }

    [[nodiscard]] vertex_type_type num_vertex_types() const { return num_vertex_types_; }

    [[nodiscard]] const v_impl &get_vertex_impl(const vertex_idx v) const { return vertices_.at(v); }

    /**
     * @brief Adds a new isolated vertex to the graph.
     *
     * @param work_weight Computational work weight.
     * @param comm_weight Communication weight.
     * @param mem_weight Memory weight.
     * @param vertex_type Type of the vertex.
     * @return The index of the newly added vertex.
     */
    vertex_idx add_vertex(const vertex_work_weight_type work_weight, const vertex_comm_weight_type comm_weight,
                          const vertex_mem_weight_type mem_weight, const vertex_type_type vertex_type = 0) {

        vertices_.emplace_back(vertices_.size(), work_weight, comm_weight, mem_weight, vertex_type);
        out_neigbors.push_back({});
        in_neigbors.push_back({});

        num_vertex_types_ = std::max(num_vertex_types_, vertex_type + 1);

        return vertices_.back().id;
    }

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

    /**
     * @brief Adds a directed edge between two vertices.
     *
     * @param source The source vertex index.
     * @param target The target vertex index.
     * @return True if the edge was added, false if it already exists or vertices are invalid.
     */
    bool add_edge(const vertex_idx source, const vertex_idx target) {

        if (source >= static_cast<vertex_idx>(vertices_.size()) || target >= static_cast<vertex_idx>(vertices_.size()) || source == target)
            return false;

        const auto &out = out_neigbors.at(source);
        if (std::find(out.begin(), out.end(), target) != out.end()) {
            return false;
        }

        out_neigbors.at(source).push_back(target);
        in_neigbors.at(target).push_back(source);
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

/**
 * @brief Default implementation of a computational DAG using unsigned integer weights.
 */
using computational_dag_vector_impl_def_t = computational_dag_vector_impl<cdag_vertex_impl_unsigned>;

/**
 * @brief Default implementation of a computational DAG using signed integer weights.
 */
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