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

@author Christos Matzoros, Toni Boehnlein, Pal Andras Papp, Raphael S. Steiner
*/

#pragma once

#ifdef EIGEN_FOUND

#include <Eigen/SparseCore>
#include "concepts/directed_graph_concept.hpp"
#include "concepts/computational_dag_concept.hpp"
#include "graph_implementations/vertex_iterator.hpp"
#include "graph_implementations/eigen_sparse_iterator.hpp"

namespace osp {

/// @brief Implementation of a lower‐triangular sparse matrix as a directed acyclic graph.
///        Wraps Eigen's sparse matrix and exposes graph-like methods for scheduling and analysis.
template<typename eigen_idx_type>
class SparseMatrixImp {
    static_assert(std::is_integral_v<eigen_idx_type>, "Eigen index type must be integral");
private:
    // Define Eigen-compatible matrix types using eigen_idx_type as the index type
    using MatrixCSR = Eigen::SparseMatrix<double, Eigen::RowMajor, eigen_idx_type>;  // For parents
    using MatrixCSC = Eigen::SparseMatrix<double, Eigen::ColMajor, eigen_idx_type>;  // For children

    // Internal pointers to the sparse matrices (not owning)
    MatrixCSR* L_csr_p = nullptr;
    MatrixCSC* L_csc_p = nullptr;

public:
    // Vertex index type must match Eigen's StorageIndex (signed 32-bit)
    using vertex_idx = size_t;

    // Required graph trait aliases (used in concept checks)
    using directed_edge_descriptor = int;
    using vertex_work_weight_type = eigen_idx_type;
    using vertex_comm_weight_type = int;
    using vertex_mem_weight_type = int;
    using vertex_type_type = unsigned;

    using eigen_idx_t = eigen_idx_type;    

    SparseMatrixImp() = default;

    // Setters for the internal CSR and CSC matrix pointers
    void setCSR(MatrixCSR* mat) { L_csr_p = mat; }
    void setCSC(MatrixCSC* mat) { L_csc_p = mat; }

    // Getters for internal matrices (used by EigenSparseRange)
    const MatrixCSR* getCSR() const { return L_csr_p; }
    const MatrixCSC* getCSC() const { return L_csc_p; }

    /// @brief Number of vertices = number of rows in the matrix
    size_t num_vertices() const noexcept {
        return static_cast<size_t>(L_csr_p->rows());
    }

    /// @brief Return a range over all vertices [0, num_vertices)
    auto vertices() const {
        return osp::vertex_range<size_t>(num_vertices());
    }

    /// @brief Number of edges = total non-zeros minus diagonal elements
    vertex_idx num_edges() const noexcept {
        return static_cast<vertex_idx>(L_csr_p->nonZeros() - L_csr_p->rows());
    }

    /// @brief In-degree = non-zero off-diagonal entries in row v (CSR)
    vertex_idx in_degree(vertex_idx v) const noexcept {
        return static_cast<vertex_idx>(L_csr_p->outerIndexPtr()[v + 1] - L_csr_p->outerIndexPtr()[v] - 1);
    }

    /// @brief Out-degree = non-zero off-diagonal entries in column v (CSC)
    vertex_idx out_degree(vertex_idx v) const noexcept {
        return static_cast<vertex_idx>(L_csc_p->outerIndexPtr()[v + 1] - L_csc_p->outerIndexPtr()[v] - 1);
    }

    /// @brief Get the children (dependents) of vertex v using CSC layout
    auto children(vertex_idx v) const {
        return osp::EigenCSCRange<SparseMatrixImp, vertex_idx>(*this, v);
    }

    /// @brief Get the parents (dependencies) of vertex v using CSR layout
    auto parents(vertex_idx v) const {
        return osp::EigenCSRRange<SparseMatrixImp, vertex_idx>(*this, v);
    }

    /// @brief Work weight of a vertex (e.g., row size)
    vertex_work_weight_type vertex_work_weight(vertex_idx v) const noexcept {
        return L_csr_p->outerIndexPtr()[v + 1] - L_csr_p->outerIndexPtr()[v];
    }

    // Default zero weights (placeholders, extend as needed)
    vertex_comm_weight_type vertex_comm_weight(vertex_idx) const noexcept { return 0; }
    vertex_mem_weight_type vertex_mem_weight(vertex_idx) const noexcept  { return 0; }

    inline unsigned num_vertex_types() const { return 1; };
    inline vertex_type_type vertex_type(const vertex_idx v) const { return v-v; }
};

using sparse_matrix_graph_int32_t = SparseMatrixImp<int32_t>;
using sparse_matrix_graph_int64_t = SparseMatrixImp<int64_t>;


// Verify that SparseMatrixImp satisfies the directed graph concept
static_assert(is_directed_graph_v<SparseMatrixImp<int32_t>>,
              "SparseMatrix must satisfy directed_graph_concept");

static_assert(is_directed_graph_v<SparseMatrixImp<int64_t>>,
              "SparseMatrix must satisfy directed_graph_concept");


static_assert(has_vertex_weights_v<SparseMatrixImp<int32_t>>, 
    "Compact_Sparse_Graph must satisfy the has_vertex_weights concept");

static_assert(has_vertex_weights_v<SparseMatrixImp<int64_t>>, 
    "Compact_Sparse_Graph must satisfy the has_vertex_weights concept");



} // namespace osp


#endif