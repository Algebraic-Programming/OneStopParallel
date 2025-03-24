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

#include <Eigen/SparseCore>

#include "iterator/vertex_iterator.hpp"
#include "concepts/directed_graph_concept.hpp"

namespace osp {

/**
 * @class SparseMatrix
 * @brief Represents a lower triangular sparse matrix as a Directed Acyclic Graph (DAG).
 *
 * Each row in the matrix corresponds to a computational task (vertex in the DAG). A non-zero entry
 * at column `u` in row `v` indicates a dependency of task `v` on task `u` (edge `u -> v` in the DAG).
 * The class provides methods to analyze dependencies, topological ordering, and graph properties.
 */
class SparseMatrix_impl {

    const Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t> *L_csr_p =
        nullptr; // CSR format for row/vertex operations
    const Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t> *L_csc_p =
        nullptr; // CSC format for column/child operations

  public:
    SparseMatrix_impl() {}

    /**
     * @brief Sets the pointer to the Compressed Sparse Row (CSR) matrix.
     * @param L_csr_p_ Pointer to the CSR matrix. Used for row-wise operations (e.g., parent access).
     */
    inline void setCSR(const Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t> *L_csr_p_) { L_csr_p = L_csr_p_; }

    /**
     * @brief Sets the pointer to the Compressed Sparse Column (CSC) matrix.
     * @param L_csc_p_ Pointer to the CSC matrix. Used for column-wise operations (e.g., child access).
     */
    inline void setCSC(const Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t> *L_csc_p_) { L_csc_p = L_csc_p_; }

    inline const Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t> *getCSR() const { return L_csr_p; }

    inline const Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t> *getCSC() const { return L_csc_p; }

    /**
     * @brief Returns the number of vertices (rows) in the matrix.
     * @return Total vertices in the CSR matrix.
     */
    inline unsigned int num_vertices() const { return static_cast<unsigned int>((*L_csr_p).rows()); }

    inline auto vertices() const { return vertex_range<vertex_idx>(num_vertices()); }

    /**
     * @brief Returns the number of edges in the DAG (non-diagonal non-zero entries).
     * @return Total edges, excluding diagonal entries.
     */
    unsigned int num_edges() const {
        return static_cast<unsigned int>((*L_csr_p).nonZeros() - (*L_csr_p).rows()); // Subtract diagonal elements
    }

    inline unsigned in_degree(const vertex_idx v) const {
        return static_cast<unsigned int>((*L_csr_p).outerIndexPtr()[v + 1] - (*L_csr_p).outerIndexPtr()[v] - 1);
    }

    /**
     * @brief Returns the number of children (outgoing edges) for a vertex.
     * @param v Vertex index.
     * @return Child count (non-zero entries in column `v`, excluding diagonal).
     */
    inline unsigned out_degree(const vertex_idx v) const {
        return static_cast<unsigned int>((*L_csc_p).outerIndexPtr()[v + 1] - (*L_csc_p).outerIndexPtr()[v] - 1);
    }

    /**
     * @brief Computes the computational work (non-zero count in row, including diagonal).
     * @param row Vertex index.
     * @return Work weight (number of non-zeros in the row).
     */
    inline int vertex_work_weight(const vertex_idx row) const {
        return static_cast<unsigned int>((*L_csr_p).outerIndexPtr()[row + 1] - (*L_csr_p).outerIndexPtr()[row]);
    }

    /**
     * @brief Placeholder for communication weight (not implemented).
     * @return Always returns 0.
     */
    inline int vertex_comm_weight(const vertex_idx v) const { return 0; }

    /**
     * @brief Placeholder for memory weight (not implemented).
     * @return Always returns 0.
     */
    inline int vertex_mem_weight(const vertex_idx v) const { return 0; }

    /**
     * @brief Placeholder for node type (not implemented).
     * @return Always returns 0.
     */
    inline unsigned vertex_type(const vertex_idx v) const { return 0; }

    inline unsigned num_vertex_types() const { return 1; }

    /**
     * @brief Retrieves parent vertices (column indices in row, excluding diagonal).
     * @param row Vertex index.
     * @return Deque of parent indices.
     */
    inline const auto parents(const unsigned int row) const {
        return Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t>::InnerIterator(*L_csr_p, row);
    }

    /**
     * @brief Retrieves parent vertices (non-const version).
     * @param row Vertex index.
     * @return Deque of parent indices.
     */
    inline auto parents(const unsigned int row) {
        return Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t>::InnerIterator(*L_csr_p, row);
    }

    /**
     * @brief Retrieves child vertices (row indices in column, excluding diagonal).
     * @param row Vertex index.
     * @return Deque of child indices.
     */
    inline const auto children(const unsigned int row) const {
        return Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>::InnerIterator(*L_csc_p, row);
    }

    /**
     * @brief Retrieves child vertices (non-const version).
     * @param row Vertex index.
     * @return Deque of child indices.
     */
    inline auto children(const unsigned int row) {
        return Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>::InnerIterator(*L_csc_p, row);
    }

    ~SparseMatrix_impl() {}
};

static_assert(is_computation_dag_typed_vertices_v<SparseMatrix_impl>,
              "SparseMatrix_impl must satisfy the is_computation_dag concept");

} // namespace osp