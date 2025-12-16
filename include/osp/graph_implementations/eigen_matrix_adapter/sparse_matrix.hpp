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

#    include <Eigen/SparseCore>

#    include "eigen_sparse_iterator.hpp"
#    include "osp/concepts/computational_dag_concept.hpp"
#    include "osp/concepts/directed_graph_concept.hpp"
#    include "osp/concepts/directed_graph_edge_desc_concept.hpp"
#    include "osp/graph_implementations/integral_range.hpp"

namespace osp {

/// @brief Implementation of a lower‐triangular sparse matrix as a directed acyclic graph.
///        Wraps Eigen's sparse matrix and exposes graph-like methods for scheduling and analysis.
template <typename EigenIdxType>
class SparseMatrixImp {
    static_assert(std::is_integral_v<EigenIdxType>, "Eigen index type must be integral");

  private:
    // Define Eigen-compatible matrix types using eigen_idx_type as the index type
    using MatrixCSR = Eigen::SparseMatrix<double, Eigen::RowMajor, EigenIdxType>;    // For parents
    using MatrixCSC = Eigen::SparseMatrix<double, Eigen::ColMajor, EigenIdxType>;    // For children

    // Internal pointers to the sparse matrices (not owning)
    MatrixCSR *lCsrP_ = nullptr;
    MatrixCSC *lCscP_ = nullptr;

  public:
    // Vertex index type must match Eigen's StorageIndex (signed 32-bit)
    using VertexIdx = size_t;

    // Required graph trait aliases (used in concept checks)
    using VertexWorkWeightType = EigenIdxType;
    using VertexCommWeightType = EigenIdxType;
    using VertexMemWeightType = int;
    using VertexTypeType = unsigned;

    using EigenIdxT = EigenIdxType;

    SparseMatrixImp() = default;

    // Setters for the internal CSR and CSC matrix pointers
    void SetCsr(MatrixCSR *mat) { lCsrP_ = mat; }

    void SetCsc(MatrixCSC *mat) { lCscP_ = mat; }

    // Getters for internal matrices (used by EigenSparseRange)
    const MatrixCSR *GetCSR() const { return lCsrP_; }

    const MatrixCSC *GetCSC() const { return lCscP_; }

    /// @brief Number of vertices = number of rows in the matrix
    size_t NumVertices() const noexcept { return static_cast<size_t>(lCsrP_->rows()); }

    /// @brief Return a range over all vertices [0, NumVertices)
    auto Vertices() const { return osp::IntegralRange<size_t>(NumVertices()); }

    /// @brief Number of edges = total non-zeros minus diagonal elements
    VertexIdx NumEdges() const noexcept { return static_cast<VertexIdx>(lCsrP_->nonZeros() - lCsrP_->rows()); }

    /// @brief In-degree = non-zero off-diagonal entries in row v (CSR)
    VertexIdx InDegree(VertexIdx v) const noexcept {
        return static_cast<VertexIdx>(lCsrP_->outerIndexPtr()[v + 1] - lCsrP_->outerIndexPtr()[v] - 1);
    }

    /// @brief Out-degree = non-zero off-diagonal entries in column v (CSC)
    VertexIdx OutDegree(VertexIdx v) const noexcept {
        return static_cast<VertexIdx>(lCscP_->outerIndexPtr()[v + 1] - lCscP_->outerIndexPtr()[v] - 1);
    }

    /// @brief Get the children (dependents) of vertex v using CSC layout
    auto Children(VertexIdx v) const {
        return osp::EigenCSCRange<SparseMatrixImp, EigenIdxType>(*this, static_cast<EigenIdxType>(v));
    }

    /// @brief Get the parents (dependencies) of vertex v using CSR layout
    auto Parents(VertexIdx v) const {
        return osp::EigenCSRRange<SparseMatrixImp, EigenIdxType>(*this, static_cast<EigenIdxType>(v));
    }

    /// @brief Work weight of a vertex (e.g., row size)
    VertexWorkWeightType VertexWorkWeight(VertexIdx v) const noexcept {
        return lCsrP_->outerIndexPtr()[v + 1] - lCsrP_->outerIndexPtr()[v];
    }

    // Default zero weights (placeholders, extend as needed)
    VertexCommWeightType VertexCommWeight(VertexIdx) const noexcept { return 0; }

    VertexMemWeightType VertexMemWeight(VertexIdx) const noexcept { return 0; }

    inline unsigned NumVertexTypes() const { return 1; };

    inline VertexTypeType VertexType(const VertexIdx) const { return 0; }
};

using SparseMatrixGraphInt32T = SparseMatrixImp<int32_t>;
using SparseMatrixGraphInt64T = SparseMatrixImp<int64_t>;

static_assert(IsDirectedGraphEdgeDescV<SparseMatrixImp<int32_t>>, "SparseMatrix must satisfy the directed_graph_edge_desc concept");

// Verify that SparseMatrixImp satisfies the directed graph concept
static_assert(IsDirectedGraphV<SparseMatrixImp<int32_t>>, "SparseMatrix must satisfy directed_graph_concept");

static_assert(IsDirectedGraphV<SparseMatrixImp<int64_t>>, "SparseMatrix must satisfy directed_graph_concept");

static_assert(hasVertexWeightsV<SparseMatrixImp<int32_t>>, "CompactSparseGraph must satisfy the has_vertex_weights concept");

static_assert(hasVertexWeightsV<SparseMatrixImp<int64_t>>, "CompactSparseGraph must satisfy the has_vertex_weights concept");

static_assert(isComputationalDagTypedVerticesV<SparseMatrixImp<int32_t>>,
              "CompactSparseGraph must satisfy the is_computation_dag concept");

}    // namespace osp

#endif
