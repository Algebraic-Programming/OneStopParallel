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

#include "boost_extensions/inv_breadth_first_search.hpp"
#include "boost_extensions/source_iterator_range.hpp"
#include <numeric>
#include <queue>
#include <vector>
#include <omp.h>
#include <Eigen/SparseCore>

#include "auxiliary/Balanced_Coin_Flips.hpp"
#include "auxiliary/auxiliary.hpp"

using SM_csr = Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t>;
using SM_csc = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>;

/**
 * @class SparseMatrix
 * @brief Represents a lower triangular sparse matrix as a Directed Acyclic Graph (DAG).
 *
 * Each row in the matrix corresponds to a computational task (vertex in the DAG). A non-zero entry
 * at column `u` in row `v` indicates a dependency of task `v` on task `u` (edge `u -> v` in the DAG).
 * The class provides methods to analyze dependencies, topological ordering, and graph properties.
 */
class SparseMatrix {
    static constexpr int DEFAULT_EDGE_COMM_WEIGHT = 1;
    const SM_csr * L_csr_p = nullptr; // CSR format for row/vertex operations
    const SM_csc * L_csc_p = nullptr; // CSC format for column/child operations

public:
    SparseMatrix() {}

    /**
     * @brief Sets the pointer to the Compressed Sparse Row (CSR) matrix.
     * @param L_csr_p_ Pointer to the CSR matrix. Used for row-wise operations (e.g., parent access).
     */
    void setCSR(const SM_csr * L_csr_p_){
        L_csr_p = L_csr_p_;
    }

    /**
     * @brief Sets the pointer to the Compressed Sparse Column (CSC) matrix.
     * @param L_csc_p_ Pointer to the CSC matrix. Used for column-wise operations (e.g., child access).
     */
    void setCSC(const SM_csc * L_csc_p_){
        L_csc_p = L_csc_p_;
    }

    const SM_csr * getCSR() const{
        return L_csr_p;
    }

    const SM_csc * getCSC() const{
        return L_csc_p;
    }

    /**
     * @brief Returns the number of vertices (rows) in the matrix.
     * @return Total vertices in the CSR matrix.
     */
    unsigned int numberOfVertices() const { 
        return static_cast<unsigned int>((*L_csr_p).rows()); 
    }

    /**
     * @brief Returns the number of edges in the DAG (non-diagonal non-zero entries).
     * @return Total edges, excluding diagonal entries.
     */
    unsigned int numberOfEdges() const {
        return (*L_csr_p).nonZeros() - (*L_csr_p).rows(); // Subtract diagonal elements
    }

    /**
     * @brief Finds source vertices (no incoming edges except diagonal).
     * @return Indices of vertices with no parents.
     */
    std::vector<unsigned int> sourceVertices() const;

    /**
     * @brief Finds sink vertices (no outgoing edges except diagonal).
     * @return Indices of vertices with no children.
     */
    std::vector<unsigned int> sinkVertices() const;

    enum TOP_SORT_ORDER { AS_IT_COMES,   // Standard topological order
                          MAX_CHILDREN,  // Prioritize vertices with most children first
                          RANDOM };      // Randomized order

    /**
     * @brief Computes a topological order of vertices.
     * @param order Sorting criterion (AS_IT_COMES, MAX_CHILDREN, RANDOM).
     * @return Vertices ordered topologically.
     */
    std::vector<unsigned int> GetTopOrder(TOP_SORT_ORDER order = AS_IT_COMES) const{
        std::vector<unsigned int> TopOrder(static_cast<unsigned int>((*L_csr_p).rows()));
        iota(std::begin(TopOrder), std::end(TopOrder), 0);

        return TopOrder;
    }
    
    /**
     * @brief Returns the number of parents (incoming edges) for a vertex.
     * @param v Vertex index.
     * @return Parent count (non-zero entries in row `v`, excluding diagonal).
     */
    inline unsigned numberOfParents(const unsigned int &v) const { 
        return (*L_csr_p).outerIndexPtr()[v+1] - (*L_csr_p).outerIndexPtr()[v] - 1;
    }

    /**
     * @brief Returns the number of children (outgoing edges) for a vertex.
     * @param v Vertex index.
     * @return Child count (non-zero entries in column `v`, excluding diagonal).
     */
    inline unsigned numberOfChildren(const unsigned int &v) const { 
        return (*L_csc_p).outerIndexPtr()[v+1] - (*L_csc_p).outerIndexPtr()[v] - 1;
    }

    /**
     * @brief Computes the computational work (non-zero count in row, including diagonal).
     * @param row Vertex index.
     * @return Work weight (number of non-zeros in the row).
     */
    int nodeWorkWeight(const size_t &row) const { 
        return (*L_csr_p).outerIndexPtr()[row+1] - (*L_csr_p).outerIndexPtr()[row];
    }

    /**
     * @brief Placeholder for communication weight (not implemented).
     * @return Always returns 0.
     */
    int nodeCommunicationWeight(const size_t &v) const { return 0; }

    /**
     * @brief Placeholder for memory weight (not implemented).
     * @return Always returns 0.
     */
    int nodeMemoryWeight(const size_t &v) const { return 0; }

    /**
     * @brief Placeholder for node type (not implemented).
     * @return Always returns 0.
     */
    unsigned nodeType(const std::size_t &v) const { return 0; }

    /**
     * @brief Retrieves parent vertices (column indices in row, excluding diagonal).
     * @param row Vertex index.
     * @return Deque of parent indices.
     */
    auto parents(const unsigned int &row) const {
        std::deque<unsigned int> parents;
        for (SM_csr::InnerIterator it(*L_csr_p, row); it; ++it){
            auto index = static_cast<unsigned int>(it.index());
            parents.push_back(index);
        }
        parents.pop_back(); // Remove diagonal entry
        return parents;
    }

    /**
     * @brief Retrieves child vertices (row indices in column, excluding diagonal).
     * @param row Vertex index.
     * @return Deque of child indices.
     */
    auto children(const unsigned int &row) const {
        std::deque<unsigned int> children;
        for (SM_csc::InnerIterator it(*L_csc_p, row); it; ++it){
            children.push_back(static_cast<unsigned int>(it.index()));
        }
        children.pop_front(); // Remove diagonal entry
        return children;
    }

    ~SparseMatrix() {}
};

#endif