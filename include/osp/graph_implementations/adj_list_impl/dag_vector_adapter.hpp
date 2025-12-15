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

#include <vector>

#include "cdag_vertex_impl.hpp"
#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/graph_implementations/integral_range.hpp"
#include "vector_cast_view.hpp"

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
 *   - `cdag_VertexTypeType`: The type used for vertex types.
 * - It must have the following public data members:
 *   - `id`: Of type `vertex_idx_type`.
 *   - `work_weight`: Of type `work_weight_type`.
 *   - `comm_weight`: Of type `comm_weight_type`.
 *   - `mem_weight`: Of type `mem_weight_type`.
 *   - `vertex_type`: Of type `cdag_VertexTypeType`.
 * - It must be constructible with the signature:
 *   `v_impl(vertex_idx_type id, work_weight_type work_weight, comm_weight_type comm_weight, mem_weight_type mem_weight,
 * cdag_VertexTypeType vertex_type)`
 *
 * @tparam index_t The type used for vertex indices in the adjacency lists.
 */
template <typename VImpl, typename IndexT>
class DagVectorAdapter {
  public:
    using VertexIdx = typename VImpl::VertexIdxType;

    using VertexWorkWeightType = typename VImpl::WorkWeightType;
    using VertexCommWeightType = typename VImpl::CommWeightType;
    using VertexMemWeightType = typename VImpl::MemWeightType;
    using VertexTypeType = typename VImpl::CDagVertexTypeType;

    DagVectorAdapter() = default;

    /**
     * @brief Constructs a dag_vector_adapter from adjacency lists.
     *
     * @param out_neigbors_ Vector of vectors representing out-neighbors for each vertex.
     * @param in_neigbors_ Vector of vectors representing in-neighbors for each vertex.
     *
     * @warning The adapter stores pointers to these vectors. They must remain valid for the lifetime of the adapter.
     */
    DagVectorAdapter(const std::vector<std::vector<IndexT>> &outNeigbors, const std::vector<std::vector<IndexT>> &inNeigbors)
        : vertices_(outNeigbors.size()), outNeigbors_(&outNeigbors), inNeigbors_(&inNeigbors), numEdges_(0), numVertexTypes_(1) {
        for (VertexIdx i = 0; i < static_cast<VertexIdx>(outNeigbors.size()); ++i) {
            vertices_[i].id = i;
            numEdges_ += outNeigbors[i].size();
        }
    }

    DagVectorAdapter(const DagVectorAdapter &other) = default;
    DagVectorAdapter &operator=(const DagVectorAdapter &other) = default;

    DagVectorAdapter(DagVectorAdapter &&other) noexcept = default;
    DagVectorAdapter &operator=(DagVectorAdapter &&other) noexcept = default;

    virtual ~DagVectorAdapter() = default;

    /**
     * @brief Re-initializes the adapter with new adjacency lists.
     *
     * @param in_neigbors_ New in-neighbors adjacency list.
     * @param out_neigbors_ New out-neighbors adjacency list.
     */
    void SetInOutNeighbors(const std::vector<std::vector<IndexT>> &inNeigbors,
                           const std::vector<std::vector<IndexT>> &outNeigbors) {
        outNeigbors_ = &outNeigbors;
        inNeigbors_ = &inNeigbors;

        vertices_.resize(outNeigbors_->size());

        numEdges_ = 0;
        for (VertexIdx i = 0; i < static_cast<VertexIdx>(outNeigbors_->size()); ++i) {
            vertices_[i].id = i;
            numEdges_ += outNeigbors[i].size();
        }

        numVertexTypes_ = 1;
    }

    /**
     * @brief Returns a range of all vertex indices.
     */
    [[nodiscard]] auto Vertices() const { return IntegralRange<VertexIdx>(static_cast<VertexIdx>(vertices_.size())); }

    /**
     * @brief Returns the total number of vertices.
     */
    [[nodiscard]] VertexIdx NumVertices() const { return static_cast<VertexIdx>(vertices_.size()); }

    /**
     * @brief Returns the total number of edges.
     */
    [[nodiscard]] VertexIdx NumEdges() const { return static_cast<VertexIdx>(numEdges_); }

    /**
     * @brief Returns a view of the parents (in-neighbors) of a vertex. Does not perform bounds checking.
     * @param v The vertex index.
     */
    [[nodiscard]] auto Parents(const VertexIdx v) const { return VectorCastView<IndexT, VertexIdx>((*inNeigbors_)[v]); }

    /**
     * @brief Returns a view of the children (out-neighbors) of a vertex. Does not perform bounds checking.
     * @param v The vertex index.
     */
    [[nodiscard]] auto Children(const VertexIdx v) const { return VectorCastView<IndexT, VertexIdx>((*outNeigbors_)[v]); }

    /**
     * @brief Returns the in-degree of a vertex. Does not perform bounds checking.
     * @param v The vertex index.
     */
    [[nodiscard]] VertexIdx InDegree(const VertexIdx v) const { return static_cast<VertexIdx>((*inNeigbors_)[v].size()); }

    /**
     * @brief Returns the out-degree of a vertex. Does not perform bounds checking.
     * @param v The vertex index.
     */
    [[nodiscard]] VertexIdx OutDegree(const VertexIdx v) const { return static_cast<VertexIdx>((*outNeigbors_)[v].size()); }

    [[nodiscard]] VertexWorkWeightType VertexWorkWeight(const VertexIdx v) const { return vertices_[v].work_weight; }

    [[nodiscard]] VertexCommWeightType VertexCommWeight(const VertexIdx v) const { return vertices_[v].comm_weight; }

    [[nodiscard]] VertexMemWeightType VertexMemWeight(const VertexIdx v) const { return vertices_[v].mem_weight; }

    [[nodiscard]] VertexTypeType VertexType(const VertexIdx v) const { return vertices_[v].vertex_type; }

    [[nodiscard]] VertexTypeType NumVertexTypes() const { return numVertexTypes_; }

    [[nodiscard]] const VImpl &GetVertexImpl(const VertexIdx v) const { return vertices_[v]; }

    void SetVertexWorkWeight(const VertexIdx v, const VertexWorkWeightType workWeight) {
        vertices_.at(v).work_weight = workWeight;
    }

    void SetVertexCommWeight(const VertexIdx v, const VertexCommWeightType commWeight) {
        vertices_.at(v).comm_weight = commWeight;
    }

    void SetVertexMemWeight(const VertexIdx v, const VertexMemWeightType memWeight) { vertices_.at(v).mem_weight = memWeight; }

    void SetVertexType(const VertexIdx v, const VertexTypeType vertexType) {
        vertices_.at(v).vertex_type = vertexType;
        numVertexTypes_ = std::max(numVertexTypes_, vertexType + 1);
    }

  private:
    std::vector<VImpl> vertices_;

    const std::vector<std::vector<IndexT>> *outNeigbors_;
    const std::vector<std::vector<IndexT>> *inNeigbors_;

    std::size_t numEdges_ = 0;
    unsigned numVertexTypes_ = 0;
};

static_assert(IsDirectedGraphEdgeDescV<dag_vector_adapter<cdag_vertex_impl_unsigned, int>>,
              "dag_vector_adapter must satisfy the directed_graph_edge_desc concept");

static_assert(HasVertexWeightsV<dag_vector_adapter<cdag_vertex_impl_unsigned, int>>,
              "dag_vector_adapter must satisfy the has_vertex_weights concept");

static_assert(IsDirectedGraphV<dag_vector_adapter<cdag_vertex_impl_unsigned, int>>,
              "dag_vector_adapter must satisfy the directed_graph concept");

static_assert(IsComputationalDagTypedVerticesV<dag_vector_adapter<cdag_vertex_impl_unsigned, int>>,
              "dag_vector_adapter must satisfy the is_computation_dag concept");

}    // namespace osp
