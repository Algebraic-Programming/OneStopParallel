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

#include <algorithm>
#include <vector>

#include "cdag_vertex_impl.hpp"
#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/concepts/directed_graph_edge_desc_concept.hpp"
#include "osp/graph_algorithms/computational_dag_construction_util.hpp"
#include "osp/graph_implementations/integral_range.hpp"

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
 * @see cdag_vertex_impl for a reference implementation of the vertex type.
 */
template <typename VImpl>
class ComputationalDagVectorImpl {
  public:
    using VertexIdx = typename VImpl::VertexIdxType;

    using VertexWorkWeightType = typename VImpl::WorkWeightType;
    using VertexCommWeightType = typename VImpl::CommWeightType;
    using VertexMemWeightType = typename VImpl::MemWeightType;
    using VertexTypeType = typename VImpl::CDagVertexTypeType;

    ComputationalDagVectorImpl() = default;

    /**
     * @brief Constructs a graph with a specified number of vertices.
     *
     * @param NumVertices The number of vertices to initialize.
     */
    explicit ComputationalDagVectorImpl(const VertexIdx numVertices)
        : vertices_(numVertices), outNeigbors_(numVertices), inNeigbors_(numVertices), numEdges_(0), numVertexTypes_(0) {
        for (VertexIdx i = 0; i < numVertices; ++i) {
            vertices_[i].id = i;
        }
    }

    ComputationalDagVectorImpl(const ComputationalDagVectorImpl &other) = default;
    ComputationalDagVectorImpl &operator=(const ComputationalDagVectorImpl &other) = default;

    /**
     * @brief Constructs a graph from another graph type.
     *
     * This constructor initializes the graph by copying the structure and properties from another graph `other`.
     * The source graph `Graph_t` must satisfy the `is_computational_dag` concept.
     *
     * @tparam Graph_t The type of the source graph. Must satisfy `is_computational_dag_v`.
     * @param other The source graph to copy from.
     */
    template <typename GraphT>
    explicit ComputationalDagVectorImpl(const GraphT &other) {
        static_assert(IsComputationalDagV<GraphT>, "Graph_t must satisfy the is_computation_dag concept");
        ConstructComputationalDag(other, *this);
    }

    ComputationalDagVectorImpl(ComputationalDagVectorImpl &&other) noexcept
        : vertices_(std::move(other.vertices_)),
          outNeigbors_(std::move(other.outNeigbors_)),
          inNeigbors_(std::move(other.inNeigbors_)),
          numEdges_(other.numEdges_),
          numVertexTypes_(other.numVertexTypes_) {
        other.numEdges_ = 0;
        other.numVertexTypes_ = 0;
    };

    ComputationalDagVectorImpl &operator=(ComputationalDagVectorImpl &&other) noexcept {
        if (this != &other) {
            vertices_ = std::move(other.vertices_);
            outNeigbors_ = std::move(other.outNeigbors_);
            inNeigbors_ = std::move(other.inNeigbors_);
            numEdges_ = other.numEdges_;
            numVertexTypes_ = other.numVertexTypes_;

            other.numEdges_ = 0;
            other.numVertexTypes_ = 0;
        }
        return *this;
    }

    virtual ~ComputationalDagVectorImpl() = default;

    /**
     * @brief Returns a range of all vertex indices.
     */
    [[nodiscard]] auto Vertices() const { return IntegralRange<VertexIdx>(static_cast<VertexIdx>(vertices_.size())); }

    /**
     * @brief Returns the total number of vertices.
     */
    [[nodiscard]] VertexIdx NumVertices() const { return static_cast<VertexIdx>(vertices_.size()); }

    /**
     * @brief Checks if the graph is empty (no vertices).
     */
    [[nodiscard]] bool empty() const { return vertices_.empty(); }

    /**
     * @brief Returns the total number of edges.
     */
    [[nodiscard]] VertexIdx NumEdges() const { return numEdges_; }

    /**
     * @brief Returns the parents (in-neighbors) of a vertex. Does not perform bounds checking.
     * @param v The vertex index.
     */
    [[nodiscard]] const std::vector<VertexIdx> &Parents(const VertexIdx v) const { return inNeigbors_[v]; }

    /**
     * @brief Returns the children (out-neighbors) of a vertex. Does not perform bounds checking.
     * @param v The vertex index.
     */
    [[nodiscard]] const std::vector<VertexIdx> &Children(const VertexIdx v) const { return outNeigbors_[v]; }

    /**
     * @brief Returns the in-degree of a vertex. Does not perform bounds checking.
     * @param v The vertex index.
     */
    [[nodiscard]] VertexIdx InDegree(const VertexIdx v) const { return static_cast<VertexIdx>(inNeigbors_[v].size()); }

    /**
     * @brief Returns the out-degree of a vertex. Does not perform bounds checking.
     * @param v The vertex index.
     */
    [[nodiscard]] VertexIdx OutDegree(const VertexIdx v) const { return static_cast<VertexIdx>(outNeigbors_[v].size()); }

    [[nodiscard]] VertexWorkWeightType VertexWorkWeight(const VertexIdx v) const { return vertices_[v].workWeight_; }

    [[nodiscard]] VertexCommWeightType VertexCommWeight(const VertexIdx v) const { return vertices_[v].commWeight_; }

    [[nodiscard]] VertexMemWeightType VertexMemWeight(const VertexIdx v) const { return vertices_[v].memWeight_; }

    [[nodiscard]] VertexTypeType VertexType(const VertexIdx v) const { return vertices_[v].vertexType_; }

    [[nodiscard]] VertexTypeType NumVertexTypes() const { return numVertexTypes_; }

    [[nodiscard]] const VImpl &GetVertexImpl(const VertexIdx v) const { return vertices_[v]; }

    /**
     * @brief Adds a new isolated vertex to the graph.
     *
     * @param work_weight Computational work weight.
     * @param comm_weight Communication weight.
     * @param mem_weight Memory weight.
     * @param vertex_type Type of the vertex.
     * @return The index of the newly added vertex.
     */
    VertexIdx AddVertex(const VertexWorkWeightType workWeight,
                        const VertexCommWeightType commWeight,
                        const VertexMemWeightType memWeight,
                        const VertexTypeType vertexType = 0) {
        vertices_.emplace_back(vertices_.size(), workWeight, commWeight, memWeight, vertexType);
        outNeigbors_.push_back({});
        inNeigbors_.push_back({});

        numVertexTypes_ = std::max(numVertexTypes_, vertexType + 1);

        return vertices_.back().id_;
    }

    void SetVertexWorkWeight(const VertexIdx v, const VertexWorkWeightType workWeight) {
        vertices_.at(v).workWeight_ = workWeight;
    }

    void SetVertexCommWeight(const VertexIdx v, const VertexCommWeightType commWeight) {
        vertices_.at(v).commWeight_ = commWeight;
    }

    void SetVertexMemWeight(const VertexIdx v, const VertexMemWeightType memWeight) { vertices_.at(v).memWeight_ = memWeight; }

    void SetVertexType(const VertexIdx v, const VertexTypeType vertexType) {
        vertices_.at(v).vertexType_ = vertexType;
        numVertexTypes_ = std::max(numVertexTypes_, vertexType + 1);
    }

    /**
     * @brief Adds a directed edge between two vertices.
     *
     * @param source The source vertex index.
     * @param target The target vertex index.
     * @return True if the edge was added, false if it already exists or vertices are invalid.
     */
    bool AddEdge(const VertexIdx source, const VertexIdx target) {
        if (source >= static_cast<VertexIdx>(vertices_.size()) || target >= static_cast<VertexIdx>(vertices_.size())
            || source == target) {
            return false;
        }

        const auto &out = outNeigbors_.at(source);
        if (std::find(out.begin(), out.end(), target) != out.end()) {
            return false;
        }

        outNeigbors_[source].push_back(target);
        inNeigbors_.at(target).push_back(source);
        numEdges_++;

        return true;
    }

  private:
    std::vector<VImpl> vertices_;

    std::vector<std::vector<VertexIdx>> outNeigbors_;
    std::vector<std::vector<VertexIdx>> inNeigbors_;

    VertexIdx numEdges_ = 0;
    unsigned numVertexTypes_ = 0;
};

/**
 * @brief Default implementation of a computational DAG using unsigned integer weights.
 */
using ComputationalDagVectorImplDefUnsignedT = ComputationalDagVectorImpl<CDagVertexImplUnsigned>;

/**
 * @brief Default implementation of a computational DAG using signed integer weights.
 */
using ComputationalDagVectorImplDefIntT = ComputationalDagVectorImpl<CDagVertexImplInt>;

static_assert(IsDirectedGraphEdgeDescV<ComputationalDagVectorImplDefUnsignedT>,
              "computational_dag_vector_impl must satisfy the directed_graph_edge_desc concept");

static_assert(HasVertexWeightsV<ComputationalDagVectorImplDefUnsignedT>,
              "computational_dag_vector_impl must satisfy the has_vertex_weights concept");

static_assert(IsDirectedGraphV<ComputationalDagVectorImplDefUnsignedT>,
              "computational_dag_vector_impl must satisfy the directed_graph concept");

static_assert(IsComputationalDagTypedVerticesV<ComputationalDagVectorImplDefUnsignedT>,
              "computational_dag_vector_impl must satisfy the is_computation_dag concept");

}    // namespace osp
