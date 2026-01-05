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
#include "edge_iterator.hpp"
#include "osp/auxiliary/hash_util.hpp"
#include "osp/graph_algorithms/computational_dag_construction_util.hpp"
#include "osp/graph_implementations/integral_range.hpp"

namespace osp {

template <typename VImpl>
struct DirectedEdgeDescriptorImpl {
    using VertexIdx = typename VImpl::VertexIdxType;

    VertexIdx idx_;

    VertexIdx source_;
    VertexIdx target_;

    DirectedEdgeDescriptorImpl() : idx_(0), source_(0), target_(0) {}

    DirectedEdgeDescriptorImpl(const DirectedEdgeDescriptorImpl<VImpl> &other) = default;
    DirectedEdgeDescriptorImpl(DirectedEdgeDescriptorImpl<VImpl> &&other) = default;
    DirectedEdgeDescriptorImpl &operator=(const DirectedEdgeDescriptorImpl<VImpl> &other) = default;
    DirectedEdgeDescriptorImpl &operator=(DirectedEdgeDescriptorImpl<VImpl> &&other) = default;

    DirectedEdgeDescriptorImpl(VertexIdx sourceArg, VertexIdx targetArg, VertexIdx idxArg)
        : idx_(idxArg), source_(sourceArg), target_(targetArg) {}

    ~DirectedEdgeDescriptorImpl() = default;

    bool operator==(const DirectedEdgeDescriptorImpl<VImpl> &other) const {
        return idx_ == other.idx_ && source_ == other.source_ && target_ == other.target_;
    }

    bool operator!=(const DirectedEdgeDescriptorImpl<VImpl> &other) const { return !(*this == other); }
};

template <typename EdgeCommWeightT>
struct CDagEdgeImpl {
    using CDagEdgeCommWeightType = EdgeCommWeightT;

    CDagEdgeImpl(EdgeCommWeightT commWeightArg = 1) : commWeight_(commWeightArg) {}

    EdgeCommWeightT commWeight_;
};

using CDagEdgeImplInt = CDagEdgeImpl<int>;
using CDagEdgeImplUnsigned = CDagEdgeImpl<unsigned>;

template <typename VImpl, typename EImpl>
class ComputationalDagEdgeIdxVectorImpl {
  public:
    // graph_traits specialization
    using VertexIdx = typename VImpl::VertexIdxType;
    using DirectedEdgeDescriptor = DirectedEdgeDescriptorImpl<VImpl>;

    using OutEdgesIteratorT = typename std::vector<DirectedEdgeDescriptor>::const_iterator;
    using InEdgesIteratorT = typename std::vector<DirectedEdgeDescriptor>::const_iterator;

    // cdag_traits specialization
    using VertexWorkWeightType = typename VImpl::WorkWeightType;
    using VertexCommWeightType = typename VImpl::CommWeightType;
    using VertexMemWeightType = typename VImpl::MemWeightType;
    using VertexTypeType = typename VImpl::CDagVertexTypeType;
    using EdgeCommWeightType = typename EImpl::CDagEdgeCommWeightType;

  private:
    using ThisT = ComputationalDagEdgeIdxVectorImpl<VImpl, EImpl>;

    std::vector<VImpl> vertices_;
    std::vector<EImpl> edges_;

    unsigned numVertexTypes_ = 0;

    std::vector<std::vector<DirectedEdgeDescriptor>> outEdges_;
    std::vector<std::vector<DirectedEdgeDescriptor>> inEdges_;

  public:
    ComputationalDagEdgeIdxVectorImpl() = default;

    ComputationalDagEdgeIdxVectorImpl(VertexIdx numVertices)
        : vertices_(numVertices), outEdges_(numVertices), inEdges_(numVertices) {
        for (VertexIdx i = 0; i < numVertices; ++i) {
            vertices_[i].id_ = i;
        }
    }

    ComputationalDagEdgeIdxVectorImpl(const ComputationalDagEdgeIdxVectorImpl &other) = default;

    template <typename GraphT>
    ComputationalDagEdgeIdxVectorImpl(const GraphT &other) {
        static_assert(isComputationalDagV<GraphT>, "GraphT must satisfy the is_computation_dag concept");

        ConstructComputationalDag(other, *this);
    }

    ComputationalDagEdgeIdxVectorImpl &operator=(const ComputationalDagEdgeIdxVectorImpl &other) = default;

    ComputationalDagEdgeIdxVectorImpl(ComputationalDagEdgeIdxVectorImpl &&other)
        : vertices_(std::move(other.vertices_)),
          edges_(std::move(other.edges_)),
          numVertexTypes_(other.numVertexTypes_),
          outEdges_(std::move(other.outEdges_)),
          inEdges_(std::move(other.inEdges_)) {
        other.numVertexTypes_ = 0;
    }

    ComputationalDagEdgeIdxVectorImpl &operator=(ComputationalDagEdgeIdxVectorImpl &&other) {
        if (this != &other) {
            vertices_ = std::move(other.vertices_);
            edges_ = std::move(other.edges_);
            outEdges_ = std::move(other.outEdges_);
            inEdges_ = std::move(other.inEdges_);
            numVertexTypes_ = other.numVertexTypes_;
            other.numVertexTypes_ = 0;
        }
        return *this;
    }

    virtual ~ComputationalDagEdgeIdxVectorImpl() = default;

    inline VertexIdx NumEdges() const { return static_cast<VertexIdx>(edges_.size()); }

    inline VertexIdx NumVertices() const { return static_cast<VertexIdx>(vertices_.size()); }

    inline auto Edges() const { return EdgeRangeVectorImpl<ThisT>(*this); }

    inline auto Parents(VertexIdx v) const { return EdgeSourceRange(inEdges_[v], *this); }

    inline auto Children(VertexIdx v) const { return EdgeTargetRange(outEdges_[v], *this); }

    inline auto Vertices() const { return IntegralRange<VertexIdx>(static_cast<VertexIdx>(vertices_.size())); }

    inline const std::vector<DirectedEdgeDescriptor> &InEdges(VertexIdx v) const { return inEdges_[v]; }

    inline const std::vector<DirectedEdgeDescriptor> &OutEdges(VertexIdx v) const { return outEdges_[v]; }

    inline VertexIdx InDegree(VertexIdx v) const { return static_cast<VertexIdx>(inEdges_[v].size()); }

    inline VertexIdx OutDegree(VertexIdx v) const { return static_cast<VertexIdx>(outEdges_[v].size()); }

    inline EdgeCommWeightType EdgeCommWeight(DirectedEdgeDescriptor e) const { return edges_[e.idx_].commWeight_; }

    inline VertexWorkWeightType VertexWorkWeight(VertexIdx v) const { return vertices_[v].workWeight_; }

    inline VertexCommWeightType VertexCommWeight(VertexIdx v) const { return vertices_[v].commWeight_; }

    inline VertexMemWeightType VertexMemWeight(VertexIdx v) const { return vertices_[v].memWeight_; }

    inline unsigned NumVertexTypes() const { return numVertexTypes_; }

    inline VertexTypeType VertexType(VertexIdx v) const { return vertices_[v].vertexType_; }

    inline VertexIdx Source(const DirectedEdgeDescriptor &e) const { return e.source_; }

    inline VertexIdx Target(const DirectedEdgeDescriptor &e) const { return e.target_; }

    VertexIdx AddVertex(VertexWorkWeightType workWeight,
                        VertexCommWeightType commWeight,
                        VertexMemWeightType memWeight,
                        VertexTypeType vertexType = 0) {
        vertices_.emplace_back(vertices_.size(), workWeight, commWeight, memWeight, vertexType);

        outEdges_.push_back({});
        inEdges_.push_back({});

        numVertexTypes_ = std::max(numVertexTypes_, vertexType + 1);

        return vertices_.back().id_;
    }

    std::pair<DirectedEdgeDescriptor, bool> AddEdge(VertexIdx source, VertexIdx target, EdgeCommWeightType commWeight = 1) {
        if (source == target) {
            return {DirectedEdgeDescriptor{}, false};
        }

        if (source >= vertices_.size() || target >= vertices_.size()) {
            return {DirectedEdgeDescriptor{}, false};
        }

        for (const auto edge : outEdges_[source]) {
            if (edge.target_ == target) {
                return {DirectedEdgeDescriptor{}, false};
            }
        }

        outEdges_[source].emplace_back(source, target, edges_.size());
        inEdges_[target].emplace_back(source, target, edges_.size());

        edges_.emplace_back(commWeight);

        return {outEdges_[source].back(), true};
    }

    inline void SetVertexWorkWeight(VertexIdx v, VertexWorkWeightType workWeight) { vertices_[v].workWeight_ = workWeight; }

    inline void SetVertexCommWeight(VertexIdx v, VertexCommWeightType commWeight) { vertices_[v].commWeight_ = commWeight; }

    inline void SetVertexMemWeight(VertexIdx v, VertexMemWeightType memWeight) { vertices_[v].memWeight_ = memWeight; }

    inline void SetVertexType(VertexIdx v, VertexTypeType vertexType) {
        vertices_[v].vertexType_ = vertexType;
        numVertexTypes_ = std::max(numVertexTypes_, vertexType + 1);
    }

    inline void SetEdgeCommWeight(DirectedEdgeDescriptor e, EdgeCommWeightType commWeight) {
        edges_[e.idx_].commWeight_ = commWeight;
    }

    inline const VImpl &GetVertexImpl(VertexIdx v) const { return vertices_[v]; }

    inline const EImpl &GetEdgeImpl(DirectedEdgeDescriptor e) const { return edges_[e.idx_]; }
};

template <typename VImpl, typename EImpl>
inline auto Edges(const ComputationalDagEdgeIdxVectorImpl<VImpl, EImpl> &graph) {
    return graph.Edges();
}

template <typename VImpl, typename EImpl>
inline auto OutEdges(VertexIdxT<ComputationalDagEdgeIdxVectorImpl<VImpl, EImpl>> v,
                     const ComputationalDagEdgeIdxVectorImpl<VImpl, EImpl> &graph) {
    return graph.OutEdges(v);
}

template <typename VImpl, typename EImpl>
inline auto InEdges(VertexIdxT<ComputationalDagEdgeIdxVectorImpl<VImpl, EImpl>> v,
                    const ComputationalDagEdgeIdxVectorImpl<VImpl, EImpl> &graph) {
    return graph.InEdges(v);
}

// default implementation to get the source of an edge
template <typename VImpl, typename EImpl>
inline VertexIdxT<ComputationalDagEdgeIdxVectorImpl<VImpl, EImpl>> Source(
    const EdgeDescT<ComputationalDagEdgeIdxVectorImpl<VImpl, EImpl>> &edge,
    const ComputationalDagEdgeIdxVectorImpl<VImpl, EImpl> &graph) {
    return graph.Source(edge);
}

// default implementation to get the target of an edge
template <typename VImpl, typename EImpl>
inline VertexIdxT<ComputationalDagEdgeIdxVectorImpl<VImpl, EImpl>> Target(
    const EdgeDescT<ComputationalDagEdgeIdxVectorImpl<VImpl, EImpl>> &edge,
    const ComputationalDagEdgeIdxVectorImpl<VImpl, EImpl> &graph) {
    return graph.Target(edge);
}

// default template specialization
using ComputationalDagEdgeIdxVectorImplDefT = ComputationalDagEdgeIdxVectorImpl<CDagVertexImplUnsigned, CDagEdgeImplUnsigned>;

using ComputationalDagEdgeIdxVectorImplDefIntT = ComputationalDagEdgeIdxVectorImpl<CDagVertexImplInt, CDagEdgeImplInt>;

static_assert(isDirectedGraphEdgeDescV<ComputationalDagEdgeIdxVectorImplDefT>,
              "computational_dag_edge_idx_vector_impl must satisfy the directed_graph_edge_desc concept");

static_assert(isComputationalDagTypedVerticesEdgeDescV<ComputationalDagEdgeIdxVectorImplDefT>,
              "computational_dag_edge_idx_vector_impl must satisfy the computation_dag_typed_vertices_edge_desc concept");

}    // namespace osp

template <typename VImpl>
struct std::hash<osp::DirectedEdgeDescriptorImpl<VImpl>> {
    using VertexIdx = typename VImpl::VertexIdxType;

    std::size_t operator()(const osp::DirectedEdgeDescriptorImpl<VImpl> &p) const noexcept {
        auto h1 = std::hash<VertexIdx>{}(p.source_);
        osp::HashCombine(h1, p.target_);

        return h1;
    }
};
