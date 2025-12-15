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

#include <stdexcept>
#include <vector>

#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/graph_algorithms/directed_graph_edge_desc_util.hpp"

namespace osp {

template <typename IndexType = std::size_t, typename WorkwType = int, typename MemwType = int, typename CommwType = int>
class Hypergraph {
    using ThisT = Hypergraph<IndexType, WorkwType, MemwType, CommwType>;

  public:
    using VertexIdx = IndexType;
    using VertexWorkWeightType = WorkwType;
    using VertexMemWeightType = MemwType;
    using VertexCommWeightType = CommwType;

    Hypergraph() = default;

    Hypergraph(IndexType numVertices, IndexType numHyperedges)
        : numVertices_(numVertices),
          numHyperedges_(numHyperedges),
          vertexWorkWeights_(numVertices, 1),
          vertexMemoryWeights_(numVertices, 1),
          hyperedgeWeights_(numHyperedges, 1),
          incidentHyperedgesToVertex_(numVertices),
          verticesInHyperedge_(numHyperedges) {}

    Hypergraph(const ThisT &other) = default;
    Hypergraph &operator=(const ThisT &other) = default;

    virtual ~Hypergraph() = default;

    inline IndexType NumVertices() const { return numVertices_; }

    inline IndexType NumHyperedges() const { return numHyperedges_; }

    inline IndexType NumPins() const { return numPins_; }

    inline WorkwType GetVertexWorkWeight(IndexType node) const { return vertexWorkWeights_[node]; }

    inline MemwType GetVertexMemoryWeight(IndexType node) const { return vertexMemoryWeights_[node]; }

    inline CommwType GetHyperedgeWeight(IndexType hyperedge) const { return hyperedgeWeights_[hyperedge]; }

    void AddPin(IndexType vertexIdx, IndexType hyperedgeIdx);
    void AddVertex(WorkwType workWeight = 1, MemwType memoryWeight = 1);
    void AddEmptyHyperedge(CommwType weight = 1);
    void AddHyperedge(const std::vector<IndexType> &pins, CommwType weight = 1);
    void SetVertexWorkWeight(IndexType vertexIdx, WorkwType weight);
    void SetVertexMemoryWeight(IndexType vertexIdx, MemwType weight);
    void SetHyperedgeWeight(IndexType hyperedgeIdx, CommwType weight);

    void Clear();
    void Reset(IndexType numVertices, IndexType numHyperedges);

    inline const std::vector<IndexType> &GetIncidentHyperedges(IndexType vertex) const {
        return incidentHyperedgesToVertex_[vertex];
    }

    inline const std::vector<IndexType> &GetVerticesInHyperedge(IndexType hyperedge) const {
        return verticesInHyperedge_[hyperedge];
    }

  private:
    IndexType numVertices_ = 0, numHyperedges_ = 0, numPins_ = 0;

    std::vector<WorkwType> vertexWorkWeights_;
    std::vector<MemwType> vertexMemoryWeights_;
    std::vector<CommwType> hyperedgeWeights_;

    std::vector<std::vector<IndexType>> incidentHyperedgesToVertex_;
    std::vector<std::vector<IndexType>> verticesInHyperedge_;
};

using HypergraphDefT = Hypergraph<size_t, int, int, int>;

template <typename IndexType, typename WorkwType, typename MemwType, typename CommwType>
void Hypergraph<IndexType, WorkwType, MemwType, CommwType>::AddPin(IndexType vertex, IndexType hyperedgeIdx) {
    if (vertex >= numVertices_) {
        throw std::invalid_argument("Invalid Argument while adding pin: vertex index out of range.");
    } else if (hyperedgeIdx >= numHyperedges_) {
        throw std::invalid_argument("Invalid Argument while adding pin: hyperedge index out of range.");
    } else {
        incidentHyperedgesToVertex_[vertex].push_back(hyperedgeIdx);
        verticesInHyperedge_[hyperedgeIdx].push_back(vertex);
        ++numPins_;
    }
}

template <typename IndexType, typename WorkwType, typename MemwType, typename CommwType>
void Hypergraph<IndexType, WorkwType, MemwType, CommwType>::AddVertex(WorkwType workWeight, MemwType memoryWeight) {
    vertexWorkWeights_.push_back(workWeight);
    vertexMemoryWeights_.push_back(memoryWeight);
    incidentHyperedgesToVertex_.emplace_back();
    ++numVertices_;
}

template <typename IndexType, typename WorkwType, typename MemwType, typename CommwType>
void Hypergraph<IndexType, WorkwType, MemwType, CommwType>::AddEmptyHyperedge(CommwType weight) {
    verticesInHyperedge_.emplace_back();
    hyperedgeWeights_.push_back(weight);
    ++numHyperedges_;
}

template <typename IndexType, typename WorkwType, typename MemwType, typename CommwType>
void Hypergraph<IndexType, WorkwType, MemwType, CommwType>::AddHyperedge(const std::vector<IndexType> &pins, CommwType weight) {
    verticesInHyperedge_.emplace_back(pins);
    hyperedgeWeights_.push_back(weight);
    for (IndexType vertex : pins) {
        incidentHyperedgesToVertex_[vertex].push_back(numHyperedges_);
    }
    ++numHyperedges_;
    numPins_ += static_cast<IndexType>(pins.size());
}

template <typename IndexType, typename WorkwType, typename MemwType, typename CommwType>
void Hypergraph<IndexType, WorkwType, MemwType, CommwType>::SetVertexWorkWeight(IndexType vertex, WorkwType weight) {
    if (vertex >= numVertices_) {
        throw std::invalid_argument("Invalid Argument while setting vertex weight: vertex index out of range.");
    } else {
        vertexWorkWeights_[vertex] = weight;
    }
}

template <typename IndexType, typename WorkwType, typename MemwType, typename CommwType>
void Hypergraph<IndexType, WorkwType, MemwType, CommwType>::SetVertexMemoryWeight(IndexType vertex, MemwType weight) {
    if (vertex >= numVertices_) {
        throw std::invalid_argument("Invalid Argument while setting vertex weight: vertex index out of range.");
    } else {
        vertexMemoryWeights_[vertex] = weight;
    }
}

template <typename IndexType, typename WorkwType, typename MemwType, typename CommwType>
void Hypergraph<IndexType, WorkwType, MemwType, CommwType>::SetHyperedgeWeight(IndexType hyperedgeIdx, CommwType weight) {
    if (hyperedgeIdx >= numHyperedges_) {
        throw std::invalid_argument("Invalid Argument while setting hyperedge weight: hyepredge index out of range.");
    } else {
        hyperedgeWeights_[hyperedgeIdx] = weight;
    }
}

template <typename IndexType, typename WorkwType, typename MemwType, typename CommwType>
void Hypergraph<IndexType, WorkwType, MemwType, CommwType>::Clear() {
    numVertices_ = 0;
    numHyperedges_ = 0;
    numPins_ = 0;

    vertexWorkWeights_.clear();
    vertexMemoryWeights_.clear();
    hyperedgeWeights_.clear();
    incidentHyperedgesToVertex_.clear();
    verticesInHyperedge_.clear();
}

template <typename IndexType, typename WorkwType, typename MemwType, typename CommwType>
void Hypergraph<IndexType, WorkwType, MemwType, CommwType>::Reset(IndexType numVertices, IndexType numHyperedges) {
    Clear();

    numVertices_ = numVertices;
    numHyperedges_ = numHyperedges;

    vertexWorkWeights_.resize(numVertices, 1);
    vertexMemoryWeights_.resize(numVertices, 1);
    hyperedgeWeights_.resize(numHyperedges, 1);
    incidentHyperedgesToVertex_.resize(numVertices);
    verticesInHyperedge_.resize(numHyperedges);
}

}    // namespace osp
