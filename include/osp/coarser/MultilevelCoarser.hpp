/*
Copyright 2025 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Pal Andras Papp, Raphael S. Steiner
*/

#pragma once

#include <memory>
#include <numeric>
#include <set>
#include <vector>

#include "osp/auxiliary/return_status.hpp"
#include "osp/bsp/model/BspInstance.hpp"
#include "osp/coarser/Coarser.hpp"
#include "osp/coarser/coarser_util.hpp"

namespace osp {

template <typename GraphT, typename GraphTCoarse>
class MultilevelCoarseAndSchedule;

template <typename GraphT, typename GraphTCoarse>
class MultilevelCoarser : public Coarser<GraphT, GraphTCoarse> {
    friend class MultilevelCoarseAndSchedule<GraphT, GraphTCoarse>;

  private:
    const GraphT *originalGraph_;

  protected:
    inline const GraphT *GetOriginalGraph() const { return originalGraph_; };

    std::vector<std::unique_ptr<GraphTCoarse>> dagHistory_;
    std::vector<std::unique_ptr<std::vector<VertexIdxT<GraphTCoarse>>>> contractionMaps_;

    ReturnStatus AddContraction(const std::vector<VertexIdxT<GraphTCoarse>> &contractionMap);
    ReturnStatus AddContraction(std::vector<VertexIdxT<GraphTCoarse>> &&contractionMap);
    ReturnStatus AddContraction(const std::vector<VertexIdxT<GraphTCoarse>> &contractionMap, const GraphTCoarse &contractedGraph);
    ReturnStatus AddContraction(std::vector<VertexIdxT<GraphTCoarse>> &&contractionMap, GraphTCoarse &&contractedGraph);
    void AddIdentityContraction();

    std::vector<VertexIdxT<GraphTCoarse>> GetCombinedContractionMap() const;

    virtual ReturnStatus RunContractions() = 0;
    void CompactifyDagHistory();

    void ClearComputationData();

  public:
    MultilevelCoarser() : originalGraph_(nullptr) {};
    MultilevelCoarser(const GraphT &graph) : originalGraph_(&graph) {};
    virtual ~MultilevelCoarser() = default;

    bool CoarsenDag(const GraphT &dagIn,
                    GraphTCoarse &coarsenedDag,
                    std::vector<VertexIdxT<GraphTCoarse>> &vertexContractionMap) override;

    ReturnStatus Run(const GraphT &graph);
    ReturnStatus Run(const BspInstance<GraphT> &inst);

    virtual std::string GetCoarserName() const override = 0;
};

template <typename GraphT, typename GraphTCoarse>
ReturnStatus MultilevelCoarser<GraphT, GraphTCoarse>::Run(const GraphT &graph) {
    ClearComputationData();
    originalGraph_ = &graph;

    ReturnStatus status = ReturnStatus::OSP_SUCCESS;
    status = std::max(status, RunContractions());

    if (dagHistory_.size() == 0) {
        AddIdentityContraction();
    }

    return status;
}

template <typename GraphT, typename GraphTCoarse>
ReturnStatus MultilevelCoarser<GraphT, GraphTCoarse>::Run(const BspInstance<GraphT> &inst) {
    return Run(inst.GetComputationalDag());
}

template <typename GraphT, typename GraphTCoarse>
void MultilevelCoarser<GraphT, GraphTCoarse>::ClearComputationData() {
    dagHistory_.clear();
    dagHistory_.shrink_to_fit();

    contractionMaps_.clear();
    contractionMaps_.shrink_to_fit();
}

template <typename GraphT, typename GraphTCoarse>
void MultilevelCoarser<GraphT, GraphTCoarse>::CompactifyDagHistory() {
    if (dagHistory_.size() < 3) {
        return;
    }

    size_t dagIndxFirst = dagHistory_.size() - 2;
    size_t mapIndxFirst = contractionMaps_.size() - 2;

    size_t dagIndxSecond = dagHistory_.size() - 1;
    size_t mapIndxSecond = contractionMaps_.size() - 1;

    if ((static_cast<double>(dagHistory_[dagIndxFirst - 1]->NumVertices())
         / static_cast<double>(dagHistory_[dagIndxSecond - 1]->NumVertices()))
        > 1.25) {
        return;
    }

    // Compute combined contraction_map
    std::unique_ptr<std::vector<VertexIdxT<GraphTCoarse>>> combiContractionMap
        = std::make_unique<std::vector<VertexIdxT<GraphTCoarse>>>(contractionMaps_[mapIndxFirst]->size());
    for (std::size_t vert = 0; vert < contractionMaps_[mapIndxFirst]->size(); ++vert) {
        combiContractionMap->at(vert) = contractionMaps_[mapIndxSecond]->at(contractionMaps_[mapIndxFirst]->at(vert));
    }

    // Delete ComputationalDag
    auto dagIt = dagHistory_.begin();
    std::advance(dagIt, dagIndxFirst);
    dagHistory_.erase(dagIt);

    // Delete contraction map
    auto contrMapIt = contractionMaps_.begin();
    std::advance(contrMapIt, mapIndxSecond);
    contractionMaps_.erase(contrMapIt);

    // Replace contraction map
    contractionMaps_[mapIndxFirst] = std::move(combiContractionMap);
}

template <typename GraphT, typename GraphTCoarse>
ReturnStatus MultilevelCoarser<GraphT, GraphTCoarse>::AddContraction(const std::vector<VertexIdxT<GraphTCoarse>> &contractionMap) {
    std::unique_ptr<GraphTCoarse> newGraph = std::make_unique<GraphTCoarse>();

    contractionMaps_.emplace_back(contractionMap);

    bool success = false;

    if (dagHistory_.size() == 0) {
        success = coarser_util::ConstructCoarseDag<GraphT, GraphTCoarse>(
            *(GetOriginalGraph()), *newGraph, *(contractionMaps_.back()));
    } else {
        success = coarser_util::ConstructCoarseDag<GraphTCoarse, GraphTCoarse>(
            *(dagHistory_.back()), *newGraph, *(contractionMaps_.back()));
    }

    dagHistory_.emplace_back(std::move(newGraph));

    if (success) {
        CompactifyDagHistory();
        return ReturnStatus::OSP_SUCCESS;
    } else {
        return ReturnStatus::ERROR;
    }
}

template <typename GraphT, typename GraphTCoarse>
ReturnStatus MultilevelCoarser<GraphT, GraphTCoarse>::AddContraction(std::vector<VertexIdxT<GraphTCoarse>> &&contractionMap) {
    std::unique_ptr<GraphTCoarse> newGraph = std::make_unique<GraphTCoarse>();

    std::unique_ptr<std::vector<VertexIdxT<GraphTCoarse>>> contrMapPtr(
        new std::vector<VertexIdxT<GraphTCoarse>>(std::move(contractionMap)));
    contractionMaps_.emplace_back(std::move(contrMapPtr));

    bool success = false;

    if (dagHistory_.size() == 0) {
        success = coarser_util::ConstructCoarseDag<GraphT, GraphTCoarse>(
            *(GetOriginalGraph()), *newGraph, *(contractionMaps_.back()));
    } else {
        success = coarser_util::ConstructCoarseDag<GraphTCoarse, GraphTCoarse>(
            *(dagHistory_.back()), *newGraph, *(contractionMaps_.back()));
    }

    dagHistory_.emplace_back(std::move(newGraph));

    if (success) {
        CompactifyDagHistory();
        return ReturnStatus::OSP_SUCCESS;
    } else {
        return ReturnStatus::ERROR;
    }
}

template <typename GraphT, typename GraphTCoarse>
ReturnStatus MultilevelCoarser<GraphT, GraphTCoarse>::AddContraction(const std::vector<VertexIdxT<GraphTCoarse>> &contractionMap,
                                                                     const GraphTCoarse &contractedGraph) {
    std::unique_ptr<GraphTCoarse> graphPtr(new GraphTCoarse(contractedGraph));
    dagHistory_.emplace_back(std::move(graphPtr));

    std::unique_ptr<std::vector<VertexIdxT<GraphTCoarse>>> contrMapPtr(new std::vector<VertexIdxT<GraphTCoarse>>(contractionMap));
    contractionMaps_.emplace_back(std::move(contrMapPtr));

    CompactifyDagHistory();
    return ReturnStatus::OSP_SUCCESS;
}

template <typename GraphT, typename GraphTCoarse>
ReturnStatus MultilevelCoarser<GraphT, GraphTCoarse>::AddContraction(std::vector<VertexIdxT<GraphTCoarse>> &&contractionMap,
                                                                     GraphTCoarse &&contractedGraph) {
    std::unique_ptr<GraphTCoarse> graphPtr(new GraphTCoarse(std::move(contractedGraph)));
    dagHistory_.emplace_back(std::move(graphPtr));

    std::unique_ptr<std::vector<VertexIdxT<GraphTCoarse>>> contrMapPtr(
        new std::vector<VertexIdxT<GraphTCoarse>>(std::move(contractionMap)));
    contractionMaps_.emplace_back(std::move(contrMapPtr));

    CompactifyDagHistory();
    return ReturnStatus::OSP_SUCCESS;
}

template <typename GraphT, typename GraphTCoarse>
std::vector<VertexIdxT<GraphTCoarse>> MultilevelCoarser<GraphT, GraphTCoarse>::GetCombinedContractionMap() const {
    std::vector<VertexIdxT<GraphTCoarse>> combinedContractionMap(originalGraph_->NumVertices());
    std::iota(combinedContractionMap.begin(), combinedContractionMap.end(), 0);

    for (std::size_t j = 0; j < contractionMaps_.size(); ++j) {
        for (std::size_t i = 0; i < combinedContractionMap.size(); ++i) {
            combinedContractionMap[i] = contractionMaps_[j]->at(combinedContractionMap[i]);
        }
    }

    return combinedContractionMap;
}

template <typename GraphT, typename GraphTCoarse>
bool MultilevelCoarser<GraphT, GraphTCoarse>::CoarsenDag(const GraphT &dagIn,
                                                         GraphTCoarse &coarsenedDag,
                                                         std::vector<VertexIdxT<GraphTCoarse>> &vertexContractionMap) {
    ClearComputationData();

    ReturnStatus status = Run(dagIn);

    if (status != ReturnStatus::OSP_SUCCESS && status != ReturnStatus::BEST_FOUND) {
        return false;
    }

    assert(dagHistory_.size() != 0);
    coarsenedDag = *(dagHistory_.back());

    vertexContractionMap = GetCombinedContractionMap();

    return true;
}

template <typename GraphT, typename GraphTCoarse>
void MultilevelCoarser<GraphT, GraphTCoarse>::AddIdentityContraction() {
    std::size_t nVert;
    if (dagHistory_.size() == 0) {
        nVert = static_cast<std::size_t>(originalGraph_->NumVertices());
    } else {
        nVert = static_cast<std::size_t>(dagHistory_.back()->NumVertices());
    }

    std::vector<VertexIdxT<GraphTCoarse>> contractionMap(nVert);
    std::iota(contractionMap.begin(), contractionMap.end(), 0);

    AddContraction(std::move(contractionMap));
    CompactifyDagHistory();
}

}    // end namespace osp
