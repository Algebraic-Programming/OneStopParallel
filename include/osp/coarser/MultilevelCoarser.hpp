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
    std::vector<std::unique_ptr<std::vector<vertex_idx_t<Graph_t_coarse>>>> contractionMaps_;

    RETURN_STATUS AddContraction(const std::vector<vertex_idx_t<Graph_t_coarse>> &contractionMap);
    RETURN_STATUS AddContraction(std::vector<vertex_idx_t<Graph_t_coarse>> &&contractionMap);
    RETURN_STATUS AddContraction(const std::vector<vertex_idx_t<Graph_t_coarse>> &contractionMap,
                                 const GraphTCoarse &contractedGraph);
    RETURN_STATUS AddContraction(std::vector<vertex_idx_t<Graph_t_coarse>> &&contractionMap, GraphTCoarse &&contractedGraph);
    void AddIdentityContraction();

    std::vector<vertex_idx_t<Graph_t_coarse>> GetCombinedContractionMap() const;

    virtual RETURN_STATUS RunContractions() = 0;
    void CompactifyDagHistory();

    void ClearComputationData();

  public:
    MultilevelCoarser() : originalGraph_(nullptr) {};
    MultilevelCoarser(const GraphT &graph) : originalGraph_(&graph) {};
    virtual ~MultilevelCoarser() = default;

    bool coarsenDag(const GraphT &dagIn,
                    GraphTCoarse &coarsenedDag,
                    std::vector<vertex_idx_t<Graph_t_coarse>> &vertexContractionMap) override;

    RETURN_STATUS Run(const GraphT &graph);
    RETURN_STATUS Run(const BspInstance<GraphT> &inst);

    virtual std::string getCoarserName() const override = 0;
};

template <typename GraphT, typename GraphTCoarse>
RETURN_STATUS MultilevelCoarser<GraphT, GraphTCoarse>::Run(const GraphT &graph) {
    ClearComputationData();
    originalGraph_ = &graph;

    RETURN_STATUS status = RETURN_STATUS::OSP_SUCCESS;
    status = std::max(status, run_contractions());

    if (dagHistory_.size() == 0) {
        AddIdentityContraction();
    }

    return status;
}

template <typename GraphT, typename GraphTCoarse>
RETURN_STATUS MultilevelCoarser<GraphT, GraphTCoarse>::Run(const BspInstance<GraphT> &inst) {
    return run(inst.getComputationalDag());
}

template <typename GraphT, typename GraphTCoarse>
void MultilevelCoarser<GraphT, GraphTCoarse>::ClearComputationData() {
    dagHistory_.clear();
    dagHistory_.shrink_to_fit();

    contraction_maps.clear();
    contraction_maps.shrink_to_fit();
}

template <typename GraphT, typename GraphTCoarse>
void MultilevelCoarser<GraphT, GraphTCoarse>::CompactifyDagHistory() {
    if (dagHistory_.size() < 3) {
        return;
    }

    size_t dagIndxFirst = dagHistory_.size() - 2;
    size_t mapIndxFirst = contraction_maps.size() - 2;

    size_t dagIndxSecond = dagHistory_.size() - 1;
    size_t mapIndxSecond = contraction_maps.size() - 1;

    if ((static_cast<double>(dagHistory_[dagIndxFirst - 1]->num_vertices())
         / static_cast<double>(dagHistory_[dagIndxSecond - 1]->num_vertices()))
        > 1.25) {
        return;
    }

    // Compute combined contraction_map
    std::unique_ptr<std::vector<vertex_idx_t<Graph_t_coarse>>> combiContractionMap
        = std::make_unique<std::vector<vertex_idx_t<Graph_t_coarse>>>(contraction_maps[map_indx_first]->size());
    for (std::size_t vert = 0; vert < contraction_maps[map_indx_first]->size(); ++vert) {
        combi_contraction_map->at(vert) = contraction_maps[map_indx_second]->at(contraction_maps[map_indx_first]->at(vert));
    }

    // Delete ComputationalDag
    auto dagIt = dagHistory_.begin();
    std::advance(dagIt, dagIndxFirst);
    dagHistory_.erase(dagIt);

    // Delete contraction map
    auto contrMapIt = contraction_maps.begin();
    std::advance(contr_map_it, mapIndxSecond);
    contraction_maps.erase(contr_map_it);

    // Replace contraction map
    contraction_maps[map_indx_first] = std::move(combi_contraction_map);
}

template <typename GraphT, typename GraphTCoarse>
RETURN_STATUS MultilevelCoarser<GraphT, GraphTCoarse>::AddContraction(
    const std::vector<vertex_idx_t<Graph_t_coarse>> &contractionMap) {
    std::unique_ptr<GraphTCoarse> newGraph = std::make_unique<GraphTCoarse>();

    contraction_maps.emplace_back(contraction_map);

    bool success = false;

    if (dagHistory_.size() == 0) {
        success = coarser_util::construct_coarse_dag<Graph_t, Graph_t_coarse>(
            *(getOriginalGraph()), *new_graph, *(contraction_maps.back()));
    } else {
        success = coarser_util::construct_coarse_dag<Graph_t_coarse, Graph_t_coarse>(
            *(dag_history.back()), *new_graph, *(contraction_maps.back()));
    }

    dagHistory_.emplace_back(std::move(newGraph));

    if (success) {
        CompactifyDagHistory();
        return RETURN_STATUS::OSP_SUCCESS;
    } else {
        return RETURN_STATUS::ERROR;
    }
}

template <typename GraphT, typename GraphTCoarse>
RETURN_STATUS MultilevelCoarser<GraphT, GraphTCoarse>::AddContraction(std::vector<vertex_idx_t<Graph_t_coarse>> &&contractionMap) {
    std::unique_ptr<GraphTCoarse> newGraph = std::make_unique<GraphTCoarse>();

    std::unique_ptr<std::vector<vertex_idx_t<Graph_t_coarse>>> contrMapPtr(
        new std::vector<vertex_idx_t<Graph_t_coarse>>(std::move(contraction_map)));
    contraction_maps.emplace_back(std::move(contr_map_ptr));

    bool success = false;

    if (dagHistory_.size() == 0) {
        success = coarser_util::construct_coarse_dag<Graph_t, Graph_t_coarse>(
            *(getOriginalGraph()), *new_graph, *(contraction_maps.back()));
    } else {
        success = coarser_util::construct_coarse_dag<Graph_t_coarse, Graph_t_coarse>(
            *(dag_history.back()), *new_graph, *(contraction_maps.back()));
    }

    dagHistory_.emplace_back(std::move(newGraph));

    if (success) {
        CompactifyDagHistory();
        return RETURN_STATUS::OSP_SUCCESS;
    } else {
        return RETURN_STATUS::ERROR;
    }
}

template <typename GraphT, typename GraphTCoarse>
RETURN_STATUS MultilevelCoarser<GraphT, GraphTCoarse>::AddContraction(
    const std::vector<vertex_idx_t<Graph_t_coarse>> &contractionMap, const GraphTCoarse &contractedGraph) {
    std::unique_ptr<GraphTCoarse> graphPtr(new GraphTCoarse(contractedGraph));
    dagHistory_.emplace_back(std::move(graphPtr));

    std::unique_ptr<std::vector<vertex_idx_t<Graph_t_coarse>>> contrMapPtr(
        new std::vector<vertex_idx_t<Graph_t_coarse>>(contraction_map));
    contraction_maps.emplace_back(std::move(contr_map_ptr));

    CompactifyDagHistory();
    return RETURN_STATUS::OSP_SUCCESS;
}

template <typename GraphT, typename GraphTCoarse>
RETURN_STATUS MultilevelCoarser<GraphT, GraphTCoarse>::AddContraction(std::vector<vertex_idx_t<Graph_t_coarse>> &&contractionMap,
                                                                      GraphTCoarse &&contractedGraph) {
    std::unique_ptr<GraphTCoarse> graphPtr(new GraphTCoarse(std::move(contractedGraph)));
    dagHistory_.emplace_back(std::move(graphPtr));

    std::unique_ptr<std::vector<vertex_idx_t<Graph_t_coarse>>> contrMapPtr(
        new std::vector<vertex_idx_t<Graph_t_coarse>>(std::move(contraction_map)));
    contraction_maps.emplace_back(std::move(contr_map_ptr));

    CompactifyDagHistory();
    return RETURN_STATUS::OSP_SUCCESS;
}

template <typename GraphT, typename GraphTCoarse>
std::vector<vertex_idx_t<Graph_t_coarse>> MultilevelCoarser<GraphT, GraphTCoarse>::GetCombinedContractionMap() const {
    std::vector<vertex_idx_t<Graph_t_coarse>> combinedContractionMap(originalGraph_->num_vertices());
    std::iota(combinedContractionMap.begin(), combinedContractionMap.end(), 0);

    for (std::size_t j = 0; j < contraction_maps.size(); ++j) {
        for (std::size_t i = 0; i < combinedContractionMap.size(); ++i) {
            combinedContractionMap[i] = contraction_maps[j]->at(combinedContractionMap[i]);
        }
    }

    return combinedContractionMap;
}

template <typename GraphT, typename GraphTCoarse>
bool MultilevelCoarser<GraphT, GraphTCoarse>::CoarsenDag(const GraphT &dagIn,
                                                         GraphTCoarse &coarsenedDag,
                                                         std::vector<vertex_idx_t<Graph_t_coarse>> &vertexContractionMap) {
    ClearComputationData();

    RETURN_STATUS status = run(dag_in);

    if (status != RETURN_STATUS::OSP_SUCCESS && status != RETURN_STATUS::BEST_FOUND) {
        return false;
    }

    assert(dagHistory_.size() != 0);
    coarsenedDag = *(dagHistory_.back());

    vertex_contraction_map = getCombinedContractionMap();

    return true;
}

template <typename GraphT, typename GraphTCoarse>
void MultilevelCoarser<GraphT, GraphTCoarse>::AddIdentityContraction() {
    std::size_t nVert;
    if (dagHistory_.size() == 0) {
        nVert = static_cast<std::size_t>(originalGraph_->num_vertices());
    } else {
        nVert = static_cast<std::size_t>(dagHistory_.back()->num_vertices());
    }

    std::vector<vertex_idx_t<Graph_t_coarse>> contractionMap(nVert);
    std::iota(contraction_map.begin(), contraction_map.end(), 0);

    add_contraction(std::move(contraction_map));
    CompactifyDagHistory();
}

}    // end namespace osp
