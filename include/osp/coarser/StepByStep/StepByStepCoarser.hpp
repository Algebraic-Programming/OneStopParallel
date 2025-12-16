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

#include "osp/coarser/Coarser.hpp"
#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/concepts/constructable_computational_dag_concept.hpp"
#include "osp/graph_algorithms/computational_dag_construction_util.hpp"
#include "osp/graph_algorithms/directed_graph_edge_desc_util.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"

namespace osp {

template <typename GraphT>
class StepByStepCoarser : public CoarserGenContractionMap<GraphT, GraphT> {
    using VertexIdx = VertexIdxT<GraphT>;

    using VertexTypeTOrDefault = std::conditional_t<isComputationalDagTypedVerticesV<GraphT>, VTypeT<GraphT>, unsigned>;
    using EdgeCommwTOrDefault = std::conditional_t<hasEdgeWeightsV<GraphT>, ECommwT<GraphT>, VCommwT<GraphT>>;

    using BoostGraphT
        = BoostGraph<VWorkwT<GraphT>, VCommwT<GraphT>, VMemwT<GraphT>, VertexTypeTOrDefault, EdgeCommwTOrDefault>;

  public:
    enum CoarseningStrategy { EDGE_BY_EDGE, BOTTOM_LEVEL_CLUSTERS };

    enum ProblemType { SCHEDULING, PEBBLING };

    struct EdgeToContract {
        std::pair<VertexIdx, VertexIdx> edge_;
        VWorkwT<GraphT> workWeight_;
        VCommwT<GraphT> commWeight_;

        EdgeToContract(const VertexIdx source,
                       const VertexIdx target,
                       const VWorkwT<GraphT> workWeight,
                       const VCommwT<GraphT> commWeight)
            : edge_(source, target), workWeight_(workWeight), commWeight_(commWeight) {}

        bool operator<(const EdgeToContract &other) const {
            return (workWeight_ < other.workWeight_ || (workWeight_ == other.workWeight_ && commWeight_ < other.commWeight_) ||
            (workWeight_ == other.workWeight_ && commWeight_ == other.commWeight_ && edge_ < other.edge_)
        );
        }
    };

  private:
    std::vector<std::pair<VertexIdx, VertexIdx>> contractionHistory_;

    CoarseningStrategy coarseningStrategy_ = CoarseningStrategy::EDGE_BY_EDGE;
    ProblemType problemType_ = ProblemType::SCHEDULING;

    unsigned targetNrOfNodes_ = 0;

    GraphT gFull_;
    BoostGraphT gCoarse_;

    std::vector<std::set<VertexIdx>> contains_;

    std::map<std::pair<VertexIdx, VertexIdx>, VCommwT<GraphT>> edgeWeights_;
    std::map<std::pair<VertexIdx, VertexIdx>, VCommwT<GraphT>> contractable_;
    std::vector<bool> nodeValid_;
    std::vector<VertexIdx> topOrderIdx_;

    VMemwT<GraphT> fastMemCapacity_ = std::numeric_limits<VMemwT<GraphT>>::max();    // for pebbling

    // Utility functions for coarsening in general
    void ContractSingleEdge(std::pair<VertexIdx, VertexIdx> edge);
    void ComputeFilteredTopOrderIdx();

    void InitializeContractableEdges();
    bool IsContractable(std::pair<VertexIdx, VertexIdx> edge) const;
    std::set<VertexIdx> GetContractableChildren(VertexIdx node) const;
    std::set<VertexIdx> GetContractableParents(VertexIdx node) const;
    void UpdateDistantEdgeContractibility(std::pair<VertexIdx, VertexIdx> edge);

    std::pair<VertexIdx, VertexIdx> PickEdgeToContract(const std::vector<EdgeToContract> &candidates) const;
    std::vector<EdgeToContract> CreateEdgeCandidateList() const;

    // Utility functions for cluster coarsening
    std::vector<std::pair<VertexIdx, VertexIdx>> ClusterCoarsen() const;
    std::vector<unsigned> ComputeFilteredTopLevel() const;

    // Utility functions for coarsening in a pebbling problem
    bool IncontractableForPebbling(const std::pair<VertexIdx, VertexIdx> &) const;
    void MergeSourcesInPebbling();

    // Utility for contracting into final format
    void SetIdVector(std::vector<VertexIdxT<GraphT>> &newVertexId) const;
    static std::vector<VertexIdx> GetFilteredTopOrderIdx(const GraphT &g, const std::vector<bool> &isValid);

  public:
    virtual ~StepByStepCoarser() = default;

    virtual std::string GetCoarserName() const override { return "StepByStepCoarsening"; }

    // DAG coarsening
    virtual std::vector<VertexIdxT<GraphT>> GenerateVertexContractionMap(const GraphT &dagIn) override;

    // Coarsening for pebbling problems - leaves source nodes intact, considers memory bound
    void CoarsenForPebbling(const GraphT &dagIn, GraphT &coarsenedDag, std::vector<VertexIdxT<GraphT>> &newVertexId);

    void SetCoarseningStrategy(CoarseningStrategy strategy) { coarseningStrategy_ = strategy; }

    void SetTargetNumberOfNodes(const unsigned nrNodes) { targetNrOfNodes_ = nrNodes; }

    void SetFastMemCapacity(const VMemwT<GraphT> capacity) { fastMemCapacity_ = capacity; }

    std::vector<std::pair<VertexIdx, VertexIdx>> GetContractionHistory() const { return contractionHistory_; }

    std::vector<VertexIdx> GetIntermediateIDs(VertexIdx untilWhichStep) const;
    GraphT Contract(const std::vector<VertexIdxT<GraphT>> &newVertexId) const;

    const GraphT &GetOriginalDag() const { return gFull_; }
};

// template<typename GraphT>
// bool StepByStepCoarser<GraphT>::coarseDag(const GraphT& dag_in, GraphT &dag_out,
//                         std::vector<std::vector<VertexIdxT<GraphT>>> &old_vertex_ids,
//                         std::vector<VertexIdxT<GraphT>> &new_vertex_id)

template <typename GraphT>
std::vector<VertexIdxT<GraphT>> StepByStepCoarser<GraphT>::GenerateVertexContractionMap(const GraphT &dagIn) {
    const unsigned n = static_cast<unsigned>(dagIn.NumVertices());

    gFull_ = dagIn;
    for (VertexIdx node = gCoarse_.NumVertices(); node > 0;) {
        --node;
        gCoarse_.RemoveVertex(node);
    }

    ConstructComputationalDag(gFull_, gCoarse_);

    contractionHistory_.clear();

    // target nr of nodes must be reasonable
    if (targetNrOfNodes_ == 0 || targetNrOfNodes_ > n) {
        targetNrOfNodes_ = std::max(n / 2, 1U);
    }

    // list of original node indices contained in each contracted node
    contains_.clear();
    contains_.resize(n);

    nodeValid_.clear();
    nodeValid_.resize(n, true);

    for (VertexIdx node = 0; node < n; ++node) {
        contains_[node].insert(node);
    }

    // used for original, slow coarsening
    edgeWeights_.clear();
    contractable_.clear();

    if (coarseningStrategy_ == CoarseningStrategy::EDGE_BY_EDGE) {
        // Init edge weights
        for (VertexIdx node = 0; node < n; ++node) {
            for (VertexIdx succ : gFull_.Children(node)) {
                edgeWeights_[std::make_pair(node, succ)] = gFull_.VertexCommWeight(node);
            }
        }

        // get original contractable edges
        InitializeContractableEdges();
    }

    for (unsigned NrOfNodes = n; NrOfNodes > targetNrOfNodes_;) {
        // Single contraction step

        std::vector<std::pair<VertexIdx, VertexIdx>> edgesToContract;

        // choose edges to contract in this step
        if (coarseningStrategy_ == CoarseningStrategy::EDGE_BY_EDGE) {
            std::vector<EdgeToContract> candidates_ = CreateEdgeCandidateList();
            if (candidates_.empty()) {
                std::cout << "Error: no more edges to contract" << std::endl;
                break;
            }
            std::pair<VertexIdx, VertexIdx> chosenEdge = PickEdgeToContract(candidates_);
            edgesToContract.push_back(chosenEdge);

            // Update far-away edges that become uncontractable now
            UpdateDistantEdgeContractibility(chosenEdge);
        } else {
            edgesToContract = ClusterCoarsen();
        }

        if (edgesToContract.empty()) {
            break;
        }

        // contract these edges
        for (const std::pair<VertexIdx, VertexIdx> &edge : edgesToContract) {
            if (coarseningStrategy_ == CoarseningStrategy::EDGE_BY_EDGE) {
                // Update contractable edges - edge.b
                for (VertexIdx pred : gCoarse_.Parents(edge.second)) {
                    contractable_.erase(std::make_pair(pred, edge.second));
                }

                for (VertexIdx succ : gCoarse_.Children(edge.second)) {
                    contractable_.erase(std::make_pair(edge.second, succ));
                }
            }

            ContractSingleEdge(edge);
            nodeValid_[edge.second] = false;

            if (coarseningStrategy_ == CoarseningStrategy::EDGE_BY_EDGE) {
                ComputeFilteredTopOrderIdx();

                // Update contractable edges - edge.a
                std::set<VertexIdx> contractableParents = GetContractableParents(edge.first);
                for (VertexIdx pred : gCoarse_.Parents(edge.first)) {
                    if (contractableParents.find(pred) != contractableParents.end()) {
                        contractable_[std::make_pair(pred, edge.first)] = edgeWeights_[std::make_pair(pred, edge.first)];
                    } else {
                        contractable_.erase(std::make_pair(pred, edge.first));
                    }
                }

                std::set<VertexIdx> contractableChildren = GetContractableChildren(edge.first);
                for (VertexIdx succ : gCoarse_.Children(edge.first)) {
                    if (contractableChildren.find(succ) != contractableChildren.end()) {
                        contractable_[std::make_pair(edge.first, succ)] = edgeWeights_[std::make_pair(edge.first, succ)];
                    } else {
                        contractable_.erase(std::make_pair(edge.first, succ));
                    }
                }
            }
            --NrOfNodes;
            if (NrOfNodes == targetNrOfNodes_) {
                break;
            }
        }
    }

    if (problemType_ == ProblemType::PEBBLING) {
        MergeSourcesInPebbling();
    }

    std::vector<VertexIdxT<GraphT>> newVertexId;
    SetIdVector(newVertexId);

    return newVertexId;
}

template <typename GraphT>
void StepByStepCoarser<GraphT>::ContractSingleEdge(std::pair<VertexIdx, VertexIdx> edge) {
    gCoarse_.SetVertexWorkWeight(edge.first, gCoarse_.VertexWorkWeight(edge.first) + gCoarse_.VertexWorkWeight(edge.second));
    gCoarse_.SetVertexWorkWeight(edge.second, 0);

    gCoarse_.SetVertexCommWeight(edge.first, gCoarse_.VertexCommWeight(edge.first) + gCoarse_.VertexCommWeight(edge.second));
    gCoarse_.SetVertexCommWeight(edge.second, 0);

    gCoarse_.SetVertexMemWeight(edge.first, gCoarse_.VertexMemWeight(edge.first) + gCoarse_.VertexMemWeight(edge.second));
    gCoarse_.SetVertexMemWeight(edge.second, 0);

    contractionHistory_.emplace_back(edge.first, edge.second);

    // process incoming edges
    std::set<VertexIdx> parentsOfSource;
    for (VertexIdx pred : gCoarse_.Parents(edge.first)) {
        parentsOfSource.insert(pred);
    }

    for (VertexIdx pred : gCoarse_.Parents(edge.second)) {
        if (pred == edge.first) {
            continue;
        }
        if (parentsOfSource.find(pred) != parentsOfSource.end())    // combine edges
        {
            edgeWeights_[std::make_pair(pred, edge.first)] = 0;
            for (VertexIdx node : contains_[pred]) {
                for (VertexIdx succ : gCoarse_.Children(node)) {
                    if (succ == edge.first || succ == edge.second) {
                        edgeWeights_[std::make_pair(pred, edge.first)] += gFull_.VertexCommWeight(node);
                    }
                }
            }

            edgeWeights_.erase(std::make_pair(pred, edge.second));
        } else    // add incoming edge
        {
            gCoarse_.AddEdge(pred, edge.first);
            edgeWeights_[std::make_pair(pred, edge.first)] = edgeWeights_[std::make_pair(pred, edge.second)];
        }
    }

    // process outgoing edges
    std::set<VertexIdx> childrenOfSource;
    for (VertexIdx succ : gCoarse_.Children(edge.first)) {
        childrenOfSource.insert(succ);
    }

    for (VertexIdx succ : gCoarse_.Children(edge.second)) {
        if (childrenOfSource.find(succ) != childrenOfSource.end())    // combine edges
        {
            edgeWeights_[std::make_pair(edge.first, succ)] += edgeWeights_[std::make_pair(edge.second, succ)];
            edgeWeights_.erase(std::make_pair(edge.second, succ));
        } else    // add outgoing edge
        {
            gCoarse_.AddEdge(edge.first, succ);
            edgeWeights_[std::make_pair(edge.first, succ)] = edgeWeights_[std::make_pair(edge.second, succ)];
        }
    }

    gCoarse_.ClearVertex(edge.second);

    for (VertexIdx node : contains_[edge.second]) {
        contains_[edge.first].insert(node);
    }

    contains_[edge.second].clear();
}

template <typename GraphT>
bool StepByStepCoarser<GraphT>::IsContractable(std::pair<VertexIdx, VertexIdx> edge) const {
    std::deque<VertexIdx> queue;
    std::set<VertexIdx> visited;
    for (VertexIdx succ : gCoarse_.Children(edge.first)) {
        if (nodeValid_[succ] && topOrderIdx_[succ] < topOrderIdx_[edge.second]) {
            queue.push_back(succ);
            visited.insert(succ);
        }
    }

    while (!queue.empty()) {
        const VertexIdx node = queue.front();
        queue.pop_front();
        for (VertexIdx succ : gCoarse_.Children(node)) {
            if (succ == edge.second) {
                return false;
            }

            if (nodeValid_[succ] && topOrderIdx_[succ] < topOrderIdx_[edge.second] && visited.count(succ) == 0) {
                queue.push_back(succ);
                visited.insert(succ);
            }
        }
    }
    return true;
}

template <typename GraphT>
std::set<VertexIdxT<GraphT>> StepByStepCoarser<GraphT>::GetContractableChildren(const VertexIdx node) const {
    std::deque<VertexIdx> queue;
    std::set<VertexIdx> visited;
    std::set<VertexIdx> succContractable;
    VertexIdx topOrderMax = topOrderIdx_[node];

    for (VertexIdx succ : gCoarse_.Children(node)) {
        if (nodeValid_[succ]) {
            succContractable.insert(succ);
        }

        if (topOrderIdx_[succ] > topOrderMax) {
            topOrderMax = topOrderIdx_[succ];
        }

        if (nodeValid_[succ]) {
            queue.push_back(succ);
            visited.insert(succ);
        }
    }

    while (!queue.empty()) {
        const VertexIdx nodeLocal = queue.front();
        queue.pop_front();
        for (VertexIdx succ : gCoarse_.Children(nodeLocal)) {
            succContractable.erase(succ);

            if (nodeValid_[succ] && topOrderIdx_[succ] < topOrderMax && visited.count(succ) == 0) {
                queue.push_back(succ);
                visited.insert(succ);
            }
        }
    }

    return succContractable;
}

template <typename GraphT>
std::set<VertexIdxT<GraphT>> StepByStepCoarser<GraphT>::GetContractableParents(const VertexIdx node) const {
    std::deque<VertexIdx> queue;
    std::set<VertexIdx> visited;
    std::set<VertexIdx> predContractable;
    VertexIdx topOrderMin = topOrderIdx_[node];

    for (VertexIdx pred : gCoarse_.Parents(node)) {
        if (nodeValid_[pred]) {
            predContractable.insert(pred);
        }

        if (topOrderIdx_[pred] < topOrderMin) {
            topOrderMin = topOrderIdx_[pred];
        }

        if (nodeValid_[pred]) {
            queue.push_back(pred);
            visited.insert(pred);
        }
    }

    while (!queue.empty()) {
        const VertexIdx nodeLocal = queue.front();
        queue.pop_front();
        for (VertexIdx pred : gCoarse_.Parents(nodeLocal)) {
            predContractable.erase(pred);

            if (nodeValid_[pred] && topOrderIdx_[pred] > topOrderMin && visited.count(pred) == 0) {
                queue.push_back(pred);
                visited.insert(pred);
            }
        }
    }

    return predContractable;
}

template <typename GraphT>
void StepByStepCoarser<GraphT>::InitializeContractableEdges() {
    ComputeFilteredTopOrderIdx();

    for (VertexIdx node = 0; node < gFull_.NumVertices(); ++node) {
        std::set<VertexIdx> succContractable = GetContractableChildren(node);
        for (VertexIdx succ : succContractable) {
            contractable_[std::make_pair(node, succ)] = gFull_.VertexCommWeight(node);
        }
    }
}

template <typename GraphT>
void StepByStepCoarser<GraphT>::UpdateDistantEdgeContractibility(std::pair<VertexIdx, VertexIdx> edge) {
    std::unordered_set<VertexIdx> ancestors, descendant;
    std::deque<VertexIdx> queue;
    for (VertexIdx succ : gCoarse_.Children(edge.first)) {
        if (succ != edge.second) {
            queue.push_back(succ);
            descendant.insert(succ);
        }
    }
    while (!queue.empty()) {
        const VertexIdx node = queue.front();
        queue.pop_front();
        for (VertexIdx succ : gCoarse_.Children(node)) {
            if (descendant.count(succ) == 0) {
                queue.push_back(succ);
                descendant.insert(succ);
            }
        }
    }

    for (VertexIdx pred : gCoarse_.Parents(edge.second)) {
        if (pred != edge.first) {
            queue.push_back(pred);
            ancestors.insert(pred);
        }
    }
    while (!queue.empty()) {
        const VertexIdx node = queue.front();
        queue.pop_front();
        for (VertexIdx pred : gCoarse_.Parents(node)) {
            if (ancestors.count(pred) == 0) {
                queue.push_back(pred);
                ancestors.insert(pred);
            }
        }
    }

    for (const VertexIdx node : ancestors) {
        for (const VertexIdx succ : gCoarse_.Children(node)) {
            if (descendant.count(succ) > 0) {
                contractable_.erase(std::make_pair(node, succ));
            }
        }
    }
}

template <typename GraphT>
std::vector<typename StepByStepCoarser<GraphT>::EdgeToContract> StepByStepCoarser<GraphT>::CreateEdgeCandidateList() const {
    std::vector<EdgeToContract> candidates;

    for (auto it = contractable_.cbegin(); it != contractable_.cend(); ++it) {
        if (problemType_ == ProblemType::PEBBLING && IncontractableForPebbling(it->first)) {
            continue;
        }

        candidates.emplace_back(
            it->first.first, it->first.second, contains_[it->first.first].size() + contains_[it->first.second].size(), it->second);
    }

    std::sort(candidates.begin(), candidates.end());
    return candidates;
}

template <typename GraphT>
std::pair<VertexIdxT<GraphT>, VertexIdxT<GraphT>> StepByStepCoarser<GraphT>::PickEdgeToContract(
    const std::vector<EdgeToContract> &candidates) const {
    size_t limit = (candidates.size() + 2) / 3;
    VWorkwT<GraphT> limitCardinality = candidates[limit].workWeight_;
    while (limit < candidates.size() - 1 && candidates[limit + 1].workWeight_ == limitCardinality) {
        ++limit;
    }

    // an edge case
    if (candidates.size() == 1) {
        limit = 0;
    }

    EdgeToContract chosen = candidates[0];
    unsigned best = 0;
    for (unsigned idx = 1; idx <= limit; ++idx) {
        if (candidates[idx].commWeight_ > candidates[best].commWeight_) {
            best = idx;
        }
    }

    chosen = candidates[best];
    return chosen.edge_;
}

/**
 * @brief Acyclic graph contractor based on (Herrmann, Julien, et al. "Acyclic partitioning of large directed acyclic graphs."
 * 2017 17th IEEE/ACM international symposium on cluster, cloud and grid computing (CCGRID). IEEE, 2017.))
 * @brief with minor changes and fixes
 *
 */
template <typename GraphT>
std::vector<std::pair<VertexIdxT<GraphT>, VertexIdxT<GraphT>>> StepByStepCoarser<GraphT>::ClusterCoarsen() const {
    std::vector<bool> singleton(gFull_.NumVertices(), true);
    std::vector<VertexIdx> leader(gFull_.NumVertices());
    std::vector<unsigned> weight(gFull_.NumVertices());
    std::vector<unsigned> nrBadNeighbors(gFull_.NumVertices());
    std::vector<VertexIdx> leaderBadNeighbors(gFull_.NumVertices());

    std::vector<unsigned> minTopLevel(gFull_.NumVertices());
    std::vector<unsigned> maxTopLevel(gFull_.NumVertices());
    std::vector<VertexIdx> clusterNewID(gFull_.NumVertices());

    std::vector<std::pair<VertexIdx, VertexIdx>> contractionSteps;
    std::vector<unsigned> topLevel = ComputeFilteredTopLevel();
    for (VertexIdx node = 0; node < gFull_.NumVertices(); ++node) {
        if (nodeValid_[node]) {
            leader[node] = node;
            weight[node] = 1 /*gCoarse_.vertex_work_weight(node)*/;
            nrBadNeighbors[node] = 0;
            leaderBadNeighbors[node] = UINT_MAX;
            clusterNewID[node] = node;
            minTopLevel[node] = topLevel[node];
            maxTopLevel[node] = topLevel[node];
        }
    }

    for (VertexIdx node = 0; node < gFull_.NumVertices(); ++node) {
        if (!nodeValid_[node] || !singleton[node]) {
            continue;
        }

        if (nrBadNeighbors[node] > 1) {
            continue;
        }

        std::vector<VertexIdx> validNeighbors;
        for (VertexIdx pred : gCoarse_.Parents(node)) {
            // direct check of condition 1
            if (topLevel[node] < maxTopLevel[leader[pred]] - 1 || topLevel[node] > minTopLevel[leader[pred]] + 1) {
                continue;
            }
            // indirect check of condition 2
            if (nrBadNeighbors[node] > 1 || (nrBadNeighbors[node] == 1 && leaderBadNeighbors[node] != leader[pred])) {
                continue;
            }
            // check condition 2 for pred if it is a singleton
            if (singleton[pred] && nrBadNeighbors[pred] > 0) {
                continue;
            }

            // check viability for pebbling
            if (problemType_ == ProblemType::PEBBLING && IncontractableForPebbling(std::make_pair(pred, node))) {
                continue;
            }

            validNeighbors.push_back(pred);
        }
        for (VertexIdx succ : gCoarse_.Children(node)) {
            // direct check of condition 1
            if (topLevel[node] < maxTopLevel[leader[succ]] - 1 || topLevel[node] > minTopLevel[leader[succ]] + 1) {
                continue;
            }
            // indirect check of condition 2
            if (nrBadNeighbors[node] > 1 || (nrBadNeighbors[node] == 1 && leaderBadNeighbors[node] != leader[succ])) {
                continue;
            }
            // check condition 2 for pred if it is a singleton
            if (singleton[succ] && nrBadNeighbors[succ] > 0) {
                continue;
            }

            // check viability for pebbling
            if (problemType_ == ProblemType::PEBBLING && IncontractableForPebbling(std::make_pair(node, succ))) {
                continue;
            }

            validNeighbors.push_back(succ);
        }

        VertexIdx bestNeighbor = std::numeric_limits<VertexIdx>::max();
        for (VertexIdx neigh : validNeighbors) {
            if (bestNeighbor == std::numeric_limits<VertexIdx>::max() || weight[leader[neigh]] < weight[leader[bestNeighbor]]) {
                bestNeighbor = neigh;
            }
        }

        if (bestNeighbor == std::numeric_limits<VertexIdx>::max()) {
            continue;
        }

        VertexIdx newLead = leader[bestNeighbor];
        leader[node] = newLead;
        weight[newLead] += weight[node];

        bool isParent = false;
        for (VertexIdx pred : gCoarse_.Parents(node)) {
            if (pred == bestNeighbor) {
                isParent = true;
            }
        }

        if (isParent) {
            contractionSteps.emplace_back(clusterNewID[newLead], node);
        } else {
            contractionSteps.emplace_back(node, clusterNewID[newLead]);
            clusterNewID[newLead] = node;
        }

        minTopLevel[newLead] = std::min(minTopLevel[newLead], topLevel[node]);
        maxTopLevel[newLead] = std::max(maxTopLevel[newLead], topLevel[node]);

        for (VertexIdx pred : gCoarse_.Parents(node)) {
            if (std::abs(static_cast<int>(topLevel[pred]) - static_cast<int>(maxTopLevel[newLead])) != 1
                && std::abs(static_cast<int>(topLevel[pred]) - static_cast<int>(minTopLevel[newLead])) != 1) {
                continue;
            }

            if (nrBadNeighbors[pred] == 0) {
                ++nrBadNeighbors[pred];
                leaderBadNeighbors[pred] = newLead;
            } else if (nrBadNeighbors[pred] == 1 && leaderBadNeighbors[pred] != newLead) {
                ++nrBadNeighbors[pred];
            }
        }
        for (VertexIdx succ : gCoarse_.Children(node)) {
            if (std::abs(static_cast<int>(topLevel[succ]) - static_cast<int>(maxTopLevel[newLead])) != 1
                && std::abs(static_cast<int>(topLevel[succ]) - static_cast<int>(minTopLevel[newLead])) != 1) {
                continue;
            }

            if (nrBadNeighbors[succ] == 0) {
                ++nrBadNeighbors[succ];
                leaderBadNeighbors[succ] = newLead;
            } else if (nrBadNeighbors[succ] == 1 && leaderBadNeighbors[succ] != newLead) {
                ++nrBadNeighbors[succ];
            }
        }

        if (singleton[bestNeighbor]) {
            for (VertexIdx pred : gCoarse_.Parents(bestNeighbor)) {
                if (std::abs(static_cast<int>(topLevel[pred]) - static_cast<int>(maxTopLevel[newLead])) != 1
                    && std::abs(static_cast<int>(topLevel[pred]) - static_cast<int>(minTopLevel[newLead])) != 1) {
                    continue;
                }

                if (nrBadNeighbors[pred] == 0) {
                    ++nrBadNeighbors[pred];
                    leaderBadNeighbors[pred] = newLead;
                } else if (nrBadNeighbors[pred] == 1 && leaderBadNeighbors[pred] != newLead) {
                    ++nrBadNeighbors[pred];
                }
            }
            for (VertexIdx succ : gCoarse_.Children(bestNeighbor)) {
                if (std::abs(static_cast<int>(topLevel[succ]) - static_cast<int>(maxTopLevel[newLead])) != 1
                    && std::abs(static_cast<int>(topLevel[succ]) - static_cast<int>(minTopLevel[newLead])) != 1) {
                    continue;
                }

                if (nrBadNeighbors[succ] == 0) {
                    ++nrBadNeighbors[succ];
                    leaderBadNeighbors[succ] = newLead;
                } else if (nrBadNeighbors[succ] == 1 && leaderBadNeighbors[succ] != newLead) {
                    ++nrBadNeighbors[succ];
                }
            }
            singleton[bestNeighbor] = false;
        }
        singleton[node] = false;
    }

    return contractionSteps;
}

template <typename GraphT>
std::vector<unsigned> StepByStepCoarser<GraphT>::ComputeFilteredTopLevel() const {
    std::vector<unsigned> topLevel(gFull_.NumVertices());
    for (const VertexIdx node : TopSortView(gCoarse_)) {
        if (!nodeValid_[node]) {
            continue;
        }

        topLevel[node] = 0;
        for (const VertexIdx pred : gCoarse_.Parents(node)) {
            topLevel[node] = std::max(topLevel[node], topLevel[pred] + 1);
        }
    }
    return topLevel;
}

template <typename GraphT>
void StepByStepCoarser<GraphT>::ComputeFilteredTopOrderIdx() {
    topOrderIdx_ = GetFilteredTopOrderIdx(gCoarse_, nodeValid_);
}

template <typename GraphT>
std::vector<VertexIdxT<GraphT>> StepByStepCoarser<GraphT>::GetFilteredTopOrderIdx(const GraphT &g,
                                                                                  const std::vector<bool> &isValid) {
    std::vector<VertexIdx> topOrder = GetFilteredTopOrder(isValid, g);
    std::vector<VertexIdx> idx(g.NumVertices());
    for (VertexIdx node = 0; node < topOrder.size(); ++node) {
        idx[topOrder[node]] = node;
    }
    return idx;
}

template <typename GraphT>
void StepByStepCoarser<GraphT>::CoarsenForPebbling(const GraphT &dagIn,
                                                   GraphT &coarsenedDag,
                                                   std::vector<VertexIdxT<GraphT>> &newVertexId) {
    problemType_ = ProblemType::PEBBLING;
    coarseningStrategy_ = CoarseningStrategy::EDGE_BY_EDGE;

    unsigned nrSources = 0;
    for (VertexIdx node = 0; node < dagIn.NumVertices(); ++node) {
        if (dagIn.InDegree(node) == 0) {
            ++nrSources;
        }
    }

    targetNrOfNodes_ = std::max(targetNrOfNodes_, nrSources + 1);

    CoarserGenContractionMap<GraphT, GraphT>::CoarsenDag(dagIn, coarsenedDag, newVertexId);
}

template <typename GraphT>
bool StepByStepCoarser<GraphT>::IncontractableForPebbling(const std::pair<VertexIdx, VertexIdx> &edge) const {
    if (gCoarse_.InDegree(edge.first) == 0) {
        return true;
    }

    VMemwT<GraphT> sumWeight = gCoarse_.VertexMemWeight(edge.first) + gCoarse_.VertexMemWeight(edge.second);
    std::set<VertexIdx> parents;
    for (VertexIdx pred : gCoarse_.Parents(edge.first)) {
        parents.insert(pred);
    }
    for (VertexIdx pred : gCoarse_.Parents(edge.second)) {
        if (pred != edge.first) {
            parents.insert(pred);
        }
    }
    for (VertexIdx node : parents) {
        sumWeight += gCoarse_.VertexMemWeight(node);
    }

    if (sumWeight > fastMemCapacity_) {
        return true;
    }

    std::set<VertexIdx> children;
    for (VertexIdx succ : gCoarse_.Children(edge.second)) {
        children.insert(succ);
    }
    for (VertexIdx succ : gCoarse_.Children(edge.first)) {
        if (succ != edge.second) {
            children.insert(succ);
        }
    }

    for (VertexIdx child : children) {
        sumWeight = gCoarse_.VertexMemWeight(edge.first) + gCoarse_.VertexMemWeight(edge.second) + gCoarse_.VertexMemWeight(child);
        for (VertexIdx pred : gCoarse_.Parents(child)) {
            if (pred != edge.first && pred != edge.second) {
                sumWeight += gCoarse_.VertexMemWeight(pred);
            }
        }

        if (sumWeight > fastMemCapacity_) {
            return true;
        }
    }
    return false;
}

template <typename GraphT>
void StepByStepCoarser<GraphT>::MergeSourcesInPebbling() {
    // initialize memory requirement sums to check viability later
    std::vector<VMemwT<GraphT>> memorySum(gCoarse_.NumVertices(), 0);
    std::vector<VertexIdx> sources;
    for (VertexIdx node = 0; node < gCoarse_.NumVertices(); ++node) {
        if (!nodeValid_[node]) {
            continue;
        }

        if (gCoarse_.InDegree(node) > 0) {
            memorySum[node] = gCoarse_.VertexMemWeight(node);
            for (VertexIdx pred : gCoarse_.Parents(node)) {
                memorySum[node] += gCoarse_.VertexMemWeight(pred);
            }
        } else {
            sources.push_back(node);
        }
    }

    std::set<VertexIdx> invalidatedSources;
    bool couldMerge = true;
    while (couldMerge) {
        couldMerge = false;
        for (unsigned idx1 = 0; idx1 < sources.size(); ++idx1) {
            VertexIdx sourceA = sources[idx1];
            if (invalidatedSources.find(sourceA) != invalidatedSources.end()) {
                continue;
            }

            for (unsigned idx2 = idx1 + 1; idx2 < sources.size(); ++idx2) {
                VertexIdx sourceB = sources[idx2];
                if (invalidatedSources.find(sourceB) != invalidatedSources.end()) {
                    continue;
                }

                // check if we can merge sourceA and sourceB
                std::set<VertexIdx> aChildren, bChildren;
                for (VertexIdx succ : gCoarse_.Children(sourceA)) {
                    aChildren.insert(succ);
                }
                for (VertexIdx succ : gCoarse_.Children(sourceB)) {
                    bChildren.insert(succ);
                }

                std::set<VertexIdx> onlyA, onlyB, both;
                for (VertexIdx succ : gCoarse_.Children(sourceA)) {
                    if (bChildren.find(succ) == bChildren.end()) {
                        onlyA.insert(succ);
                    } else {
                        both.insert(succ);
                    }
                }
                for (VertexIdx succ : gCoarse_.Children(sourceB)) {
                    if (aChildren.find(succ) == aChildren.end()) {
                        onlyB.insert(succ);
                    }
                }

                bool violatesConstraint = false;
                for (VertexIdx node : onlyA) {
                    if (memorySum[node] + gCoarse_.VertexMemWeight(sourceB) > fastMemCapacity_) {
                        violatesConstraint = true;
                    }
                }
                for (VertexIdx node : onlyB) {
                    if (memorySum[node] + gCoarse_.VertexMemWeight(sourceA) > fastMemCapacity_) {
                        violatesConstraint = true;
                    }
                }

                if (violatesConstraint) {
                    continue;
                }

                // check if we want to merge sourceA and sourceB
                double simDiff = (onlyA.size() + onlyB.size() == 0) ? 0.0001
                                                                       : static_cast<double>(onlyA.size() + onlyB.size());
                double ratio = static_cast<double>(both.size()) / simDiff;

                if (ratio > 2) {
                    ContractSingleEdge(std::make_pair(sourceA, sourceB));
                    invalidatedSources.insert(sourceB);
                    couldMerge = true;

                    for (VertexIdx node : onlyA) {
                        memorySum[node] += gCoarse_.VertexMemWeight(sourceB);
                    }
                    for (VertexIdx node : onlyB) {
                        memorySum[node] += gCoarse_.VertexMemWeight(sourceA);
                    }
                }
            }
        }
    }
}

template <typename GraphT>
GraphT StepByStepCoarser<GraphT>::Contract(const std::vector<VertexIdxT<GraphT>> &newVertexId) const {
    GraphT gContracted;
    std::vector<bool> isValid(gFull_.NumVertices(), false);
    for (VertexIdx node = 0; node < gFull_.NumVertices(); ++node) {
        isValid[newVertexId[node]] = true;
    }

    for (VertexIdx node = 0; node < gFull_.NumVertices(); ++node) {
        if (isValid[node]) {
            gContracted.AddVertex(0, 0, 0, 0);
        }
    }

    for (VertexIdx node = 0; node < gFull_.NumVertices(); ++node) {
        gContracted.SetVertexWorkWeight(newVertexId[node],
                                         gContracted.VertexWorkWeight(newVertexId[node]) + gFull_.VertexWorkWeight(node));
        gContracted.SetVertexCommWeight(newVertexId[node],
                                         gContracted.VertexCommWeight(newVertexId[node]) + gFull_.VertexCommWeight(node));
        gContracted.SetVertexMemWeight(newVertexId[node],
                                        gContracted.VertexMemWeight(newVertexId[node]) + gFull_.VertexMemWeight(node));
        gContracted.SetVertexType(newVertexId[node], gFull_.VertexType(node));
    }

    for (VertexIdx node = 0; node < gFull_.NumVertices(); ++node) {
        for (const auto &outEdge : OutEdges(node, gFull_)) {
            const VertexIdx succ = Target(outEdge, gFull_);

            if (newVertexId[node] == newVertexId[succ]) {
                continue;
            }

            if constexpr (hasEdgeWeightsV<GraphT>) {
                const auto pair = EdgeDesc(newVertexId[node], newVertexId[succ], gContracted);

                if (pair.second) {
                    gContracted.SetEdgeCommWeight(pair.first,
                                                   gContracted.EdgeCommWeight(pair.first) + gFull_.EdgeCommWeight(outEdge));
                } else {
                    gContracted.AddEdge(newVertexId[node], newVertexId[succ], gFull_.EdgeCommWeight(outEdge));
                }

            } else {
                if (not Edge(newVertexId[node], newVertexId[succ], gContracted)) {
                    gContracted.AddEdge(newVertexId[node], newVertexId[succ]);
                }
            }
        }
    }

    return gContracted;
}

template <typename GraphT>
void StepByStepCoarser<GraphT>::SetIdVector(std::vector<VertexIdxT<GraphT>> &newVertexId) const {
    newVertexId.clear();
    newVertexId.resize(gFull_.NumVertices());

    newVertexId = GetIntermediateIDs(contractionHistory_.size());
}

template <typename GraphT>
std::vector<VertexIdxT<GraphT>> StepByStepCoarser<GraphT>::GetIntermediateIDs(VertexIdx untilWhichStep) const {
    std::vector<VertexIdx> target(gFull_.NumVertices()), pointsTo(gFull_.NumVertices(), std::numeric_limits<VertexIdx>::max());

    for (VertexIdx iterate = 0; iterate < contractionHistory_.size() && iterate < untilWhichStep; ++iterate) {
        const std::pair<VertexIdx, VertexIdx> &contractionStep = contractionHistory_[iterate];
        pointsTo[contractionStep.second] = contractionStep.first;
    }

    for (VertexIdx node = 0; node < gFull_.NumVertices(); ++node) {
        target[node] = node;
        while (pointsTo[target[node]] != std::numeric_limits<VertexIdx>::max()) {
            target[node] = pointsTo[target[node]];
        }
    }

    if (contractionHistory_.empty() || untilWhichStep == 0) {
        return target;
    }

    std::vector<bool> isValid(gFull_.NumVertices(), false);
    for (VertexIdx node = 0; node < gFull_.NumVertices(); ++node) {
        isValid[target[node]] = true;
    }

    std::vector<VertexIdx> newId(gFull_.NumVertices());
    VertexIdx currentIndex = 0;
    for (VertexIdx node = 0; node < gFull_.NumVertices(); ++node) {
        if (isValid[node]) {
            newId[node] = currentIndex++;
        }
    }

    for (VertexIdx node = 0; node < gFull_.NumVertices(); ++node) {
        target[node] = newId[target[node]];
    }

    BoostGraphT tempDag;
    tempDag = Contract(target);
    std::vector<bool> allValid(tempDag.NumVertices(), true);
    std::vector<VertexIdx> topIdx = GetFilteredTopOrderIdx(tempDag, allValid);

    for (VertexIdx node = 0; node < gFull_.NumVertices(); ++node) {
        target[node] = topIdx[target[node]];
    }

    return target;
}

}    // namespace osp
