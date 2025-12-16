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

    using VertexTypeTOrDefault = std::conditional_t<IsComputationalDagTypedVerticesV<GraphT>, VTypeT<GraphT>, unsigned>;
    using EdgeCommwTOrDefault = std::conditional_t<HasEdgeWeightsV<GraphT>, ECommwT<GraphT>, VCommwT<GraphT>>;

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
    std::vector<std::pair<VertexIdx, VertexIdx>> contractionHistory;

    CoarseningStrategy coarseningStrategy_ = CoarseningStrategy::EDGE_BY_EDGE;
    ProblemType problemType_ = ProblemType::SCHEDULING;

    unsigned target_nr_of_nodes = 0;

    GraphT G_full;
    BoostGraphT G_coarse;

    std::vector<std::set<VertexIdx>> contains;

    std::map<std::pair<VertexIdx, VertexIdx>, VCommwT<GraphT>> edgeWeights;
    std::map<std::pair<VertexIdx, VertexIdx>, VCommwT<GraphT>> contractable;
    std::vector<bool> node_valid;
    std::vector<VertexIdx> top_order_idx;

    VMemwT<GraphT> fast_mem_capacity = std::numeric_limits<VMemwT<GraphT>>::max();    // for pebbling

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
    void SetIdVector(std::vector<VertexIdxT<GraphT>> &new_vertex_id) const;
    static std::vector<VertexIdx> GetFilteredTopOrderIdx(const GraphT &G, const std::vector<bool> &is_valid);

  public:
    virtual ~StepByStepCoarser() = default;

    virtual std::string GetCoarserName() const override { return "StepByStepCoarsening"; }

    // DAG coarsening
    virtual std::vector<VertexIdxT<GraphT>> GenerateVertexContractionMap(const GraphT &dag_in) override;

    // Coarsening for pebbling problems - leaves source nodes intact, considers memory bound
    void CoarsenForPebbling(const GraphT &dag_in, GraphT &coarsened_dag, std::vector<VertexIdxT<GraphT>> &new_vertex_id);

    void SetCoarseningStrategy(CoarseningStrategy strategy) { coarseningStrategy_ = strategy; }

    void SetTargetNumberOfNodes(const unsigned nr_nodes_) { target_nr_of_nodes = nr_nodes_; }

    void SetFastMemCapacity(const VMemwT<GraphT> capacity_) { fast_mem_capacity = capacity_; }

    std::vector<std::pair<VertexIdx, VertexIdx>> GetContractionHistory() const { return contractionHistory; }

    std::vector<VertexIdx> GetIntermediateIDs(VertexIdx until_which_step) const;
    GraphT Contract(const std::vector<VertexIdxT<GraphT>> &new_vertex_id) const;

    const GraphT &GetOriginalDag() const { return G_full; }
};

// template<typename GraphT>
// bool StepByStepCoarser<GraphT>::coarseDag(const GraphT& dag_in, GraphT &dag_out,
//                         std::vector<std::vector<VertexIdxT<GraphT>>> &old_vertex_ids,
//                         std::vector<VertexIdxT<GraphT>> &new_vertex_id)

template <typename GraphT>
std::vector<VertexIdxT<GraphT>> StepByStepCoarser<GraphT>::GenerateVertexContractionMap(const GraphT &dag_in) {
    const unsigned N = static_cast<unsigned>(dag_in.NumVertices());

    G_full = dag_in;
    for (VertexIdx node = G_coarse.NumVertices(); node > 0;) {
        --node;
        G_coarse.RemoveVertex(node);
    }

    ConstructComputationalDag(G_full, G_coarse);

    contractionHistory.clear();

    // target nr of nodes must be reasonable
    if (target_nr_of_nodes == 0 || target_nr_of_nodes > N) {
        target_nr_of_nodes = std::max(N / 2, 1U);
    }

    // list of original node indices contained in each contracted node
    contains.clear();
    contains.resize(N);

    node_valid.clear();
    node_valid.resize(N, true);

    for (VertexIdx node = 0; node < N; ++node) {
        contains[node].insert(node);
    }

    // used for original, slow coarsening
    edgeWeights.clear();
    contractable.clear();

    if (coarseningStrategy_ == CoarseningStrategy::EDGE_BY_EDGE) {
        // Init edge weights
        for (VertexIdx node = 0; node < N; ++node) {
            for (VertexIdx succ : G_full.Children(node)) {
                edgeWeights[std::make_pair(node, succ)] = G_full.VertexCommWeight(node);
            }
        }

        // get original contractable edges
        InitializeContractableEdges();
    }

    for (unsigned NrOfNodes = N; NrOfNodes > target_nr_of_nodes;) {
        // Single contraction step

        std::vector<std::pair<VertexIdx, VertexIdx>> edgesToContract;

        // choose edges to contract in this step
        if (coarseningStrategy_ == CoarseningStrategy::EDGE_BY_EDGE) {
            std::vector<EdgeToContract> candidates = CreateEdgeCandidateList();
            if (candidates.empty()) {
                std::cout << "Error: no more edges to contract" << std::endl;
                break;
            }
            std::pair<VertexIdx, VertexIdx> chosenEdge = PickEdgeToContract(candidates);
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
                for (VertexIdx pred : G_coarse.Parents(edge.second)) {
                    contractable.erase(std::make_pair(pred, edge.second));
                }

                for (VertexIdx succ : G_coarse.Children(edge.second)) {
                    contractable.erase(std::make_pair(edge.second, succ));
                }
            }

            ContractSingleEdge(edge);
            node_valid[edge.second] = false;

            if (coarseningStrategy_ == CoarseningStrategy::EDGE_BY_EDGE) {
                ComputeFilteredTopOrderIdx();

                // Update contractable edges - edge.a
                std::set<VertexIdx> contractableParents = GetContractableParents(edge.first);
                for (VertexIdx pred : G_coarse.Parents(edge.first)) {
                    if (contractableParents.find(pred) != contractableParents.end()) {
                        contractable[std::make_pair(pred, edge.first)] = edgeWeights[std::make_pair(pred, edge.first)];
                    } else {
                        contractable.erase(std::make_pair(pred, edge.first));
                    }
                }

                std::set<VertexIdx> contractableChildren = GetContractableChildren(edge.first);
                for (VertexIdx succ : G_coarse.Children(edge.first)) {
                    if (contractableChildren.find(succ) != contractableChildren.end()) {
                        contractable[std::make_pair(edge.first, succ)] = edgeWeights[std::make_pair(edge.first, succ)];
                    } else {
                        contractable.erase(std::make_pair(edge.first, succ));
                    }
                }
            }
            --NrOfNodes;
            if (NrOfNodes == target_nr_of_nodes) {
                break;
            }
        }
    }

    if (problemType_ == ProblemType::PEBBLING) {
        MergeSourcesInPebbling();
    }

    std::vector<VertexIdxT<GraphT>> new_vertex_id;
    SetIdVector(new_vertex_id);

    return new_vertex_id;
}

template <typename GraphT>
void StepByStepCoarser<GraphT>::ContractSingleEdge(std::pair<VertexIdx, VertexIdx> edge) {
    G_coarse.SetVertexWorkWeight(edge.first, G_coarse.VertexWorkWeight(edge.first) + G_coarse.VertexWorkWeight(edge.second));
    G_coarse.SetVertexWorkWeight(edge.second, 0);

    G_coarse.SetVertexCommWeight(edge.first, G_coarse.VertexCommWeight(edge.first) + G_coarse.VertexCommWeight(edge.second));
    G_coarse.SetVertexCommWeight(edge.second, 0);

    G_coarse.SetVertexMemWeight(edge.first, G_coarse.VertexMemWeight(edge.first) + G_coarse.VertexMemWeight(edge.second));
    G_coarse.SetVertexMemWeight(edge.second, 0);

    contractionHistory.emplace_back(edge.first, edge.second);

    // process incoming edges
    std::set<VertexIdx> parents_of_source;
    for (VertexIdx pred : G_coarse.Parents(edge.first)) {
        parents_of_source.insert(pred);
    }

    for (VertexIdx pred : G_coarse.Parents(edge.second)) {
        if (pred == edge.first) {
            continue;
        }
        if (parents_of_source.find(pred) != parents_of_source.end())    // combine edges
        {
            edgeWeights[std::make_pair(pred, edge.first)] = 0;
            for (VertexIdx node : contains[pred]) {
                for (VertexIdx succ : G_coarse.Children(node)) {
                    if (succ == edge.first || succ == edge.second) {
                        edgeWeights[std::make_pair(pred, edge.first)] += G_full.VertexCommWeight(node);
                    }
                }
            }

            edgeWeights.erase(std::make_pair(pred, edge.second));
        } else    // add incoming edge
        {
            G_coarse.AddEdge(pred, edge.first);
            edgeWeights[std::make_pair(pred, edge.first)] = edgeWeights[std::make_pair(pred, edge.second)];
        }
    }

    // process outgoing edges
    std::set<VertexIdx> children_of_source;
    for (VertexIdx succ : G_coarse.Children(edge.first)) {
        children_of_source.insert(succ);
    }

    for (VertexIdx succ : G_coarse.Children(edge.second)) {
        if (children_of_source.find(succ) != children_of_source.end())    // combine edges
        {
            edgeWeights[std::make_pair(edge.first, succ)] += edgeWeights[std::make_pair(edge.second, succ)];
            edgeWeights.erase(std::make_pair(edge.second, succ));
        } else    // add outgoing edge
        {
            G_coarse.AddEdge(edge.first, succ);
            edgeWeights[std::make_pair(edge.first, succ)] = edgeWeights[std::make_pair(edge.second, succ)];
        }
    }

    G_coarse.ClearVertex(edge.second);

    for (VertexIdx node : contains[edge.second]) {
        contains[edge.first].insert(node);
    }

    contains[edge.second].clear();
}

template <typename GraphT>
bool StepByStepCoarser<GraphT>::IsContractable(std::pair<VertexIdx, VertexIdx> edge) const {
    std::deque<VertexIdx> Queue;
    std::set<VertexIdx> visited;
    for (VertexIdx succ : G_coarse.Children(edge.first)) {
        if (node_valid[succ] && top_order_idx[succ] < top_order_idx[edge.second]) {
            Queue.push_back(succ);
            visited.insert(succ);
        }
    }

    while (!Queue.empty()) {
        const VertexIdx node = Queue.front();
        Queue.pop_front();
        for (VertexIdx succ : G_coarse.Children(node)) {
            if (succ == edge.second) {
                return false;
            }

            if (node_valid[succ] && top_order_idx[succ] < top_order_idx[edge.second] && visited.count(succ) == 0) {
                Queue.push_back(succ);
                visited.insert(succ);
            }
        }
    }
    return true;
}

template <typename GraphT>
std::set<VertexIdxT<GraphT>> StepByStepCoarser<GraphT>::GetContractableChildren(const VertexIdx node) const {
    std::deque<VertexIdx> Queue;
    std::set<VertexIdx> visited;
    std::set<VertexIdx> succ_contractable;
    VertexIdx topOrderMax = top_order_idx[node];

    for (VertexIdx succ : G_coarse.Children(node)) {
        if (node_valid[succ]) {
            succ_contractable.insert(succ);
        }

        if (top_order_idx[succ] > topOrderMax) {
            topOrderMax = top_order_idx[succ];
        }

        if (node_valid[succ]) {
            Queue.push_back(succ);
            visited.insert(succ);
        }
    }

    while (!Queue.empty()) {
        const VertexIdx node_local = Queue.front();
        Queue.pop_front();
        for (VertexIdx succ : G_coarse.Children(node_local)) {
            succ_contractable.erase(succ);

            if (node_valid[succ] && top_order_idx[succ] < topOrderMax && visited.count(succ) == 0) {
                Queue.push_back(succ);
                visited.insert(succ);
            }
        }
    }

    return succ_contractable;
}

template <typename GraphT>
std::set<VertexIdxT<GraphT>> StepByStepCoarser<GraphT>::GetContractableParents(const VertexIdx node) const {
    std::deque<VertexIdx> Queue;
    std::set<VertexIdx> visited;
    std::set<VertexIdx> pred_contractable;
    VertexIdx topOrderMin = top_order_idx[node];

    for (VertexIdx pred : G_coarse.Parents(node)) {
        if (node_valid[pred]) {
            pred_contractable.insert(pred);
        }

        if (top_order_idx[pred] < topOrderMin) {
            topOrderMin = top_order_idx[pred];
        }

        if (node_valid[pred]) {
            Queue.push_back(pred);
            visited.insert(pred);
        }
    }

    while (!Queue.empty()) {
        const VertexIdx node_local = Queue.front();
        Queue.pop_front();
        for (VertexIdx pred : G_coarse.Parents(node_local)) {
            pred_contractable.erase(pred);

            if (node_valid[pred] && top_order_idx[pred] > topOrderMin && visited.count(pred) == 0) {
                Queue.push_back(pred);
                visited.insert(pred);
            }
        }
    }

    return pred_contractable;
}

template <typename GraphT>
void StepByStepCoarser<GraphT>::InitializeContractableEdges() {
    ComputeFilteredTopOrderIdx();

    for (VertexIdx node = 0; node < G_full.NumVertices(); ++node) {
        std::set<VertexIdx> succ_contractable = GetContractableChildren(node);
        for (VertexIdx succ : succ_contractable) {
            contractable[std::make_pair(node, succ)] = G_full.VertexCommWeight(node);
        }
    }
}

template <typename GraphT>
void StepByStepCoarser<GraphT>::UpdateDistantEdgeContractibility(std::pair<VertexIdx, VertexIdx> edge) {
    std::unordered_set<VertexIdx> ancestors, descendant;
    std::deque<VertexIdx> Queue;
    for (VertexIdx succ : G_coarse.Children(edge.first)) {
        if (succ != edge.second) {
            Queue.push_back(succ);
            descendant.insert(succ);
        }
    }
    while (!Queue.empty()) {
        const VertexIdx node = Queue.front();
        Queue.pop_front();
        for (VertexIdx succ : G_coarse.Children(node)) {
            if (descendant.count(succ) == 0) {
                Queue.push_back(succ);
                descendant.insert(succ);
            }
        }
    }

    for (VertexIdx pred : G_coarse.Parents(edge.second)) {
        if (pred != edge.first) {
            Queue.push_back(pred);
            ancestors.insert(pred);
        }
    }
    while (!Queue.empty()) {
        const VertexIdx node = Queue.front();
        Queue.pop_front();
        for (VertexIdx pred : G_coarse.Parents(node)) {
            if (ancestors.count(pred) == 0) {
                Queue.push_back(pred);
                ancestors.insert(pred);
            }
        }
    }

    for (const VertexIdx node : ancestors) {
        for (const VertexIdx succ : G_coarse.Children(node)) {
            if (descendant.count(succ) > 0) {
                contractable.erase(std::make_pair(node, succ));
            }
        }
    }
}

template <typename GraphT>
std::vector<typename StepByStepCoarser<GraphT>::EdgeToContract> StepByStepCoarser<GraphT>::CreateEdgeCandidateList() const {
    std::vector<EdgeToContract> candidates;

    for (auto it = contractable.cbegin(); it != contractable.cend(); ++it) {
        if (problemType_ == ProblemType::PEBBLING && IncontractableForPebbling(it->first)) {
            continue;
        }

        candidates.emplace_back(
            it->first.first, it->first.second, contains[it->first.first].size() + contains[it->first.second].size(), it->second);
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
    std::vector<bool> singleton(G_full.NumVertices(), true);
    std::vector<VertexIdx> leader(G_full.NumVertices());
    std::vector<unsigned> weight(G_full.NumVertices());
    std::vector<unsigned> nrBadNeighbors(G_full.NumVertices());
    std::vector<VertexIdx> leaderBadNeighbors(G_full.NumVertices());

    std::vector<unsigned> minTopLevel(G_full.NumVertices());
    std::vector<unsigned> maxTopLevel(G_full.NumVertices());
    std::vector<VertexIdx> clusterNewID(G_full.NumVertices());

    std::vector<std::pair<VertexIdx, VertexIdx>> contractionSteps;
    std::vector<unsigned> topLevel = ComputeFilteredTopLevel();
    for (VertexIdx node = 0; node < G_full.NumVertices(); ++node) {
        if (node_valid[node]) {
            leader[node] = node;
            weight[node] = 1 /*G_coarse.vertex_work_weight(node)*/;
            nrBadNeighbors[node] = 0;
            leaderBadNeighbors[node] = UINT_MAX;
            clusterNewID[node] = node;
            minTopLevel[node] = topLevel[node];
            maxTopLevel[node] = topLevel[node];
        }
    }

    for (VertexIdx node = 0; node < G_full.NumVertices(); ++node) {
        if (!node_valid[node] || !singleton[node]) {
            continue;
        }

        if (nrBadNeighbors[node] > 1) {
            continue;
        }

        std::vector<VertexIdx> validNeighbors;
        for (VertexIdx pred : G_coarse.Parents(node)) {
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
        for (VertexIdx succ : G_coarse.Children(node)) {
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

        bool is_parent = false;
        for (VertexIdx pred : G_coarse.Parents(node)) {
            if (pred == bestNeighbor) {
                is_parent = true;
            }
        }

        if (is_parent) {
            contractionSteps.emplace_back(clusterNewID[newLead], node);
        } else {
            contractionSteps.emplace_back(node, clusterNewID[newLead]);
            clusterNewID[newLead] = node;
        }

        minTopLevel[newLead] = std::min(minTopLevel[newLead], topLevel[node]);
        maxTopLevel[newLead] = std::max(maxTopLevel[newLead], topLevel[node]);

        for (VertexIdx pred : G_coarse.Parents(node)) {
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
        for (VertexIdx succ : G_coarse.Children(node)) {
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
            for (VertexIdx pred : G_coarse.Parents(bestNeighbor)) {
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
            for (VertexIdx succ : G_coarse.Children(bestNeighbor)) {
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
    std::vector<unsigned> TopLevel(G_full.NumVertices());
    for (const VertexIdx node : TopSortView(G_coarse)) {
        if (!node_valid[node]) {
            continue;
        }

        TopLevel[node] = 0;
        for (const VertexIdx pred : G_coarse.Parents(node)) {
            TopLevel[node] = std::max(TopLevel[node], TopLevel[pred] + 1);
        }
    }
    return TopLevel;
}

template <typename GraphT>
void StepByStepCoarser<GraphT>::ComputeFilteredTopOrderIdx() {
    top_order_idx = GetFilteredTopOrderIdx(G_coarse, node_valid);
}

template <typename GraphT>
std::vector<VertexIdxT<GraphT>> StepByStepCoarser<GraphT>::GetFilteredTopOrderIdx(const GraphT &G,
                                                                                  const std::vector<bool> &is_valid) {
    std::vector<VertexIdx> top_order = GetFilteredTopOrder(is_valid, G);
    std::vector<VertexIdx> idx(G.NumVertices());
    for (VertexIdx node = 0; node < top_order.size(); ++node) {
        idx[top_order[node]] = node;
    }
    return idx;
}

template <typename GraphT>
void StepByStepCoarser<GraphT>::CoarsenForPebbling(const GraphT &dag_in,
                                                   GraphT &coarsened_dag,
                                                   std::vector<VertexIdxT<GraphT>> &new_vertex_id) {
    problemType_ = ProblemType::PEBBLING;
    coarseningStrategy_ = CoarseningStrategy::EDGE_BY_EDGE;

    unsigned nr_sources = 0;
    for (VertexIdx node = 0; node < dag_in.NumVertices(); ++node) {
        if (dag_in.InDegree(node) == 0) {
            ++nr_sources;
        }
    }

    target_nr_of_nodes = std::max(target_nr_of_nodes, nr_sources + 1);

    CoarserGenContractionMap<GraphT, GraphT>::CoarsenDag(dag_in, coarsened_dag, new_vertex_id);
}

template <typename GraphT>
bool StepByStepCoarser<GraphT>::IncontractableForPebbling(const std::pair<VertexIdx, VertexIdx> &edge) const {
    if (G_coarse.InDegree(edge.first) == 0) {
        return true;
    }

    VMemwT<GraphT> sum_weight = G_coarse.VertexMemWeight(edge.first) + G_coarse.VertexMemWeight(edge.second);
    std::set<VertexIdx> parents;
    for (VertexIdx pred : G_coarse.Parents(edge.first)) {
        parents.insert(pred);
    }
    for (VertexIdx pred : G_coarse.Parents(edge.second)) {
        if (pred != edge.first) {
            parents.insert(pred);
        }
    }
    for (VertexIdx node : parents) {
        sum_weight += G_coarse.VertexMemWeight(node);
    }

    if (sum_weight > fast_mem_capacity) {
        return true;
    }

    std::set<VertexIdx> children;
    for (VertexIdx succ : G_coarse.Children(edge.second)) {
        children.insert(succ);
    }
    for (VertexIdx succ : G_coarse.Children(edge.first)) {
        if (succ != edge.second) {
            children.insert(succ);
        }
    }

    for (VertexIdx child : children) {
        sum_weight = G_coarse.VertexMemWeight(edge.first) + G_coarse.VertexMemWeight(edge.second) + G_coarse.VertexMemWeight(child);
        for (VertexIdx pred : G_coarse.Parents(child)) {
            if (pred != edge.first && pred != edge.second) {
                sum_weight += G_coarse.VertexMemWeight(pred);
            }
        }

        if (sum_weight > fast_mem_capacity) {
            return true;
        }
    }
    return false;
}

template <typename GraphT>
void StepByStepCoarser<GraphT>::MergeSourcesInPebbling() {
    // initialize memory requirement sums to check viability later
    std::vector<VMemwT<GraphT>> memory_sum(G_coarse.NumVertices(), 0);
    std::vector<VertexIdx> sources;
    for (VertexIdx node = 0; node < G_coarse.NumVertices(); ++node) {
        if (!node_valid[node]) {
            continue;
        }

        if (G_coarse.InDegree(node) > 0) {
            memory_sum[node] = G_coarse.VertexMemWeight(node);
            for (VertexIdx pred : G_coarse.Parents(node)) {
                memory_sum[node] += G_coarse.VertexMemWeight(pred);
            }
        } else {
            sources.push_back(node);
        }
    }

    std::set<VertexIdx> invalidated_sources;
    bool could_merge = true;
    while (could_merge) {
        could_merge = false;
        for (unsigned idx1 = 0; idx1 < sources.size(); ++idx1) {
            VertexIdx source_a = sources[idx1];
            if (invalidated_sources.find(source_a) != invalidated_sources.end()) {
                continue;
            }

            for (unsigned idx2 = idx1 + 1; idx2 < sources.size(); ++idx2) {
                VertexIdx source_b = sources[idx2];
                if (invalidated_sources.find(source_b) != invalidated_sources.end()) {
                    continue;
                }

                // check if we can merge source_a and source_b
                std::set<VertexIdx> a_children, b_children;
                for (VertexIdx succ : G_coarse.Children(source_a)) {
                    a_children.insert(succ);
                }
                for (VertexIdx succ : G_coarse.Children(source_b)) {
                    b_children.insert(succ);
                }

                std::set<VertexIdx> only_a, only_b, both;
                for (VertexIdx succ : G_coarse.Children(source_a)) {
                    if (b_children.find(succ) == b_children.end()) {
                        only_a.insert(succ);
                    } else {
                        both.insert(succ);
                    }
                }
                for (VertexIdx succ : G_coarse.Children(source_b)) {
                    if (a_children.find(succ) == a_children.end()) {
                        only_b.insert(succ);
                    }
                }

                bool violates_constraint = false;
                for (VertexIdx node : only_a) {
                    if (memory_sum[node] + G_coarse.VertexMemWeight(source_b) > fast_mem_capacity) {
                        violates_constraint = true;
                    }
                }
                for (VertexIdx node : only_b) {
                    if (memory_sum[node] + G_coarse.VertexMemWeight(source_a) > fast_mem_capacity) {
                        violates_constraint = true;
                    }
                }

                if (violates_constraint) {
                    continue;
                }

                // check if we want to merge source_a and source_b
                double sim_diff = (only_a.size() + only_b.size() == 0) ? 0.0001
                                                                       : static_cast<double>(only_a.size() + only_b.size());
                double ratio = static_cast<double>(both.size()) / sim_diff;

                if (ratio > 2) {
                    ContractSingleEdge(std::make_pair(source_a, source_b));
                    invalidated_sources.insert(source_b);
                    could_merge = true;

                    for (VertexIdx node : only_a) {
                        memory_sum[node] += G_coarse.VertexMemWeight(source_b);
                    }
                    for (VertexIdx node : only_b) {
                        memory_sum[node] += G_coarse.VertexMemWeight(source_a);
                    }
                }
            }
        }
    }
}

template <typename GraphT>
GraphT StepByStepCoarser<GraphT>::Contract(const std::vector<VertexIdxT<GraphT>> &new_vertex_id) const {
    GraphT G_contracted;
    std::vector<bool> is_valid(G_full.NumVertices(), false);
    for (VertexIdx node = 0; node < G_full.NumVertices(); ++node) {
        is_valid[new_vertex_id[node]] = true;
    }

    for (VertexIdx node = 0; node < G_full.NumVertices(); ++node) {
        if (is_valid[node]) {
            G_contracted.AddVertex(0, 0, 0, 0);
        }
    }

    for (VertexIdx node = 0; node < G_full.NumVertices(); ++node) {
        G_contracted.SetVertexWorkWeight(new_vertex_id[node],
                                         G_contracted.VertexWorkWeight(new_vertex_id[node]) + G_full.VertexWorkWeight(node));
        G_contracted.SetVertexCommWeight(new_vertex_id[node],
                                         G_contracted.VertexCommWeight(new_vertex_id[node]) + G_full.VertexCommWeight(node));
        G_contracted.SetVertexMemWeight(new_vertex_id[node],
                                        G_contracted.VertexMemWeight(new_vertex_id[node]) + G_full.VertexMemWeight(node));
        G_contracted.SetVertexType(new_vertex_id[node], G_full.VertexType(node));
    }

    for (VertexIdx node = 0; node < G_full.NumVertices(); ++node) {
        for (const auto &out_edge : OutEdges(node, G_full)) {
            const VertexIdx succ = Target(out_edge, G_full);

            if (new_vertex_id[node] == new_vertex_id[succ]) {
                continue;
            }

            if constexpr (HasEdgeWeightsV<GraphT>) {
                const auto pair = EdgeDesc(new_vertex_id[node], new_vertex_id[succ], G_contracted);

                if (pair.second) {
                    G_contracted.SetEdgeCommWeight(pair.first,
                                                   G_contracted.EdgeCommWeight(pair.first) + G_full.EdgeCommWeight(out_edge));
                } else {
                    G_contracted.AddEdge(new_vertex_id[node], new_vertex_id[succ], G_full.EdgeCommWeight(out_edge));
                }

            } else {
                if (not Edge(new_vertex_id[node], new_vertex_id[succ], G_contracted)) {
                    G_contracted.AddEdge(new_vertex_id[node], new_vertex_id[succ]);
                }
            }
        }
    }

    return G_contracted;
}

template <typename GraphT>
void StepByStepCoarser<GraphT>::SetIdVector(std::vector<VertexIdxT<GraphT>> &new_vertex_id) const {
    new_vertex_id.clear();
    new_vertex_id.resize(G_full.NumVertices());

    new_vertex_id = GetIntermediateIDs(contractionHistory.size());
}

template <typename GraphT>
std::vector<VertexIdxT<GraphT>> StepByStepCoarser<GraphT>::GetIntermediateIDs(VertexIdx until_which_step) const {
    std::vector<VertexIdx> target(G_full.NumVertices()), pointsTo(G_full.NumVertices(), std::numeric_limits<VertexIdx>::max());

    for (VertexIdx iterate = 0; iterate < contractionHistory.size() && iterate < until_which_step; ++iterate) {
        const std::pair<VertexIdx, VertexIdx> &contractionStep = contractionHistory[iterate];
        pointsTo[contractionStep.second] = contractionStep.first;
    }

    for (VertexIdx node = 0; node < G_full.NumVertices(); ++node) {
        target[node] = node;
        while (pointsTo[target[node]] != std::numeric_limits<VertexIdx>::max()) {
            target[node] = pointsTo[target[node]];
        }
    }

    if (contractionHistory.empty() || until_which_step == 0) {
        return target;
    }

    std::vector<bool> is_valid(G_full.NumVertices(), false);
    for (VertexIdx node = 0; node < G_full.NumVertices(); ++node) {
        is_valid[target[node]] = true;
    }

    std::vector<VertexIdx> new_id(G_full.NumVertices());
    VertexIdx current_index = 0;
    for (VertexIdx node = 0; node < G_full.NumVertices(); ++node) {
        if (is_valid[node]) {
            new_id[node] = current_index++;
        }
    }

    for (VertexIdx node = 0; node < G_full.NumVertices(); ++node) {
        target[node] = new_id[target[node]];
    }

    BoostGraphT temp_dag;
    temp_dag = Contract(target);
    std::vector<bool> all_valid(temp_dag.NumVertices(), true);
    std::vector<VertexIdx> top_idx = GetFilteredTopOrderIdx(temp_dag, all_valid);

    for (VertexIdx node = 0; node < G_full.NumVertices(); ++node) {
        target[node] = top_idx[target[node]];
    }

    return target;
}

}    // namespace osp
