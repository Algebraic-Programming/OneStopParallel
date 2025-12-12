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
    using vertex_idx = vertex_idx_t<Graph_t>;

    using vertex_type_t_or_default = std::conditional_t<IsComputationalDagTypedVerticesV<Graph_t>, v_type_t<Graph_t>, unsigned>;
    using edge_commw_t_or_default = std::conditional_t<HasEdgeWeightsV<Graph_t>, ECommwT<Graph_t>, VCommwT<Graph_t>>;

    using boost_graph_t
        = boost_graph<VWorkwT<Graph_t>, VCommwT<Graph_t>, VMemwT<Graph_t>, vertex_type_t_or_default, edge_commw_t_or_default>;

  public:
    enum CoarseningStrategy { EDGE_BY_EDGE, BOTTOM_LEVEL_CLUSTERS };

    enum ProblemType { SCHEDULING, PEBBLING };

    struct EdgeToContract {
        std::pair<vertex_idx, vertex_idx> edge_;
        VWorkwT<Graph_t> workWeight_;
        VCommwT<Graph_t> commWeight_;

        EdgeToContract(const vertex_idx source,
                       const vertex_idx target,
                       const VWorkwT<Graph_t> workWeight,
                       const VCommwT<Graph_t> commWeight)
            : edge(source, target), work_weight(work_weight_), comm_weight(comm_weight_) {}

        bool operator<(const EdgeToContract &other) const {
            return (work_weight < other.work_weight || (work_weight == other.work_weight && comm_weight < other.comm_weight));
        }
    };

  private:
    std::vector<std::pair<vertex_idx, vertex_idx>> contractionHistory_;

    CoarseningStrategy coarseningStrategy_ = CoarseningStrategy::EDGE_BY_EDGE;
    ProblemType problemType_ = ProblemType::SCHEDULING;

    unsigned targetNrOfNodes_ = 0;

    GraphT gFull_;
    boost_graph_t gCoarse_;

    std::vector<std::set<vertex_idx>> contains_;

    std::map<std::pair<vertex_idx, vertex_idx>, VCommwT<Graph_t>> edgeWeights;
    std::map<std::pair<vertex_idx, vertex_idx>, VCommwT<Graph_t>> contractable;
    std::vector<bool> nodeValid_;
    std::vector<vertex_idx> topOrderIdx_;

    VMemwT<Graph_t> fastMemCapacity_ = std::numeric_limits<VMemwT<Graph_t>>::max();    // for pebbling

    // Utility functions for coarsening in general
    void ContractSingleEdge(std::pair<vertex_idx, vertex_idx> edge);
    void ComputeFilteredTopOrderIdx();

    void InitializeContractableEdges();
    bool IsContractable(std::pair<vertex_idx, vertex_idx> edge) const;
    std::set<vertex_idx> GetContractableChildren(vertex_idx node) const;
    std::set<vertex_idx> GetContractableParents(vertex_idx node) const;
    void UpdateDistantEdgeContractibility(std::pair<vertex_idx, vertex_idx> edge);

    std::pair<vertex_idx, vertex_idx> PickEdgeToContract(const std::vector<EdgeToContract> &candidates) const;
    std::vector<EdgeToContract> CreateEdgeCandidateList() const;

    // Utility functions for cluster coarsening
    std::vector<std::pair<vertex_idx, vertex_idx>> ClusterCoarsen() const;
    std::vector<unsigned> ComputeFilteredTopLevel() const;

    // Utility functions for coarsening in a pebbling problem
    bool IncontractableForPebbling(const std::pair<vertex_idx, vertex_idx> &) const;
    void MergeSourcesInPebbling();

    // Utility for contracting into final format
    void SetIdVector(std::vector<vertex_idx_t<Graph_t>> &newVertexId) const;
    static std::vector<vertex_idx> GetFilteredTopOrderIdx(const GraphT &g, const std::vector<bool> &isValid);

  public:
    virtual ~StepByStepCoarser() = default;

    virtual std::string getCoarserName() const override { return "StepByStepCoarsening"; }

    // DAG coarsening
    virtual std::vector<vertex_idx_t<Graph_t>> generate_vertex_contraction_map(const GraphT &dagIn) override;

    // Coarsening for pebbling problems - leaves source nodes intact, considers memory bound
    void CoarsenForPebbling(const GraphT &dagIn, GraphT &coarsenedDag, std::vector<vertex_idx_t<Graph_t>> &newVertexId);

    void SetCoarseningStrategy(CoarseningStrategy strategy) { coarseningStrategy_ = strategy; }

    void SetTargetNumberOfNodes(const unsigned nrNodes) { targetNrOfNodes_ = nrNodes; }

    void SetFastMemCapacity(const VMemwT<Graph_t> capacity) { fast_mem_capacity = capacity_; }

    std::vector<std::pair<vertex_idx, vertex_idx>> GetContractionHistory() const { return contractionHistory; }

    std::vector<vertex_idx> GetIntermediateIDs(vertex_idx untilWhichStep) const;
    GraphT Contract(const std::vector<vertex_idx_t<Graph_t>> &newVertexId) const;

    const GraphT &GetOriginalDag() const { return gFull_; }
};

// template<typename Graph_t>
// bool StepByStepCoarser<Graph_t>::coarseDag(const Graph_t& dag_in, Graph_t &dag_out,
//                         std::vector<std::vector<vertex_idx_t<Graph_t>>> &old_vertex_ids,
//                         std::vector<vertex_idx_t<Graph_t>> &new_vertex_id)

template <typename GraphT>
std::vector<vertex_idx_t<Graph_t>> StepByStepCoarser<GraphT>::GenerateVertexContractionMap(const GraphT &dagIn) {
    const unsigned n = static_cast<unsigned>(dagIn.NumVertices());

    gFull_ = dagIn;
    for (vertex_idx node = G_coarse.NumVertices(); node > 0;) {
        --node;
        G_coarse.remove_vertex(node);
    }

    constructComputationalDag(G_full, G_coarse);

    contractionHistory.clear();

    // target nr of nodes must be reasonable
    if (targetNrOfNodes_ == 0 || targetNrOfNodes_ > n) {
        targetNrOfNodes_ = std::max(n / 2, 1U);
    }

    // list of original node indices contained in each contracted node
    contains.clear();
    contains.resize(N);

    nodeValid_.clear();
    nodeValid_.resize(n, true);

    for (vertex_idx node = 0; node < n; ++node) {
        contains[node].insert(node);
    }

    // used for original, slow coarsening
    edgeWeights.clear();
    contractable.clear();

    if (coarseningStrategy_ == CoarseningStrategy::EDGE_BY_EDGE) {
        // Init edge weights
        for (vertex_idx node = 0; node < n; ++node) {
            for (vertex_idx succ : G_full.Children(node)) {
                edgeWeights[std::make_pair(node, succ)] = G_full.VertexCommWeight(node);
            }
        }

        // get original contractable edges
        InitializeContractableEdges();
    }

    for (unsigned nrOfNodes = n; nrOfNodes > targetNrOfNodes_;) {
        // Single contraction step

        std::vector<std::pair<vertex_idx, vertex_idx>> edgesToContract;

        // choose edges to contract in this step
        if (coarseningStrategy_ == CoarseningStrategy::EDGE_BY_EDGE) {
            std::vector<EdgeToContract> candidates = CreateEdgeCandidateList();
            if (candidates.empty()) {
                std::cout << "Error: no more edges to contract" << std::endl;
                break;
            }
            std::pair<vertex_idx, vertex_idx> chosenEdge = PickEdgeToContract(candidates);
            edgesToContract.push_back(chosenEdge);

            // Update far-away edges that become uncontractable now
            updateDistantEdgeContractibility(chosenEdge);
        } else {
            edgesToContract = ClusterCoarsen();
        }

        if (edgesToContract.empty()) {
            break;
        }

        // contract these edges
        for (const std::pair<vertex_idx, vertex_idx> &edge : edgesToContract) {
            if (coarsening_strategy == COARSENING_STRATEGY::EDGE_BY_EDGE) {
                // Update contractable edges - edge.b
                for (vertex_idx pred : G_coarse.Parents(edge.second)) {
                    contractable.erase(std::make_pair(pred, edge.second));
                }

                for (vertex_idx succ : G_coarse.Children(edge.second)) {
                    contractable.erase(std::make_pair(edge.second, succ));
                }
            }

            ContractSingleEdge(edge);
            node_valid[edge.second] = false;

            if (coarsening_strategy == COARSENING_STRATEGY::EDGE_BY_EDGE) {
                ComputeFilteredTopOrderIdx();

                // Update contractable edges - edge.a
                std::set<vertex_idx> contractableParents = getContractableParents(edge.first);
                for (vertex_idx pred : G_coarse.Parents(edge.first)) {
                    if (contractableParents.find(pred) != contractableParents.end()) {
                        contractable[std::make_pair(pred, edge.first)] = edgeWeights[std::make_pair(pred, edge.first)];
                    } else {
                        contractable.erase(std::make_pair(pred, edge.first));
                    }
                }

                std::set<vertex_idx> contractableChildren = getContractableChildren(edge.first);
                for (vertex_idx succ : G_coarse.Children(edge.first)) {
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

    std::vector<vertex_idx_t<Graph_t>> newVertexId;
    SetIdVector(new_vertex_id);

    return new_vertex_id;
}

template <typename GraphT>
void StepByStepCoarser<GraphT>::ContractSingleEdge(std::pair<vertex_idx, vertex_idx> edge) {
    G_coarse.SetVertexWorkWeight(edge.first, G_coarse.VertexWorkWeight(edge.first) + G_coarse.VertexWorkWeight(edge.second));
    G_coarse.SetVertexWorkWeight(edge.second, 0);

    G_coarse.SetVertexCommWeight(edge.first, G_coarse.VertexCommWeight(edge.first) + G_coarse.VertexCommWeight(edge.second));
    G_coarse.SetVertexCommWeight(edge.second, 0);

    G_coarse.SetVertexMemWeight(edge.first, G_coarse.VertexMemWeight(edge.first) + G_coarse.VertexMemWeight(edge.second));
    G_coarse.SetVertexMemWeight(edge.second, 0);

    contractionHistory.emplace_back(edge.first, edge.second);

    // process incoming edges
    std::set<vertex_idx> parentsOfSource;
    for (vertex_idx pred : G_coarse.Parents(edge.first)) {
        parents_of_source.insert(pred);
    }

    for (vertex_idx pred : G_coarse.Parents(edge.second)) {
        if (pred == edge.first) {
            continue;
        }
        if (parents_of_source.find(pred) != parents_of_source.end())    // combine edges
        {
            edgeWeights[std::make_pair(pred, edge.first)] = 0;
            for (vertex_idx node : contains[pred]) {
                for (vertex_idx succ : G_coarse.Children(node)) {
                    if (succ == edge.first || succ == edge.second) {
                        edgeWeights[std::make_pair(pred, edge.first)] += G_full.VertexCommWeight(node);
                    }
                }
            }

            edgeWeights.erase(std::make_pair(pred, edge.second));
        } else    // add incoming edge
        {
            G_coarse.add_edge(pred, edge.first);
            edgeWeights[std::make_pair(pred, edge.first)] = edgeWeights[std::make_pair(pred, edge.second)];
        }
    }

    // process outgoing edges
    std::set<vertex_idx> childrenOfSource;
    for (vertex_idx succ : G_coarse.Children(edge.first)) {
        children_of_source.insert(succ);
    }

    for (vertex_idx succ : G_coarse.Children(edge.second)) {
        if (children_of_source.find(succ) != children_of_source.end())    // combine edges
        {
            edgeWeights[std::make_pair(edge.first, succ)] += edgeWeights[std::make_pair(edge.second, succ)];
            edgeWeights.erase(std::make_pair(edge.second, succ));
        } else    // add outgoing edge
        {
            G_coarse.add_edge(edge.first, succ);
            edgeWeights[std::make_pair(edge.first, succ)] = edgeWeights[std::make_pair(edge.second, succ)];
        }
    }

    G_coarse.clear_vertex(edge.second);

    for (vertex_idx node : contains[edge.second]) {
        contains[edge.first].insert(node);
    }

    contains[edge.second].clear();
}

template <typename GraphT>
bool StepByStepCoarser<GraphT>::IsContractable(std::pair<vertex_idx, vertex_idx> edge) const {
    std::deque<vertex_idx> queue;
    std::set<vertex_idx> visited;
    for (vertex_idx succ : G_coarse.Children(edge.first)) {
        if (node_valid[succ] && top_order_idx[succ] < top_order_idx[edge.second]) {
            Queue.push_back(succ);
            visited.insert(succ);
        }
    }

    while (!Queue.empty()) {
        const vertex_idx node = Queue.front();
        queue.pop_front();
        for (vertex_idx succ : G_coarse.Children(node)) {
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
std::set<vertex_idx_t<Graph_t>> StepByStepCoarser<GraphT>::GetContractableChildren(const vertex_idx node) const {
    std::deque<vertex_idx> queue;
    std::set<vertex_idx> visited;
    std::set<vertex_idx> succContractable;
    vertex_idx topOrderMax = top_order_idx[node];

    for (vertex_idx succ : G_coarse.Children(node)) {
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
        const vertex_idx nodeLocal = Queue.front();
        queue.pop_front();
        for (vertex_idx succ : G_coarse.Children(node_local)) {
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
std::set<vertex_idx_t<Graph_t>> StepByStepCoarser<GraphT>::GetContractableParents(const vertex_idx node) const {
    std::deque<vertex_idx> queue;
    std::set<vertex_idx> visited;
    std::set<vertex_idx> predContractable;
    vertex_idx topOrderMin = top_order_idx[node];

    for (vertex_idx pred : G_coarse.Parents(node)) {
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
        const vertex_idx nodeLocal = Queue.front();
        queue.pop_front();
        for (vertex_idx pred : G_coarse.Parents(node_local)) {
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

    for (vertex_idx node = 0; node < gFull_.NumVertices(); ++node) {
        std::set<vertex_idx> succContractable = getContractableChildren(node);
        for (vertex_idx succ : succ_contractable) {
            contractable[std::make_pair(node, succ)] = G_full.VertexCommWeight(node);
        }
    }
}

template <typename GraphT>
void StepByStepCoarser<GraphT>::UpdateDistantEdgeContractibility(std::pair<vertex_idx, vertex_idx> edge) {
    std::unordered_set<vertex_idx> ancestors, descendant;
    std::deque<vertex_idx> queue;
    for (vertex_idx succ : G_coarse.Children(edge.first)) {
        if (succ != edge.second) {
            Queue.push_back(succ);
            descendant.insert(succ);
        }
    }
    while (!Queue.empty()) {
        const vertex_idx node = Queue.front();
        queue.pop_front();
        for (vertex_idx succ : G_coarse.Children(node)) {
            if (descendant.count(succ) == 0) {
                Queue.push_back(succ);
                descendant.insert(succ);
            }
        }
    }

    for (vertex_idx pred : G_coarse.Parents(edge.second)) {
        if (pred != edge.first) {
            Queue.push_back(pred);
            ancestors.insert(pred);
        }
    }
    while (!Queue.empty()) {
        const vertex_idx node = Queue.front();
        queue.pop_front();
        for (vertex_idx pred : G_coarse.Parents(node)) {
            if (ancestors.count(pred) == 0) {
                Queue.push_back(pred);
                ancestors.insert(pred);
            }
        }
    }

    for (const vertex_idx node : ancestors) {
        for (const vertex_idx succ : G_coarse.Children(node)) {
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
        if (problem_type == PROBLEM_TYPE::PEBBLING && IncontractableForPebbling(it->first)) {
            continue;
        }

        candidates.emplace_back(
            it->first.first, it->first.second, contains[it->first.first].size() + contains[it->first.second].size(), it->second);
    }

    std::sort(candidates.begin(), candidates.end());
    return candidates;
}

template <typename Graph_t>
std::pair<vertex_idx_t<Graph_t>, vertex_idx_t<Graph_t>> StepByStepCoarser<Graph_t>::PickEdgeToContract(
    const std::vector<EdgeToContract> &candidates) const {
    size_t limit = (candidates.size() + 2) / 3;
    VWorkwT<Graph_t> limitCardinality = candidates[limit].work_weight;
    while (limit < candidates.size() - 1 && candidates[limit + 1].work_weight == limitCardinality) {
        ++limit;
    }

    // an edge case
    if (candidates.size() == 1) {
        limit = 0;
    }

    EdgeToContract chosen = candidates[0];
    unsigned best = 0;
    for (unsigned idx = 1; idx <= limit; ++idx) {
        if (candidates[idx].comm_weight > candidates[best].comm_weight) {
            best = idx;
        }
    }

    chosen = candidates[best];
    return chosen.edge;
}

/**
 * @brief Acyclic graph contractor based on (Herrmann, Julien, et al. "Acyclic partitioning of large directed acyclic graphs."
 * 2017 17th IEEE/ACM international symposium on cluster, cloud and grid computing (CCGRID). IEEE, 2017.))
 * @brief with minor changes and fixes
 *
 */
template <typename Graph_t>
std::vector<std::pair<vertex_idx_t<Graph_t>, vertex_idx_t<Graph_t>>> StepByStepCoarser<Graph_t>::ClusterCoarsen() const {
    std::vector<bool> singleton(G_full.NumVertices(), true);
    std::vector<vertex_idx> leader(G_full.NumVertices());
    std::vector<unsigned> weight(G_full.NumVertices());
    std::vector<unsigned> nrBadNeighbors(G_full.NumVertices());
    std::vector<vertex_idx> leaderBadNeighbors(G_full.NumVertices());

    std::vector<unsigned> minTopLevel(G_full.NumVertices());
    std::vector<unsigned> maxTopLevel(G_full.NumVertices());
    std::vector<vertex_idx> clusterNewID(G_full.NumVertices());

    std::vector<std::pair<vertex_idx, vertex_idx>> contractionSteps;
    std::vector<unsigned> topLevel = ComputeFilteredTopLevel();
    for (vertex_idx node = 0; node < G_full.NumVertices(); ++node) {
        if (node_valid[node]) {
            leader[node] = node;
            weight[node] = 1 /*G_coarse.VertexWorkWeight(node)*/;
            nrBadNeighbors[node] = 0;
            leaderBadNeighbors[node] = UINT_MAX;
            clusterNewID[node] = node;
            minTopLevel[node] = topLevel[node];
            maxTopLevel[node] = topLevel[node];
        }
    }

    for (vertex_idx node = 0; node < G_full.NumVertices(); ++node) {
        if (!node_valid[node] || !singleton[node]) {
            continue;
        }

        if (nrBadNeighbors[node] > 1) {
            continue;
        }

        std::vector<vertex_idx> validNeighbors;
        for (vertex_idx pred : G_coarse.Parents(node)) {
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
            if (problem_type == PROBLEM_TYPE::PEBBLING && IncontractableForPebbling(std::make_pair(pred, node))) {
                continue;
            }

            validNeighbors.push_back(pred);
        }
        for (vertex_idx succ : G_coarse.Children(node)) {
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
            if (problem_type == PROBLEM_TYPE::PEBBLING && IncontractableForPebbling(std::make_pair(node, succ))) {
                continue;
            }

            validNeighbors.push_back(succ);
        }

        vertex_idx bestNeighbor = std::numeric_limits<vertex_idx>::max();
        for (vertex_idx neigh : validNeighbors) {
            if (bestNeighbor == std::numeric_limits<vertex_idx>::max() || weight[leader[neigh]] < weight[leader[bestNeighbor]]) {
                bestNeighbor = neigh;
            }
        }

        if (bestNeighbor == std::numeric_limits<vertex_idx>::max()) {
            continue;
        }

        vertex_idx newLead = leader[bestNeighbor];
        leader[node] = newLead;
        weight[newLead] += weight[node];

        bool is_parent = false;
        for (vertex_idx pred : G_coarse.Parents(node)) {
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

        for (vertex_idx pred : G_coarse.Parents(node)) {
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
        for (vertex_idx succ : G_coarse.Children(node)) {
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
            for (vertex_idx pred : G_coarse.Parents(bestNeighbor)) {
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
            for (vertex_idx succ : G_coarse.Children(bestNeighbor)) {
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
    for (const vertex_idx node : top_sort_view(G_coarse)) {
        if (!node_valid[node]) {
            continue;
        }

        TopLevel[node] = 0;
        for (const vertex_idx pred : G_coarse.Parents(node)) {
            TopLevel[node] = std::max(TopLevel[node], TopLevel[pred] + 1);
        }
    }
    return topLevel;
}

template <typename GraphT>
void StepByStepCoarser<GraphT>::ComputeFilteredTopOrderIdx() {
    top_order_idx = GetFilteredTopOrderIdx(G_coarse, node_valid);
}

template <typename GraphT>
std::vector<vertex_idx_t<Graph_t>> StepByStepCoarser<GraphT>::GetFilteredTopOrderIdx(const GraphT &g,
                                                                                     const std::vector<bool> &isValid) {
    std::vector<vertex_idx> topOrder = GetFilteredTopOrder(isValid, g);
    std::vector<vertex_idx> idx(g.NumVertices());
    for (vertex_idx node = 0; node < top_order.size(); ++node) {
        idx[top_order[node]] = node;
    }
    return idx;
}

template <typename GraphT>
void StepByStepCoarser<GraphT>::CoarsenForPebbling(const GraphT &dagIn,
                                                   GraphT &coarsenedDag,
                                                   std::vector<vertex_idx_t<Graph_t>> &newVertexId) {
    problemType_ = ProblemType::PEBBLING;
    coarseningStrategy_ = CoarseningStrategy::EDGE_BY_EDGE;

    unsigned nrSources = 0;
    for (vertex_idx node = 0; node < dagIn.NumVertices(); ++node) {
        if (dagIn.in_degree(node) == 0) {
            ++nrSources;
        }
    }

    targetNrOfNodes_ = std::max(targetNrOfNodes_, nrSources + 1);

    CoarserGenContractionMap<GraphT, GraphT>::coarsenDag(dagIn, coarsenedDag, new_vertex_id);
}

template <typename GraphT>
bool StepByStepCoarser<GraphT>::IncontractableForPebbling(const std::pair<vertex_idx, vertex_idx> &edge) const {
    if (G_coarse.in_degree(edge.first) == 0) {
        return true;
    }

    VMemwT<Graph_t> sumWeight = G_coarse.VertexMemWeight(edge.first) + G_coarse.VertexMemWeight(edge.second);
    std::set<vertex_idx> parents;
    for (vertex_idx pred : G_coarse.Parents(edge.first)) {
        parents.insert(pred);
    }
    for (vertex_idx pred : G_coarse.Parents(edge.second)) {
        if (pred != edge.first) {
            parents.insert(pred);
        }
    }
    for (vertex_idx node : parents) {
        sum_weight += G_coarse.VertexMemWeight(node);
    }

    if (sum_weight > fast_mem_capacity) {
        return true;
    }

    std::set<vertex_idx> children;
    for (vertex_idx succ : G_coarse.Children(edge.second)) {
        children.insert(succ);
    }
    for (vertex_idx succ : G_coarse.Children(edge.first)) {
        if (succ != edge.second) {
            children.insert(succ);
        }
    }

    for (vertex_idx child : children) {
        sum_weight = G_coarse.VertexMemWeight(edge.first) + G_coarse.VertexMemWeight(edge.second) + G_coarse.VertexMemWeight(child);
        for (vertex_idx pred : G_coarse.Parents(child)) {
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
    std::vector<VMemwT<Graph_t>> memorySum(G_coarse.NumVertices(), 0);
    std::vector<vertex_idx> sources;
    for (vertex_idx node = 0; node < G_coarse.NumVertices(); ++node) {
        if (!nodeValid_[node]) {
            continue;
        }

        if (G_coarse.in_degree(node) > 0) {
            memory_sum[node] = G_coarse.VertexMemWeight(node);
            for (vertex_idx pred : G_coarse.Parents(node)) {
                memory_sum[node] += G_coarse.VertexMemWeight(pred);
            }
        } else {
            sources.push_back(node);
        }
    }

    std::set<vertex_idx> invalidatedSources;
    bool couldMerge = true;
    while (couldMerge) {
        couldMerge = false;
        for (unsigned idx1 = 0; idx1 < sources.size(); ++idx1) {
            vertex_idx sourceA = sources[idx1];
            if (invalidatedSources.find(source_a) != invalidated_sources.end()) {
                continue;
            }

            for (unsigned idx2 = idx1 + 1; idx2 < sources.size(); ++idx2) {
                vertex_idx sourceB = sources[idx2];
                if (invalidatedSources.find(source_b) != invalidated_sources.end()) {
                    continue;
                }

                // check if we can merge source_a and source_b
                std::set<vertex_idx> aChildren, b_children;
                for (vertex_idx succ : G_coarse.Children(source_a)) {
                    a_children.insert(succ);
                }
                for (vertex_idx succ : G_coarse.Children(source_b)) {
                    b_children.insert(succ);
                }

                std::set<vertex_idx> onlyA, only_b, both;
                for (vertex_idx succ : G_coarse.Children(source_a)) {
                    if (b_children.find(succ) == b_children.end()) {
                        only_a.insert(succ);
                    } else {
                        both.insert(succ);
                    }
                }
                for (vertex_idx succ : G_coarse.Children(source_b)) {
                    if (a_children.find(succ) == a_children.end()) {
                        only_b.insert(succ);
                    }
                }

                bool violatesConstraint = false;
                for (vertex_idx node : only_a) {
                    if (memory_sum[node] + G_coarse.VertexMemWeight(source_b) > fast_mem_capacity) {
                        violates_constraint = true;
                    }
                }
                for (vertex_idx node : only_b) {
                    if (memory_sum[node] + G_coarse.VertexMemWeight(source_a) > fast_mem_capacity) {
                        violates_constraint = true;
                    }
                }

                if (violatesConstraint) {
                    continue;
                }

                // check if we want to merge source_a and source_b
                double simDiff = (only_a.size() + only_b.size() == 0) ? 0.0001
                                                                      : static_cast<double>(only_a.size() + only_b.size());
                double ratio = static_cast<double>(both.size()) / sim_diff;

                if (ratio > 2) {
                    ContractSingleEdge(std::make_pair(source_a, source_b));
                    invalidatedSources.insert(source_b);
                    couldMerge = true;

                    for (vertex_idx node : only_a) {
                        memory_sum[node] += G_coarse.VertexMemWeight(source_b);
                    }
                    for (vertex_idx node : only_b) {
                        memory_sum[node] += G_coarse.VertexMemWeight(source_a);
                    }
                }
            }
        }
    }
}

template <typename GraphT>
GraphT StepByStepCoarser<GraphT>::Contract(const std::vector<vertex_idx_t<Graph_t>> &newVertexId) const {
    GraphT gContracted;
    std::vector<bool> isValid(gFull_.NumVertices(), false);
    for (vertex_idx node = 0; node < gFull_.NumVertices(); ++node) {
        isValid[new_vertex_id[node]] = true;
    }

    for (vertex_idx node = 0; node < gFull_.NumVertices(); ++node) {
        if (isValid[node]) {
            gContracted.add_vertex(0, 0, 0, 0);
        }
    }

    for (vertex_idx node = 0; node < gFull_.NumVertices(); ++node) {
        gContracted.SetVertexWorkWeight(new_vertex_id[node],
                                        gContracted.VertexWorkWeight(new_vertex_id[node]) + gFull_.VertexWorkWeight(node));
        gContracted.SetVertexCommWeight(new_vertex_id[node],
                                        gContracted.VertexCommWeight(new_vertex_id[node]) + gFull_.VertexCommWeight(node));
        gContracted.SetVertexMemWeight(new_vertex_id[node],
                                       gContracted.VertexMemWeight(new_vertex_id[node]) + gFull_.VertexMemWeight(node));
        gContracted.SetVertexType(new_vertex_id[node], gFull_.VertexType(node));
    }

    for (vertex_idx node = 0; node < gFull_.NumVertices(); ++node) {
        for (const auto &out_edge : OutEdges(node, G_full)) {
            const vertex_idx succ = Traget(out_edge, G_full);

            if (new_vertex_id[node] == new_vertex_id[succ]) {
                continue;
            }

            if constexpr (HasEdgeWeightsV<Graph_t>) {
                const auto pair = edge_desc(new_vertex_id[node], new_vertex_id[succ], G_contracted);

                if (pair.second) {
                    G_contracted.SetEdgeCommWeight(pair.first,
                                                   G_contracted.EdgeCommWeight(pair.first) + G_full.EdgeCommWeight(out_edge));
                } else {
                    G_contracted.add_edge(new_vertex_id[node], new_vertex_id[succ], G_full.EdgeCommWeight(out_edge));
                }

            } else {
                if (not edge(new_vertex_id[node], new_vertex_id[succ], G_contracted)) {
                    G_contracted.add_edge(new_vertex_id[node], new_vertex_id[succ]);
                }
            }
        }
    }

    return gContracted;
}

template <typename GraphT>
void StepByStepCoarser<GraphT>::SetIdVector(std::vector<vertex_idx_t<Graph_t>> &newVertexId) const {
    newVertexId.clear();
    newVertexId.resize(gFull_.NumVertices());

    new_vertex_id = GetIntermediateIDs(contractionHistory.size());
}

template <typename GraphT>
std::vector<vertex_idx_t<Graph_t>> StepByStepCoarser<GraphT>::GetIntermediateIDs(vertex_idx untilWhichStep) const {
    std::vector<vertex_idx> Traget(gFull_.NumVertices()), pointsTo(G_full.NumVertices(), std::numeric_limits<vertex_idx>::max());

    for (vertex_idx iterate = 0; iterate < contractionHistory.size() && iterate < until_which_step; ++iterate) {
        const std::pair<vertex_idx, vertex_idx> &contractionStep = contractionHistory[iterate];
        pointsTo[contractionStep.second] = contractionStep.first;
    }

    for (vertex_idx node = 0; node < gFull_.NumVertices(); ++node) {
        target[node] = node;
        while (pointsTo[target[node]] != std::numeric_limits<vertex_idx>::max()) {
            target[node] = pointsTo[target[node]];
        }
    }

    if (contractionHistory.empty() || until_which_step == 0) {
        return target;
    }

    std::vector<bool> isValid(gFull_.NumVertices(), false);
    for (vertex_idx node = 0; node < gFull_.NumVertices(); ++node) {
        isValid[target[node]] = true;
    }

    std::vector<vertex_idx> newId(gFull_.NumVertices());
    vertex_idx currentIndex = 0;
    for (vertex_idx node = 0; node < gFull_.NumVertices(); ++node) {
        if (isValid[node]) {
            newId[node] = current_index++;
        }
    }

    for (vertex_idx node = 0; node < gFull_.NumVertices(); ++node) {
        target[node] = new_id[target[node]];
    }

    boost_graph_t tempDag;
    temp_dag = Contract(target);
    std::vector<bool> allValid(tempDag.NumVertices(), true);
    std::vector<vertex_idx> topIdx = GetFilteredTopOrderIdx(temp_dag, all_valid);

    for (vertex_idx node = 0; node < gFull_.NumVertices(); ++node) {
        target[node] = top_idx[target[node]];
    }

    return target;
}

}    // namespace osp
