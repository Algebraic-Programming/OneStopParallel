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

#include "concepts/computational_dag_concept.hpp"
#include "concepts/constructable_computational_dag_concept.hpp"
#include "graph_algorithms/directed_graph_edge_desc_util.hpp"
#include "graph_algorithms/computational_dag_construction_util.hpp"
#include "graph_algorithms/directed_graph_top_sort.hpp"
#include "coarser/Coarser.hpp"
#include "graph_implementations/boost_graphs/boost_graph.hpp"

namespace osp {

template<typename Graph_t>
class StepByStepCoarser : public CoarserGenContractionMap<Graph_t, Graph_t> {

    using vertex_idx = vertex_idx_t<Graph_t>;

    using vertex_type_t_or_default   = std::conditional_t<is_computational_dag_typed_vertices_v<Graph_t>, v_type_t<Graph_t>, unsigned>;
    using edge_commw_t_or_default   = std::conditional_t<is_computational_dag_edge_desc_v<Graph_t>, e_commw_t<Graph_t>, v_commw_t<Graph_t>>;
    
    using boost_graph_t = boost_graph<v_workw_t<Graph_t>, v_commw_t<Graph_t>, v_memw_t<Graph_t>, vertex_type_t_or_default, edge_commw_t_or_default >;

  public:
    enum COARSENING_STRATEGY
    {
        EDGE_BY_EDGE,
        BOTTOM_LEVEL_CLUSTERS
    };

    enum PROBLEM_TYPE
    {
        SCHEDULING,
        PEBBLING
    };

    struct EdgeToContract{
        std::pair<vertex_idx, vertex_idx> edge;
        v_workw_t<Graph_t> work_weight;
        v_commw_t<Graph_t> comm_weight;

        EdgeToContract(const vertex_idx source, const vertex_idx target, const v_workw_t<Graph_t> work_weight_, const v_commw_t<Graph_t> comm_weight_)
            : edge(source, target), work_weight(work_weight_), comm_weight(comm_weight_) {}

        bool operator<(const EdgeToContract &other) const {
            return (work_weight < other.work_weight || (work_weight == other.work_weight && comm_weight < other.comm_weight));
        }
    };

  private:

    std::vector<std::pair<vertex_idx, vertex_idx> > contractionHistory;

    COARSENING_STRATEGY coarsening_strategy = COARSENING_STRATEGY::EDGE_BY_EDGE;
    PROBLEM_TYPE problem_type = PROBLEM_TYPE::SCHEDULING;

    unsigned target_nr_of_nodes = 0;

    Graph_t G_full;
    boost_graph_t G_coarse;

    std::vector<std::set<vertex_idx>> contains;

    std::map<std::pair<vertex_idx, vertex_idx>, v_commw_t<Graph_t> > edgeWeights;
    std::map<std::pair<vertex_idx, vertex_idx>, v_commw_t<Graph_t> > contractable;
    std::vector<bool> node_valid;
    std::vector<vertex_idx> top_order_idx;

    v_memw_t<Graph_t> fast_mem_capacity = std::numeric_limits<v_memw_t<Graph_t>>::max(); // for pebbling

    // Utility functions for coarsening in general
    void ContractSingleEdge(std::pair<vertex_idx, vertex_idx> edge);
    void ComputeFilteredTopOrderIdx();

    void InitializeContractableEdges();
    bool isContractable(std::pair<vertex_idx, vertex_idx> edge) const;
    std::set<vertex_idx> getContractableChildren(vertex_idx node) const;
    std::set<vertex_idx> getContractableParents(vertex_idx node) const;
    void updateDistantEdgeContractibility(std::pair<vertex_idx, vertex_idx> edge);

    std::pair<vertex_idx, vertex_idx> PickEdgeToContract(const std::vector<EdgeToContract>& candidates) const;
    std::vector<EdgeToContract> CreateEdgeCandidateList() const;

    // Utility functions for cluster coarsening
    std::vector<std::pair<vertex_idx, vertex_idx> > ClusterCoarsen() const;
    std::vector<unsigned> ComputeFilteredTopLevel() const;

    // Utility functions for coarsening in a pebbling problem
    bool IncontractableForPebbling(const std::pair<vertex_idx, vertex_idx>&) const;
    void MergeSourcesInPebbling();

    // Utility for contracting into final format
    void SetIdVector(std::vector<vertex_idx_t<Graph_t>> &new_vertex_id) const;
    static std::vector<vertex_idx> GetFilteredTopOrderIdx(const Graph_t& G, const std::vector<bool>& is_valid);


  public:
    virtual ~StepByStepCoarser() = default;

    virtual std::string getCoarserName() const override { return "StepByStepCoarsening"; }



    // DAG coarsening
    virtual std::vector<vertex_idx_t<Graph_t>> generate_vertex_contraction_map(const Graph_t &dag_in) override;



    // Coarsening for pebbling problems - leaves source nodes intact, considers memory bound
    void coarsenForPebbling(const Graph_t& dag_in, Graph_t &coarsened_dag,
                           std::vector<vertex_idx_t<Graph_t>> &new_vertex_id);



    void setCoarseningStrategy(COARSENING_STRATEGY strategy_){ coarsening_strategy = strategy_;}
    void setTargetNumberOfNodes(const unsigned nr_nodes_){ target_nr_of_nodes = nr_nodes_;}
    void setFastMemCapacity(const v_memw_t<Graph_t> capacity_){ fast_mem_capacity = capacity_;}

    std::vector<std::pair<vertex_idx, vertex_idx> > getContractionHistory() const {return contractionHistory;}
    std::vector<vertex_idx> GetIntermediateIDs(vertex_idx until_which_step) const;
    Graph_t Contract(const std::vector<vertex_idx_t<Graph_t>> &new_vertex_id) const;
    const Graph_t& getOriginalDag() const {return G_full;}
};

// template<typename Graph_t>
// bool StepByStepCoarser<Graph_t>::coarseDag(const Graph_t& dag_in, Graph_t &dag_out,
//                         std::vector<std::vector<vertex_idx_t<Graph_t>>> &old_vertex_ids,
//                         std::vector<vertex_idx_t<Graph_t>> &new_vertex_id)

template<typename Graph_t>
std::vector<vertex_idx_t<Graph_t>> StepByStepCoarser<Graph_t>::generate_vertex_contraction_map(const Graph_t &dag_in)
{
    const unsigned N = static_cast<unsigned>(dag_in.num_vertices());

    G_full = dag_in;
    for(vertex_idx node = G_coarse.num_vertices(); node > 0;)
    {
        --node;
        G_coarse.remove_vertex(node);
    }

    construct_computational_dag(G_full, G_coarse);

    contractionHistory.clear();

    // target nr of nodes must be reasonable
    if(target_nr_of_nodes == 0 || target_nr_of_nodes > N)
        target_nr_of_nodes = std::max(N/2, 1U);

    // list of original node indices contained in each contracted node
    contains.clear();
    contains.resize(N);

    node_valid.clear();
    node_valid.resize(N, true);

    for (vertex_idx node = 0; node < N; ++node)
        contains[node].insert(node);

    //used for original, slow coarsening
    edgeWeights.clear();
    contractable.clear();
    
    if(coarsening_strategy == COARSENING_STRATEGY::EDGE_BY_EDGE)
    {
        // Init edge weights
        for (vertex_idx node = 0; node < N; ++node)
            for (vertex_idx succ: G_full.children(node))
                edgeWeights[std::make_pair(node, succ)] = G_full.vertex_comm_weight(node);

        // get original contractable edges
        InitializeContractableEdges();
    }

    for (unsigned NrOfNodes = N; NrOfNodes > target_nr_of_nodes; ) {
        // Single contraction step

        std::vector<std::pair<vertex_idx, vertex_idx>> edgesToContract;

        // choose edges to contract in this step
        if(coarsening_strategy == COARSENING_STRATEGY::EDGE_BY_EDGE)
        {
            std::vector<EdgeToContract> candidates = CreateEdgeCandidateList();
            if(candidates.empty())
            {
                std::cout<<"Error: no more edges to contract"<<std::endl;
                break;
            }
            std::pair<vertex_idx, vertex_idx> chosenEdge = PickEdgeToContract(candidates);
            edgesToContract.push_back(chosenEdge);

            // Update far-away edges that become uncontractable now
            updateDistantEdgeContractibility(chosenEdge);
        }
        else
            edgesToContract = ClusterCoarsen();

        if(edgesToContract.empty())
            break;
        
        // contract these edges
        for(const std::pair<vertex_idx, vertex_idx>& edge : edgesToContract)
        {
            if(coarsening_strategy == COARSENING_STRATEGY::EDGE_BY_EDGE)
            {
                //Update contractable edges - edge.b
                for(vertex_idx pred : G_coarse.parents(edge.second))
                    contractable.erase(std::make_pair(pred, edge.second));
                
                for(vertex_idx succ : G_coarse.children(edge.second))
                    contractable.erase(std::make_pair(edge.second, succ));
            }

            ContractSingleEdge(edge);
            node_valid[edge.second] = false;

            if(coarsening_strategy == COARSENING_STRATEGY::EDGE_BY_EDGE)
            {
                ComputeFilteredTopOrderIdx();

                //Update contractable edges - edge.a
                std::set<vertex_idx> contractableParents = getContractableParents(edge.first);
                for (vertex_idx pred : G_coarse.parents(edge.first))
                {
                    if(contractableParents.find(pred) != contractableParents.end())
                        contractable[std::make_pair(pred, edge.first)] = edgeWeights[std::make_pair(pred, edge.first)];
                    else
                        contractable.erase(std::make_pair(pred, edge.first));
                }
                
                std::set<vertex_idx> contractableChildren = getContractableChildren(edge.first);
                for (vertex_idx succ : G_coarse.children(edge.first))
                {
                    if(contractableChildren.find(succ) != contractableChildren.end())
                        contractable[std::make_pair(edge.first, succ)] = edgeWeights[std::make_pair(edge.first, succ)];
                    else
                        contractable.erase(std::make_pair(edge.first, succ));
                }
            }
            --NrOfNodes;
            if(NrOfNodes == target_nr_of_nodes)
                break;
        }
    }

    if(problem_type == PROBLEM_TYPE::PEBBLING)
        MergeSourcesInPebbling();

    std::vector<vertex_idx_t<Graph_t>> new_vertex_id;
    SetIdVector(new_vertex_id);

    return new_vertex_id;
}

template<typename Graph_t>
void StepByStepCoarser<Graph_t>::ContractSingleEdge(std::pair<vertex_idx, vertex_idx> edge)
{
    G_coarse.set_vertex_work_weight(edge.first, G_coarse.vertex_work_weight(edge.first) + G_coarse.vertex_work_weight(edge.second));
    G_coarse.set_vertex_work_weight(edge.second, 0);

    G_coarse.set_vertex_comm_weight(edge.first, G_coarse.vertex_comm_weight(edge.first) + G_coarse.vertex_comm_weight(edge.second));
    G_coarse.set_vertex_comm_weight(edge.second, 0);
    
    G_coarse.set_vertex_mem_weight(edge.first, G_coarse.vertex_mem_weight(edge.first) + G_coarse.vertex_mem_weight(edge.second));
    G_coarse.set_vertex_mem_weight(edge.second, 0);

    contractionHistory.emplace_back(edge.first, edge.second);

    // process incoming edges
    std::set<vertex_idx> parents_of_source;
    for(vertex_idx pred : G_coarse.parents(edge.first))
        parents_of_source.insert(pred);

    for(vertex_idx pred : G_coarse.parents(edge.second))
    {
        if(pred == edge.first)
            continue;
        if(parents_of_source.find(pred) != parents_of_source.end()) // combine edges
        {
            edgeWeights[std::make_pair(pred, edge.first)] = 0;
            for (vertex_idx node: contains[pred])
                for (vertex_idx succ: G_coarse.children(node))
                    if (succ == edge.first || succ == edge.second)
                        edgeWeights[std::make_pair(pred, edge.first)] += G_full.vertex_comm_weight(node);
            
            edgeWeights.erase(std::make_pair(pred, edge.second));
        }
        else // add incoming edge
        {
            G_coarse.add_edge(pred, edge.first);
            edgeWeights[std::make_pair(pred, edge.first)] = edgeWeights[std::make_pair(pred, edge.second)];
        }
    }

    // process outgoing edges
    std::set<vertex_idx> children_of_source;
    for(vertex_idx succ : G_coarse.children(edge.first))
        children_of_source.insert(succ);

    for(vertex_idx succ : G_coarse.children(edge.second))
    {
        if(children_of_source.find(succ) != children_of_source.end()) // combine edges
        {
            edgeWeights[std::make_pair(edge.first, succ)] += edgeWeights[std::make_pair(edge.second, succ)]; 
            edgeWeights.erase(std::make_pair(edge.second, succ));
        }
        else // add outgoing edge
        {
            G_coarse.add_edge(edge.first, succ);
            edgeWeights[std::make_pair(edge.first, succ)] = edgeWeights[std::make_pair(edge.second, succ)];
        }
    }

    G_coarse.clear_vertex(edge.second);

    for (vertex_idx node: contains[edge.second])
        contains[edge.first].insert(node);

    contains[edge.second].clear();
}

template<typename Graph_t>
bool StepByStepCoarser<Graph_t>::isContractable(std::pair<vertex_idx, vertex_idx> edge) const
{
    
    std::deque<vertex_idx> Queue;
    std::set<vertex_idx> visited;
    for (vertex_idx succ : G_coarse.children(edge.first))
        if (node_valid[succ] && top_order_idx[succ] < top_order_idx[edge.second]) {
            Queue.push_back(succ);
            visited.insert(succ);
        }

    while (!Queue.empty()) {
        const vertex_idx node = Queue.front();
        Queue.pop_front();
        for (vertex_idx succ : G_coarse.children(node)) {
            if (succ == edge.second)
                return false;

            if (node_valid[succ] && top_order_idx[succ] < top_order_idx[edge.second] && visited.count(succ) == 0) {
                Queue.push_back(succ);
                visited.insert(succ);
            }
        }
    }
    return true;
}

template<typename Graph_t>
std::set<vertex_idx_t<Graph_t> > StepByStepCoarser<Graph_t>::getContractableChildren(const vertex_idx node) const
{
    std::deque<vertex_idx> Queue;
    std::set<vertex_idx> visited;
    std::set<vertex_idx> succ_contractable;
    vertex_idx topOrderMax = top_order_idx[node];

    for (vertex_idx succ : G_coarse.children(node))
    {
        if(node_valid[succ])
            succ_contractable.insert(succ);
        
        if(top_order_idx[succ] > topOrderMax)
            topOrderMax = top_order_idx[succ];

        if (node_valid[succ]) {
            Queue.push_back(succ);
            visited.insert(succ);
        }
    }

    while (!Queue.empty()) {
        const vertex_idx node = Queue.front();
        Queue.pop_front();
        for (vertex_idx succ : G_coarse.children(node)) {
            
            succ_contractable.erase(succ);

            if (node_valid[succ] && top_order_idx[succ] < topOrderMax && visited.count(succ) == 0) {
                Queue.push_back(succ);
                visited.insert(succ);
            }
        }
    }

    return succ_contractable;
}

template<typename Graph_t>
std::set<vertex_idx_t<Graph_t> > StepByStepCoarser<Graph_t>::getContractableParents(const vertex_idx node) const
{
    std::deque<vertex_idx> Queue;
    std::set<vertex_idx> visited;
    std::set<vertex_idx> pred_contractable;
    vertex_idx topOrderMin = top_order_idx[node];

    for (vertex_idx pred : G_coarse.parents(node))
    {
        if(node_valid[pred])
            pred_contractable.insert(pred);
        
        if(top_order_idx[pred] < topOrderMin)
            topOrderMin = top_order_idx[pred];

        if (node_valid[pred]) {
            Queue.push_back(pred);
            visited.insert(pred);
        }
    }

    while (!Queue.empty()) {
        const vertex_idx node = Queue.front();
        Queue.pop_front();
        for (vertex_idx pred : G_coarse.parents(node)) {
            
            pred_contractable.erase(pred);

            if (node_valid[pred] && top_order_idx[pred] > topOrderMin && visited.count(pred) == 0) {
                Queue.push_back(pred);
                visited.insert(pred);
            }
        }
    }

    return pred_contractable;
}

template<typename Graph_t>
void StepByStepCoarser<Graph_t>::InitializeContractableEdges() {

    ComputeFilteredTopOrderIdx();

    for (vertex_idx node = 0; node < G_full.num_vertices(); ++node)
    {
        std::set<vertex_idx> succ_contractable = getContractableChildren(node);
        for(vertex_idx succ : succ_contractable)          
            contractable[std::make_pair(node, succ)] = G_full.vertex_comm_weight(node);
    }
}

template<typename Graph_t>
void StepByStepCoarser<Graph_t>::updateDistantEdgeContractibility(std::pair<vertex_idx, vertex_idx> edge)
{

    std::unordered_set<vertex_idx> ancestors, descendant;
    std::deque<vertex_idx> Queue;
    for (vertex_idx succ : G_coarse.children(edge.first))
        if (succ != edge.second) {
            Queue.push_back(succ);
            descendant.insert(succ);
        }
    while (!Queue.empty()) {
        const vertex_idx node = Queue.front();
        Queue.pop_front();
        for (vertex_idx succ : G_coarse.children(node))
            if (descendant.count(succ) == 0) {
                Queue.push_back(succ);
                descendant.insert(succ);
            }
    }

    for (vertex_idx pred : G_coarse.parents(edge.second))
        if (pred != edge.first) {
            Queue.push_back(pred);
            ancestors.insert(pred);
        }
    while (!Queue.empty()) {
        const vertex_idx node = Queue.front();
        Queue.pop_front();
        for (vertex_idx pred : G_coarse.parents(node))
            if (ancestors.count(pred) == 0) {
                Queue.push_back(pred);
                ancestors.insert(pred);
            }
    }

    for (const vertex_idx node : ancestors)
        for (const vertex_idx succ : G_coarse.children(node))
            if (descendant.count(succ) > 0)
                contractable.erase(std::make_pair(node, succ));
}

template<typename Graph_t>
std::vector<typename StepByStepCoarser<Graph_t>::EdgeToContract> StepByStepCoarser<Graph_t>::CreateEdgeCandidateList() const
{
    std::vector<EdgeToContract> candidates;

    for (auto it = contractable.cbegin(); it != contractable.cend(); ++it)
    {
        if(problem_type == PROBLEM_TYPE::PEBBLING && IncontractableForPebbling(it->first))
            continue;

        candidates.emplace_back(it->first.first, it->first.second, contains[it->first.first].size() + contains[it->first.second].size(), it->second);
    }

    std::sort(candidates.begin(), candidates.end());
    return candidates;
}

template<typename Graph_t>
std::pair<vertex_idx_t<Graph_t>, vertex_idx_t<Graph_t>> StepByStepCoarser<Graph_t>::PickEdgeToContract(const std::vector<EdgeToContract>& candidates) const
{
    size_t limit = (candidates.size() + 2) / 3;
    v_workw_t<Graph_t> limitCardinality = candidates[limit].work_weight;
    while (limit < candidates.size() - 1 && candidates[limit + 1].work_weight == limitCardinality)
        ++limit;

    // an edge case
    if (candidates.size() == 1)
        limit = 0;

    EdgeToContract chosen = candidates[0];
    unsigned best = 0;
    for (unsigned idx = 1; idx <= limit; ++idx)
        if (candidates[idx].comm_weight > candidates[best].comm_weight)
            best = idx;

    chosen = candidates[best];
    return chosen.edge;
}

/**
 * @brief Acyclic graph contractor based on (Herrmann, Julien, et al. "Acyclic partitioning of large directed acyclic graphs." 2017 17th IEEE/ACM international symposium on cluster, cloud and grid computing (CCGRID). IEEE, 2017.))
 * @brief with minor changes and fixes
 * 
 */
template<typename Graph_t>
std::vector<std::pair<vertex_idx_t<Graph_t>, vertex_idx_t<Graph_t> > > StepByStepCoarser<Graph_t>::ClusterCoarsen() const
{
    std::vector<bool> singleton(G_full.num_vertices(), true);
    std::vector<vertex_idx> leader(G_full.num_vertices());
    std::vector<unsigned> weight(G_full.num_vertices());
    std::vector<unsigned> nrBadNeighbors(G_full.num_vertices());
    std::vector<vertex_idx> leaderBadNeighbors(G_full.num_vertices());

    std::vector<unsigned> minTopLevel(G_full.num_vertices());
    std::vector<unsigned> maxTopLevel(G_full.num_vertices());
    std::vector<vertex_idx> clusterNewID(G_full.num_vertices());

    std::vector<std::pair<vertex_idx, vertex_idx> > contractionSteps;
    std::vector<unsigned> topLevel = ComputeFilteredTopLevel();
    for(vertex_idx node = 0; node < G_full.num_vertices(); ++node)
        if(node_valid[node])
        {
            leader[node]=node;
            weight[node]=1 /*G_coarse.vertex_work_weight(node)*/;
            nrBadNeighbors[node]=0;
            leaderBadNeighbors[node]=UINT_MAX;
            clusterNewID[node]=node;
            minTopLevel[node]=topLevel[node];
            maxTopLevel[node]=topLevel[node];
        }

    for(vertex_idx node = 0; node < G_full.num_vertices(); ++node)
    {
        if(!node_valid[node] || !singleton[node])
            continue;

        if(nrBadNeighbors[node] > 1)
            continue;

        std::vector<vertex_idx> validNeighbors;
        for(vertex_idx pred: G_coarse.parents(node))
        {
            // direct check of condition 1
            if(topLevel[node] < maxTopLevel[leader[pred]]-1 || topLevel[node] > minTopLevel[leader[pred]]+1)
                continue;
            // indirect check of condition 2
            if(nrBadNeighbors[node] > 1 || (nrBadNeighbors[node] == 1 && leaderBadNeighbors[node] != leader[pred]))
                continue;
            //check condition 2 for pred if it is a singleton
            if(singleton[pred] && nrBadNeighbors[pred] > 0)
                continue;

            // check viability for pebbling
            if(problem_type == PROBLEM_TYPE::PEBBLING && IncontractableForPebbling(std::make_pair(pred, node)))
                continue;

            validNeighbors.push_back(pred);
        }
        for(vertex_idx succ: G_coarse.children(node))
        {
            // direct check of condition 1
            if(topLevel[node] < maxTopLevel[leader[succ]]-1 || topLevel[node] > minTopLevel[leader[succ]]+1)
                continue;
            // indirect check of condition 2
            if(nrBadNeighbors[node] > 1 || (nrBadNeighbors[node] == 1 && leaderBadNeighbors[node] != leader[succ]))
                continue;
            //check condition 2 for pred if it is a singleton
            if(singleton[succ] && nrBadNeighbors[succ] > 0)
                continue;

            // check viability for pebbling
            if(problem_type == PROBLEM_TYPE::PEBBLING && IncontractableForPebbling(std::make_pair(node, succ)))
                continue;

            validNeighbors.push_back(succ);
        }

        vertex_idx bestNeighbor = std::numeric_limits<vertex_idx>::max();
        for(vertex_idx neigh : validNeighbors)
            if(bestNeighbor == std::numeric_limits<vertex_idx>::max() || weight[leader[neigh]] < weight[leader[bestNeighbor]])
                bestNeighbor = neigh;

        if(bestNeighbor == std::numeric_limits<vertex_idx>::max())
            continue;

        vertex_idx newLead = leader[bestNeighbor];
        leader[node] = newLead;
        weight[newLead] += weight[node];

        bool is_parent = false;
        for(vertex_idx pred : G_coarse.parents(node))
            if(pred == bestNeighbor)
                is_parent = true;

        if(is_parent)
            contractionSteps.emplace_back(clusterNewID[newLead], node);
        else
        {
            contractionSteps.emplace_back(node, clusterNewID[newLead]);
            clusterNewID[newLead] = node;
        }

        minTopLevel[newLead] = std::min(minTopLevel[newLead], topLevel[node]);
        maxTopLevel[newLead] = std::max(maxTopLevel[newLead], topLevel[node]);

        for(vertex_idx pred: G_coarse.parents(node))
        {
            if(std::abs( static_cast<int>(topLevel[pred]) - static_cast<int>(maxTopLevel[newLead]) ) != 1 &&
                std::abs( static_cast<int>(topLevel[pred]) - static_cast<int>(minTopLevel[newLead]) ) != 1)
                continue;

            if(nrBadNeighbors[pred] == 0)
            {
                ++nrBadNeighbors[pred];
                leaderBadNeighbors[pred] = newLead;
            }
            else if(nrBadNeighbors[pred] == 1 && leaderBadNeighbors[pred] != newLead)
                ++nrBadNeighbors[pred];
        }
        for(vertex_idx succ: G_coarse.children(node))
        {
            if(std::abs( static_cast<int>(topLevel[succ]) - static_cast<int>(maxTopLevel[newLead]) ) != 1 &&
                std::abs( static_cast<int>(topLevel[succ]) - static_cast<int>(minTopLevel[newLead]) ) != 1)
                continue;

            if(nrBadNeighbors[succ]==0)
            {
                ++nrBadNeighbors[succ];
                leaderBadNeighbors[succ] = newLead;
            }
            else if(nrBadNeighbors[succ] == 1 && leaderBadNeighbors[succ] != newLead)
                ++nrBadNeighbors[succ];
        }

        if(singleton[bestNeighbor])
        {
            for(vertex_idx pred: G_coarse.parents(bestNeighbor) )
            {
                if(std::abs( static_cast<int>(topLevel[pred]) - static_cast<int>(maxTopLevel[newLead]) ) != 1 &&
                    std::abs( static_cast<int>(topLevel[pred]) - static_cast<int>(minTopLevel[newLead]) ) != 1)
                    continue;

                if(nrBadNeighbors[pred] == 0)
                {
                    ++nrBadNeighbors[pred];
                    leaderBadNeighbors[pred] = newLead;
                }
                else if(nrBadNeighbors[pred] == 1 && leaderBadNeighbors[pred] != newLead)
                    ++nrBadNeighbors[pred];
            }
            for(vertex_idx succ: G_coarse.children(bestNeighbor))
            {
                if(std::abs( static_cast<int>(topLevel[succ]) - static_cast<int>(maxTopLevel[newLead]) ) != 1 &&
                    std::abs( static_cast<int>(topLevel[succ]) - static_cast<int>(minTopLevel[newLead]) ) != 1)
                    continue;

                if(nrBadNeighbors[succ]==0)
                {
                    ++nrBadNeighbors[succ];
                    leaderBadNeighbors[succ] = newLead;
                }
                else if(nrBadNeighbors[succ] == 1 && leaderBadNeighbors[succ] != newLead)
                    ++nrBadNeighbors[succ];
            }
            singleton[bestNeighbor] = false;
        }
        singleton[node] = false;
    }

    return contractionSteps;
}

template<typename Graph_t> 
std::vector<unsigned> StepByStepCoarser<Graph_t>::ComputeFilteredTopLevel() const
{
    std::vector<unsigned> TopLevel(G_full.num_vertices());
    for (const vertex_idx node : top_sort_view(G_coarse)) {
        if(!node_valid[node])
            continue;

        TopLevel[node] = 0;
        for (const vertex_idx pred: G_coarse.parents(node) )
            TopLevel[node] = std::max(TopLevel[node], TopLevel[pred] + 1);

    }
    return TopLevel;
}

template<typename Graph_t> 
void StepByStepCoarser<Graph_t>::ComputeFilteredTopOrderIdx() {
    top_order_idx = GetFilteredTopOrderIdx(G_coarse, node_valid);
}

template<typename Graph_t> 
std::vector<vertex_idx_t<Graph_t> > StepByStepCoarser<Graph_t>::GetFilteredTopOrderIdx(const Graph_t& G, const std::vector<bool>& is_valid) {
    std::vector<vertex_idx> top_order = GetFilteredTopOrder(is_valid, G);
    std::vector<vertex_idx> idx(G.num_vertices());
    for (vertex_idx node = 0; node < top_order.size(); ++node)
        idx[top_order[node]] = node;
    return idx;
}


template<typename Graph_t> 
void StepByStepCoarser<Graph_t>::coarsenForPebbling(const Graph_t& dag_in, Graph_t &coarsened_dag,
                           std::vector<vertex_idx_t<Graph_t>> &new_vertex_id)
{

    problem_type = PROBLEM_TYPE::PEBBLING;
    coarsening_strategy = COARSENING_STRATEGY::EDGE_BY_EDGE;

    unsigned nr_sources = 0;
    for(vertex_idx node = 0; node < dag_in.num_vertices(); ++node)
        if(dag_in.in_degree(node) == 0)
            ++nr_sources;

    target_nr_of_nodes = std::max(target_nr_of_nodes, nr_sources + 1);

    CoarserGenContractionMap<Graph_t, Graph_t>::coarsenDag(dag_in, coarsened_dag, new_vertex_id);
}

template<typename Graph_t> 
bool StepByStepCoarser<Graph_t>::IncontractableForPebbling(const std::pair<vertex_idx, vertex_idx>& edge) const
{
    if(G_coarse.in_degree(edge.first) == 0)
        return true;

    v_memw_t<Graph_t> sum_weight = G_coarse.vertex_mem_weight(edge.first) + G_coarse.vertex_mem_weight(edge.second);
    std::set<vertex_idx> parents;
    for(vertex_idx pred : G_coarse.parents(edge.first))
        parents.insert(pred);
    for(vertex_idx pred : G_coarse.parents(edge.second))
        if(pred != edge.first)
            parents.insert(pred);
    for(vertex_idx node : parents)
        sum_weight += G_coarse.vertex_mem_weight(node);

    if(sum_weight > fast_mem_capacity)
        return true;
    
    std::set<vertex_idx> children;
    for(vertex_idx succ: G_coarse.children(edge.second))
        children.insert(succ);
    for(vertex_idx succ: G_coarse.children(edge.first))
        if(succ != edge.second)
            children.insert(succ);

    for(vertex_idx child : children)
    {
        sum_weight = G_coarse.vertex_mem_weight(edge.first) + G_coarse.vertex_mem_weight(edge.second) + G_coarse.vertex_mem_weight(child);
        for(vertex_idx pred: G_coarse.parents(child))
        {
            if(pred != edge.first && pred != edge.second)
                sum_weight += G_coarse.vertex_mem_weight(pred);
        }
        
        if(sum_weight > fast_mem_capacity)
            return true;
    }
    return false;
}

template<typename Graph_t> 
void StepByStepCoarser<Graph_t>::MergeSourcesInPebbling()
{
    // initialize memory requirement sums to check viability later
    std::vector<v_memw_t<Graph_t> > memory_sum(G_coarse.num_vertices(), 0);
    std::vector<vertex_idx> sources;
    for(vertex_idx node = 0; node < G_coarse.num_vertices(); ++node)
    {
        if(!node_valid[node])
            continue;

        if(G_coarse.in_degree(node)>0)
        {
            memory_sum[node] = G_coarse.vertex_mem_weight(node);
            for(vertex_idx pred: G_coarse.parents(node))
                memory_sum[node] += G_coarse.vertex_mem_weight(pred);
        }
        else 
            sources.push_back(node);
    }
    
    std::set<vertex_idx> invalidated_sources;
    bool could_merge = true;
    while(could_merge)
    {
        could_merge = false;
        for(unsigned idx1 = 0; idx1 < sources.size(); ++idx1)
        {
            vertex_idx source_a = sources[idx1];
            if(invalidated_sources.find(source_a) != invalidated_sources.end())
                continue;
            
            for(unsigned idx2 = idx1 + 1; idx2 < sources.size(); ++idx2)
            {
                vertex_idx source_b = sources[idx2];
                if(invalidated_sources.find(source_b) != invalidated_sources.end())
                    continue;
                
                // check if we can merge source_a and source_b
                std::set<vertex_idx> a_children, b_children;
                for(vertex_idx succ: G_coarse.children(source_a))
                    a_children.insert(succ);
                for(vertex_idx succ: G_coarse.children(source_b))
                    b_children.insert(succ);
                
                std::set<vertex_idx> only_a, only_b, both;
                for(vertex_idx succ: G_coarse.children(source_a))
                {
                    if(b_children.find(succ) == b_children.end())
                        only_a.insert(succ);
                    else
                        both.insert(succ);
                }
                for(vertex_idx succ: G_coarse.children(source_b))
                {
                    if(a_children.find(succ) == a_children.end())
                        only_b.insert(succ);
                }

                bool violates_constraint = false;
                for(vertex_idx node : only_a)
                    if(memory_sum[node] + G_coarse.vertex_mem_weight(source_b) > fast_mem_capacity)
                        violates_constraint = true;
                for(vertex_idx node : only_b)
                    if(memory_sum[node] + G_coarse.vertex_mem_weight(source_a) > fast_mem_capacity)
                        violates_constraint = true;

                if(violates_constraint)
                    continue;

                // check if we want to merge source_a and source_b
                double sim_diff = (only_a.size() + only_b.size() == 0) ? 0.0001 : static_cast<double>(only_a.size() + only_b.size());
                double ratio = static_cast<double>(both.size()) / sim_diff;
                
                if(ratio > 2)
                {
                    ContractSingleEdge(std::make_pair(source_a, source_b));
                    invalidated_sources.insert(source_b);
                    could_merge = true;

                    for(vertex_idx node : only_a)
                        memory_sum[node] += G_coarse.vertex_mem_weight(source_b);
                    for(vertex_idx node : only_b)
                        memory_sum[node] += G_coarse.vertex_mem_weight(source_a);
                }
            }
        }
    }
}

template<typename Graph_t> 
Graph_t StepByStepCoarser<Graph_t>::Contract(const std::vector<vertex_idx_t<Graph_t>> &new_vertex_id) const
{
    Graph_t G_contracted;
    std::vector<bool> is_valid(G_full.num_vertices(), false);
    for (vertex_idx node = 0; node < G_full.num_vertices(); ++node)
        is_valid[new_vertex_id[node]] = true;

    for (vertex_idx node = 0; node < G_full.num_vertices(); ++node)
        if(is_valid[node])
            G_contracted.add_vertex(0, 0, 0, 0);

    for (vertex_idx node = 0; node < G_full.num_vertices(); ++node)
    {
        G_contracted.set_vertex_work_weight(new_vertex_id[node],
                G_contracted.vertex_work_weight(new_vertex_id[node]) + G_full.vertex_work_weight(node));
        G_contracted.set_vertex_comm_weight(new_vertex_id[node],
                G_contracted.vertex_comm_weight(new_vertex_id[node]) + G_full.vertex_comm_weight(node));
        G_contracted.set_vertex_mem_weight(new_vertex_id[node],
                G_contracted.vertex_mem_weight(new_vertex_id[node]) + G_full.vertex_mem_weight(node));
        G_contracted.set_vertex_type(new_vertex_id[node], G_full.vertex_type(node));
    }

    for (vertex_idx node = 0; node < G_full.num_vertices(); ++node)
        for (const auto &out_edge : G_full.out_edges(node))
        {
            const vertex_idx succ = target(out_edge, G_full);

            if (new_vertex_id[node] == new_vertex_id[succ])
                continue;
            
            if constexpr (has_edge_weights_v<Graph_t>) {

                const auto pair = edge_desc(new_vertex_id[node], new_vertex_id[succ], G_contracted);

                if (pair.second) {
                    G_contracted.set_edge_comm_weight(pair.first, G_contracted.edge_comm_weight(pair.first) +
                                                                    G_full.edge_comm_weight(out_edge));
                } else {
                    G_contracted.add_edge(new_vertex_id[node], new_vertex_id[succ], G_full.edge_comm_weight(out_edge));
                }

            } else {

                if (not edge(new_vertex_id[node], new_vertex_id[succ], G_contracted)) {
                    G_contracted.add_edge(new_vertex_id[node], new_vertex_id[succ]);
                }
            }
        }
    
    return G_contracted;
}

template<typename Graph_t> 
void StepByStepCoarser<Graph_t>::SetIdVector(std::vector<vertex_idx_t<Graph_t>> &new_vertex_id) const
{
    new_vertex_id.clear();
    new_vertex_id.resize(G_full.num_vertices());

    new_vertex_id = GetIntermediateIDs(contractionHistory.size());
}

template<typename Graph_t>
std::vector<vertex_idx_t<Graph_t> > StepByStepCoarser<Graph_t>::GetIntermediateIDs(vertex_idx until_which_step) const {
    
    std::vector<vertex_idx> target(G_full.num_vertices()), pointsTo(G_full.num_vertices(), std::numeric_limits<vertex_idx>::max());

    for(vertex_idx iterate = 0; iterate < contractionHistory.size() && iterate < until_which_step; ++iterate)
    {
        const std::pair<vertex_idx, vertex_idx>& contractionStep = contractionHistory[iterate];
        pointsTo[contractionStep.second] = contractionStep.first;
    }

    for (vertex_idx node = 0; node < G_full.num_vertices(); ++node) {
        target[node] = node;
        while (pointsTo[target[node]] != std::numeric_limits<vertex_idx>::max())
            target[node] = pointsTo[target[node]];
    }

    if (contractionHistory.empty() || until_which_step == 0) 
        return target;

    std::vector<bool> is_valid(G_full.num_vertices(), false);
    for (vertex_idx node = 0; node < G_full.num_vertices(); ++node)
        is_valid[target[node]] = true;

    std::vector<vertex_idx> new_id(G_full.num_vertices());
    vertex_idx current_index = 0;
    for (vertex_idx node = 0; node < G_full.num_vertices(); ++node)
        if(is_valid[node])
            new_id[node] = current_index++;

    for (vertex_idx node = 0; node < G_full.num_vertices(); ++node)
        target[node] = new_id[target[node]];

    boost_graph_t temp_dag;
    temp_dag = Contract(target);
    std::vector<bool> all_valid(temp_dag.num_vertices(), true);
    std::vector<vertex_idx> top_idx = GetFilteredTopOrderIdx(temp_dag, all_valid);

    for (vertex_idx node = 0; node < G_full.num_vertices(); ++node)
        target[node] = top_idx[target[node]];

    return target;
};

} // namespace osp