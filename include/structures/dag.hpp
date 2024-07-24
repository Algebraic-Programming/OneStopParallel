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
#include <random>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "auxiliary/auxiliary.hpp"
#include "structures/union_find.hpp"
#include "model/ComputationalDag.hpp"
#include "auxiliary/Balanced_Coin_Flips.hpp"

enum contract_edge_sort { Contract_Edge_Decrease, Contract_Edge_Weight };

/**
 * @brief Parameters used for coarsening
 * @param geom_decay_num_nodes Cap on node decrease in coarsening steps
 * @param poisson_par poisson parameter for extra minute shifts in poset map generation
 * @param noise extra noise shifts for sinks and sources in poset map generation
 * @param edge_sort_ration ratio Contract_Edge_Decrease : Contract_Edge_Weight
 * @param num_rep_without_node_decrease number of repetitions without change limit that causes termination of coarsening
 * 
 */
struct CoarsenParams {
    const float geom_decay_num_nodes;
    const double poisson_par;
    const unsigned noise;
    const std::pair<unsigned, unsigned> edge_sort_ratio;
    const int num_rep_without_node_decrease;
    const float temperature_multiplier;
    const float number_of_temperature_increases;

    CoarsenParams(
        const float geom_decay_num_nodes_ = 17.0/16.0,
        const double poisson_par_ = 0,
        const unsigned noise_ = 0,
        const std::pair<unsigned, unsigned> edge_sort_ratio_ = std::make_pair(3,2),
        const int num_rep_without_node_decrease_ = 4,
        const float temperature_multiplier_ = 1.125, // not more than 2
        const float number_of_temperature_increases_ = 14 // temperature_multiplier_ ^ number_of_temperature_increases_ should be at least 2 
    ) :
        geom_decay_num_nodes(geom_decay_num_nodes_),
        poisson_par(poisson_par_),
        noise(noise_),
        edge_sort_ratio(edge_sort_ratio_),
        num_rep_without_node_decrease(num_rep_without_node_decrease_),
        temperature_multiplier(temperature_multiplier_),
        number_of_temperature_increases(number_of_temperature_increases_)
        { }
};

struct Edge_Weighted {
    const std::pair<int, int> edge_pair;
    const int weight;

    Edge_Weighted( const std::pair<int, int> edge_pair_, const int weight_ ) : edge_pair(edge_pair_), weight(weight_) { }

    struct Comparator{
        constexpr bool operator()(const Edge_Weighted& a ,const Edge_Weighted& b) const {return (a.weight > b.weight); };
    };
};

// TODO add decent constructor, etc.
struct SubDAG;

struct DAG {
    unsigned int n;
    // in-neighbors and out-neighbors of each node
    std::vector<std::vector<int>> In, Out;

    // work and communication weight of each node
    std::vector<int> workW, commW;

    // memory weight of each node
    std::vector<int> memW;

    // communication weights of each edge
    std::unordered_map< std::pair<int, int> , int, pair_hash > comm_edge_W;

    DAG(const std::vector<std::vector<int>> &in_, const std::vector<std::vector<int>> &out_,
        const std::vector<int> &workW_, const std::vector<int> &commW_,
        const std::unordered_map< std::pair<int, int>, int, pair_hash >& comm_edge_W_ )
        : n(in_.size()), In(in_), Out(out_), workW(workW_), commW(commW_), comm_edge_W(comm_edge_W_) {
        assert(in_.size() == out_.size());
        assert(in_.size() == workW_.size());
        assert(in_.size() == commW_.size());
        for (unsigned i = 0; i<n; i++) {
            for(int j : Out[i]) {
                assert( comm_edge_W.find( std::make_pair(i,j) ) != comm_edge_W.cend() );
            }
        }
    }

    explicit DAG(const std::vector<std::vector<int>> &in_, const std::vector<std::vector<int>> &out_,
        const std::vector<int> &workW_, const std::vector<int> &commW_ )
        : n(in_.size()), In(in_), Out(out_), workW(workW_), commW(commW_) {
        assert(in_.size() == out_.size());
        assert(in_.size() == workW_.size());
        assert(in_.size() == commW_.size());
        for (unsigned i = 0; i<n; i++) {
            for(int j : Out[i]) {
                comm_edge_W[std::make_pair(i,j)] = 1;
            }
        }
        for (unsigned i = 0; i<n; i++) {
            for(int j : Out[i]) {
                assert( comm_edge_W.find( std::make_pair(i,j) ) != comm_edge_W.cend() );
            }
        }
    }

    explicit DAG(const ComputationalDag& cdag) : n(cdag.numberOfVertices()) {
        Resize(n);
        
        for (unsigned node = 0; node < n; ++node) {
            workW[node] = cdag.nodeWorkWeight(node);
            commW[node] = cdag.nodeCommunicationWeight(node);
            memW[node] = cdag.nodeMemoryWeight(node);
        }

        for(auto edge : cdag.edges()) {
            In[edge.m_target].push_back(edge.m_source);
            Out[edge.m_source].push_back(edge.m_target);
            comm_edge_W[ std::make_pair(edge.m_source, edge.m_target ) ] = cdag.edgeCommunicationWeight(edge);
        }

    }

    // explicit DAG(const DAG& other) :n(other.n), In(other.In), Out(other.Out), workW(other.workW),
    //                                 commW(other.commW), comm_edge_W(other.comm_edge_W) { };

    // explicit DAG(DAG& other) :n(other.n), In(other.In), Out(other.Out), workW(other.workW),
    //                                 commW(other.commW), comm_edge_W(other.comm_edge_W) { };

    // DAG& operator=(DAG &other) {
    //     n = other.n;
    //     In = other.In;
    //     Out = other.Out;
    //     workW = other.workW;
    //     commW = other.commW;
    //     comm_edge_W = other.comm_edge_W;
    //     return *this;
    // };

    // DAG& operator=(const DAG &other) {
    //     n = other.n;
    //     In = other.In;
    //     Out = other.Out;
    //     workW = other.workW;
    //     commW = other.commW;
    //     comm_edge_W = other.comm_edge_W;
    //     return *this;
    // };

    explicit DAG() : DAG({}, {}, {}, {}, {}) {}

    virtual ~DAG() = default;

    void Resize(int N);

    void addEdge(int v1, int v2, int comm_edge_weight = 1, bool noPrint = false);  // depreciated

    // checks if graph is acyclic
    bool is_acyclic() const;

    // compute topological ordering
    std::vector<int> GetTopOrder() const;
    std::vector<int> GetTopOrderIdx(const std::vector<bool> &filter = std::vector<bool>()) const;
    // get topological order for a subset of (valid) nodes
    std::vector<int> GetFilteredTopOrder(const std::vector<bool> &valid) const;

    // read DAG from file in hyperDAG format
    bool read(std::ifstream &infile);
    bool read(const std::string &filename);

    // write DAG to file in hyperDAG format
    void write(std::ofstream &outfile) const;

    // ensure that the list of in- and out-neighbors for nodes follows a topological ordering
    void ReOrderEdgeLists();

    // calculates number of edges in the longest path
    // returns 0 if no vertices
    int getLongestPath(const std::set<int> &nodes) const;

    // longest chain of nodes (measured by number of nodes in chain)
    // Returns list of nodes in longest chain
    virtual std::vector<int> longest_chain() const;

    // get ancestors of a node (including itself)
    virtual std::unordered_set<int> ancestors(int node) const;

    // get descendants of a node (including itself)
    virtual std::unordered_set<int> descendants(int node) const;

    // create SubDag from DAG (of itself)
    virtual SubDAG toSubDAG() const;

    // work weight of collection of nodes
    virtual int workW_of_node_set(const std::unordered_set<int> &node_set) const;

    // computes the node sets of weakly connected components
    virtual std::vector<std::unordered_set<int>> weakly_connected_components() const;

    // computes the node sets of weakly connected components of subset of nodes
    virtual std::vector<std::unordered_set<int>> weakly_connected_components(const std::unordered_set<int>& node_set) const;

    // computes bottom node distance
    std::vector<int> get_bottom_node_distance() const;

    // computes top node distance
    std::vector<int> get_top_node_distance() const;

    // Generates f:V \to Z such that (v,w) \in E \Rightarrow f(v) < f(w)
    // TODO: possibly make it less random? by either biasing a coinflip or make high chance to repeat the last
    std::vector<int> get_strict_poset_integer_map(unsigned const noise = 0, double const poisson_param = 0 ) const;

    // Generates a list of edges, that individually can be contracted and the contracted graph remains acyclic
    std::multiset<Edge_Weighted, Edge_Weighted::Comparator> get_contractable_edges(const contract_edge_sort edge_sort_alg, const std::vector<int>& poset_int_mapping ) const;

    int count_common_parents_plus_common_children(int vert_1, int vert_2) const;

    /**
     * @brief Takes in a partition and returns the quotient graph with self-loops deleted together with the contraction map
     * 
     * @param partition A partition of the vertex/node set - singletons can be left out.
     */
    std::pair<DAG, std::unordered_map<int, int >> contracted_graph_without_loops(const std::vector<std::unordered_set<int>>& partition ) const;

    /**
     * @brief Takes in a partition and returns the quotient graph with self-loops deleted together with the contraction map
     * 
     * @param partition A partition of the vertex/node set - singletons can be left out.
     */
    std::pair<DAG, std::unordered_map<int, int >> contracted_graph_without_loops(const std::vector<std::vector<int>>& partition ) const;
 
    //conversion to/from new DAG format
    ComputationalDag ConvertToNewDAG() const;

    // void ConvertFromNewDAG(const ComputationalDag& cdag); use constructor instead
};

struct SubDAG : DAG {
    // pointer full graph
    const DAG *dagptr;

    // translation from super-graph nodes to subgraph nodes
    std::unordered_map<int, int> super_to_sub;

    // translation from sub-graph nodes to supergraph nodes
    std::unordered_map<int, int> sub_to_super;

    // Constructing induced SubDAG from DAG
    explicit SubDAG(const DAG &super_graph, const std::unordered_set<int> &node_set);
    explicit SubDAG(const DAG &super_graph);

    // Constructing induced SubDAG from SubDAG with node references from original DAG(!)
    explicit SubDAG(const SubDAG &super_graph, const std::unordered_set<int> &node_set);

    // longest chain of nodes (measured by number of nodes in chain)
    // Returns list of nodes in longest chain, with node names in the super-graph
    std::vector<int> longest_chain() const override;

    // get ancestors of a node (including itself), with node names in the super-graph
    // takes in node from super-sgraph
    std::unordered_set<int> ancestors(int node) const override;

    // get descendants of a node (including itself), with node names in the super-graph
    // takes in node from super-graph
    std::unordered_set<int> descendants(int node) const override;

    SubDAG toSubDAG() const override { return *this; };

    // work weight of collection of nodes with node_set from original DAG
    int workW_of_node_set(const std::unordered_set<int> &node_set) const override;

    // computes the node (with node names of super dag) sets of weakly connected components
    std::vector<std::unordered_set<int>> weakly_connected_components() const override;

    // computes the node (with node names of super dag) sets of weakly connected components of subset of nodes (with node names in super dag)
    std::vector<std::unordered_set<int>> weakly_connected_components(const std::unordered_set<int>& node_set) const override;

    SubDAG& operator=(const SubDAG &other) {
        if (this == &other) return *this;

        dagptr = other.dagptr;
        n = other.n;
        In = other.In;
        Out = other.Out;
        workW = other.workW;
        commW = other.commW;
        comm_edge_W = other.comm_edge_W;
        super_to_sub = other.super_to_sub;
        sub_to_super = other.sub_to_super;
        return *this;
    }
};
