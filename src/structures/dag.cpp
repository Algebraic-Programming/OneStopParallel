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

#include "structures/dag.hpp"
#include "model/ComputationalDag.hpp"

#include <algorithm>
#include <iostream>
#include <list>

void DAG::Resize(const int N) {
    n = N;
    In.clear();
    In.resize(n);
    Out.clear();
    Out.resize(n);
    workW.clear();
    workW.resize(n);
    commW.clear();
    commW.resize(n);
    comm_edge_W.clear();
};

void DAG::addEdge(const int v1, const int v2, int comm_edge_weight, bool noPrint) {
    if (v1 >= v2)
        // std::cout << "DAG edge addition error. (insert everything is fine meme)" << std::endl;

    if (v2 >= n)
        std::cout << "Error: node index out of range." << std::endl;

    In[v2].push_back(v1);
    Out[v1].push_back(v2);
    comm_edge_W[std::make_pair(v1, v2)] = comm_edge_weight;
};

std::vector<int> DAG::GetTopOrder() const {
    std::vector<int> predecessors(n, 0);
    std::deque<int> next;

    std::vector<int> TopOrder;

    // Find source nodes
    for (unsigned i = 0; i < n; ++i)
        if (In[i].empty())
            next.push_back(i);

    // Execute BFS
    while (!next.empty()) {
        int node = next.front();
        next.pop_front();
        TopOrder.push_back(node);

        for (size_t i = 0; i < Out[node].size(); ++i) {
            int current = Out[node][i];
            ++predecessors[current];
            if (predecessors[current] == In[current].size())
                next.push_back(current);
        }
    }

    if (TopOrder.size() != n)
        std::cout << "Error during topological ordering!" << std::endl;

    return TopOrder;
};

std::vector<int> DAG::GetTopOrderIdx(const std::vector<bool> &valid) const {
    std::vector<int> topOrder = valid.empty() ? GetTopOrder() : GetFilteredTopOrder(valid);
    std::vector<int> topOrderIdx(n);
    for (int i = 0; i < topOrder.size(); ++i)
        topOrderIdx[topOrder[i]] = i;

    return topOrderIdx;
}

bool DAG::read(std::ifstream &infile) {
    std::string line;
    getline(infile, line);
    while (!infile.eof() && line.at(0) == '%')
        getline(infile, line);

    int hEdges, pins, N;
    sscanf(line.c_str(), "%d %d %d", &hEdges, &N, &pins);

    if (N <= 0 || hEdges <= 0 || pins <= 0) {
        std::cout << "Incorrect input file format (number of nodes/hyperedges/pins "
                     "is not positive).\n";
        return false;
    }

    Resize(N);
    std::vector<int> edgeSource(hEdges, -1);
    // read edges
    for (int i = 0; i < pins; ++i) {
        if (infile.eof()) {
            std::cout << "Incorrect input file format (file terminated too early).\n";
            return false;
        }
        getline(infile, line);
        while (!infile.eof() && line.at(0) == '%')
            getline(infile, line);

        int hEdge, node;
        sscanf(line.c_str(), "%d %d", &hEdge, &node);

        if (hEdge < 0 || node < 0 || hEdge >= hEdges || node >= N) {
            std::cout << "Incorrect input file format (index out of range).\n";
            return false;
        }

        if (edgeSource[hEdge] == -1)
            edgeSource[hEdge] = node;
        else
            addEdge(edgeSource[hEdge], node);
    }

    ReOrderEdgeLists();

    for (int i = 0; i < N; ++i) {
        if (infile.eof()) {
            std::cout << "Incorrect input file format (file terminated too early).\n";
            return false;
        }

        getline(infile, line);
        while (!infile.eof() && line.at(0) == '%')
            getline(infile, line);

        int node, work, comm;
        sscanf(line.c_str(), "%d %d %d", &node, &work, &comm);

        if (node < 0 || work < 0 || comm < 0 || node >= N) {
            std::cout << "Incorrect input file format (index out of range, our "
                         "weight below 0).\n";
            return false;
        }

        workW[node] = work;
        commW[node] = comm;
    }

    return true;
}

bool DAG::read(const std::string &filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cout << "Unable to find/open input schedule file.\n";
        return false;
    }

    read(infile);
    infile.close();
    return true;
}

void DAG::write(std::ofstream &outfile) const {
    int sinks = 0, pins = 0;
    for (unsigned i = 0; i < n; ++i)
        if (!Out[i].empty())
            pins += 1 + Out[i].size();
        else
            ++sinks;

    outfile << n - sinks << " " << n << " " << pins << "\n";

    int edgeIndex = 0;
    for (unsigned i = 0; i < n; ++i)
        if (!Out[i].empty()) {
            outfile << edgeIndex << " " << i << "\n";
            for (const int j : Out[i])
                outfile << edgeIndex << " " << j << "\n";

            ++edgeIndex;
        }

    for (unsigned i = 0; i < n; ++i)
        outfile << i << " " << workW[i] << " " << commW[i] << "\n";
};

void DAG::ReOrderEdgeLists() {
    const std::vector<int> topOrder = GetTopOrder();
    std::vector<std::vector<int>> newIn, newOut;
    newIn.resize(n);
    newOut.resize(n);
    for (unsigned i = 0; i < n; ++i) {
        int node = topOrder[i];
        for (const int j : In[node])
            newOut[j].push_back(node);
        for (const int j : Out[node])
            newIn[j].push_back(node);
    }
    In = newIn;
    Out = newOut;
}

std::vector<int> DAG::GetFilteredTopOrder(const std::vector<bool> &valid) const {
    const std::vector<int> TopOrder = GetTopOrder();
    std::vector<int> filteredOrder;
    for (int node : TopOrder)
        if (valid[node])
            filteredOrder.push_back(node);

    return filteredOrder;
}

int DAG::getLongestPath(const std::set<int> &nodes) const {
    std::list<int> Q;
    std::map<int, int> dist, inDegree, visited;

    // Find source nodes
    for (int node : nodes) {
        int indeg = 0;
        for (int pred : In[node])
            if (nodes.count(pred) == 1)
                ++indeg;

        if (indeg == 0) {
            Q.push_back(node);
            dist[node] = 0;
        }
        inDegree[node] = indeg;
        visited[node] = 0;
    }

    // Execute BFS
    while (!Q.empty()) {
        int node = Q.front();
        Q.pop_front();

        for (int succ : Out[node]) {
            if (nodes.count(succ) == 0)
                continue;

            ++visited[succ];
            if (visited[succ] == inDegree[succ]) {
                Q.push_back(succ);
                dist[succ] = dist[node] + 1;
            }
        }
    }

    int mx = 0;
    for (int node : nodes)
        mx = std::max(mx, dist[node]);

    return mx;
};

// longest chain of nodes (measured by number of nodes in chain)
// Returns list of nodes in longest chain
std::vector<int> DAG::longest_chain() const {
    std::vector<int> top_length(n, 0);
    std::vector<int> chain;
    int running_longest_chain = -1;
    int end_longest_chain = -1;

    // calculating lenght of longest path
    const std::vector<int> top_order = GetTopOrder();
    for (const auto &node : top_order) {
        int max_temp = 0;
        for (const int &in_node : In[node]) {
            max_temp = std::max(max_temp, top_length[in_node]);
        }
        top_length[node] = ++max_temp;
        if (top_length[node] > running_longest_chain) {
            end_longest_chain = node;
            running_longest_chain = top_length[node];
        };
    }

    // no nodes
    if (running_longest_chain == -1)
        return chain;

    // reconstructing longest path
    chain.push_back(end_longest_chain);
    while (!In[end_longest_chain].empty()) {
        for (const int in_node : In[end_longest_chain]) {
            if (top_length[in_node] == top_length[end_longest_chain] - 1) {
                end_longest_chain = in_node;
                chain.push_back(end_longest_chain);
                break;
            };
        };
    };

    std::reverse(chain.begin(), chain.end());
    return chain;
};

// get ancestors of a node (including itself)
std::unordered_set<int> DAG::ancestors(const int node) const {
    std::unordered_set<int> active = {node};
    std::vector<int> new_active;
    std::unordered_set<int> anc = active;
    while (!active.empty()) {
        for (auto &v : active) {
            for (auto &w : In[v]) {
                if (anc.find(w) == anc.end()) {
                    new_active.emplace_back(w);
                };
            };
        };
        active = std::unordered_set<int>{new_active.begin(), new_active.end()};
        anc.insert(active.begin(), active.end());
        new_active.clear();
    };
    return anc;
};

// get descendants of a node (including itself)
std::unordered_set<int> DAG::descendants(int node) const {
    std::unordered_set<int> active = {node};
    std::vector<int> new_active;
    std::unordered_set<int> desc = active;
    while (!active.empty()) {
        for (auto &v : active) {
            for (auto &w : Out[v]) {
                if (desc.find(w) == desc.end()) {
                    new_active.emplace_back(w);
                };
            };
        };
        active = std::unordered_set<int>{new_active.begin(), new_active.end()};
        desc.insert(active.begin(), active.end());
        new_active.clear();
    };
    return desc;
};

// create SubDag from DAG (of itself)
SubDAG DAG::toSubDAG() const {
    std::unordered_set<int> node_set;
    node_set.reserve(n);
    for (unsigned i = 0; i < n; i++) {
        node_set.emplace(i);
    }

    SubDAG G = SubDAG(*this, node_set);

    for (int i = 0; i<G.n; i++) {
        for(int j : G.Out[i]) {
            assert( G.comm_edge_W.find( std::make_pair(i,j) ) != G.comm_edge_W.cend() );
        }
    }

    return G;
}

// Constructing induced SubDAG from DAG
SubDAG::SubDAG(const DAG &super_graph, const std::unordered_set<int> &node_set) : dagptr(&super_graph) {
    n = node_set.size();

    In.resize(n);
    Out.resize(n);
    workW.resize(n);
    commW.resize(n);
    super_to_sub.reserve(n);
    sub_to_super.reserve(n);

    // creating translation
    int i = 0;
    for (auto &node : node_set) {
        super_to_sub[node] = i;
        sub_to_super[i] = node;

        workW[i] = super_graph.workW[node];
        commW[i] = super_graph.commW[node];

        i++;
    }

    // in and out edge lists
    for (unsigned ind = 0; ind < n; ind++) {
        for (auto &in_node : super_graph.In[sub_to_super[ind]]) {
            if (node_set.find(in_node) != node_set.cend()) {
                In[ind].push_back(super_to_sub[in_node]);
            }
        }
        for (auto &out_node : super_graph.Out[sub_to_super[ind]]) {
            if (node_set.find(out_node) != node_set.cend()) {
                Out[ind].push_back(super_to_sub[out_node]);
            }
        }
    }

    // comm edge weights
    for (unsigned ind = 0; ind < n; ind++) {
        for (int out_node : super_graph.Out[sub_to_super[ind]]) {
            if (node_set.find(out_node) != node_set.cend()) {
                assert( super_graph.comm_edge_W.find(std::make_pair(sub_to_super[ind], out_node)) != super_graph.comm_edge_W.cend() );
                comm_edge_W[ std::make_pair(ind, super_to_sub[out_node] ) ] = super_graph.comm_edge_W.at(std::make_pair(sub_to_super[ind], out_node));
            }
        }
    }

    // comm edge weight tests
    for (unsigned ind = 0; ind<n; ind++) {
        for(int j : Out[ind]) {
            assert( comm_edge_W.find( std::make_pair(ind,j) ) != comm_edge_W.cend() );
        }
    }
};

// Constructing induced SubDAG from DAG
SubDAG::SubDAG(const DAG &super_graph) : dagptr(&super_graph) {
    n = super_graph.n;

    In = super_graph.In;
    Out = super_graph.Out;
    workW = super_graph.workW;
    commW = super_graph.commW;
    comm_edge_W = super_graph.comm_edge_W;

    super_to_sub.reserve(n);
    sub_to_super.reserve(n);

    // creating translation
    for (unsigned i = 0; i< n; i++) {
        super_to_sub[i] = i;
        sub_to_super[i] = i;
    }

    // comm edge weight tests
    for (unsigned i = 0; i<n; i++) {
        for(int j : Out[i]) {
            assert( comm_edge_W.find( std::make_pair(i,j) ) != comm_edge_W.cend() );
        }
    }
};

// Constructing induced SubDAG from SubDAG with node references from original DAG(!)
SubDAG::SubDAG(const SubDAG &middle_graph, const std::unordered_set<int> &node_set) : dagptr(middle_graph.dagptr) {
    n = node_set.size();

    In.resize(n);
    Out.resize(n);
    workW.resize(n);
    commW.resize(n);
    super_to_sub.reserve(n);
    sub_to_super.reserve(n);

    // creating translation
    int i = 0;
    for (auto &node : node_set) {
        super_to_sub[node] = i;
        sub_to_super[i] = node;

        workW[i] = middle_graph.dagptr->workW[node];
        commW[i] = middle_graph.dagptr->commW[node];

        i++;
    }

    // in and out edge lists
    for (unsigned ind = 0; ind < n; ind++) {
        const int middle_dag_node = middle_graph.super_to_sub.at(sub_to_super[ind]);

        for (auto &middle_in_node : middle_graph.In[middle_dag_node]) {
            int dag_in_node = middle_graph.sub_to_super.at(middle_in_node);
            if (node_set.find(dag_in_node) != node_set.cend()) {
                In[ind].push_back(super_to_sub[dag_in_node]);
            }
        }

        for (auto &middle_out_node : middle_graph.Out[middle_dag_node]) {
            int dag_out_node = middle_graph.sub_to_super.at(middle_out_node);
            
            if (node_set.find(dag_out_node) != node_set.cend()) {
                Out[ind].push_back(super_to_sub[dag_out_node]);
            }
        }
    }

    // comm edge weights
    for (unsigned ind = 0; ind < n; ind++) {
        const int middle_dag_node = middle_graph.super_to_sub.at(sub_to_super[ind]);
        for (int middle_out_node : middle_graph.Out[middle_dag_node]) {
            int dag_out_node = middle_graph.sub_to_super.at(middle_out_node);
            
            if (node_set.find(dag_out_node) != node_set.cend()) {
                // assert( dagptr->comm_edge_W.find( std::make_pair(  sub_to_super[ind] , dag_out_node )) != dagptr->comm_edge_W.cend() );
                // comm_edge_W[ std::make_pair(ind, super_to_sub[dag_out_node] ) ] = middle_graph.comm_edge_W.at( std::make_pair( sub_to_super[ind], dag_out_node ) );
                assert( middle_graph.comm_edge_W.find( std::make_pair( middle_dag_node, middle_out_node )) != middle_graph.comm_edge_W.cend() );
                comm_edge_W[ std::make_pair(ind, super_to_sub[dag_out_node] ) ] = middle_graph.comm_edge_W.at( std::make_pair( middle_dag_node, middle_out_node ) );
            }
        }
    }

    // comm weight tests
    for (unsigned ind = 0; ind<n; ind++) {
        for(int j : Out[ind]) {
            assert( comm_edge_W.find( std::make_pair(ind,j) ) != comm_edge_W.cend() );
        }
    }
};

// longest chain of nodes (measured by number of nodes in chain)
// Returns list of nodes in longest chain, with node names in the super-graph
std::vector<int> SubDAG::longest_chain() const {
    const std::vector<int> sub_chain = DAG::longest_chain();
    std::vector<int> super_chain;

    super_chain.resize(sub_chain.size());

    for (size_t i = 0; i < sub_chain.size(); i++) {
        super_chain[i] = sub_to_super.at(sub_chain[i]);
    }

    return super_chain;
}

// get ancestors of a node (including itself), with node names in the super-graph
// takes in node from super-graph
std::unordered_set<int> SubDAG::ancestors(int node) const {
    const std::unordered_set<int> sub_anc = DAG::ancestors(super_to_sub.at(node));
    std::unordered_set<int> super_anc;

    super_anc.reserve(sub_anc.size());
    for (auto &i : sub_anc) {
        super_anc.emplace(sub_to_super.at(i));
    }

    return super_anc;
}

// get descendants of a node (including itself), with node names in the super-graph
// takes in node from super-graph
std::unordered_set<int> SubDAG::descendants(int node) const {
    const std::unordered_set<int> sub_desc = DAG::descendants(super_to_sub.at(node));
    std::unordered_set<int> super_desc;

    super_desc.reserve(sub_desc.size());
    for (auto &i : sub_desc) {
        super_desc.emplace(sub_to_super.at(i));
    }

    return super_desc;
}

// work weight of collection of nodes
int DAG::workW_of_node_set(const std::unordered_set<int> &node_set) const {
    int output = 0;
    for (auto &node : node_set) {
        output += workW[node];
    }

    return output;
}

// work weight of collection of nodes with node_set from original DAG
int SubDAG::workW_of_node_set(const std::unordered_set<int> &node_set) const {
    int output = 0;
    for (auto &node : node_set) {
        output += dagptr->workW[node];
    }

    return output;
}

// computes the node sets of weakly connected components
std::vector<std::unordered_set<int>> DAG::weakly_connected_components() const {
    if (n == 0)
        return std::vector<std::unordered_set<int>>({{}});
    std::vector<std::unordered_set<int>> components;

    Union_Find_Universe<int> uf;

    for (unsigned i = 0; i < n; i++) {
        uf.add_object(i);
    }

    for (unsigned i = 0; i < n; i++) {
        for (auto &j : In[i]) {
            uf.join_by_name(i, j);
        }
    }

    std::vector<std::vector<int>> comp_vectors = uf.get_connected_components();
    components.resize(comp_vectors.size());
    for (int i = 0; i < comp_vectors.size(); i++) {
        components[i].reserve(comp_vectors[i].size());
        for (auto &elem : comp_vectors[i]) {
            components[i].emplace(elem);
        }
    }

    return components;
}

std::vector<std::unordered_set<int>> SubDAG::weakly_connected_components() const {
    const std::vector<std::unordered_set<int>> components = DAG::weakly_connected_components();
    std::vector<std::unordered_set<int>> output;
    output.resize(components.size());

    for (size_t i = 0; i < components.size(); i++) {
        output[i].reserve(components[i].size());
        for (int j : components[i]) {
            output[i].emplace(sub_to_super.at(j));
        }
    }

    return output;
}

// computes bottom node distance
std::vector<int> DAG::get_bottom_node_distance() const {
    std::vector<int> bottom_distance(n, 0);
    std::vector<int> top_order = GetTopOrder();
    for (int i = top_order.size() - 1; i >= 0; i--) {
        int max_temp = 0;
        for (auto &j : Out[top_order[i]]) {
            max_temp = std::max(max_temp, bottom_distance[j]);
        }
        bottom_distance[top_order[i]] = ++max_temp;
    }
    return bottom_distance;
}

// computes top node distance
std::vector<int> DAG::get_top_node_distance() const {
    std::vector<int> top_distance(n, 0);
    std::vector<int> top_order = GetTopOrder();
    for (int i = 0; i < top_order.size(); i++) {
        int max_temp = 0;
        for (auto &j : In[top_order[i]]) {
            max_temp = std::max(max_temp, top_distance[j]);
        }
        top_distance[top_order[i]] = ++max_temp;
    }
    return top_distance;
}


// computes the node sets of weakly connected components of subset of nodes
std::vector<std::unordered_set<int>> DAG::weakly_connected_components(const std::unordered_set<int>& node_set) const {
    if (node_set.size() == 0)
        return std::vector<std::unordered_set<int>>({{}});
    std::vector<std::unordered_set<int>> components;

    Union_Find_Universe<int> uf;

    for (auto& node : node_set) {
        uf.add_object(node);
    }

    for (auto& node : node_set) {
        for (auto& in_node : In[node]) {
            if ( node_set.find(in_node) != node_set.cend() ) {
                uf.join_by_name(node, in_node);
            }
        }
    }

    std::vector<std::vector<int>> comp_vectors = uf.get_connected_components();
    components.resize(comp_vectors.size());
    for (int i = 0; i < comp_vectors.size(); i++) {
        components[i].reserve(comp_vectors[i].size());
        for (auto &elem : comp_vectors[i]) {
            components[i].emplace(elem);
        }
    }

    return components;
}


// computes the node (with node names of super dag) sets of weakly connected components of subset of nodes (with node names in super dag)
std::vector<std::unordered_set<int>> SubDAG::weakly_connected_components(const std::unordered_set<int>& node_set) const {
    std::unordered_set<int> sub_node_set;
    sub_node_set.reserve(node_set.size());
    for (auto& super_node: node_set) {
        sub_node_set.emplace(super_to_sub.at(super_node));
    }

    const std::vector<std::unordered_set<int>> components = DAG::weakly_connected_components(sub_node_set);
    
    std::vector<std::unordered_set<int>> output;
    output.resize(components.size());

    for (size_t i = 0; i < components.size(); i++) {
        output[i].reserve(components[i].size());
        for (int j : components[i]) {
            output[i].emplace(sub_to_super.at(j));
        }
    }

    return output;
}
ComputationalDag DAG::ConvertToNewDAG() const
{
    return ComputationalDag(Out, workW, commW, comm_edge_W);
};

// void DAG::ConvertFromNewDAG(const ComputationalDag& cdag)
// {
//     n = cdag.numberOfVertices();
//     Resize(n);
//     for (int node = 0; node < n; ++node)
//     {
//         workW[node] = cdag.nodeWorkWeight(node);
//         commW[node] = cdag.nodeWorkWeight(node);
//     }
//     auto edges = cdag.edges();
//     for(auto edge : edges)
//         addEdge(edge.m_source, edge.m_target);
// };

std::vector<int> DAG::get_strict_poset_integer_map( unsigned const noise, double const poisson_param) const
{
    std::vector<int> top_order = GetTopOrder();

    Repeat_Chance repeater_coin;

    std::unordered_map< std::pair<int, int>, bool, pair_hash>  up_or_down;

    for (auto& i : top_order) {
        for (auto j : Out[i]) {
            bool val = repeater_coin.get_flip();
            up_or_down.emplace( std::make_pair(i,j) , val );
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::poisson_distribution<> poisson_gen( poisson_param );
    

    std::vector<int> top = get_top_node_distance();
    std::vector<int> bot = get_bottom_node_distance();
    std::vector<int> new_top(n);
    std::vector<int> new_bot(n);

    int max_path = INT_MIN;
    for (unsigned i = 0; i<n ; i++) {
        max_path = std::max(max_path, top[i]);
    }

    std::vector<int> sources;
    std::vector<int> sinks;

    for (unsigned i = 0; i<n; i++) {
        if (In[i].size() == 0)
            sources.emplace_back(i);
        if (Out[i].size() == 0)
            sinks.emplace_back(i);
    }

    for (auto& source: sources) {
        new_top[source] = randInt(max_path-bot[source] + 1+2*noise)-noise;
    }
    for (auto& sink: sinks) {
        new_bot[sink] = randInt(max_path-top[sink] + 1+2*noise)-noise;
    }

    for (int i = 0; i< top_order.size(); i++) {
        if (In[top_order[i]].empty()) continue;
        int max_temp = INT_MIN;
        for (auto &j : In[top_order[i]]) {
            int temp = new_top[j];
            std::pair<int, int> edge = std::make_pair(j, top_order[i]);
            if (up_or_down.at( edge )) {
                temp += 1 + poisson_gen(gen);
            }
            max_temp = std::max(max_temp, temp);
        }
        new_top[top_order[i]] = max_temp;
    }
    for (int i = top_order.size()-1 ; i>= 0; i--) {
        if (Out[top_order[i]].empty()) continue;
        int max_temp = INT_MIN;
        for (auto &j : Out[top_order[i]]) {
            int temp = new_bot[j];
            std::pair<int, int> edge = std::make_pair(top_order[i], j);
            if (! up_or_down.at( edge )) {
                temp += 1 + poisson_gen(gen);
            }
            max_temp = std::max(max_temp, temp);
        }
        new_bot[top_order[i]] = max_temp;
    }

    std::vector<int> output;
    output.reserve(n);
    for (unsigned i = 0 ; i < n; i++) {
        output.emplace_back( new_top[i] - new_bot[i] );
    }
    return output;
}

std::multiset<Edge_Weighted, Edge_Weighted::Comparator> DAG::get_contractable_edges(const contract_edge_sort edge_sort_alg, const std::vector<int>& poset_int_mapping ) const
{
    std::multiset<Edge_Weighted, Edge_Weighted::Comparator> contractable_edge_list;

    // const std::vector<int> poset_int_mapping = get_strict_poset_integer_map( coarsen_para.noise, coarsen_para.poisson_par );

    for (int i = 0; i<n; i++) {
        for (auto& j : Out[i]) {
            if (poset_int_mapping[j] == poset_int_mapping[i] + 1 ) {
                int edge_weight;
                if (edge_sort_alg == Contract_Edge_Decrease) {
                    edge_weight = count_common_parents_plus_common_children(i,j);
                }
                else if ( edge_sort_alg == Contract_Edge_Weight ) {
                    edge_weight = comm_edge_W.at( std::make_pair(i,j) );
                }
                contractable_edge_list.emplace( std::make_pair(i,j) , edge_weight );
            }
        }
    }

    return contractable_edge_list;
}

int DAG::count_common_parents_plus_common_children(int vert_1, int vert_2) const
{
    int out = 0;

    // Common Parents
    std::set<int> par_1( In[vert_1].cbegin(), In[vert_1].cend() );
    std::set<int> par_2( In[vert_2].cbegin(), In[vert_2].cend() );

    auto par_it_1 = par_1.begin();
    auto par_it_2 = par_2.begin();

    while( par_it_1 != par_1.cend() && par_it_2 != par_2.cend() ) {
        if ( *par_it_1 == *par_it_2 )
        {
            out++;
            std::advance(par_it_1, 1);
            std::advance(par_it_2, 1);
        }
        else if ( *par_it_1 < *par_it_2 )
            std::advance(par_it_1, 1);
        else
            std::advance(par_it_2,1);
    }

    // Common Children
    std::set<int> chld_1( Out[vert_1].cbegin(), Out[vert_1].cend() );
    std::set<int> chld_2( Out[vert_2].cbegin(), Out[vert_2].cend() );

    auto chld_it_1 = chld_1.begin();
    auto chld_it_2 = chld_2.begin();

    while( chld_it_1 != chld_1.cend() && chld_it_2 != chld_2.cend() ) {
        if ( *chld_it_1 == *chld_it_2 )
        {
            out++;
            std::advance(chld_it_1, 1);
            std::advance(chld_it_2, 1);
        }
        else if ( *chld_it_1 < *chld_it_2 )
            std::advance(chld_it_1, 1);
        else
            std::advance(chld_it_2,1);
    }

    return out;
}

std::pair<DAG, std::unordered_map<int, int >> DAG::contracted_graph_without_loops( const std::vector<std::unordered_set<int>>& partition ) const
{
    DAG new_graph;
    std::unordered_map<int, int > contraction_map;
    contraction_map.reserve(n);

    std::vector<bool> allocated_into_new_graph(n, false);

    int new_node = 0;
    for (auto& part : partition) {
        for (auto& node : part) {
            contraction_map[node] = new_node;
            allocated_into_new_graph[node] = true;
        }
        new_node++;
    }
    for (unsigned i = 0; i<n; i++) {
        if (allocated_into_new_graph[i]) continue;

        contraction_map[i] = new_node;
        allocated_into_new_graph[i] = true;

        new_node++;
    }

    assert( std::all_of(allocated_into_new_graph.begin(), allocated_into_new_graph.end(), [](const bool& has_been){ return has_been; } ) );

    // Making new graph
    new_graph.n = new_node;
    new_graph.In.resize(new_graph.n);
    new_graph.Out.resize(new_graph.n);
    new_graph.workW.resize(new_graph.n,0);
    new_graph.commW.resize(new_graph.n,0);

    std::vector<std::set<int>> in_set(new_graph.n);
    std::vector<std::set<int>> out_set(new_graph.n);
    for (unsigned i = 0; i<n; i++) {
        // In edges
        for (auto& j : In[i]) {
            if ( contraction_map.at(i) == contraction_map.at(j) ) continue;
            in_set[contraction_map.at(i)].emplace(contraction_map.at(j));
        }

        // Out edges
        for (auto& j : Out[i]) {
            if ( contraction_map.at(i) == contraction_map.at(j) ) continue;
            out_set[contraction_map.at(i)].emplace(contraction_map.at(j));
        }

        // Work weights
        new_graph.workW[contraction_map.at(i)] += workW[i];
        
        // Comm node weights
        new_graph.commW[contraction_map.at(i)] += commW[i];

        // Comm edge weights
        for (int j : Out[i]) {
            if ( contraction_map.at(i) == contraction_map.at(j) ) continue;
            new_graph.comm_edge_W[ std::make_pair(contraction_map.at(i),contraction_map.at(j)) ] = 0;
        }
        for (int j : Out[i]) {
            if ( contraction_map.at(i) == contraction_map.at(j) ) continue;
            new_graph.comm_edge_W[ std::make_pair(contraction_map.at(i),contraction_map.at(j)) ] +=  comm_edge_W.at( std::make_pair(i,j) );
        }
    }
    for (unsigned i = 0; i<new_graph.n; i++) {
        // In edges continued
        for (auto& j : in_set[i]) {
            new_graph.In[i].emplace_back(j);
        }
        // Out edges continued
        for (auto& j : out_set[i]) {
            new_graph.Out[i].emplace_back(j);
        }
    }

    for (unsigned i = 0; i<new_graph.n; i++) {
        for(int j : new_graph.Out[i]) {
            assert( new_graph.comm_edge_W.find( std::make_pair(i,j) ) != new_graph.comm_edge_W.cend() );
        }
    }

    return std::make_pair(new_graph, contraction_map);
}


std::pair<DAG, std::unordered_map<int, int >> DAG::contracted_graph_without_loops( const std::vector<std::vector<int>>& partition ) const
{
    DAG new_graph;
    std::unordered_map<int, int > contraction_map;
    contraction_map.reserve(n);

    std::vector<bool> allocated_into_new_graph(n, false);

    int new_node = 0;
    for (auto& part : partition) {
        for (auto& node : part) {
            contraction_map[node] = new_node;
            allocated_into_new_graph[node] = true;
        }
        new_node++;
    }
    for (unsigned i = 0; i<n; i++) {
        if (allocated_into_new_graph[i]) continue;

        contraction_map[i] = new_node;
        allocated_into_new_graph[i] = true;

        new_node++;
    }

    assert( std::all_of(allocated_into_new_graph.begin(), allocated_into_new_graph.end(), [](const bool& has_been){ return has_been; } ) );

    // Making new graph
    new_graph.n = new_node;
    new_graph.In.resize(new_graph.n);
    new_graph.Out.resize(new_graph.n);
    new_graph.workW.resize(new_graph.n,0);
    new_graph.commW.resize(new_graph.n,0);

    std::vector<std::set<int>> in_set(new_graph.n);
    std::vector<std::set<int>> out_set(new_graph.n);
    for (unsigned i = 0; i<n; i++) {
        // In edges
        for (auto& j : In[i]) {
            if ( contraction_map.at(i) == contraction_map.at(j) ) continue;
            in_set[contraction_map.at(i)].emplace(contraction_map.at(j));
        }

        // Out edges
        for (auto& j : Out[i]) {
            if ( contraction_map.at(i) == contraction_map.at(j) ) continue;
            out_set[contraction_map.at(i)].emplace(contraction_map.at(j));
        }

        // Work weights
        new_graph.workW[contraction_map.at(i)] += workW[i];
        
        // Comm node weights
        new_graph.commW[contraction_map.at(i)] += commW[i];

        // Comm edge weights
        for (int j : Out[i]) {
            if ( contraction_map.at(i) == contraction_map.at(j) ) continue;
            new_graph.comm_edge_W[ std::make_pair(contraction_map.at(i),contraction_map.at(j)) ] = 0;
        }
        for (int j : Out[i]) {
            if ( contraction_map.at(i) == contraction_map.at(j) ) continue;
            new_graph.comm_edge_W[ std::make_pair(contraction_map.at(i),contraction_map.at(j)) ] +=  comm_edge_W.at( std::make_pair(i,j) );
        }
    }
    for (unsigned i = 0; i<new_graph.n; i++) {
        // In edges continued
        for (auto& j : in_set[i]) {
            new_graph.In[i].emplace_back(j);
        }
        // Out edges continued
        for (auto& j : out_set[i]) {
            new_graph.Out[i].emplace_back(j);
        }
    }

    for (unsigned i = 0; i<new_graph.n; i++) {
        for(int j : new_graph.Out[i]) {
            assert( new_graph.comm_edge_W.find( std::make_pair(i,j) ) != new_graph.comm_edge_W.cend() );
        }
    }

    return std::make_pair(new_graph, contraction_map);
}

bool DAG::is_acyclic() const
{
    std::vector<int> in_count(n);
    int num_processed = 0;
    std::queue<int> queue;

    for (unsigned i = 0; i< n; i++) {
        in_count[i] = In[i].size();
        if (In[i].empty())
            queue.emplace(i);
    }

    while (! queue.empty())
    {
        int lead = queue.front();
        queue.pop();
        num_processed++;

        for (auto& j : Out[lead]) {
            in_count[j]--;
            if (in_count[j] == 0) {
                queue.emplace(j);
            }
        }
    }

    if (num_processed != n)
        return false;
    return true;
}