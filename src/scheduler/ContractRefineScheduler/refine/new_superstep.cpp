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

#include "scheduler/ContractRefineScheduler/refine/new_superstep.hpp"

// binary search for weight-balanced cut decision
// returns cut (cut between cut-1 and cut), new bias
int binary_search_weight_bal_cut_along_chain(const SubDAG &graph, const std::vector<int> chain, const int bias) {
    int left = 0;
    int right = chain.size();

    while (left < right) {
        int mid = left + (right - left) / 2; // left <= mid < right unless left==right
        assert(left <= mid && mid < right);
        // std::cerr << left << " " << mid << " " << right << std::endl;

        // calculating weight of nodes of ancestors of chain[mid-1] (if exists)
        int weight_top = 0;
        if (mid > 0) {
            weight_top = graph.workW_of_node_set(graph.ancestors(chain[mid - 1]));
        }

        // calculating weight of nodes of descendants of chain[mid]
        int weight_bottom = graph.workW_of_node_set(
            graph.descendants(chain[mid])); // mid < chain.size() unless left = right = chain.size()

        if (weight_top + bias >= weight_bottom) {
            right = mid;
        } else {
            left = mid + 1;
        }
    };
    // cut between left-1 and left // it is guaranteed that weight_top[chain[left-1]] + bias >=
    // weight_bottom[chain[left]]

    return left;
};

// Introduces a weight-balanced superstep into a directed acyclic graph thus cutting it into into multiple
// weakly-connected components Returns node assigments (top, bottom) by node name of super-dag
std::pair<std::unordered_set<int>, std::unordered_set<int>> dag_weight_bal_cut(const SubDAG &graph, int bias) {
    std::unordered_set<int> top_assigned_nodes;
    std::unordered_set<int> bottom_assigned_nodes;
    std::unordered_set<int> unassigned_nodes;
    unassigned_nodes.reserve(graph.n);
    for (unsigned i = 0; i < graph.n; i++) {
        unassigned_nodes.emplace(graph.sub_to_super.at(i));
    }

    for (int i = 0; i<graph.n; i++) {
        for(int j : graph.Out[i]) {
            assert( graph.comm_edge_W.find( std::make_pair(i,j) ) != graph.comm_edge_W.cend() );
        }
    }
    SubDAG active_dag = graph;
    for (int i = 0; i<active_dag.n; i++) {
        for(int j : active_dag.Out[i]) {
            assert( active_dag.comm_edge_W.find( std::make_pair(i,j) ) != active_dag.comm_edge_W.cend() );
        }
    }

    while (!(unassigned_nodes.empty())) {
        std::vector<int> chain = active_dag.longest_chain();
        int cut = binary_search_weight_bal_cut_along_chain(active_dag, chain, bias);

        std::unordered_set<int> pre_cut;
        if (cut > 0) {
            pre_cut = active_dag.ancestors(chain[cut - 1]);
            top_assigned_nodes = get_union(top_assigned_nodes, pre_cut);
            for (auto &node : pre_cut) {
                unassigned_nodes.erase(node);
            }
        }

        std::unordered_set<int> post_cut;
        if (cut < chain.size()) {
            post_cut = active_dag.descendants(chain[cut]);
            bottom_assigned_nodes = get_union(bottom_assigned_nodes, post_cut);
            for (auto &node : post_cut) {
                unassigned_nodes.erase(node);
            }
        }

        // update bias
        bias = graph.workW_of_node_set(pre_cut) + bias - graph.workW_of_node_set(post_cut);

        // update active dag
        active_dag = SubDAG(active_dag, unassigned_nodes);
    }

    return {top_assigned_nodes, bottom_assigned_nodes};
};

// TODO later: reimplement with passing on information about connected components
// Introduces a superstep, by shaving from the top, in order to increase source nodes of bottom
// Returns node assigments (top, bottom) by node name of super-dag
std::pair<std::unordered_set<int>, std::unordered_set<int>> top_shave_few_sources(const SubDAG &graph,
                                                                                  const int min_comp_generation,
                                                                                  const float mult_size_cap) {
    if (graph.n == 0) {
        return {std::unordered_set<int>(), std::unordered_set<int>()};
    }

    std::unordered_set<int> top_assigned_nodes_sub;
    int top_weight = 0;

    int max_top_weight = 0;
    for (unsigned i = 0; i < graph.n; i++) {
        max_top_weight += graph.workW[i];
    }
    max_top_weight = int(max_top_weight * mult_size_cap);

    // computing distance from bottom for priority
    const std::vector<int> bottom_distance = graph.get_bottom_node_distance();

    std::vector<int> sources;
    for (int i = 0; i < graph.n; i++) {
        if (graph.In[i].empty()) {
            sources.emplace_back(i);
        }
    }

    if (sources.size() == 1) {
        const auto cmp = [bottom_distance](int a, int b) {
            return ((bottom_distance[a] > bottom_distance[b]) ||
                    ((bottom_distance[a] == bottom_distance[b]) && (a > b)));
        };                                                                         // highest bottom first
        std::set<int, decltype(cmp)> movable(sources.begin(), sources.end(), cmp); // this changes sources in place

        bool change = true;

        while (change && (top_weight <= max_top_weight) && (top_assigned_nodes_sub.size() + 1 != graph.n)) {
            change = false;

            int chosen_node = -1;
            std::vector<int> to_moveable_children;

            for (auto &node : movable) {
                to_moveable_children.clear();
                for (auto &child : graph.Out[node]) {
                    if (top_assigned_nodes_sub.find(child) != top_assigned_nodes_sub.end())
                        continue;
                    if (std::all_of(graph.In[child].cbegin(), graph.In[child].cend(),
                                    [node, top_assigned_nodes_sub](int par) {
                                        return ((par == node) ||
                                                (top_assigned_nodes_sub.find(par) != top_assigned_nodes_sub.end()));
                                    })) {
                        to_moveable_children.emplace_back(child);
                    }
                }
                if (to_moveable_children.size() >= 1) {
                    chosen_node = node;
                    change = true;
                    break;
                }
            }

            if (change) {
                assert(chosen_node != -1);
                top_assigned_nodes_sub.emplace(chosen_node);
                top_weight += graph.workW[chosen_node];

                movable.erase(chosen_node);

                for (auto &child : to_moveable_children) {
                    movable.emplace(child);
                }
            }
        }
    }

    if (sources.size() >= 2) {
        int top_weight = 0;

        Union_Find_Universe<int> uf_universe;
        std::multiset<int> components_top_sizes_sorted; // for median computation

        const auto cmp = [bottom_distance](int a, int b) {
            return ((bottom_distance[a] > bottom_distance[b]) ||
                    ((bottom_distance[a] == bottom_distance[b]) && (a > b)));
        }; // highest bottom first
        std::set<int, decltype(cmp)> movable(cmp);

        for (auto &source : sources) {
            movable.emplace(source);
        }

        bool change = true;

        while (change && (top_weight <= max_top_weight) && (top_assigned_nodes_sub.size() + 1 != graph.n)) {
            assert(uf_universe.get_number_of_connected_components() == components_top_sizes_sorted.size());

            change = false;

            // computing median
            int median;
            if (!components_top_sizes_sorted.empty()) {
                median = Get_Lower_Median(components_top_sizes_sorted);
            }
            int moving_node = -1;
            std::set<int> par_components;

            for (auto &node : movable) {
                // reset
                par_components.clear();
                moving_node = -1;

                // computing parent components
                for (auto &par : graph.In[node]) {
                    assert(top_assigned_nodes_sub.find(par) != top_assigned_nodes_sub.cend());
                    par_components.emplace(uf_universe.find_origin_by_name(par));
                }

                // checking for enough remaining components
                if (uf_universe.get_number_of_connected_components() <= 1) {
                    if (par_components.size() > 0) continue;
                } else if (uf_universe.get_number_of_connected_components() <= min_comp_generation) {
                    if (par_components.size() > 1) continue;
                } else {
                    if (uf_universe.get_number_of_connected_components() + 1 < min_comp_generation + par_components.size()) continue;
                }
                // if (par_components.size() == uf_universe.get_number_of_connected_components() &&
                //     uf_universe.get_number_of_connected_components() != 0)
                //     continue;

                // checking if adding node makes a too fat component
                // only one parent
                if (par_components.size() == 1) {
                    if (uf_universe.get_weight_of_component_by_name(*par_components.begin()) > 2 * median)
                        continue;
                }
                // more than one parent
                if (par_components.size() >= 2) {
                    int par_comp_weight = 0;
                    for (auto &par : par_components) {
                        par_comp_weight += uf_universe.get_weight_of_component_by_name(par);
                    }
                    if (par_comp_weight > (median + 1) / 2 * 3)
                        continue;
                }

                change = true;
                moving_node = node;
                break;
            }

            if (change) {
                assert(moving_node != -1);
                top_assigned_nodes_sub.emplace(moving_node);
                top_weight += graph.workW[moving_node];
                movable.erase(moving_node);

                uf_universe.add_object(moving_node, graph.workW[moving_node]);
                assert(uf_universe.find_origin_by_name(moving_node) == moving_node);
                assert(uf_universe.get_weight_of_component_by_name(moving_node) == graph.workW[moving_node]);

                for (auto &par : par_components) {
                    int weight_to_delete = uf_universe.get_weight_of_component_by_name(par);
                    components_top_sizes_sorted.erase(components_top_sizes_sorted.find(weight_to_delete));
                }

                for (auto &par : par_components) {
                    uf_universe.join_by_name(par, moving_node);
                }
                components_top_sizes_sorted.emplace(uf_universe.get_weight_of_component_by_name(moving_node));

                for (auto &child : graph.Out[moving_node]) {
                    if (top_assigned_nodes_sub.find(child) != top_assigned_nodes_sub.end())
                        continue;
                    if (std::all_of(graph.In[child].cbegin(), graph.In[child].cend(),
                                    [moving_node, top_assigned_nodes_sub](int par) {
                                        return ((par == moving_node) ||
                                                (top_assigned_nodes_sub.find(par) != top_assigned_nodes_sub.end()));
                                    })) {
                        movable.emplace(child);
                    }
                }
            }
        }
    }

    // conversion to super nodes
    std::unordered_set<int> top_assigned_nodes;
    top_assigned_nodes.reserve(top_assigned_nodes_sub.size());
    std::unordered_set<int> bottom_assigned_nodes;
    bottom_assigned_nodes.reserve(graph.n - top_assigned_nodes_sub.size());
    for (unsigned i = 0; i < graph.n; i++) {
        int sup_node = graph.sub_to_super.at(i);
        if (top_assigned_nodes_sub.find(i) == top_assigned_nodes_sub.end()) {
            bottom_assigned_nodes.emplace(sup_node);
        } else {
            top_assigned_nodes.emplace(sup_node);
        }
    }

    return {top_assigned_nodes, bottom_assigned_nodes};
}

// TODO: reimplement using the already computed information about connected components
// Introduces a superstep, by shaving from the bottom, to either increase source nodes of top or to have multiple
// components in bottom Returns node assigments (top, bottom) by node name of super-dag
std::pair<std::unordered_set<int>, std::unordered_set<int>> bottom_shave_few_sinks(const SubDAG &graph,
                                                                                   const int min_comp_generation,
                                                                                   const float mult_size_cap) {
    if (graph.n == 0) {
        return {std::unordered_set<int>(), std::unordered_set<int>()};
    }

    std::unordered_set<int> bottom_assigned_nodes_sub;
    int bottom_weight = 0;

    int max_bottom_weight = 0;
    for (unsigned i = 0; i < graph.n; i++) {
        max_bottom_weight += graph.workW[i];
    }
    max_bottom_weight = int(max_bottom_weight * mult_size_cap);

    // computing distance from top for priority
    const std::vector<int> top_distance = graph.get_top_node_distance();

    std::vector<int> sinks;
    for (int i = 0; i < graph.n; i++) {
        if (graph.Out[i].empty()) {
            sinks.emplace_back(i);
        }
    }

    if (sinks.size() == 1) {
        const auto cmp = [top_distance](int a, int b) {
            return ((top_distance[a] > top_distance[b]) || ((top_distance[a] == top_distance[b]) && (a > b)));
        };                                                                     // highest top first
        std::set<int, decltype(cmp)> movable(sinks.begin(), sinks.end(), cmp); // this changes sinks in place

        bool change = true;

        while (change && (bottom_weight <= max_bottom_weight) && (bottom_assigned_nodes_sub.size() + 1 != graph.n)) {
            change = false;

            int chosen_node = -1;
            std::vector<int> to_moveable_parent;

            for (auto &node : movable) {
                to_moveable_parent.clear();
                for (auto &parent : graph.In[node]) {
                    if (bottom_assigned_nodes_sub.find(parent) != bottom_assigned_nodes_sub.end())
                        continue;
                    if (std::all_of(graph.Out[parent].cbegin(), graph.Out[parent].cend(),
                                    [node, bottom_assigned_nodes_sub](int chld) {
                                        return ((chld == node) || (bottom_assigned_nodes_sub.find(chld) !=
                                                                   bottom_assigned_nodes_sub.end()));
                                    })) {
                        to_moveable_parent.emplace_back(parent);
                    }
                }
                if (to_moveable_parent.size() >= 1) {
                    chosen_node = node;
                    change = true;
                    break;
                }
            }

            if (change) {
                assert(chosen_node != -1);
                bottom_assigned_nodes_sub.emplace(chosen_node);
                bottom_weight += graph.workW[chosen_node];

                movable.erase(chosen_node);

                for (auto &par : to_moveable_parent) {
                    movable.emplace(par);
                }
            }
        }
    }

    if (sinks.size() >= 2) {
        int bottom_weight = 0;

        Union_Find_Universe<int> uf_universe;
        std::multiset<int> components_bottom_sizes_sorted; // for median computation

        const auto cmp = [top_distance](int a, int b) {
            return ((top_distance[a] > top_distance[b]) || ((top_distance[a] == top_distance[b]) && (a > b)));
        }; // highest top first
        std::set<int, decltype(cmp)> movable(cmp);

        for (auto &sink : sinks) {
            movable.emplace(sink);
        }

        bool change = true;

        while (change && (bottom_weight <= max_bottom_weight) && (bottom_assigned_nodes_sub.size() + 1 != graph.n)) {
            assert(uf_universe.get_number_of_connected_components() == components_bottom_sizes_sorted.size());

            change = false;

            // computing median
            int median;
            if (!components_bottom_sizes_sorted.empty()) {
                median = Get_Lower_Median(components_bottom_sizes_sorted);
            }
            int moving_node = -1;
            std::set<int> child_components;

            for (auto &node : movable) {
                // reset
                child_components.clear();
                moving_node = -1;

                // computing parent components
                for (auto &chld : graph.Out[node]) {
                    assert(bottom_assigned_nodes_sub.find(chld) != bottom_assigned_nodes_sub.end());
                    child_components.emplace(uf_universe.find_origin_by_name(chld));
                }

                // checking for enough remaining components
                if (uf_universe.get_number_of_connected_components() <= 1) {
                    if (child_components.size() > 0) continue;
                } else if (uf_universe.get_number_of_connected_components() <= min_comp_generation) {
                    if (child_components.size() > 1) continue;
                } else {
                    if (uf_universe.get_number_of_connected_components() + 1 < min_comp_generation + child_components.size()) continue;
                }
                // if (child_components.size() == uf_universe.get_number_of_connected_components() &&
                //     uf_universe.get_number_of_connected_components() != 0)
                //     continue;

                // checking if adding node makes a too fat component
                // only one parent
                if (child_components.size() == 1) {
                    if (uf_universe.get_weight_of_component_by_name(*child_components.begin()) > 2 * median)
                        continue;
                }
                // more than one parent
                if (child_components.size() >= 2) {
                    int chld_comp_weight = 0;
                    for (auto &chld : child_components) {
                        chld_comp_weight += uf_universe.get_weight_of_component_by_name(chld);
                    }
                    if (chld_comp_weight > (median + 1) / 2 * 3)
                        continue;
                }

                change = true;
                moving_node = node;
                break;
            }

            if (change) {
                assert(moving_node != -1);
                bottom_assigned_nodes_sub.emplace(moving_node);
                bottom_weight += graph.workW[moving_node];
                movable.erase(moving_node);

                uf_universe.add_object(moving_node, graph.workW[moving_node]);
                assert(uf_universe.find_origin_by_name(moving_node) == moving_node);
                assert(uf_universe.get_weight_of_component_by_name(moving_node) == graph.workW[moving_node]);

                for (auto &chld : child_components) {
                    int weight_to_delete = uf_universe.get_weight_of_component_by_name(chld);
                    components_bottom_sizes_sorted.erase(components_bottom_sizes_sorted.find(weight_to_delete));
                }

                for (auto &chld : child_components) {
                    assert(bottom_assigned_nodes_sub.find(chld) != bottom_assigned_nodes_sub.end());
                    uf_universe.join_by_name(chld, moving_node);
                }
                components_bottom_sizes_sorted.emplace(uf_universe.get_weight_of_component_by_name(moving_node));

                for (auto &par : graph.In[moving_node]) {
                    if (bottom_assigned_nodes_sub.find(par) != bottom_assigned_nodes_sub.end())
                        continue;
                    if (std::all_of(graph.Out[par].cbegin(), graph.Out[par].cend(),
                                    [moving_node, bottom_assigned_nodes_sub](int chld) {
                                        return ((chld == moving_node) || (bottom_assigned_nodes_sub.find(chld) !=
                                                                          bottom_assigned_nodes_sub.end()));
                                    })) {
                        movable.emplace(par);
                    }
                }
            }
        }
    }

    // conversion to super nodes
    std::unordered_set<int> top_assigned_nodes;
    top_assigned_nodes.reserve(graph.n - bottom_assigned_nodes_sub.size());
    std::unordered_set<int> bottom_assigned_nodes;
    bottom_assigned_nodes.reserve(bottom_assigned_nodes_sub.size());
    for (int i = 0; i < graph.n; i++) {
        int sup_node = graph.sub_to_super.at(i);
        if (bottom_assigned_nodes_sub.find(i) != bottom_assigned_nodes_sub.end()) {
            bottom_assigned_nodes.emplace(sup_node);
        } else {
            top_assigned_nodes.emplace(sup_node);
        }
    }

    return {top_assigned_nodes, bottom_assigned_nodes};
}