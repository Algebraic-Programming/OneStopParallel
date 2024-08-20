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

#include "scheduler/SchedulePermutations/ScheduleNodePermuter.hpp"


void topological_sort_for_data_locality_interior(std::vector<size_t>& nodes, const BspSchedule& sched, unsigned cache_line_size) {
    if (nodes.empty()) return;

    const ComputationalDag& graph = sched.getInstance().getComputationalDag();
    
    std::unordered_map<size_t, size_t> nodes_to_position; //position in vector called nodes
    for (size_t i = 0; i < nodes.size(); i++) {
        nodes_to_position.emplace(nodes[i], i);
    }

    std::vector<std::vector<size_t>> out_edges_temp(nodes.size(), std::vector<size_t>({})); // first index position to vector nodes and then list of positions of vector nodes (of children)
    std::vector<size_t> number_of_unallocated_parents(nodes.size()); // same order as nodes
    for (size_t i = 0; i < nodes.size(); i++) {
        for (const auto& out_edge : graph.out_edges(nodes[i])) {
            if ( nodes_to_position.find(out_edge.m_target) != nodes_to_position.cend() ) {
                number_of_unallocated_parents[ nodes_to_position.at(out_edge.m_target) ]++;
                out_edges_temp[i].emplace_back(nodes_to_position.at(out_edge.m_target));
            }
        }
    }

    // queue ordered by vector lexicographically with parents_in_cache, number_of_children_, node_number
    std::vector<std::vector<size_t>> node_queue; // needs to be sorted every time it has been updated
    auto queue_cmp = [](const std::vector<size_t>& a, const std::vector<size_t>& b) {
                            return  (a[0] < b[0])
                                    || ((a[0] == b[0]) && (a[1] < b[1]))
                                    || ((a[0] == b[0]) && (a[1] == b[1]) && (a[2] > b[2]));
                        };
    for  (size_t i = 0; i < nodes.size(); i++) {
        if ( number_of_unallocated_parents[i] == 0 ) {
            node_queue.push_back( std::vector<size_t>({0, out_edges_temp[i].size(), nodes[i]}) );
        }
    }
    std::sort(node_queue.begin(), node_queue.end(), queue_cmp);

    std::vector<size_t> new_ordering;
    new_ordering.reserve(nodes.size());
    while ( !node_queue.empty() ) {
        size_t recently_added_node = node_queue.back()[2];
        new_ordering.emplace_back( recently_added_node );
        node_queue.pop_back();

        // new ones to the queue
        for (const auto& chld_ind : out_edges_temp[ nodes_to_position.at(recently_added_node) ] ) {
            number_of_unallocated_parents[chld_ind]--;
            if (number_of_unallocated_parents[chld_ind] == 0) {
                node_queue.push_back(std::vector<size_t>({0, out_edges_temp[chld_ind].size(), nodes[chld_ind]}));
            }
        }

        // updating queue
        for (auto& elem : node_queue) {
            elem[0] = 0;
            for (size_t i = new_ordering.size()-1; (i >= 0) && (i >= new_ordering.size()-cache_line_size); i--) {
                for (auto& chld_ind : out_edges_temp[nodes_to_position.at(new_ordering[i])]) {
                    if ( nodes[chld_ind] == elem[2]) {
                        elem[0]++;
                    }
                }
                if (i == 0) break;
            }
        }
        std::sort(node_queue.begin(), node_queue.end(), queue_cmp);
    }

    if (new_ordering.size() != nodes.size()) {
        throw std::runtime_error("topological_sort_for_data_locality_interior failed");
    }

    // if ( ! std::is_permutation(new_ordering.cbegin(), new_ordering.cend(), nodes.cbegin(), nodes.cend()) ) {
    //     throw std::runtime_error("topological_sort_for_data_locality_interior failed");
    // }

    nodes = new_ordering;
}

void topological_sort_for_data_locality_begin(std::vector<size_t>& nodes, const BspSchedule& sched, unsigned cache_line_size, const std::vector<std::vector<std::vector<size_t>>>& allocation) {
    if (nodes.empty()) return;
    if ( sched.assignedSuperstep(nodes[0]) == 0 ) {
        topological_sort_for_data_locality_interior(nodes, sched, cache_line_size);
        return;
    }

    const ComputationalDag& graph = sched.getInstance().getComputationalDag();
    
    std::unordered_map<size_t, size_t> nodes_to_position; //position in vector called nodes
    for (size_t i = 0; i < nodes.size(); i++) {
        nodes_to_position.emplace(nodes[i], i);
    }

    std::vector<std::vector<size_t>> out_edges_temp(nodes.size(), std::vector<size_t>({})); // first index position to vector nodes and then list of positions of vector nodes (of children)
    std::vector<size_t> number_of_unallocated_parents(nodes.size()); // same order as nodes
    for (size_t i = 0; i < nodes.size(); i++) {
        for (const auto& out_edge : graph.out_edges(nodes[i])) {
            if ( nodes_to_position.find(out_edge.m_target) != nodes_to_position.cend() ) {
                number_of_unallocated_parents[ nodes_to_position.at(out_edge.m_target) ]++;
                out_edges_temp[i].emplace_back(nodes_to_position.at(out_edge.m_target));
            }
        }
    }

    // counting parents in same processor but prior superstep
    size_t curr_superstep = sched.assignedSuperstep(nodes[0]);
    size_t curr_processor = sched.assignedProcessor(nodes[0]);
    std::vector<size_t> num_parents_prev_superstep(nodes.size(), 0);
    for (size_t i = 0; i < nodes.size(); i++) {
        for (const auto& in_edge : graph.in_edges(nodes[i])) {
            if ( (sched.assignedProcessor(in_edge.m_source) == curr_processor) && (sched.assignedSuperstep(in_edge.m_source) == curr_superstep-1) ) {
                num_parents_prev_superstep[i]++;
            }
        }
    }

    // queue vector with parents_previous_superstep, parents_in_cache, number_of_children_, node_number
    std::vector<std::vector<size_t>> node_queue; // needs to be sorted every time it has been updated
    auto queue_cmp = [](const std::vector<size_t>& a, const std::vector<size_t>& b) {
                            return  (a[0] < b[0])
                                    || ((a[0] == b[0]) && (a[1] < b[1]))
                                    || ((a[0] == b[0]) && (a[1] == b[1]) && (a[2] < b[2]))
                                    || ((a[0] == b[0]) && (a[1] == b[1]) && (a[2] == b[2]) && (a[3] > b[3]));
                        };
    for  (size_t i = 0; i < nodes.size(); i++) {
        if ( number_of_unallocated_parents[i] == 0 ) {
            node_queue.push_back( std::vector<size_t>({num_parents_prev_superstep[i], 0, out_edges_temp[i].size(), nodes[i]}) );
        }
    }
    std::sort(node_queue.begin(), node_queue.end(), queue_cmp);

    std::vector<size_t> new_ordering;
    new_ordering.reserve(nodes.size());
    while ( !node_queue.empty() ) {
        size_t recently_added_node = node_queue.back()[3];
        new_ordering.emplace_back( recently_added_node );
        node_queue.pop_back();

        // new ones to the queue
        for (const auto& chld_ind : out_edges_temp[ nodes_to_position.at(recently_added_node) ] ) {
            number_of_unallocated_parents[chld_ind]--;
            if (number_of_unallocated_parents[chld_ind] == 0) {
                node_queue.push_back(std::vector<size_t>({num_parents_prev_superstep[chld_ind], 0, out_edges_temp[chld_ind].size(), nodes[chld_ind]}));
            }
        }

        // updating queue
        for (auto& elem : node_queue) {
            elem[1] = 0;
            for (size_t i = new_ordering.size()-1; (i >= 0) && (i >= new_ordering.size()-cache_line_size); i--) {
                for (auto& chld_ind : out_edges_temp[nodes_to_position.at(new_ordering[i])]) {
                    if ( nodes[chld_ind] == elem[3]) {
                        elem[1]++;
                    }
                }
                if (i == 0) break;
            }
        }
        std::sort(node_queue.begin(), node_queue.end(), queue_cmp);
    }

    if (new_ordering.size() != nodes.size()) {
        throw std::runtime_error("topological_sort_for_data_locality_interior failed");
    }

    // if ( ! std::is_permutation(new_ordering.cbegin(), new_ordering.cend(), nodes.cbegin(), nodes.cend()) ) {
    //     throw std::runtime_error("topological_sort_for_data_locality_begin failed");
    // }

    nodes = new_ordering;
}

void topological_sort_for_data_locality_end(std::vector<size_t>& nodes, const BspSchedule& sched, unsigned cache_line_size, const std::vector<std::vector<std::vector<size_t>>>& allocation) {
    if (nodes.empty()) return;
    if ( sched.assignedSuperstep(nodes[0]) == sched.numberOfSupersteps()-1 ) {
        topological_sort_for_data_locality_interior(nodes, sched, cache_line_size);
        return;
    }

    const ComputationalDag& graph = sched.getInstance().getComputationalDag();
    
    std::unordered_map<size_t, size_t> nodes_to_position; //position in vector called nodes
    for (size_t i = 0; i < nodes.size(); i++) {
        nodes_to_position.emplace(nodes[i], i);
    }

    std::vector<std::vector<size_t>> out_edges_temp(nodes.size(), std::vector<size_t>({})); // first index position to vector nodes and then list of positions of vector nodes (of children)
    std::vector<size_t> number_of_unallocated_parents(nodes.size()); // same order as nodes
    for (size_t i = 0; i < nodes.size(); i++) {
        for (const auto& out_edge : graph.out_edges(nodes[i])) {
            if ( nodes_to_position.find(out_edge.m_target) != nodes_to_position.cend() ) {
                number_of_unallocated_parents[ nodes_to_position.at(out_edge.m_target) ]++;
                out_edges_temp[i].emplace_back(nodes_to_position.at(out_edge.m_target));
            }
        }
    }

    // counting children in same processor but subsequent superstep
    size_t curr_superstep = sched.assignedSuperstep(nodes[0]);
    size_t curr_processor = sched.assignedProcessor(nodes[0]);
    std::vector<size_t> num_children_next_superstep(nodes.size(), 0);
    for (size_t i = 0; i < nodes.size(); i++) {
        for (const auto& out_edge : graph.out_edges(nodes[i])) {
            if ( (sched.assignedProcessor(out_edge.m_target) == curr_processor) && (sched.assignedSuperstep(out_edge.m_target) == curr_superstep+1) ) {
                num_children_next_superstep[i]++;
            }
        }
    }

    // queue vector with children_in_next_superstep, parents_in_cache, number_of_children_, node_number
    std::vector<std::vector<size_t>> node_queue; // needs to be sorted every time it has been updated
    auto queue_cmp = [](const std::vector<size_t>& a, const std::vector<size_t>& b) {
                            return  (a[0] > b[0])
                                    || ((a[0] == b[0]) && (a[1] < b[1]))
                                    || ((a[0] == b[0]) && (a[1] == b[1]) && (a[2] < b[2]))
                                    || ((a[0] == b[0]) && (a[1] == b[1]) && (a[2] == b[2]) && (a[3] > b[3]));
                        };
    for  (size_t i = 0; i < nodes.size(); i++) {
        if ( number_of_unallocated_parents[i] == 0 ) {
            node_queue.push_back( std::vector<size_t>({num_children_next_superstep[i], 0, out_edges_temp[i].size(), nodes[i]}) );
        }
    }
    std::sort(node_queue.begin(), node_queue.end(), queue_cmp);

    std::vector<size_t> new_ordering;
    new_ordering.reserve(nodes.size());
    while ( !node_queue.empty() ) {
        size_t recently_added_node = node_queue.back()[3];
        new_ordering.emplace_back( recently_added_node );
        node_queue.pop_back();

        // new ones to the queue
        for (const auto& chld_ind : out_edges_temp[ nodes_to_position.at(recently_added_node) ] ) {
            number_of_unallocated_parents[chld_ind]--;
            if (number_of_unallocated_parents[chld_ind] == 0) {
                node_queue.push_back(std::vector<size_t>({num_children_next_superstep[chld_ind], 0, out_edges_temp[chld_ind].size(), nodes[chld_ind]}));
            }
        }

        // updating queue
        for (auto& elem : node_queue) {
            elem[1] = 0;
            for (size_t i = new_ordering.size()-1; (i >= 0) && (i >= new_ordering.size()-cache_line_size); i--) {
                for (auto& chld_ind : out_edges_temp[nodes_to_position.at(new_ordering[i])]) {
                    if ( nodes[chld_ind] == elem[3]) {
                        elem[1]++;
                    }
                }
                if (i == 0) break;
            }
        }
        std::sort(node_queue.begin(), node_queue.end(), queue_cmp);
    }

    if (new_ordering.size() != nodes.size()) {
        throw std::runtime_error("topological_sort_for_data_locality_interior failed");
    }

    // if ( ! std::is_permutation(new_ordering.cbegin(), new_ordering.cend(), nodes.cbegin(), nodes.cend()) ) {
    //     throw std::runtime_error("topological_sort_for_data_locality_end failed");
    // }

    nodes = new_ordering;
}

std::vector<size_t> schedule_node_permuter(const BspSchedule& sched, unsigned cache_line_size, const SCHEDULE_NODE_PERMUTATION_MODES mode, const bool simplified) {
    // superstep, processor, nodes
    std::vector<std::vector<std::vector<size_t>>> allocation(sched.numberOfSupersteps(),
                                                                std::vector<std::vector<size_t>>(sched.getInstance().numberOfProcessors(),
                                                                    std::vector<size_t>({})));
    for (size_t node = 0; node < sched.getInstance().numberOfVertices(); node++) {
        allocation[ sched.assignedSuperstep(node) ][ sched.assignedProcessor(node) ].emplace_back(node);
    }

    // reordering and allocating into permutation
    std::vector<size_t> permutation(sched.getInstance().numberOfVertices());

    if(mode == LOOP_PROCESSORS || mode == SNAKE_PROCESSORS) {
        bool forward = true;
        size_t counter = 0;
        for (auto step_it = allocation.begin(); step_it != allocation.cend(); step_it++) {
            if (forward) {
                for (auto proc_it = step_it->begin(); proc_it != step_it->cend(); proc_it++) {
                    if (simplified) {
                        topological_sort_for_data_locality_interior(*proc_it, sched, cache_line_size);
                    } else if ( proc_it == step_it->cbegin() ) {
                        topological_sort_for_data_locality_begin(*proc_it, sched, cache_line_size, allocation);
                    } else if ( proc_it == std::prev(step_it->end())) {
                        topological_sort_for_data_locality_end(*proc_it, sched, cache_line_size, allocation);
                    } else {
                        topological_sort_for_data_locality_interior(*proc_it, sched, cache_line_size);
                    }
                    for (const auto& node : *proc_it) {
                        permutation[node] = counter;
                        counter++;
                    }
                }
            } else {
                for (auto proc_it = step_it->rbegin(); proc_it != step_it->crend(); proc_it++) {
                    if (simplified) {
                        topological_sort_for_data_locality_interior(*proc_it, sched, cache_line_size);
                    } else if ( proc_it == step_it->crbegin() ) {
                        topological_sort_for_data_locality_begin(*proc_it, sched, cache_line_size, allocation);
                    } else if ( proc_it == std::prev(step_it->rend())) {
                        topological_sort_for_data_locality_end(*proc_it, sched, cache_line_size, allocation);
                    } else {
                        topological_sort_for_data_locality_interior(*proc_it, sched, cache_line_size);
                    }
                    for (const auto& node : *proc_it) {
                        permutation[node] = counter;
                        counter++;
                    }
                }
            }
            
            if (mode == SNAKE_PROCESSORS) {
                forward = !forward;
            }
        }
    } else if (mode == PROCESSOR_FIRST) {
        size_t counter = 0;
        for (size_t proc = 0; proc < sched.getInstance().numberOfProcessors(); proc++) {
            for (auto step_it = allocation.begin(); step_it != allocation.cend(); step_it++) {
                topological_sort_for_data_locality_interior((*step_it)[proc], sched, cache_line_size);
                for (const auto& node : (*step_it)[proc]) {
                    permutation[node] = counter;
                    counter++;
                }
            }
        }
    } else {
        throw std::logic_error("Permutation mode not implemented.");
    }


    return permutation;
}

std::vector<size_t> schedule_node_permuter_basic(const BspSchedule& sched, const SCHEDULE_NODE_PERMUTATION_MODES mode) {
    // superstep, processor, nodes
    std::vector<std::vector<std::vector<size_t>>> allocation(sched.numberOfSupersteps(),
                                                                std::vector<std::vector<size_t>>(sched.getInstance().numberOfProcessors(),
                                                                    std::vector<size_t>({})));
    for (size_t node = 0; node < sched.getInstance().numberOfVertices(); node++) {
        allocation[ sched.assignedSuperstep(node) ][ sched.assignedProcessor(node) ].emplace_back(node);
    }

    // reordering and allocating into permutation
    std::vector<size_t> permutation(sched.getInstance().numberOfVertices());

    if(mode == LOOP_PROCESSORS || mode == SNAKE_PROCESSORS) {
        bool forward = true;
        size_t counter = 0;
        for (auto step_it = allocation.begin(); step_it != allocation.cend(); step_it++) {
            if (forward) {
                for (auto proc_it = step_it->begin(); proc_it != step_it->cend(); proc_it++) {
                    topological_sort_for_data_locality_interior_basic(*proc_it, sched);
                    
                    for (const auto& node : *proc_it) {
                        permutation[node] = counter;
                        counter++;
                    }
                }
            } else {
                for (auto proc_it = step_it->rbegin(); proc_it != step_it->crend(); proc_it++) {
                    topological_sort_for_data_locality_interior_basic(*proc_it, sched);
                    
                    for (const auto& node : *proc_it) {
                        permutation[node] = counter;
                        counter++;
                    }
                }
            }
            
            if (mode == SNAKE_PROCESSORS) {
                forward = !forward;
            }
        }
    } else {
        throw std::logic_error("Permutation mode not implemented.");
    }

    return permutation;
}

void topological_sort_for_data_locality_interior_basic(std::vector<size_t>& nodes, const BspSchedule& sched) {
    if (nodes.empty()) return;

    const ComputationalDag& graph = sched.getInstance().getComputationalDag();
    
    std::unordered_map<size_t, size_t> nodes_to_position; //position in vector called nodes
    for (size_t i = 0; i < nodes.size(); i++) {
        nodes_to_position.emplace(nodes[i], i);
    }

    std::vector<std::vector<size_t>> out_edges_temp(nodes.size(), std::vector<size_t>({})); // first index position to vector nodes and then list of positions of vector nodes (of children)
    std::vector<size_t> number_of_unallocated_parents(nodes.size()); // same order as nodes
    for (size_t i = 0; i < nodes.size(); i++) {
        for (const auto& out_edge : graph.out_edges(nodes[i])) {
            if ( nodes_to_position.find(out_edge.m_target) != nodes_to_position.cend() ) {
                number_of_unallocated_parents[ nodes_to_position.at(out_edge.m_target) ]++;
                out_edges_temp[i].emplace_back(nodes_to_position.at(out_edge.m_target));
            }
        }
    }

    // queue ordered by node_number (decreasingly)
    std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>> node_queue;
    for  (size_t i = 0; i < nodes.size(); i++) {
        if ( number_of_unallocated_parents[i] == 0 ) {
            node_queue.push( nodes[i] );
        }
    }

    std::vector<size_t> new_ordering;
    new_ordering.reserve(nodes.size());
    while ( !node_queue.empty() ) {
        size_t recently_added_node = node_queue.top();
        new_ordering.emplace_back( recently_added_node );
        node_queue.pop();

        // new ones to the queue
        for (const auto& chld_ind : out_edges_temp[ nodes_to_position.at(recently_added_node) ] ) {
            number_of_unallocated_parents[chld_ind]--;
            if (number_of_unallocated_parents[chld_ind] == 0) {
                node_queue.push(nodes[chld_ind]);
            }
        }
    }

    if (new_ordering.size() != nodes.size()) {
        throw std::runtime_error("topological_sort_for_data_locality_interior_basic failed");
    }

    // if ( ! std::is_permutation(new_ordering.cbegin(), new_ordering.cend(), nodes.cbegin(), nodes.cend()) ) {
    //     throw std::runtime_error("topological_sort_for_data_locality_interior failed");
    // }

    nodes = new_ordering;
}

void topological_sort_for_data_locality_interior_small_relations(std::vector<size_t>& nodes, const BspSchedule& sched) {
    if (nodes.empty()) return;

    std::unordered_set<size_t> nodes_set;
    nodes_set.reserve(nodes.size());
    for (const auto& node : nodes) {
        nodes_set.emplace(node);
    }

    const ComputationalDag& graph = sched.getInstance().getComputationalDag();
    
    std::unordered_map<size_t, unsigned> number_children_in_superstep;
    for (const auto& node : nodes) {
        number_children_in_superstep.emplace(node,0);
        for (const auto& child : graph.children(node)) {
            if (nodes_set.find(child) == nodes_set.cend()) continue;
            number_children_in_superstep[node] += 1;
        }
    }

    std::unordered_map<size_t, unsigned> number_of_unallocated_parents_in_superstep;
    for (const auto& node : nodes) {
            number_of_unallocated_parents_in_superstep.emplace(node,0);
    }
    for (const auto& node : nodes) {
        for (const auto& child : graph.children(node)) {
            if (nodes_set.find(child) == nodes_set.cend()) continue;
            number_of_unallocated_parents_in_superstep[child] += 1;
        }
    }
    
    // smallest allocated parent in superstep, number of children in superstep, number of children, vertex id
    std::set<std::tuple<unsigned, unsigned, unsigned, size_t>> smallest_allocated_parent_in_superstep;

    // smallest allocated child's parent in superstep, number of children in superstep, number of children, vertex id
    std::set<std::tuple<unsigned, unsigned, unsigned, size_t>> smallest_allocated_childs_parent_in_superstep;
    std::unordered_map<size_t, unsigned> smallest_allocated_childs_parent_in_superstep_first_coord;

    // number of children in superstep, number of children, vertex id
    std::set<std::tuple<unsigned, unsigned, size_t>> smallest_number_of_children_in_superstep;
    for (const auto& node : nodes) {
        if (number_of_unallocated_parents_in_superstep.at(node) == 0) {
            smallest_number_of_children_in_superstep.emplace(number_children_in_superstep.at(node), graph.numberOfChildren(node), node);
            }
    }

    std::unordered_set<size_t> child_visited;
    std::unordered_set<size_t> already_scheduled;

    std::vector<size_t> new_ordering;
    new_ordering.reserve(nodes.size());

    for (size_t i = 0; i < nodes.size(); i++) {
        // getting next node
        size_t next_node; 
        if (!smallest_allocated_parent_in_superstep.empty()) {
            next_node = std::get<3>(*smallest_allocated_parent_in_superstep.begin());
            // erasing
            smallest_allocated_parent_in_superstep.erase(smallest_allocated_parent_in_superstep.begin());
            if (smallest_allocated_childs_parent_in_superstep_first_coord.find(next_node) != smallest_allocated_childs_parent_in_superstep_first_coord.cend()) {
                smallest_allocated_childs_parent_in_superstep.erase(std::make_tuple(smallest_allocated_childs_parent_in_superstep_first_coord.at(next_node), number_children_in_superstep.at(next_node), graph.numberOfChildren(next_node), next_node));
            }
            smallest_number_of_children_in_superstep.erase(std::make_tuple(number_children_in_superstep.at(next_node), graph.numberOfChildren(next_node), next_node));
        } else if (!smallest_allocated_childs_parent_in_superstep.empty()) {
            next_node = std::get<3>(*smallest_allocated_childs_parent_in_superstep.begin());
            // erasing
            smallest_allocated_childs_parent_in_superstep.erase(smallest_allocated_childs_parent_in_superstep.begin());
            smallest_number_of_children_in_superstep.erase(std::make_tuple(number_children_in_superstep.at(next_node), graph.numberOfChildren(next_node), next_node));
        } else if (!smallest_number_of_children_in_superstep.empty()) {
            next_node = std::get<2>(*smallest_number_of_children_in_superstep.begin());
            // erasing
            smallest_number_of_children_in_superstep.erase(smallest_number_of_children_in_superstep.begin());
        } else {
            throw std::runtime_error("topological_sort_for_data_locality_interior_small_relations failed");    
        }

        // adding to order
        new_ordering.emplace_back(next_node);
        already_scheduled.emplace(next_node);

        // adding to the queues
        for (const auto& child : graph.children(next_node)) {
            if (nodes_set.find(child) != nodes_set.cend()) {
                number_of_unallocated_parents_in_superstep[child] -= 1;
                if (number_of_unallocated_parents_in_superstep.at(child) == 0) {
                    smallest_number_of_children_in_superstep.emplace(number_children_in_superstep.at(child), graph.numberOfChildren(child), child);
                    smallest_allocated_parent_in_superstep.emplace(i, number_children_in_superstep.at(child), graph.numberOfChildren(child), child);
                }
            }
            if (child_visited.find(child) == child_visited.cend()) {
                child_visited.emplace(child);
                for (const auto& parent : graph.parents(child)) {
                    if (nodes_set.find(parent) == nodes_set.cend()) continue;
                    if (already_scheduled.find(parent) != already_scheduled.cend()) continue;
                    if (smallest_allocated_childs_parent_in_superstep_first_coord.find(parent) != smallest_allocated_childs_parent_in_superstep_first_coord.cend()) continue;
                    if (number_of_unallocated_parents_in_superstep.at(parent) != 0) continue;
                    smallest_allocated_childs_parent_in_superstep.emplace(i, number_children_in_superstep.at(parent), graph.numberOfChildren(parent), parent);
                    smallest_allocated_childs_parent_in_superstep_first_coord.emplace(parent, i);
                }
            }
        }
    }





    if (new_ordering.size() != nodes.size()) {
        throw std::runtime_error("topological_sort_for_data_locality_interior_small_relations failed");
    }

    // if ( ! std::is_permutation(new_ordering.cbegin(), new_ordering.cend(), nodes.cbegin(), nodes.cend()) ) {
    //     throw std::runtime_error("topological_sort_for_data_locality_interior_small_relations failed failed");
    // }

    nodes = new_ordering;
}


std::vector<size_t> schedule_node_permuter_small_relation(const BspSchedule& sched, const SCHEDULE_NODE_PERMUTATION_MODES mode) {
    // superstep, processor, nodes
    std::vector<std::vector<std::vector<size_t>>> allocation(sched.numberOfSupersteps(),
                                                                std::vector<std::vector<size_t>>(sched.getInstance().numberOfProcessors(),
                                                                    std::vector<size_t>({})));
    for (size_t node = 0; node < sched.getInstance().numberOfVertices(); node++) {
        allocation[ sched.assignedSuperstep(node) ][ sched.assignedProcessor(node) ].emplace_back(node);
    }

    // reordering and allocating into permutation
    std::vector<size_t> permutation(sched.getInstance().numberOfVertices());

    if(mode == LOOP_PROCESSORS || mode == SNAKE_PROCESSORS) {
        bool forward = true;
        size_t counter = 0;
        for (auto step_it = allocation.begin(); step_it != allocation.cend(); step_it++) {
            if (forward) {
                for (auto proc_it = step_it->begin(); proc_it != step_it->cend(); proc_it++) {
                    topological_sort_for_data_locality_interior_small_relations(*proc_it, sched);
                    
                    for (const auto& node : *proc_it) {
                        permutation[node] = counter;
                        counter++;
                    }
                }
            } else {
                for (auto proc_it = step_it->rbegin(); proc_it != step_it->crend(); proc_it++) {
                    topological_sort_for_data_locality_interior_small_relations(*proc_it, sched);
                    
                    for (const auto& node : *proc_it) {
                        permutation[node] = counter;
                        counter++;
                    }
                }
            }
            
            if (mode == SNAKE_PROCESSORS) {
                forward = !forward;
            }
        }
    } else {
        throw std::logic_error("Permutation mode not implemented.");
    }

    return permutation;
}


std::vector<size_t> schedule_node_permuter_dfs(const BspSchedule& sched, const SCHEDULE_NODE_PERMUTATION_MODES mode) {
    const ComputationalDag& graph = sched.getInstance().getComputationalDag();
    
    const unsigned& num_steps = sched.numberOfSupersteps();
    const unsigned& num_procs = sched.getInstance().numberOfProcessors();
    const unsigned& num_vertices = sched.getInstance().numberOfVertices();

    // superstep, processor, nodes
    std::vector<std::vector<std::vector<size_t>>> allocation(num_steps,
                                                                std::vector<std::vector<size_t>>(num_procs,
                                                                    std::vector<size_t>({})));
    for (size_t node = 0; node < num_vertices; node++) {
        allocation[ sched.assignedSuperstep(node) ][ sched.assignedProcessor(node) ].emplace_back(node);
    }

    std::vector<unsigned> number_of_unallocated_parents(num_vertices);
    for (const auto& vert : graph.vertices()) {
        number_of_unallocated_parents[vert] = graph.numberOfParents(vert);
    }

    std::vector<unsigned> num_children_same_superstep(num_vertices,0);
    std::vector<unsigned> num_children_next_superstep(num_vertices,0);
    for (const auto& vert : graph.vertices()) {
        for (const auto& child : graph.children(vert)) {
            if (sched.assignedSuperstep(vert) ==  sched.assignedSuperstep(child)) {
                num_children_same_superstep[vert] += 1;
            }
            if (sched.assignedSuperstep(vert) + 1 ==  sched.assignedSuperstep(child)) {
                num_children_next_superstep[vert] += 1;
            }
        }
    }

    // parent with maximal superstep+1 and then earliest occurrence
    // 0 superstep if no parent
    std::vector<std::pair<unsigned, unsigned>> priority(num_vertices, std::make_pair(0, UINT_MAX));

    // reordering and allocating into permutation
    std::vector<size_t> permutation(num_vertices);

    if(mode == LOOP_PROCESSORS || mode == SNAKE_PROCESSORS) {
        bool forward = true;
        size_t counter = 0;
        for (size_t step = 0; step < num_steps; step++) {
            if (forward) {
                unsigned node_superstep_counter = 0;
                for (unsigned proc = 0; proc < num_procs; proc++) {
                    // parent with minimal (total_supersteps - superstep), earliest occurrence, number of children in same superstep, number of children in next superstep, vertex_id
                    std::set<std::tuple<unsigned, unsigned, unsigned, unsigned, size_t>> node_queue;
                    for (const auto& vert : allocation[step][proc]) {
                        if (number_of_unallocated_parents[vert] == 0) {
                            const std::pair<unsigned, unsigned>& priority_pair = priority[vert];
                            node_queue.emplace(num_steps - priority_pair.first, priority_pair.second, num_children_same_superstep[vert], num_children_next_superstep[vert], vert);
                        }
                    }
                    while (! node_queue.empty()) {
                        size_t next_node = std::get<4>(*node_queue.begin());
                        node_queue.erase(node_queue.begin());
                        permutation[next_node] = counter;
                        counter++;
                        // resolve children
                        for (const auto& child : graph.children(next_node)) {
                            unsigned occurrence = sched.assignedProcessor(next_node) < sched.assignedProcessor(child) ? node_superstep_counter + num_vertices: node_superstep_counter;
                            if ((step+1 > priority[child].first) || ((step+1 == priority[child].first) && ( occurrence < priority[child].second ))) {
                                priority[child] = std::make_pair(step+1, occurrence);
                            }
                            number_of_unallocated_parents[child] -= 1;
                            if (number_of_unallocated_parents[child] == 0 && sched.assignedSuperstep(child) == step) {
                                const std::pair<unsigned, unsigned>& priority_pair = priority[child];
                                node_queue.emplace(num_steps - priority_pair.first, priority_pair.second, num_children_same_superstep[child], num_children_next_superstep[child], child);
                            }
                        }
                        node_superstep_counter++;
                    }
                }
            } else {
                unsigned node_superstep_counter = 0;
                for (unsigned proc = num_procs-1; proc < num_procs; proc--) {
                    // parent with minimal (total_supersteps - superstep), earliest occurrence, number of children in same superstep, number of children in next superstep, vertex_id
                    std::set<std::tuple<unsigned, unsigned, unsigned, unsigned, size_t>> node_queue;
                    for (const auto& vert : allocation[step][proc]) {
                        if (number_of_unallocated_parents[vert] == 0) {
                            const std::pair<unsigned, unsigned>& priority_pair = priority[vert];
                            node_queue.emplace(num_steps - priority_pair.first, priority_pair.second, num_children_same_superstep[vert], num_children_next_superstep[vert], vert);
                        }
                    }
                    while (! node_queue.empty()) {
                        size_t next_node = std::get<4>(*node_queue.begin());
                        node_queue.erase(node_queue.begin());
                        permutation[next_node] = counter;
                        counter++;
                        // resolve children
                        for (const auto& child : graph.children(next_node)) {
                            unsigned occurrence = sched.assignedProcessor(next_node) > sched.assignedProcessor(child) ? node_superstep_counter + num_vertices: node_superstep_counter;
                            if ((step+1 > priority[child].first) || ((step+1 == priority[child].first) && ( occurrence < priority[child].second ))) {
                                priority[child] = std::make_pair(step+1, occurrence);
                            }
                            number_of_unallocated_parents[child] -= 1;
                            if (number_of_unallocated_parents[child] == 0 && sched.assignedSuperstep(child) == step) {
                                const std::pair<unsigned, unsigned>& priority_pair = priority[child];
                                node_queue.emplace(num_steps - priority_pair.first, priority_pair.second, num_children_same_superstep[child], num_children_next_superstep[child], child);
                            }
                        }
                        node_superstep_counter++;
                    }
                }
            }
            
            if (mode == SNAKE_PROCESSORS) {
                forward = !forward;
            }
        }
    } else {
        throw std::logic_error("Permutation mode not implemented.");
    }

    return permutation;
}