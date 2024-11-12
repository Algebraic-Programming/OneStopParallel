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

#include "scheduler/Coarsers/FunnelSSP.hpp"

bool FunnelSSP::isCompatibleNodeType(const VertexType& new_node, const VertexType& old_node, const BspInstance& instance) {
    for (unsigned proc_type = 0; proc_type < instance.getArchitecture().getNumberOfProcessorTypes(); proc_type++) {
        if (instance.isCompatibleType(instance.getComputationalDag().nodeType(old_node), proc_type)
            && (!instance.isCompatibleType(instance.getComputationalDag().nodeType(new_node), proc_type))) {
            return false;
        }
    }
    return true;
}

void FunnelSSP::expand_in_group_dfs(const std::unordered_set<EdgeType, EdgeType_hash>& edge_mask, std::unordered_set<VertexType>& group, std::unordered_map<VertexType, unsigned>& children_not_in_group, long unsigned& group_weight, const double& max_weight, const VertexType active_node, const VertexType sink_node, bool& failed_to_add) {
    if (failed_to_add) return;
    const ComputationalDag& graph = dag_history.back()->getComputationalDag();

    for (const EdgeType in_edge : graph.in_edges(active_node)) {
        if (parameters.use_approx_transitive_reduction && (edge_mask.find(in_edge) != edge_mask.cend())) continue;
        const VertexType& par = in_edge.m_source;
        if (children_not_in_group.find(par) != children_not_in_group.cend()) {
            children_not_in_group[par] -= 1;
        } else {
            if (parameters.use_approx_transitive_reduction) {
                children_not_in_group[par] = 0;
                for (const EdgeType out_edge : graph.out_edges(par)) {
                    if (edge_mask.find(out_edge) != edge_mask.cend()) continue;
                    children_not_in_group[par] += 1;
                }
            } else {
                children_not_in_group[par] = graph.numberOfChildren(par);
            }
            children_not_in_group[par] -= 1;
        }
    }
    for (const EdgeType in_edge : graph.in_edges(active_node)) {
        if (parameters.use_approx_transitive_reduction && (edge_mask.find(in_edge) != edge_mask.cend())) continue;
        const VertexType& par = in_edge.m_source;
        if (children_not_in_group[par] == 0) {
            if (!isCompatibleNodeType(par, sink_node, *dag_history.back())) continue;
            if (group_weight + graph.nodeWorkWeight(par) > max_weight) {
                failed_to_add = true;
                return;
            } else {
                group_weight += graph.nodeWorkWeight(par);
                group.emplace(par);
                expand_in_group_dfs(edge_mask, group, children_not_in_group, group_weight, max_weight, par, sink_node, failed_to_add);
                if (failed_to_add) return;
            }
        }
    }
}


void FunnelSSP::expand_out_group_dfs(const std::unordered_set<EdgeType, EdgeType_hash>& edge_mask, std::unordered_set<VertexType>& group, std::unordered_map<VertexType, unsigned>& parents_not_in_group, long unsigned& group_weight, const double& max_weight, const VertexType active_node, const VertexType source_node, bool& failed_to_add) {
    if (failed_to_add) return;
    const ComputationalDag& graph = dag_history.back()->getComputationalDag();

    for (const EdgeType out_edge : graph.out_edges(active_node)) {
        if (parameters.use_approx_transitive_reduction && (edge_mask.find(out_edge) != edge_mask.cend())) continue;
        const VertexType& child = out_edge.m_target;
        if (parents_not_in_group.find(child) != parents_not_in_group.cend()) {
            parents_not_in_group[child] -= 1;
        } else {
            if (parameters.use_approx_transitive_reduction) {
                parents_not_in_group[child] = 0;
                for (const EdgeType in_edge : graph.in_edges(child)) {
                    if (edge_mask.find(in_edge) != edge_mask.cend()) continue;
                    parents_not_in_group[child] += 1;
                }
            } else {
                parents_not_in_group[child] = graph.numberOfParents(child);
            }
            parents_not_in_group[child] -= 1;
        }
    }
    for (const EdgeType out_edge : graph.out_edges(active_node)) {
        if (parameters.use_approx_transitive_reduction && (edge_mask.find(out_edge) != edge_mask.cend() )) continue;
        const VertexType& child = out_edge.m_target;
        if (parents_not_in_group[child] == 0) {
            if (!isCompatibleNodeType(child, source_node, *dag_history.back())) continue;
            if (group_weight + graph.nodeWorkWeight(child) > max_weight) {
                failed_to_add = true;
                return;
            } else {
                group_weight += graph.nodeWorkWeight(child);
                group.emplace(child);
                expand_in_group_dfs(edge_mask, group, parents_not_in_group, group_weight, max_weight, child, source_node, failed_to_add);
                if (failed_to_add) return;
            }
        }
    }
}


void FunnelSSP::run_in_contraction() {
    const ComputationalDag& graph = dag_history.back()->getComputationalDag();

    const std::unordered_set<EdgeType, EdgeType_hash> edge_mask = parameters.use_approx_transitive_reduction ? graph.long_edges_in_triangles_parallel() : std::unordered_set<EdgeType, EdgeType_hash>();
    
    std::multiset<int> weights;
    for (VertexType node = 0; node < graph.numberOfVertices(); node++) {
        weights.emplace(graph.nodeWorkWeight(node));
    }
    int median = Get_Median(weights);

    std::vector<std::unordered_set<VertexType>> partition;
    std::vector<bool> visited(graph.numberOfVertices(), false);

    std::vector<VertexType> top_order = graph.GetTopOrder();

    for (auto rev_top_it = top_order.rbegin(); rev_top_it != top_order.crend(); rev_top_it++) {
        const VertexType& bottom_node = *rev_top_it;
        if (visited[bottom_node]) continue;

        bool failed_to_add = false;
        long unsigned weight_of_group = graph.nodeWorkWeight(bottom_node);
        std::unordered_map<VertexType, unsigned> children_not_in_group;
        std::unordered_set<VertexType> group({bottom_node});

        expand_in_group_dfs(edge_mask, group, children_not_in_group, weight_of_group, parameters.max_relative_weight * (double)median, bottom_node, bottom_node, failed_to_add);
        partition.push_back(group);

        for (const auto& node : group) {
            visited[node] = true;
        }
    }

    add_contraction(partition);
}


void FunnelSSP::run_out_contraction() {
    const ComputationalDag& graph = dag_history.back()->getComputationalDag();

    const std::unordered_set<EdgeType, EdgeType_hash> edge_mask = parameters.use_approx_transitive_reduction ? graph.long_edges_in_triangles_parallel() : std::unordered_set<EdgeType, EdgeType_hash>();
    
    std::multiset<int> weights;
    for (VertexType node = 0; node < graph.numberOfVertices(); node++) {
        weights.emplace(graph.nodeWorkWeight(node));
    }
    int median = Get_Median(weights);

    std::vector<std::unordered_set<VertexType>> partition;
    std::vector<bool> visited(graph.numberOfVertices(), false);

    std::vector<VertexType> top_order = graph.GetTopOrder();

    for (auto top_it = top_order.begin(); top_it != top_order.cend(); top_it++) {
        const VertexType& top_node = *top_it;
        if (visited[top_node]) continue;

        bool failed_to_add = false;
        long unsigned weight_of_group = graph.nodeWorkWeight(top_node);
        std::unordered_map<VertexType, unsigned> parents_not_in_group;
        std::unordered_set<VertexType> group({top_node});

        std::unordered_set<VertexType> sack;
        for (const VertexType& par : graph.parents(top_node)) {
            sack.emplace(par);
        }

        expand_out_group_dfs(edge_mask, group, parents_not_in_group, weight_of_group, parameters.max_relative_weight * (double)median, top_node, top_node, failed_to_add);
        partition.push_back(group);

        for (const auto& node : group) {
            visited[node] = true;
        }
    }

    add_contraction(partition);
}




RETURN_STATUS FunnelSSP::run_contractions() {
    std::cout   << "Coarsen Step: " << dag_history.size()
                << ", Number of nodes: " << dag_history.back()->numberOfVertices()
                << ", Number of edges: " << dag_history.back()->getComputationalDag().numberOfEdges()
                << ", Log ratio: " << std::log(dag_history.back()->getComputationalDag().numberOfEdges()) / std::log(dag_history.back()->numberOfVertices()) << std::endl;

    if (parameters.funnel_incoming && parameters.first_funnel_incoming) {
        run_in_contraction();

        std::cout   << "Coarsen Step: " << dag_history.size()
                << ", Number of nodes: " << dag_history.back()->numberOfVertices()
                << ", Number of edges: " << dag_history.back()->getComputationalDag().numberOfEdges()
                << ", Log ratio: " << std::log(dag_history.back()->getComputationalDag().numberOfEdges()) / std::log(dag_history.back()->numberOfVertices()) << std::endl;
    }

    if (parameters.funnel_outgoing) {
        run_out_contraction();

        std::cout   << "Coarsen Step: " << dag_history.size()
                << ", Number of nodes: " << dag_history.back()->numberOfVertices()
                << ", Number of edges: " << dag_history.back()->getComputationalDag().numberOfEdges()
                << ", Log ratio: " << std::log(dag_history.back()->getComputationalDag().numberOfEdges()) / std::log(dag_history.back()->numberOfVertices()) << std::endl;
    }

    if (parameters.funnel_incoming && (! parameters.first_funnel_incoming)) {
        run_in_contraction();

        std::cout   << "Coarsen Step: " << dag_history.size()
                << ", Number of nodes: " << dag_history.back()->numberOfVertices()
                << ", Number of edges: " << dag_history.back()->getComputationalDag().numberOfEdges()
                << ", Log ratio: " << std::log(dag_history.back()->getComputationalDag().numberOfEdges()) / std::log(dag_history.back()->numberOfVertices()) << std::endl;
    }

    return SUCCESS;
}
