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

#include "scheduler/Coarsers/FunnelBfsSSP.hpp"

bool FunnelBfsSSP::isCompatibleNodeType(const VertexType& new_node, const VertexType& old_node, const BspInstance& instance) {
    for (unsigned proc_type = 0; proc_type < instance.getArchitecture().getNumberOfProcessorTypes(); proc_type++) {
        if (instance.isCompatibleType(instance.getComputationalDag().nodeType(old_node), proc_type)
            && (!(instance.isCompatibleType(instance.getComputationalDag().nodeType(new_node), proc_type)))) {
            return false;
        }
    }
    return true;
}



void FunnelBfsSSP::run_in_contraction() {
    const ComputationalDag& graph = dag_history.back()->getComputationalDag();
    const BspArchitecture &arch = dag_history.back()->getArchitecture();

    const std::unordered_set<EdgeType, EdgeType_hash> edge_mask = parameters.use_approx_transitive_reduction ? graph.long_edges_in_triangles_parallel() : std::unordered_set<EdgeType, EdgeType_hash>();

    std::vector<std::unordered_set<VertexType>> partition;
    std::vector<bool> visited(graph.numberOfVertices(), false);

    std::vector<unsigned> max_memory_per_vertex_type;
    if (use_architecture_memory_contraints) {
        max_memory_per_vertex_type = std::vector<unsigned>(graph.getNumberOfNodeTypes(), 0);
        for (unsigned proc = 0; proc < arch.numberOfProcessors(); proc++) {
            for (unsigned vert_type = 0; vert_type < graph.getNumberOfNodeTypes(); vert_type++) {
                if (dag_history.back()->isCompatibleType(vert_type, arch.processorType(proc))) {
                    max_memory_per_vertex_type[vert_type] = std::max(max_memory_per_vertex_type[vert_type], arch.memoryBound(proc));
                }
            }
        }
    }

    std::vector<VertexType> top_order = graph.GetTopOrder();

    for (auto rev_top_it = top_order.rbegin(); rev_top_it != top_order.crend(); rev_top_it++) {
        const VertexType& bottom_node = *rev_top_it;
        if (visited[bottom_node]) continue;

        long unsigned work_weight_of_group = 0;
        long unsigned memory_weight_of_group = 0;
        std::unordered_map<VertexType, unsigned> children_not_in_group;
        std::unordered_set<VertexType> group;

        std::deque<VertexType> vertex_processing_fifo({bottom_node});
        std::deque<VertexType> next_vertex_processing_fifo;
        unsigned depth_counter = 0;

        while ((!vertex_processing_fifo.empty()) || (!next_vertex_processing_fifo.empty())) {
            if (vertex_processing_fifo.empty()) {
                vertex_processing_fifo = next_vertex_processing_fifo;
                next_vertex_processing_fifo.clear();
                depth_counter++;
                if (depth_counter > parameters.max_depth) {
                    break;
                }
            }
            VertexType active_node = vertex_processing_fifo.front();
            vertex_processing_fifo.pop_front();

            if (!isCompatibleNodeType(active_node, bottom_node, *(dag_history.back()))) continue;
            if (work_weight_of_group + graph.nodeWorkWeight(active_node) > parameters.max_work_weight) continue;
            if (memory_weight_of_group + graph.nodeMemoryWeight(active_node) > parameters.max_memory_weight) continue;
            if (use_architecture_memory_contraints && (memory_weight_of_group + graph.nodeMemoryWeight(active_node) > max_memory_per_vertex_type[graph.nodeType(bottom_node)])) continue;

            group.emplace(active_node);
            work_weight_of_group += graph.nodeWorkWeight(active_node);
            memory_weight_of_group += graph.nodeMemoryWeight(active_node);

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
                    next_vertex_processing_fifo.emplace_back(par);
                }
            }
        }
        
        partition.push_back(group);

        for (const auto& node : group) {
            visited[node] = true;
        }
    }

    add_contraction(partition);
}


void FunnelBfsSSP::run_out_contraction() {
    const ComputationalDag& graph = dag_history.back()->getComputationalDag();
    const BspArchitecture &arch = dag_history.back()->getArchitecture();

    const std::unordered_set<EdgeType, EdgeType_hash> edge_mask = parameters.use_approx_transitive_reduction ? graph.long_edges_in_triangles_parallel() : std::unordered_set<EdgeType, EdgeType_hash>();

    std::vector<std::unordered_set<VertexType>> partition;
    std::vector<bool> visited(graph.numberOfVertices(), false);

    std::vector<unsigned> max_memory_per_vertex_type;
    if (use_architecture_memory_contraints) {
        max_memory_per_vertex_type = std::vector<unsigned>(graph.getNumberOfNodeTypes(), 0);
        for (unsigned proc = 0; proc < arch.numberOfProcessors(); proc++) {
            for (unsigned vert_type = 0; vert_type < graph.getNumberOfNodeTypes(); vert_type++) {
                if (dag_history.back()->isCompatibleType(vert_type, arch.processorType(proc))) {
                    max_memory_per_vertex_type[vert_type] = std::max(max_memory_per_vertex_type[vert_type], arch.memoryBound(proc));
                }
            }
        }
    }

    std::vector<VertexType> top_order = graph.GetTopOrder();

    for (auto top_it = top_order.begin(); top_it != top_order.cend(); top_it++) {
        const VertexType& top_node = *top_it;
        if (visited[top_node]) continue;

        long unsigned work_weight_of_group = 0;
        long unsigned memory_weight_of_group = 0;
        std::unordered_map<VertexType, unsigned> parents_not_in_group;
        std::unordered_set<VertexType> group;

        std::deque<VertexType> vertex_processing_fifo({top_node});
        std::deque<VertexType> next_vertex_processing_fifo;
        unsigned depth_counter = 0;

        while ((!vertex_processing_fifo.empty()) || (!next_vertex_processing_fifo.empty())) {
            if (vertex_processing_fifo.empty()) {
                vertex_processing_fifo = next_vertex_processing_fifo;
                next_vertex_processing_fifo.clear();
                depth_counter++;
                if (depth_counter > parameters.max_depth) {
                    break;
                }
            }
            VertexType active_node = vertex_processing_fifo.front();
            vertex_processing_fifo.pop_front();

            if (!isCompatibleNodeType(active_node, top_node, *(dag_history.back()))) continue;
            if (work_weight_of_group + graph.nodeWorkWeight(active_node) > parameters.max_work_weight) continue;
            if (memory_weight_of_group + graph.nodeMemoryWeight(active_node) > parameters.max_memory_weight) continue;
            if (use_architecture_memory_contraints && (memory_weight_of_group + graph.nodeMemoryWeight(active_node) > max_memory_per_vertex_type[graph.nodeType(top_node)])) continue;

            group.emplace(active_node);
            work_weight_of_group += graph.nodeWorkWeight(active_node);
            memory_weight_of_group += graph.nodeMemoryWeight(active_node);

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
                    next_vertex_processing_fifo.emplace_back(child);
                }
            }
        }

        partition.push_back(group);

        for (const auto& node : group) {
            visited[node] = true;
        }
    }

    add_contraction(partition);
}




RETURN_STATUS FunnelBfsSSP::run_contractions() {
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
