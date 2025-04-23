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

#include "model/dag_algorithms/top_sort.hpp"

std::vector<VertexType> dag_algorithms::top_sort_dfs(const ComputationalDag &dag) {

    std::vector<VertexType> top_order;
    top_order.reserve(dag.numberOfVertices());
    std::vector<bool> visited(dag.numberOfVertices(), false);

    std::function<void(VertexType)> dfs_visit = [&](VertexType node) {
        visited[node] = true;
        for (const VertexType &child : dag.children(node)) {
            if (!visited[child])
                dfs_visit(child);
        }
        top_order.push_back(node);
    };

    for (const VertexType &i : dag.sourceVertices())
        dfs_visit(i);

    std::reverse(top_order.begin(), top_order.end());
    return top_order;
}

std::vector<VertexType> dag_algorithms::top_sort_dfs(const ComputationalDag &dag, std::vector<unsigned> &source_node_dist) {

    std::vector<VertexType> top_order;
    top_order.reserve(dag.numberOfVertices());
    source_node_dist.resize(dag.numberOfVertices(), 0);

    std::vector<bool> visited(dag.numberOfVertices(), false);

    std::function<void(VertexType,unsigned)> dfs_visit = [&](VertexType node, unsigned depth) {
        visited[node] = true;
        
        for (const VertexType &child : dag.children(node)) {
            
            source_node_dist[child] = std::max(depth+1, source_node_dist[child]);
            
            if (!visited[child])
                dfs_visit(child, depth + 1);
        }
        top_order.push_back(node);
    };

    for (const VertexType &i : dag.sourceVertices())
        dfs_visit(i, 0);

    std::reverse(top_order.begin(), top_order.end());
    return top_order;
}

std::vector<VertexType> dag_algorithms::top_sort_bfs(const ComputationalDag &dag) {

    std::vector<VertexType> predecessors_count(dag.numberOfVertices(), 0);
    std::vector<VertexType> top_order(dag.numberOfVertices(), 0);
    std::vector<std::queue<VertexType>> next(dag.getNumberOfNodeTypes());

    for (const VertexType &i : dag.sourceVertices()) {
        next[dag.nodeType(i)].push(i);
    }

    size_t idx = 0;
    unsigned current_node_type = 0;

    while (idx < dag.numberOfVertices()) {

        while (not next[current_node_type].empty()) {
            const VertexType node = next[current_node_type].front();
            next[current_node_type].pop();
            top_order[idx++] = node;

            for (const VertexType &current : dag.children(node)) {
                ++predecessors_count[current];
                if (predecessors_count[current] == dag.numberOfParents(current))
                    next[dag.nodeType(current)].push(current);
            }
        }

        current_node_type = (current_node_type + 1) % dag.getNumberOfNodeTypes();
    }

    return top_order;
}

std::vector<VertexType> dag_algorithms::top_sort_locality(const ComputationalDag &dag) {

    std::vector<unsigned> priority(dag.numberOfVertices(), 0);
    std::iota(priority.begin(), priority.end(), 0);

    return top_sort_priority_node_type<unsigned>(dag, priority);
}

std::vector<VertexType> dag_algorithms::top_sort_max_children(const ComputationalDag &dag) {

    std::vector<unsigned> priority(dag.numberOfVertices(), 0);

    for (const VertexType &v : dag.vertices()) {
        priority[v] = dag.numberOfChildren(v);
    }

    return top_sort_priority_node_type<unsigned>(dag, priority);
}

std::vector<VertexType> dag_algorithms::top_sort_random(const ComputationalDag &dag) {

    std::vector<unsigned> priority(dag.numberOfVertices(), 0);

    std::iota(priority.begin(), priority.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(priority.begin(), priority.end(), g);

    return top_sort_priority_node_type<unsigned>(dag, priority);
}

std::vector<VertexType> dag_algorithms::top_sort_heavy_edges(const ComputationalDag &dag, bool sum) {

    std::vector<int> priority(dag.numberOfVertices(), 0);

    if (sum) {
        for (const VertexType &v : dag.vertices()) {
            int com = 0;

            for (const EdgeType &e : dag.in_edges(v))
                com += dag.edgeCommunicationWeight(e);

            priority[v] = com;
        }

    } else {
        
        for (const VertexType &v : dag.vertices()) {
            int com = 0;

            for (const EdgeType &e : dag.in_edges(v))
                com = std::max(com, dag.edgeCommunicationWeight(e));

            priority[v] = com;
        }
    }

    return top_sort_priority_node_type<int>(dag, priority);
}
