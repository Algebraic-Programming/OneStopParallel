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
#include <unordered_map>
#include <vector>

#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"

namespace osp {

template <typename GraphT>
struct CmVertex {
    using VertexType = VertexIdxT<GraphT>;
    VertexType vertex_;

    VertexType parentPosition_;

    VertexType degree_;

    CmVertex() : vertex_(0), parentPosition_(0), degree_(0) {}

    CmVertex(VertexType vertex, VertexType degree, VertexType parentPosition)
        : vertex_(vertex), parentPosition_(parentPosition), degree_(degree) {}

    bool operator<(CmVertex const &rhs) const {
        return (parentPosition_ < rhs.parentPosition_) || (parentPosition_ == rhs.parentPosition_ and degree_ < rhs.degree_)
               || (parentPosition_ == rhs.parentPosition_ and degree_ == rhs.degree_ and vertex_ < rhs.vertex_);
    }
};

template <typename GraphT>
std::vector<VertexIdxT<GraphT>> CuthillMckeeWavefront(const GraphT &dag, bool permutation = false) {
    using VertexType = VertexIdxT<GraphT>;
    using CmVertex = CmVertex<GraphT>;

    std::vector<VertexType> result(dag.NumVertices());
    std::vector<unsigned> predecessorsCount(dag.NumVertices(), 0);
    std::vector<VertexType> predecessorsPosition(dag.NumVertices(), dag.NumVertices());

    std::vector<CmVertex> currentWavefront;
    for (const auto &source : source_vertices_view(dag)) {
        currentWavefront.push_back(CmVertex(source, dag.OutDegree(source), 0));
    }

    std::vector<CmVertex> newWavefront;
    VertexType nodeCounter = 0;
    while (node_counter < dag.NumVertices()) {
        newWavefront.clear();
        std::sort(currentWavefront.begin(), currentWavefront.end());

        if (permutation) {
            for (VertexType i = 0; i < static_cast<VertexType>(currentWavefront.size()); i++) {
                result[currentWavefront[i].vertex] = node_counter + i;
            }
        } else {
            for (size_t i = 0; i < currentWavefront.size(); i++) {
                result[node_counter + i] = currentWavefront[i].vertex;
            }
        }

        if (nodeCounter + static_cast<VertexType>(currentWavefront.size()) == dag.NumVertices()) {
            break;
        }

        for (VertexType i = 0; i < static_cast<VertexType>(currentWavefront.size()); i++) {
            for (const auto &child : dag.Children(current_wavefront[i].vertex)) {
                predecessors_count[child]++;
                predecessors_position[child] = std::min(predecessors_position[child], node_counter + i);

                if (predecessors_count[child] == dag.InDegree(child)) {
                    new_wavefront.push_back(cm_vertex(child, dag.OutDegree(child), predecessors_position[child]));
                }
            }
        }

        nodeCounter += static_cast<VertexType>(currentWavefront.size());

        std::swap(currentWavefront, newWavefront);
    }

    return result;
}

template <typename GraphT>
std::vector<VertexIdxT<GraphT>> CuthillMckeeUndirected(const GraphT &dag, bool startAtSink, bool perm = false) {
    using VertexType = VertexIdxT<GraphT>;
    using CmVertex = CmVertex<GraphT>;

    std::vector<VertexType> cmOrder(dag.NumVertices());

    std::unordered_map<VertexType, unsigned> maxNodeDistances;
    VertexType firstNode = 0;

    // compute bottom or top node distances of sink or source nodes, store node with the largest distance in first_node
    if (startAtSink) {
        unsigned maxDistance = 0;
        const std::vector<unsigned> topNodeDistance = GetTopNodeDistance(dag);
        for (const auto &i : dag.Vertices()) {
            if (IsSink(i, dag)) {
                maxNodeDistances[i] = topNodeDistance[i];

                if (topNodeDistance[i] > maxDistance) {
                    maxDistance = topNodeDistance[i];
                    firstNode = i;
                }
            }
        }
    } else {
        unsigned maxDistance = 0;
        const std::vector<unsigned> bottomNodeDistance = GetBottomNodeDistance(dag);
        for (const auto &i : dag.Vertices()) {
            if (IsSource(i, dag)) {
                maxNodeDistances[i] = bottomNodeDistance[i];

                if (bottomNodeDistance[i] > maxDistance) {
                    maxDistance = bottomNodeDistance[i];
                    firstNode = i;
                }
            }
        }
    }

    if (perm) {
        cmOrder[firstNode] = 0;
    } else {
        cmOrder[0] = firstNode;
    }

    std::unordered_set<VertexType> visited;
    visited.insert(firstNode);

    std::vector<CmVertex> currentLevel;
    currentLevel.reserve(dag.InDegree(firstNode) + dag.OutDegree(firstNode));

    for (const auto &child : dag.Children(firstNode)) {
        currentLevel.push_back(CmVertex(child, dag.InDegree(child) + dag.OutDegree(child), 0));
        visited.insert(child);
    }

    for (const auto &parent : dag.Parents(firstNode)) {
        currentLevel.push_back(CmVertex(parent, dag.InDegree(parent) + dag.OutDegree(parent), 0));
        visited.insert(parent);
    }

    VertexType nodeCounter = 1;
    while (nodeCounter < dag.NumVertices()) {
        std::sort(currentLevel.begin(), currentLevel.end());

        if (perm) {
            for (VertexType i = 0; i < currentLevel.size(); i++) {
                cmOrder[currentLevel[i].vertex] = nodeCounter + i;
            }
        } else {
            for (VertexType i = 0; i < currentLevel.size(); i++) {
                cmOrder[nodeCounter + i] = currentLevel[i].vertex;
            }
        }

        if (nodeCounter + currentLevel.size() == dag.NumVertices()) {
            break;
        }

        std::unordered_map<VertexType, VertexType> nodePriority;

        for (VertexType i = 0; i < currentLevel.size(); i++) {
            for (const auto &child : dag.Children(currentLevel[i].vertex)) {
                if (visited.find(child) == visited.end()) {
                    if (nodePriority.find(child) == nodePriority.end()) {
                        nodePriority[child] = nodeCounter + i;
                    } else {
                        nodePriority[child] = std::min(nodePriority[child], nodeCounter + i);
                    }
                }
            }

            for (const auto &parent : dag.Parents(currentLevel[i].vertex)) {
                if (visited.find(parent) == visited.end()) {
                    if (nodePriority.find(parent) == nodePriority.end()) {
                        nodePriority[parent] = nodeCounter + i;
                    } else {
                        nodePriority[parent] = std::min(nodePriority[parent], nodeCounter + i);
                    }
                }
            }
        }

        nodeCounter += currentLevel.size();

        if (nodePriority.empty()) {    // the dag has more than one connected components

            unsigned maxDistance = 0;
            for (const auto [node, distance] : maxNodeDistances) {
                if (visited.find(node) == visited.end() and distance > maxDistance) {
                    maxDistance = distance;
                    firstNode = node;
                }
            }

            if (perm) {
                cmOrder[firstNode] = nodeCounter;
            } else {
                cmOrder[nodeCounter] = firstNode;
            }
            visited.insert(firstNode);

            currentLevel.clear();
            currentLevel.reserve(dag.InDegree(firstNode) + dag.OutDegree(firstNode));

            for (const auto &child : dag.Children(firstNode)) {
                currentLevel.push_back(CmVertex(child, dag.InDegree(child) + dag.OutDegree(child), nodeCounter));
                visited.insert(child);
            }

            for (const auto &parent : dag.Parents(firstNode)) {
                currentLevel.push_back(CmVertex(parent, dag.InDegree(parent) + dag.OutDegree(parent), nodeCounter));
                visited.insert(parent);
            }

            nodeCounter++;

        } else {
            currentLevel.clear();
            currentLevel.reserve(nodePriority.size());

            for (const auto &[node, priority] : nodePriority) {
                currentLevel.push_back(CmVertex(node, dag.InDegree(node) + dag.OutDegree(node), priority));
                visited.insert(node);
            }
        }
    }

    return cm_order;
}

// Cuthill-McKee Wavefront
template <typename GraphT>
inline std::vector<VertexIdxT<GraphT>> GetTopOrderCuthillMcKeeWavefront(const GraphT &dag) {
    std::vector<VertexIdxT<GraphT>> order;
    if (dag.NumVertices() > 0) {
        std::vector<VertexIdxT<GraphT>> priority = CuthillMcKeeWavefront(dag);
        order.reserve(dag.NumVertices());
        for (const auto &v : PriorityVecTopSortView(dag, priority)) {
            order.push_back(v);
        }
    }
    return order;
}

// Cuthill-McKee Undirected
template <typename GraphT>
inline std::vector<VertexIdxT<GraphT>> GetTopOrderCuthillMcKeeUndirected(const GraphT &dag) {
    std::vector<VertexIdxT<GraphT>> order;
    if (dag.NumVertices() > 0) {
        std::vector<VertexIdxT<GraphT>> priority = CuthillMcKeeUndirected(dag, true, true);
        order.reserve(dag.NumVertices());
        for (const auto &v : PriorityVecTopSortView(dag, priority)) {
            order.push_back(v);
        }
    }
    return order;
}

}    // namespace osp
