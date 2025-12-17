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

#include <limits>
#include <queue>
#include <vector>

#include "DagDivider.hpp"
#include "osp/concepts/graph_traits.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"

namespace osp {

template <typename GraphT, typename ConstrGraphT>
class ConnectedComponentDivider : public IDagDivider<GraphT> {
    static_assert(isComputationalDagV<GraphT>, "Graph must be a computational DAG");
    static_assert(isComputationalDagV<ConstrGraphT>, "ConstrGraphT must be a computational DAG");
    static_assert(isConstructableCdagV<ConstrGraphT>, "ConstrGraphT must satisfy the constructable_cdag_vertex concept");
    static_assert(std::is_same_v<VertexIdxT<GraphT>, VertexIdxT<ConstrGraphT>>,
                  "GraphT and ConstrGraphT must have the same VertexIdx types");

  private:
    using VertexIdx = VertexIdxT<GraphT>;

    std::vector<ConstrGraphT> subDags_;

    // For each component: local_idx -> global vertex
    std::vector<std::vector<VertexIdx>> vertexMapping_;

    // Global vertex -> local index
    std::vector<VertexIdx> vertexMap_;

    // Global vertex -> component id
    std::vector<unsigned> component_;

  public:
    inline std::vector<ConstrGraphT> &GetSubDags() { return subDags_; }

    inline const std::vector<ConstrGraphT> &GetSubDags() const { return subDags_; }

    inline const std::vector<std::vector<VertexIdx>> &GetVertexMapping() const { return vertexMapping_; }

    inline const std::vector<unsigned> &GetComponent() const { return component_; }

    inline const std::vector<VertexIdx> &GetVertexMap() const { return vertexMap_; }

    virtual std::vector<std::vector<std::vector<VertexIdxT<GraphT>>>> Divide(const GraphT &dag) override {
        if (dag.NumVertices() == 0) {
            return {};
        }

        bool hasMoreThanOneConnectedComponent = ComputeConnectedComponents(dag);

        std::vector<std::vector<std::vector<VertexIdxT<GraphT>>>> vertexMaps(1);

        if (hasMoreThanOneConnectedComponent) {
            vertexMaps[0].resize(subDags_.size());
            for (unsigned i = 0; i < subDags_.size(); ++i) {
                vertexMaps[0][i].resize(subDags_[i].NumVertices());
            }

            for (const auto &v : dag.Vertices()) {
                vertexMaps[0][component_[v]][vertexMap_[v]] = v;
            }
        } else {
            subDags_.resize(1);
            subDags_[0] = dag;
            vertexMapping_.resize(1);
            vertexMapping_[0].resize(dag.NumVertices());
            vertexMap_.resize(dag.NumVertices());

            vertexMaps[0].resize(1);
            vertexMaps[0][0].resize(dag.NumVertices());
            for (const auto &v : dag.Vertices()) {
                vertexMaps[0][0][v] = v;
                vertexMap_[v] = v;
                vertexMapping_[0][v] = v;
            }
        }

        return vertexMaps;
    }

    std::vector<std::vector<std::vector<VertexIdxT<GraphT>>>> ComputeVertexMaps(const GraphT &dag) {
        std::vector<std::vector<std::vector<VertexIdxT<GraphT>>>> vertexMaps(1);

        vertexMaps[0].resize(subDags_.size());
        for (unsigned i = 0; i < subDags_.size(); ++i) {
            vertexMaps[0][i].resize(subDags_[i].NumVertices());
        }

        for (const auto &v : dag.Vertices()) {
            vertexMaps[0][component_[v]][vertexMap_[v]] = v;
        }

        return vertexMaps;
    }

    bool ComputeConnectedComponents(const GraphT &dag) {
        // Clear previous state
        subDags_.clear();
        vertexMapping_.clear();
        vertexMap_.clear();
        component_.assign(dag.NumVertices(), std::numeric_limits<unsigned>::max());

        if (dag.NumVertices() == 0) {
            return false;
        }

        unsigned componentId = 0;
        for (const auto &v : dag.Vertices()) {
            if (component_[v] == std::numeric_limits<unsigned>::max()) {
                component_[v] = componentId;

                // BFS for weakly connected component
                std::queue<VertexIdx> q;
                q.push(v);

                while (!q.empty()) {
                    VertexIdx current = q.front();
                    q.pop();

                    for (const auto &child : dag.Children(current)) {
                        if (component_[child] == std::numeric_limits<unsigned>::max()) {
                            q.push(child);
                            component_[child] = componentId;
                        }
                    }

                    for (const auto &parent : dag.Parents(current)) {
                        if (component_[parent] == std::numeric_limits<unsigned>::max()) {
                            q.push(parent);
                            component_[parent] = componentId;
                        }
                    }
                }

                ++componentId;
            }
        }

        if (componentId == 1) {
            // Single component: no need to build sub_dags or maps
            return false;
        }

        subDags_ = CreateInducedSubgraphs<GraphT, ConstrGraphT>(dag, component_);

        // Create the mappings between global and local vertex indices.
        vertexMapping_.resize(subDags_.size());
        vertexMap_.resize(dag.NumVertices());

        std::vector<VertexIdx> currentIndexInSubdag(subDags_.size(), 0);
        for (const auto &v : dag.Vertices()) {
            unsigned compId = component_[v];
            VertexIdx localIdx = currentIndexInSubdag[compId]++;
            vertexMap_[v] = localIdx;

            if (vertexMapping_[compId].empty()) {
                vertexMapping_[compId].resize(subDags_[compId].NumVertices());
            }

            vertexMapping_[compId][localIdx] = v;
        }

        return true;
    }
};

}    // namespace osp
