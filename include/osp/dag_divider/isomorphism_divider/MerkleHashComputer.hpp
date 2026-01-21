/*
Copyright 2024 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

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
#include <set>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "osp/auxiliary/hash_util.hpp"
#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/dag_divider/isomorphism_divider/HashComputer.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"

namespace osp {

/**
 * @class MerkleHashComputer
 * @brief Computes Merkle hashes for graph vertices to identify isomorphic orbits.
 *
 * The Merkle hash of a vertex is computed recursively based on its own properties
 * and the sorted hashes of its parents (or children, depending on the `forward` template parameter).
 * This allows for the identification of structurally isomorphic subgraphs.
 *
 * @tparam GraphT The type of the graph, must satisfy the `directed_graph` concept.
 * @tparam NodeHashFuncT A functor that computes a hash for a single node.
 *                       Defaults to `UniformNodeHashFunc`.
 * @tparam forward If true, hashes are computed based on parents (top-down).
 *                 If false, hashes are computed based on children (bottom-up).
 */
template <typename GraphT, typename NodeHashFuncT = UniformNodeHashFunc<VertexIdxT<GraphT>>, bool forward = true>
class MerkleHashComputer : public HashComputer<VertexIdxT<GraphT>> {
    static_assert(isDirectedGraphV<GraphT>, "GraphT must satisfy the directed_graph concept");
    static_assert(std::is_invocable_r<std::size_t, NodeHashFuncT, VertexIdxT<GraphT>>::value,
                  "NodeHashFuncT must be invocable with one VertexIdxT<GraphT> argument and return std::size_t.");

    using VertexType = VertexIdxT<GraphT>;

    std::vector<std::size_t> vertexHashes_;
    std::unordered_map<std::size_t, std::vector<VertexType>> orbits_;

    NodeHashFuncT nodeHashFunc_;

    void ComputeHashesHelper(const VertexType &v, std::vector<std::size_t> &parentChildHashes) {
        std::sort(parentChildHashes.begin(), parentChildHashes.end());

        std::size_t hash = nodeHashFunc_(v);
        for (const auto &pcHash : parentChildHashes) {
            HashCombine(hash, pcHash);
        }

        vertexHashes_[v] = hash;

        if (orbits_.find(hash) == orbits_.end()) {
            orbits_[hash] = {v};
        } else {
            orbits_[hash].push_back(v);
        }
    }

    template <typename RetT = void>
    std::enable_if_t<forward, RetT> ComputeHashes(const GraphT &graph) {
        const size_t numVertices = graph.NumVertices();
        vertexHashes_.resize(numVertices);
        std::vector<std::size_t> neighborHashes;

        for (const VertexType &v : TopSortView(graph)) {
            neighborHashes.clear();
            for (const VertexType &parent : graph.Parents(v)) {
                neighborHashes.push_back(vertexHashes_[parent]);
            }
            ComputeHashesHelper(v, neighborHashes);
        }
    }

    template <typename RetT = void>
    std::enable_if_t<not forward, RetT> ComputeHashes(const GraphT &graph) {
        const size_t numVertices = graph.NumVertices();
        vertexHashes_.resize(numVertices);
        std::vector<std::size_t> neighborHashes;

        const auto topSort = GetTopOrderReverse(graph);
        for (auto it = topSort.cbegin(); it != topSort.cend(); ++it) {
            const VertexType &v = *it;
            neighborHashes.clear();
            for (const VertexType &child : graph.Children(v)) {
                neighborHashes.push_back(vertexHashes_[child]);
            }
            ComputeHashesHelper(v, neighborHashes);
        }
    }

  public:
    /**
     * @brief Constructs the MerkleHashComputer and immediately computes the hashes.
     * @tparam Args Arguments forwarded to the NodeHashFuncT constructor.
     * @param graph The graph to process.
     * @param args Arguments for the node hash function.
     */
    template <typename... Args>
    MerkleHashComputer(const GraphT &graph, Args &&...args)
        : HashComputer<VertexType>(), nodeHashFunc_(std::forward<Args>(args)...) {
        ComputeHashes(graph);
    }

    ~MerkleHashComputer() override = default;

    std::size_t GetVertexHash(const VertexType &v) const override { return vertexHashes_[v]; }

    const std::vector<std::size_t> &GetVertexHashes() const override { return vertexHashes_; }

    std::size_t NumOrbits() const override { return orbits_.size(); }

    const std::vector<VertexType> &GetOrbit(const VertexType &v) const override {
        return this->GetOrbitFromHash(this->GetVertexHash(v));
    }

    const std::unordered_map<std::size_t, std::vector<VertexType>> &GetOrbits() const override { return orbits_; }

    const std::vector<VertexType> &GetOrbitFromHash(const std::size_t &hash) const override { return orbits_.at(hash); }
};

/**
 * @brief Checks if two graphs are isomorphic based on Merkle hashes.
 * @note This is a necessary but not sufficient condition for graph isomorphism in general cases,
 *       but sufficient for the kinds of DAGs often encountered in this context.
 *
 * @tparam GraphT The graph type.
 * @tparam NodeHashFuncT The node hash function type.
 * @tparam forward Direction of hash computation.
 * @param g1 The first graph.
 * @param g2 The second graph.
 * @return True if they have the same orbit structure, false otherwise.
 */
template <typename GraphT, typename NodeHashFuncT = UniformNodeHashFunc<VertexIdxT<GraphT>>, bool forward = true>
bool AreIsomorphicByMerkleHash(const GraphT &g1, const GraphT &g2) {
    if (g1.NumVertices() != g2.NumVertices() || g1.NumEdges() != g2.NumEdges()) {
        return false;
    }

    MerkleHashComputer<GraphT, NodeHashFuncT, forward> hash1(g1);
    MerkleHashComputer<GraphT, NodeHashFuncT, forward> hash2(g2);

    const auto &orbits1 = hash1.GetOrbits();
    const auto &orbits2 = hash2.GetOrbits();

    if (orbits1.size() != orbits2.size()) {
        return false;
    }

    for (const auto &pair : orbits1) {
        const std::size_t hash = pair.first;
        const auto &orbitVec = pair.second;

        auto it = orbits2.find(hash);
        if (it == orbits2.end() || it->second.size() != orbitVec.size()) {
            return false;
        }
    }

    return true;
}

template <typename GraphT>
struct BwdMerkleNodeHashFunc {
    MerkleHashComputer<GraphT, UniformNodeHashFunc<VertexIdxT<GraphT>>, false> bwMerkleHash_;

    BwdMerkleNodeHashFunc(const GraphT &graph) : bwMerkleHash_(graph) {}

    std::size_t operator()(const VertexIdxT<GraphT> &v) const { return bwMerkleHash_.GetVertexHash(v); }
};

template <typename GraphT>
struct PrecomBwdMerkleNodeHashFunc {
    MerkleHashComputer<GraphT, VectorNodeHashFunc<VertexIdxT<GraphT>>, false> bwMerkleHash_;

    PrecomBwdMerkleNodeHashFunc(const GraphT &graph, const std::vector<std::size_t> &nodeHashes)
        : bwMerkleHash_(graph, nodeHashes) {}

    std::size_t operator()(const VertexIdxT<GraphT> &v) const { return bwMerkleHash_.GetVertexHash(v); }
};

}    // namespace osp
