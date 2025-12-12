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

#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "osp/auxiliary/hash_util.hpp"
#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/dag_divider/isomorphism_divider/HashComputer.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"

namespace osp {

/**
 * @brief Computes Merkle hashes for graph vertices to identify isomorphic orbits.
 *
 * The Merkle hash of a vertex is computed recursively based on its own properties
 * and the sorted hashes of its parents (or children, depending on the `forward` template parameter).
 * This allows for the identification of structurally isomorphic subgraphs.
 *
 * @tparam Graph_t The type of the graph, must satisfy the `directed_graph` concept.
 * @tparam node_hash_func_t A functor that computes a hash for a single node.
 *                          Defaults to `uniform_node_hash_func`.
 * @tparam forward If true, hashes are computed based on parents (top-down).
 *                 If false, hashes are computed based on children (bottom-up).
 */
template <typename GraphT, typename NodeHashFuncT = uniform_node_hash_func<VertexIdxT<GraphT>>, bool forward = true>
class MerkleHashComputer : public HashComputer<VertexIdxT<GraphT>> {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");
    static_assert(std::is_invocable_r<std::size_t, node_hash_func_t, VertexIdxT<GraphT>>::value,
                  "node_hash_func_t must be invocable with one VertexIdxT<GraphT> argument and return std::size_t.");

    using VertexType = VertexIdxT<GraphT>;

    std::vector<std::size_t> vertexHashes_;
    std::unordered_map<std::size_t, std::vector<VertexType>> orbits_;

    NodeHashFuncT nodeHashFunc_;

    inline void ComputeHashesHelper(const VertexType &v, std::vector<std::size_t> &parentChildHashes) {
        std::sort(parentChildHashes.begin(), parentChildHashes.end());

        std::size_t hash = nodeHashFunc_(v);
        for (const auto &pcHash : parentChildHashes) {
            HashCombine(hash, pcHash);
        }

        vertexHashes_[v] = hash;

        if (orbits.find(hash) == orbits.end()) {
            orbits[hash] = {v};
        } else {
            orbits[hash].push_back(v);
        }
    }

    template <typename RetT = void>
    std::enable_if_t<forward, RetT> ComputeHashes(const GraphT &graph) {
        vertexHashes_.resize(graph.NumVertices());

        for (const VertexType &v : top_sort_view(graph)) {
            std::vector<std::size_t> parent_hashes;
            for (const VertexType &parent : graph.Parents(v)) {
                parent_hashes.push_back(vertex_hashes[parent]);
            }
            compute_hashes_helper(v, parent_hashes);
        }
    }

    template <typename RetT = void>
    std::enable_if_t<not forward, RetT> ComputeHashes(const GraphT &graph) {
        vertexHashes_.resize(graph.NumVertices());

        const auto topSort = GetTopOrderReverse(graph);
        for (auto it = topSort.cbegin(); it != topSort.cend(); ++it) {
            const VertexType &v = *it;
            std::vector<std::size_t> childHashes;
            for (const VertexType &child : graph.Children(v)) {
                child_hashes.push_back(vertex_hashes[child]);
            }
            compute_hashes_helper(v, child_hashes);
        }
    }

  public:
    template <typename... Args>
    MerkleHashComputer(const GraphT &graph, Args &&...args)
        : HashComputer<VertexType>(), nodeHashFunc_(std::forward<Args>(args)...) {
        compute_hashes(graph);
    }

    virtual ~MerkleHashComputer() override = default;

    inline std::size_t get_vertex_hash(const VertexType &v) const override { return vertexHashes_[v]; }

    inline const std::vector<std::size_t> &GetVertexHashes() const override { return vertexHashes_; }

    inline std::size_t NumOrbits() const override { return orbits.size(); }

    inline const std::vector<VertexType> &get_orbit(const VertexType &v) const override {
        return this->get_orbit_from_hash(this->get_vertex_hash(v));
    }

    inline const std::unordered_map<std::size_t, std::vector<VertexType>> &get_orbits() const override { return orbits; }

    inline const std::vector<VertexType> &get_orbit_from_hash(const std::size_t &hash) const override { return orbits.at(hash); }
};

template <typename GraphT, typename NodeHashFuncT = uniform_node_hash_func<VertexIdxT<GraphT>>, bool forward = true>
bool AreIsomorphicByMerkleHash(const GraphT &g1, const GraphT &g2) {
    // Basic check: Different numbers of vertices or edges mean they can't be isomorphic.
    if (g1.NumVertices() != g2.NumVertices() || g1.NumEdges() != g2.NumEdges()) {
        return false;
    }

    // --- Compute Hashes in the Specified Direction ---
    MerkleHashComputer<GraphT, NodeHashFuncT, forward> hash1(g1);
    MerkleHashComputer<GraphT, NodeHashFuncT, forward> hash2(g2);

    const auto &orbits1 = hash1.get_orbits();
    const auto &orbits2 = hash2.get_orbits();

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
    MerkleHashComputer<Graph_t, uniform_node_hash_func<VertexIdxT<GraphT>>, false> bwMerkleHash_;

    BwdMerkleNodeHashFunc(const GraphT &graph) : bw_merkle_hash(graph) {}

    std::size_t operator()(const VertexIdxT<GraphT> &v) const { return bw_merkle_hash.get_vertex_hash(v); }
};

template <typename GraphT>
struct PrecomBwdMerkleNodeHashFunc {
    MerkleHashComputer<Graph_t, vector_node_hash_func<VertexIdxT<GraphT>>, false> bwMerkleHash_;

    PrecomBwdMerkleNodeHashFunc(const GraphT &graph, const std::vector<std::size_t> &nodeHashes)
        : bw_merkle_hash(graph, node_hashes) {}

    std::size_t operator()(const VertexIdxT<GraphT> &v) const { return bw_merkle_hash.get_vertex_hash(v); }
};

}    // namespace osp
