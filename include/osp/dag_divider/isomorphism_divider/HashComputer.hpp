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

#include <unordered_map>
#include <vector>

namespace osp {

/**
 * @class HashComputer
 * @brief Abstract base class for computing and managing hash values and orbits for graph vertices.
 *
 * This class provides an interface for obtaining hash values for individual vertices,
 * the full list of vertex hashes, the number of unique orbits, and the vertices belonging to specific orbits.
 *
 * @tparam index_type The type used for indexing vertices in the graph.
 */
template <typename IndexType>
class HashComputer {
  public:
    virtual ~HashComputer() = default;

    virtual std::size_t GetVertexHash(const IndexType &v) const = 0;
    virtual const std::vector<std::size_t> &GetVertexHashes() const = 0;
    virtual std::size_t NumOrbits() const = 0;

    virtual const std::vector<IndexType> &GetOrbit(const IndexType &v) const = 0;
    virtual const std::unordered_map<std::size_t, std::vector<IndexType>> &GetOrbits() const = 0;

    virtual const std::vector<IndexType> &GetOrbitFromHash(const std::size_t &hash) const = 0;
};

}    // namespace osp
