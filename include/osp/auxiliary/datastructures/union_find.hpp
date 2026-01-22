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
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "osp/concepts/graph_traits.hpp"

namespace osp {

/**
 * @brief Structure to represent an object in the union-find universe.
 *
 * @tparam T Type of the unique identifier (name).
 * @tparam IndexT Type of the index used for internal references.
 * @tparam WorkwT Type of the weight associated with the object.
 * @tparam MemwT Type of the memory associated with the object.
 */
template <typename T, typename IndexT, typename WorkwT, typename MemwT>
struct UnionFindObject {
    const T name_;       /** Unique identifier of the object. */
    IndexT parentIndex_; /** Index of the parent object in the union-find tree. */
    unsigned rank_;      /** Rank of the object, used for union operation optimization. */
    WorkwT weight_;      /** Weight associated with the object. */
    MemwT memory_;       /** Memory associated with the object. */

    /**
     * @brief Constructs a new UnionFindObject.
     *
     * @param name Unique identifier.
     * @param parentIndex Index of the parent object.
     * @param weight Weight of the object. Default is 0.
     * @param memory Memory of the object. Default is 0.
     */
    explicit UnionFindObject(const T &name, IndexT parentIndex, WorkwT weight = 0, MemwT memory = 0)
        : name_(name), parentIndex_(parentIndex), rank_(1), weight_(weight), memory_(memory) {}

    UnionFindObject(const UnionFindObject &other) = default;
    UnionFindObject &operator=(const UnionFindObject &other) = default;
};

/**
 * @brief Class to execute a union-find algorithm with path compression and union by rank.
 *
 * This class manages a set of elements partitioned into disjoint sets. It supports adding elements
 * and merging sets (finding connected components).
 *
 * @tparam T Type of the unique identifier (name).
 * @tparam IndexT Type of the index used for internal references.
 * @tparam WorkwT Type of the weight associated with the object.
 * @tparam MemwT Type of the memory associated with the object.
 */
template <typename T, typename IndexT, typename WorkwT, typename MemwT>
class UnionFindUniverse {
  private:
    std::vector<UnionFindObject<T, IndexT, WorkwT, MemwT>> universe_;
    std::unordered_map<T, IndexT> namesToIndices_;
    std::set<IndexT> componentIndices_;

    IndexT FindOrigin(IndexT index) {
        while (index != universe_[index].parentIndex_) {
            universe_[index].parentIndex_ = universe_[universe_[index].parentIndex_].parentIndex_;
            index = universe_[index].parentIndex_;
        }
        return index;
    }

    int Join(IndexT index, IndexT otherIndex) {
        index = FindOrigin(index);
        otherIndex = FindOrigin(otherIndex);

        if (index == otherIndex) {
            return 0;
        }

        if (universe_[index].rank_ >= universe_[otherIndex].rank_) {
            universe_[otherIndex].parentIndex_ = index;
            universe_[index].weight_ += universe_[otherIndex].weight_;
            universe_[index].memory_ += universe_[otherIndex].memory_;
            componentIndices_.erase(otherIndex);

            if (universe_[index].rank_ == universe_[otherIndex].rank_) {
                universe_[index].rank_++;
            }
        } else {
            universe_[index].parentIndex_ = otherIndex;
            universe_[otherIndex].weight_ += universe_[index].weight_;
            universe_[otherIndex].memory_ += universe_[index].memory_;
            componentIndices_.erase(index);
        }
        return -1;
    }

    IndexT GetIndexFromName(const T &name) const { return namesToIndices_.at(name); }

    void ReserveAdditional(std::size_t additionalSize) {
        IndexT addSize = static_cast<IndexT>(additionalSize);
        IndexT currentSize = static_cast<IndexT>(universe_.size());
        IndexT currentCapacity = static_cast<IndexT>(universe_.capacity());

        if (addSize + currentSize > currentCapacity) {
            IndexT newMinCapacity = std::max((currentCapacity + 1) / 2 * 3, currentSize + addSize);
            universe_.reserve(newMinCapacity);
        }

        // Reserve map to avoid rehashes
        IndexT currentMapSize = static_cast<IndexT>(namesToIndices_.size());
        IndexT currentMapCapacity
            = static_cast<IndexT>(static_cast<float>(namesToIndices_.bucket_count()) * namesToIndices_.max_load_factor());

        if (currentMapSize + addSize > currentMapCapacity) {
            IndexT newMinMapCapacity = std::max((currentMapCapacity + 1) / 2 * 3, currentMapSize + addSize);
            namesToIndices_.reserve(newMinMapCapacity);
        }
    }

    void AddObjectInternal(const T &name, WorkwT weight, MemwT memory) {
        if (namesToIndices_.find(name) != namesToIndices_.end()) {
            throw std::runtime_error("This name already exists in the universe.");
        }
        IndexT newIndex = static_cast<IndexT>(universe_.size());
        universe_.emplace_back(name, newIndex, weight, memory);
        namesToIndices_[name] = newIndex;
        componentIndices_.emplace(newIndex);
    }

  public:
    explicit UnionFindUniverse() = default;

    explicit UnionFindUniverse(const std::vector<T> &names) { AddObject(names); }

    /**
     * @brief Initiates a union-find structure and adds objects with weights.
     * @param names List of object names.
     * @param weights List of object weights.
     */
    explicit UnionFindUniverse(const std::vector<T> &names, const std::vector<WorkwT> &weights) { AddObject(names, weights); }

    /**
     * @brief Initiates a union-find structure and adds objects with weights and memory.
     * @param names List of object names.
     * @param weights List of object weights.
     * @param memories List of object memories.
     */
    explicit UnionFindUniverse(const std::vector<T> &names, const std::vector<WorkwT> &weights, const std::vector<MemwT> &memories) {
        AddObject(names, weights, memories);
    }

    UnionFindUniverse(const UnionFindUniverse &other) = default;
    UnionFindUniverse &operator=(const UnionFindUniverse &other) = default;
    UnionFindUniverse(UnionFindUniverse &&other) noexcept = default;
    UnionFindUniverse &operator=(UnionFindUniverse &&other) noexcept = default;
    ~UnionFindUniverse() = default;

    /**
     * @brief Resets the universe, clearing all objects and components.
     */
    void Reset() {
        universe_.clear();
        namesToIndices_.clear();
        componentIndices_.clear();
    }

    /**
     * @brief Checks if an object exists in the universe.
     * @param name The name of the object.
     * @return True if the object exists, false otherwise.
     */
    [[nodiscard]] bool IsInUniverse(const T &name) const noexcept { return namesToIndices_.find(name) != namesToIndices_.end(); }

    /**
     * @brief Finds the representative name of the component containing the object.
     * @param name The name of the object.
     * @return The name of the component's representative.
     */
    [[nodiscard]] T FindOriginByName(const T &name) { return universe_[FindOrigin(namesToIndices_.at(name))].name_; }

    /**
     * @brief Joins the components containing the two objects.
     * @param name Name of the first object.
     * @param otherName Name of the second object.
     */
    void JoinByName(const T &name, const T &otherName) { Join(namesToIndices_.at(name), namesToIndices_.at(otherName)); }

    /**
     * @brief Retrieves the current number of connected components.
     * @return Number of disjoint sets.
     */
    [[nodiscard]] std::size_t GetNumberOfConnectedComponents() const noexcept { return componentIndices_.size(); }

    /**
     * @brief Retrieves the names of all component representatives.
     * @return Vector of names.
     */
    [[nodiscard]] std::vector<T> GetComponentNames() const {
        std::vector<T> componentNames;
        componentNames.reserve(componentIndices_.size());
        for (auto &indx : componentIndices_) {
            componentNames.emplace_back(universe_[indx].name_);
        }
        return componentNames;
    }

    /**
     * @brief Retrieves the names of all component representatives together with their weights.
     * @return Vector of pairs (name, weight).
     */
    [[nodiscard]] std::vector<std::pair<T, WorkwT>> GetComponentNamesAndWeights() const {
        std::vector<std::pair<T, WorkwT>> componentNamesAndWeights;
        componentNamesAndWeights.reserve(componentIndices_.size());
        for (auto &indx : componentIndices_) {
            componentNamesAndWeights.emplace_back(universe_[indx].name_, universe_[indx].weight_);
        }
        return componentNamesAndWeights;
    }

    /**
     * @brief Retrieves the names of all component representatives together with their weight and memory.
     * @return Vector of tuples (name, weight, memory).
     */
    [[nodiscard]] std::vector<std::tuple<T, WorkwT, MemwT>> GetComponentNamesWeightsAndMemory() const {
        std::vector<std::tuple<T, WorkwT, MemwT>> componentNamesWeightsAndMemory;
        componentNamesWeightsAndMemory.reserve(componentIndices_.size());
        for (auto &indx : componentIndices_) {
            componentNamesWeightsAndMemory.emplace_back(universe_[indx].name_, universe_[indx].weight_, universe_[indx].memory_);
        }
        return componentNamesWeightsAndMemory;
    }

    /**
     * @brief Retrieves the weight of the component containing the given object.
     * @param name Name of the object.
     * @return Total weight of the component.
     */
    [[nodiscard]] WorkwT GetWeightOfComponentByName(const T &name) {
        IndexT index = GetIndexFromName(name);
        index = FindOrigin(index);
        return universe_[index].weight_;
    }

    /**
     * @brief Retrieves the memory of the component containing the given object.
     * @param name Name of the object.
     * @return Total memory of the component.
     */
    [[nodiscard]] MemwT GetMemoryOfComponentByName(const T &name) {
        IndexT index = GetIndexFromName(name);
        index = FindOrigin(index);
        return universe_[index].memory_;
    }

    /**
     * @brief Retrieves all connected components grouping member names.
     * @return Vector of components, where each component is a vector of names.
     */
    [[nodiscard]] std::vector<std::vector<T>> GetConnectedComponents() {
        std::vector<std::vector<IndexT>> connectedComponentsByIndex;
        connectedComponentsByIndex.resize(universe_.size());
        for (IndexT i = 0; i < static_cast<IndexT>(universe_.size()); i++) {
            connectedComponentsByIndex[FindOrigin(i)].emplace_back(i);
        }

        std::vector<std::vector<T>> connectedComponentsByName;
        connectedComponentsByName.reserve(componentIndices_.size());

        for (auto &comp : connectedComponentsByIndex) {
            if (comp.empty()) {
                continue;
            }
            std::vector<T> namesInComp;
            namesInComp.reserve(comp.size());
            for (const auto &indx : comp) {
                namesInComp.emplace_back(universe_[indx].name_);
            }
            connectedComponentsByName.push_back(std::move(namesInComp));
        }

        return connectedComponentsByName;
    }

    /**
     * @brief Retrieves all connected components with their total weights.
     * @return Vector of pairs (component members, total weight).
     */
    [[nodiscard]] std::vector<std::pair<std::vector<T>, WorkwT>> GetConnectedComponentsAndWeights() {
        std::vector<std::vector<IndexT>> connectedComponentsByIndex;
        connectedComponentsByIndex.resize(universe_.size());
        for (IndexT i = 0; i < static_cast<IndexT>(universe_.size()); i++) {
            connectedComponentsByIndex[FindOrigin(i)].emplace_back(i);
        }

        std::vector<std::pair<std::vector<T>, WorkwT>> connectedComponentsByNameInclWeight;
        connectedComponentsByNameInclWeight.reserve(componentIndices_.size());

        for (auto &comp : connectedComponentsByIndex) {
            if (comp.empty()) {
                continue;
            }

            WorkwT compWeight = universe_[FindOrigin(comp[0])].weight_;

            std::vector<T> namesInComp;
            namesInComp.reserve(comp.size());
            for (auto &indx : comp) {
                namesInComp.emplace_back(universe_[indx].name_);
            }
            connectedComponentsByNameInclWeight.emplace_back(std::move(namesInComp), compWeight);
        }

        return connectedComponentsByNameInclWeight;
    }

    /**
     * @brief Retrieves all connected components with their total weights and memories.
     * @return Vector of tuples (component members, total weight, total memory).
     */
    [[nodiscard]] std::vector<std::tuple<std::vector<T>, WorkwT, MemwT>> GetConnectedComponentsWeightsAndMemories() {
        std::vector<std::vector<IndexT>> connectedComponentsByIndex;
        connectedComponentsByIndex.resize(universe_.size());
        for (IndexT i = 0; i < static_cast<IndexT>(universe_.size()); i++) {
            connectedComponentsByIndex[FindOrigin(i)].emplace_back(i);
        }

        std::vector<std::tuple<std::vector<T>, WorkwT, MemwT>> connectedComponentsByNameInclWeightMemory;
        connectedComponentsByNameInclWeightMemory.reserve(componentIndices_.size());

        for (auto &comp : connectedComponentsByIndex) {
            if (comp.empty()) {
                continue;
            }

            WorkwT compWeight = universe_[FindOrigin(comp[0])].weight_;
            MemwT compMemory = universe_[FindOrigin(comp[0])].memory_;

            std::vector<T> namesInComp;
            namesInComp.reserve(comp.size());
            for (auto &indx : comp) {
                namesInComp.emplace_back(universe_[indx].name_);
            }
            connectedComponentsByNameInclWeightMemory.emplace_back(std::move(namesInComp), compWeight, compMemory);
        }

        return connectedComponentsByNameInclWeightMemory;
    }

    /**
     * @brief Adds a single object to the universe.
     * @param name Name of the object.
     */
    void AddObject(const T &name) { AddObjectInternal(name, 0, 0); }

    /**
     * @brief Adds a single object with weight.
     * @param name Name of the object.
     * @param weight Weight of the object.
     */
    void AddObject(const T &name, const WorkwT weight) { AddObjectInternal(name, weight, 0); }

    /**
     * @brief Adds a single object with weight and memory.
     * @param name Name of the object.
     * @param weight Weight of the object.
     * @param memory Memory of the object.
     */
    void AddObject(const T &name, const WorkwT weight, const MemwT memory) { AddObjectInternal(name, weight, memory); }

    /**
     * @brief Adds multiple objects to the universe.
     * @param names Vector of names.
     */
    void AddObject(const std::vector<T> &names) {
        ReserveAdditional(names.size());
        for (auto &name : names) {
            AddObjectInternal(name, 0, 0);
        }
    }

    /**
     * @brief Adds multiple objects with weights.
     * @param names Vector of names.
     * @param weights Vector of weights.
     * @throws std::runtime_error If vectors have different sizes.
     */
    void AddObject(const std::vector<T> &names, const std::vector<WorkwT> &weights) {
        if (names.size() != weights.size()) {
            throw std::runtime_error("Vectors of names and weights must be of equal length.");
        }
        ReserveAdditional(names.size());
        for (std::size_t i = 0; i < names.size(); i++) {
            AddObjectInternal(names[i], weights[i], 0);
        }
    }

    /**
     * @brief Adds multiple objects with weights and memories.
     * @param names Vector of names.
     * @param weights Vector of weights.
     * @param memories Vector of memories.
     * @throws std::runtime_error If vectors have different sizes.
     */
    void AddObject(const std::vector<T> &names, const std::vector<WorkwT> &weights, const std::vector<MemwT> &memories) {
        if (names.size() != weights.size() || names.size() != memories.size()) {
            throw std::runtime_error("Vectors of names, weights, and memories must be of equal length.");
        }
        ReserveAdditional(names.size());
        for (size_t i = 0; i < names.size(); i++) {
            AddObjectInternal(names[i], weights[i], memories[i]);
        }
    }
};

/**
 * @brief Alias for a UnionFindUniverse using GraphT properties.
 */
template <typename GraphT>
using UnionFindUniverseT = UnionFindUniverse<VertexIdxT<GraphT>, VertexIdxT<GraphT>, VWorkwT<GraphT>, VMemwT<GraphT>>;

}    // namespace osp
