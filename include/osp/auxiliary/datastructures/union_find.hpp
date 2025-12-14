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
#include <unordered_map>
#include <vector>

#include "osp/concepts/graph_traits.hpp"

namespace osp {

/// @brief Structure to execute a union-find algorithm
template <typename T, typename IndexT, typename WorkwT, typename MemwT>
struct UnionFindObject {
    const T name_;    // unique identifier
    IndexT parentIndex_;
    unsigned rank_;
    WorkwT weight_;
    MemwT memory_;

    explicit UnionFindObject(const T &name, IndexT parentIndex, WorkwT weight = 0, MemwT memory = 0)
        : name_(name), parentIndex_(parentIndex), weight_(weight), memory_(memory) {
        rank_ = 1;
    }

    UnionFindObject(const UnionFindObject &other) = default;
    UnionFindObject &operator=(const UnionFindObject &other) = default;
};

/// @brief Class to execute a union-find algorithm
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

  public:
    void Reset() {
        universe_.clear();
        namesToIndices_.clear();
        componentIndices_.clear();
    }

    bool IsInUniverse(const T &name) const { return namesToIndices_.find(name) != namesToIndices_.end(); }

    /// @brief Loops till object is its own parent
    /// @param name of object
    /// @return returns (current) name of component
    T FindOriginByName(const T &name) { return universe_[FindOrigin(namesToIndices_.at(name))].name_; }

    /// @brief Joins two components
    /// @param name of object to join
    /// @param other_name of object to join
    void JoinByName(const T &name, const T &otherName) { Join(namesToIndices_.at(name), namesToIndices_.at(otherName)); }

    /// @brief Retrieves the current number of connected components
    std::size_t GetNumberOfConnectedComponents() const { return componentIndices_.size(); }

    /// @brief Retrieves the (current) names of components
    std::vector<T> GetComponentNames() const {
        std::vector<T> componentNames;
        componentNames.reserve(componentIndices_.size());
        for (auto &indx : componentIndices_) {
            componentNames.emplace_back(universe_[indx].name_);
        }
        return componentNames;
    }

    /// @brief Retrieves the (current) names of components together with their weight
    std::vector<std::pair<T, WorkwT>> GetComponentNamesAndWeights() const {
        std::vector<std::pair<T, WorkwT>> componentNamesAndWeights;
        componentNamesAndWeights.reserve(componentIndices_.size());
        for (auto &indx : componentIndices_) {
            componentNamesAndWeights.emplace_back({universe_[indx].name_, universe_[indx].weight});
        }
        return componentNamesAndWeights;
    }

    /// @brief Retrieves the (current) names of components together with their weight and memory
    std::vector<std::tuple<T, WorkwT, MemwT>> GetComponentNamesWeightsAndMemory() const {
        std::vector<std::tuple<T, WorkwT, MemwT>> componentNamesWeightsAndMemory;
        componentNamesWeightsAndMemory.reserve(componentIndices_.size());
        for (auto &indx : componentIndices_) {
            componentNamesWeightsAndMemory.emplace_back({universe_[indx].name_, universe_[indx].weight, universe_[indx].memory});
        }
        return componentNamesWeightsAndMemory;
    }

    /// @brief Retrieves the weight of the component containing the given object
    /// @param name of object
    WorkwT GetWeightOfComponentByName(const T &name) {
        IndexT index = GetIndexFromName(name);
        index = FindOrigin(index);
        return universe_[index].weight_;
    }

    /// @brief Retrieves the memory of the component containing the given object
    /// @param name of object
    MemwT GetMemoryOfComponentByName(const T &name) {
        IndexT index = GetIndexFromName(name);
        index = FindOrigin(index);
        return universe_[index].memory_;
    }

    /// @brief Retrieves the connected components
    /// @return Partition of the names of objects according to the connected components
    std::vector<std::vector<T>> GetConnectedComponents() {
        std::vector<std::vector<IndexT>> connectedComponentsByIndex;
        connectedComponentsByIndex.resize(universe_.size());
        for (IndexT i = 0; i < static_cast<IndexT>(universe_.size()); i++) {
            connectedComponentsByIndex[FindOrigin(i)].emplace_back(i);
        }

        std::vector<std::vector<T>> connectedComponentsByName;
        for (auto &comp : connectedComponentsByIndex) {
            if (comp.empty()) {
                continue;
            }
            std::vector<T> namesInComp;
            namesInComp.reserve(comp.size());
            for (const auto &indx : comp) {
                namesInComp.emplace_back(universe_[indx].name_);
            }
            connectedComponentsByName.push_back(namesInComp);
        }

        return connectedComponentsByName;
    }

    /// @brief Retrieves the connected components and their respective weights
    /// @return Partition of the names of objects according to the connected components together with their respective
    /// weight
    std::vector<std::pair<std::vector<T>, WorkwT>> GetConnectedComponentsAndWeights() {
        std::vector<std::vector<WorkwT>> connectedComponentsByIndex;
        connectedComponentsByIndex.resize(universe_.size());
        for (IndexT i = 0; i < static_cast<IndexT>(universe_.size()); i++) {
            connectedComponentsByIndex[FindOrigin(i)].emplace_back(i);
        }

        std::vector<std::pair<std::vector<T>, WorkwT>> connectedComponentsByNameInclWeight;
        for (auto &comp : connectedComponentsByIndex) {
            if (comp.empty()) {
                continue;
            }

            WorkwT compWeight = universe_[FindOrigin(comp[0])].weight_;

            std::vector<T> namesInComp;
            namesInComp.reserve(comp.size());
            for (auto &indx : comp) {
                namesInComp.emplace_back(universe_[indx].name__);
            }
            connectedComponentsByNameInclWeight.emplace_back(namesInComp, compWeight);
        }

        return connectedComponentsByNameInclWeight;
    }

    /// @brief Retrieves the connected components and their respective weights and memories
    /// @return Partition of the names of objects according to the connected components together with their respective
    /// weight and memory
    std::vector<std::tuple<std::vector<T>, WorkwT, MemwT>> GetConnectedComponentsWeightsAndMemories() {
        std::vector<std::vector<IndexT>> connectedComponentsByIndex;
        connectedComponentsByIndex.resize(universe_.size());
        for (IndexT i = 0; i < static_cast<IndexT>(universe_.size()); i++) {
            connectedComponentsByIndex[FindOrigin(i)].emplace_back(i);
        }

        std::vector<std::tuple<std::vector<T>, WorkwT, MemwT>> connectedComponentsByNameInclWeightMemory;
        for (auto &comp : connectedComponentsByIndex) {
            if (comp.empty()) {
                continue;
            }

            WorkwT compWeight = universe_[FindOrigin(comp[0])].weight_;
            MemwT compMemory = universe_[FindOrigin(comp[0])].memory_;

            std::vector<T> namesInComp;
            namesInComp.reserve(comp.size());
            for (auto &indx : comp) {
                namesInComp.emplace_back(universe_[indx].name__);
            }
            connectedComponentsByNameInclWeightMemory.emplace_back(namesInComp, compWeight, compMemory);
        }

        return connectedComponentsByNameInclWeightMemory;
    }

    /// @brief Adds object to the union-find structure
    /// @param name of object
    void AddObject(const T &name) {
        if (namesToIndices_.find(name) != namesToIndices_.end()) {
            throw std::runtime_error("This name already exists in the universe.");
        }
        IndexT newIndex = static_cast<IndexT>(universe_.size());
        universe_.emplace_back(name, newIndex);
        namesToIndices_[name] = newIndex;
        componentIndices_.emplace(newIndex);
    }

    /// @brief Adds object to the union-find structure with given weight
    /// @param name of object
    /// @param weight of object
    void AddObject(const T &name, const WorkwT weight) {
        if (namesToIndices_.find(name) != namesToIndices_.end()) {
            throw std::runtime_error("This name already exists in the universe.");
        }
        IndexT newIndex = static_cast<IndexT>(universe_.size());
        universe_.emplace_back(name, newIndex, weight);
        namesToIndices_[name] = newIndex;
        componentIndices_.emplace(newIndex);
    }

    /// @brief Adds object to the union-find structure with given weight and memory
    /// @param name of object
    /// @param weight of object
    /// @param memory of object
    void AddObject(const T &name, const WorkwT weight, const MemwT memory) {
        if (namesToIndices_.find(name) != namesToIndices_.end()) {
            throw std::runtime_error("This name already exists in the universe.");
        }
        IndexT newIndex = static_cast<IndexT>(universe_.size());
        universe_.emplace_back(name, newIndex, weight, memory);
        namesToIndices_[name] = newIndex;
        componentIndices_.emplace(newIndex);
    }

    /// @brief Adds objects to the union-find structure
    /// @param names of objects
    void AddObject(const std::vector<T> &names) {
        // adjusting universe capacity
        IndexT additionalSize = static_cast<IndexT>(names.size());
        IndexT currentSize = static_cast<IndexT>(universe_.size());
        IndexT currentCapacity = static_cast<IndexT>(universe_.capacity());
        if (additionalSize + currentSize > currentCapacity) {
            IndexT newMinCapacity = std::max((currentCapacity + 1) / 2 * 3, currentSize + additionalSize);
            universe_.reserve(newMinCapacity);
        }

        // adjusting names_to_indices capacity
        currentSize = static_cast<IndexT>(namesToIndices_.size());
        if (additionalSize + currentSize > currentCapacity) {
            IndexT newMinCapacity = std::max((currentCapacity + 1) / 2 * 3, currentSize + additionalSize);
            namesToIndices_.reserve(newMinCapacity);
        }

        for (auto &name : names) {
            AddObject(name);
        }
    }

    /// @brief Adds objects to the union-find structure
    /// @param names of objects
    /// @param weights of objects
    void AddObject(const std::vector<T> &names, const std::vector<WorkwT> &weights) {
        if (names.size() != weights.size()) {
            throw std::runtime_error("Vectors of names and weights must be of equal length.");
        }

        // adjusting universe capacity
        IndexT additionalSize = static_cast<IndexT>(names.size());
        IndexT currentSize = static_cast<IndexT>(universe_.size());
        IndexT currentCapacity = static_cast<IndexT>(universe_.capacity());
        if (additionalSize + currentSize > currentCapacity) {
            IndexT newMinCapacity = std::max((currentCapacity + 1) / 2 * 3, currentSize + additionalSize);
            universe_.reserve(newMinCapacity);
        }

        // adjusting names_to_indices capacity
        currentSize = static_cast<IndexT>(namesToIndices_.size());
        if (additionalSize + currentSize > currentCapacity) {
            IndexT newMinCapacity = std::max((currentCapacity + 1) / 2 * 3, currentSize + additionalSize);
            namesToIndices_.reserve(newMinCapacity);
        }

        for (std::size_t i = 0; i < names.size(); i++) {
            AddObject(names[i], weights[i]);
        }
    }

    /// @brief Adds objects to the union-find structure
    /// @param names of objects
    /// @param weights of objects
    /// @param memories of objects
    void AddObject(const std::vector<T> &names, const std::vector<unsigned> &weights, const std::vector<MemwT> &memories) {
        if (names.size() != weights.size()) {
            throw std::runtime_error("Vectors of names and weights must be of equal length.");
        }

        // adjusting universe capacity
        IndexT additionalSize = static_cast<IndexT>(names.size());
        IndexT currentSize = static_cast<IndexT>(universe_.size());
        IndexT currentCapacity = static_cast<IndexT>(universe_.capacity());
        if (additionalSize + currentSize > currentCapacity) {
            unsigned newMinCapacity = std::max((currentCapacity + 1) / 2 * 3, currentSize + additionalSize);
            universe_.reserve(newMinCapacity);
        }

        // adjusting names_to_indices capacity
        currentSize = static_cast<IndexT>(namesToIndices_.size());
        if (additionalSize + currentSize > currentCapacity) {
            IndexT newMinCapacity = std::max((currentCapacity + 1) / 2 * 3, currentSize + additionalSize);
            namesToIndices_.reserve(newMinCapacity);
        }

        for (size_t i = 0; i < names.size(); i++) {
            AddObject(names[i], weights[i], memories[i]);
        }
    }

    /// @brief Initiates a union-find structure
    explicit UnionFindUniverse() {}

    /// @brief Initiates a union-find structure
    /// @param names of objects
    explicit UnionFindUniverse(const std::vector<T> &names) { AddObject(names); }

    /// @brief Initiates a union-find structure
    /// @param names of objects
    /// @param weights of objects
    explicit UnionFindUniverse(const std::vector<T> &names, const std::vector<WorkwT> &weights) { AddObject(names, weights); }

    /// @brief Initiates a union-find structure
    /// @param names of objects
    /// @param weights of objects
    explicit UnionFindUniverse(const std::vector<T> &names, const std::vector<WorkwT> &weights, const std::vector<MemwT> &memories) {
        AddObject(names, weights, memories);
    }

    UnionFindUniverse(const UnionFindUniverse &other) = default;
    UnionFindUniverse &operator=(const UnionFindUniverse &other) = default;
};

template <typename GraphT>
using UnionFindUniverseT = UnionFindUniverse<VertexIdxT<GraphT>, VertexIdxT<GraphT>, VWorkwT<GraphT>, VMemwT<GraphT>>;

}    // namespace osp
