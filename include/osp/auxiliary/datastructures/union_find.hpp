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
template<typename T, typename index_t, typename workw_t, typename memw_t>
struct union_find_object {
    const T name; // unique identifier
    index_t parent_index;
    unsigned rank;
    workw_t weight;
    memw_t memory;

    explicit union_find_object(const T &name_, index_t parent_index_, workw_t weight_ = 0, memw_t memory_ = 0)
        : name(name_), parent_index(parent_index_), weight(weight_), memory(memory_) {
        rank = 1;
    }
};

/// @brief Class to execute a union-find algorithm
template<typename T, typename index_t, typename workw_t, typename memw_t>
class Union_Find_Universe {
  private:
    std::vector<union_find_object<T, index_t, workw_t, memw_t>> universe;
    std::unordered_map<T, index_t> names_to_indices;
    std::set<index_t> component_indices;

    index_t find_origin(index_t index) {
        while (index != universe[index].parent_index) {
            universe[index].parent_index = universe[universe[index].parent_index].parent_index;
            index = universe[index].parent_index;
        }
        return index;
    }

    int join(index_t index, index_t other_index) {
        index = find_origin(index);
        other_index = find_origin(other_index);

        if (index == other_index) {
            return 0;
        }

        if (universe[index].rank >= universe[other_index].rank) {
            universe[other_index].parent_index = index;
            universe[index].weight += universe[other_index].weight;
            universe[index].memory += universe[other_index].memory;
            component_indices.erase(other_index);

            if (universe[index].rank == universe[other_index].rank) {
                universe[index].rank++;
            }
        } else {
            universe[index].parent_index = other_index;
            universe[other_index].weight += universe[index].weight;
            universe[other_index].memory += universe[index].memory;
            component_indices.erase(index);
        }
        return -1;
    }

    index_t get_index_from_name(const T &name) const { return names_to_indices.at(name); }

  public:
    bool is_in_universe(const T &name) const { return names_to_indices.find(name) != names_to_indices.end(); }

    /// @brief Loops till object is its own parent
    /// @param name of object
    /// @return returns (current) name of component
    T find_origin_by_name(const T &name) { return universe[find_origin(names_to_indices.at(name))].name; }

    /// @brief Joins two components
    /// @param name of object to join
    /// @param other_name of object to join
    void join_by_name(const T &name, const T &other_name) {
        join(names_to_indices.at(name), names_to_indices.at(other_name));
    }

    /// @brief Retrieves the current number of connected components
    std::size_t get_number_of_connected_components() const { return component_indices.size(); }

    /// @brief Retrieves the (current) names of components
    std::vector<T> get_component_names() const {
        std::vector<T> component_names;
        component_names.reserve(component_indices.size());
        for (auto &indx : component_indices) {
            component_names.emplace_back(universe[indx].name);
        }
        return component_names;
    }

    /// @brief Retrieves the (current) names of components together with their weight
    std::vector<std::pair<T, workw_t>> get_component_names_and_weights() const {
        std::vector<std::pair<T, workw_t>> component_names_and_weights;
        component_names_and_weights.reserve(component_indices.size());
        for (auto &indx : component_indices) {
            component_names_and_weights.emplace_back({universe[indx].name, universe[indx].weight});
        }
        return component_names_and_weights;
    }

    /// @brief Retrieves the (current) names of components together with their weight and memory
    std::vector<std::tuple<T, workw_t, memw_t>> get_component_names_weights_and_memory() const {
        std::vector<std::tuple<T, workw_t, memw_t>> component_names_weights_and_memory;
        component_names_weights_and_memory.reserve(component_indices.size());
        for (auto &indx : component_indices) {
            component_names_weights_and_memory.emplace_back(
                {universe[indx].name, universe[indx].weight, universe[indx].memory});
        }
        return component_names_weights_and_memory;
    }

    /// @brief Retrieves the weight of the component containing the given object
    /// @param name of object
    workw_t get_weight_of_component_by_name(const T &name) {
        index_t index = get_index_from_name(name);
        index = find_origin(index);
        return universe[index].weight;
    }

    /// @brief Retrieves the memory of the component containing the given object
    /// @param name of object
    memw_t get_memory_of_component_by_name(const T &name) {
        index_t index = get_index_from_name(name);
        index = find_origin(index);
        return universe[index].memory;
    }

    /// @brief Retrieves the connected components
    /// @return Partition of the names of objects according to the connected components
    std::vector<std::vector<T>> get_connected_components() {
        std::vector<std::vector<index_t>> connected_components_by_index;
        connected_components_by_index.resize(universe.size());
        for (index_t i = 0; i < static_cast<index_t>(universe.size()); i++) {
            connected_components_by_index[find_origin(i)].emplace_back(i);
        }

        std::vector<std::vector<T>> connected_components_by_name;
        for (auto &comp : connected_components_by_index) {
            if (comp.empty()) {
                continue;
            }
            std::vector<T> names_in_comp;
            names_in_comp.reserve(comp.size());
            for (const auto &indx : comp) {
                names_in_comp.emplace_back(universe[indx].name);
            }
            connected_components_by_name.push_back(names_in_comp);
        }

        return connected_components_by_name;
    }

    /// @brief Retrieves the connected components and their respective weights
    /// @return Partition of the names of objects according to the connected components together with their respective
    /// weight
    std::vector<std::pair<std::vector<T>, workw_t>> get_connected_components_and_weights() {
        std::vector<std::vector<workw_t>> connected_components_by_index;
        connected_components_by_index.resize(universe.size());
        for (index_t i = 0; i < static_cast<index_t>(universe.size()); i++) {
            connected_components_by_index[find_origin(i)].emplace_back(i);
        }

        std::vector<std::pair<std::vector<T>, workw_t>> connected_components_by_name_incl_weight;
        for (auto &comp : connected_components_by_index) {
            if (comp.empty()) {
                continue;
            }

            workw_t comp_weight = universe[find_origin(comp[0])].weight;

            std::vector<T> names_in_comp;
            names_in_comp.reserve(comp.size());
            for (auto &indx : comp) {
                names_in_comp.emplace_back(universe[indx].name);
            }
            connected_components_by_name_incl_weight.emplace_back(names_in_comp, comp_weight);
        }

        return connected_components_by_name_incl_weight;
    }

    /// @brief Retrieves the connected components and their respective weights and memories
    /// @return Partition of the names of objects according to the connected components together with their respective
    /// weight and memory
    std::vector<std::tuple<std::vector<T>, workw_t, memw_t>> get_connected_components_weights_and_memories() {
        std::vector<std::vector<index_t>> connected_components_by_index;
        connected_components_by_index.resize(universe.size());
        for (index_t i = 0; i < static_cast<index_t>(universe.size()); i++) {
            connected_components_by_index[find_origin(i)].emplace_back(i);
        }

        std::vector<std::tuple<std::vector<T>, workw_t, memw_t>> connected_components_by_name_incl_weight_memory;
        for (auto &comp : connected_components_by_index) {
            if (comp.empty()) {
                continue;
            }

            workw_t comp_weight = universe[find_origin(comp[0])].weight;
            memw_t comp_memory = universe[find_origin(comp[0])].memory;

            std::vector<T> names_in_comp;
            names_in_comp.reserve(comp.size());
            for (auto &indx : comp) {
                names_in_comp.emplace_back(universe[indx].name);
            }
            connected_components_by_name_incl_weight_memory.emplace_back(names_in_comp, comp_weight, comp_memory);
        }

        return connected_components_by_name_incl_weight_memory;
    }

    /// @brief Adds object to the union-find structure
    /// @param name of object
    void add_object(const T &name) {
        if (names_to_indices.find(name) != names_to_indices.end()) {
            throw std::runtime_error("This name already exists in the universe.");
        }
        index_t new_index = static_cast<index_t>(universe.size());
        universe.emplace_back(name, new_index);
        names_to_indices[name] = new_index;
        component_indices.emplace(new_index);
    }

    /// @brief Adds object to the union-find structure with given weight
    /// @param name of object
    /// @param weight of object
    void add_object(const T &name, const workw_t weight) {
        if (names_to_indices.find(name) != names_to_indices.end()) {
            throw std::runtime_error("This name already exists in the universe.");
        }
        index_t new_index = static_cast<index_t>(universe.size());
        universe.emplace_back(name, new_index, weight);
        names_to_indices[name] = new_index;
        component_indices.emplace(new_index);
    }

    /// @brief Adds object to the union-find structure with given weight and memory
    /// @param name of object
    /// @param weight of object
    /// @param memory of object
    void add_object(const T &name, const workw_t weight, const memw_t memory) {
        if (names_to_indices.find(name) != names_to_indices.end()) {
            throw std::runtime_error("This name already exists in the universe.");
        }
        index_t new_index = static_cast<index_t>(universe.size());
        universe.emplace_back(name, new_index, weight, memory);
        names_to_indices[name] = new_index;
        component_indices.emplace(new_index);
    }

    /// @brief Adds objects to the union-find structure
    /// @param names of objects
    void add_object(const std::vector<T> &names) {
        // adjusting universe capacity
        index_t additional_size = static_cast<index_t>(names.size());
        index_t current_size = static_cast<index_t>(universe.size());
        index_t current_capacity = static_cast<index_t>(universe.capacity());
        if (additional_size + current_size > current_capacity) {
            index_t new_min_capacity = std::max((current_capacity + 1) / 2 * 3, current_size + additional_size);
            universe.reserve(new_min_capacity);
        }

        // adjusting names_to_indices capacity
        current_size = static_cast<index_t>(names_to_indices.size());
        if (additional_size + current_size > current_capacity) {
            index_t new_min_capacity = std::max((current_capacity + 1) / 2 * 3, current_size + additional_size);
            names_to_indices.reserve(new_min_capacity);
        }

        for (auto &name : names) {
            add_object(name);
        }
    }

    /// @brief Adds objects to the union-find structure
    /// @param names of objects
    /// @param weights of objects
    void add_object(const std::vector<T> &names, const std::vector<workw_t> &weights) {
        if (names.size() != weights.size()) {
            throw std::runtime_error("Vectors of names and weights must be of equal length.");
        }

        // adjusting universe capacity
        index_t additional_size = static_cast<index_t>(names.size());
        index_t current_size = static_cast<index_t>(universe.size());
        index_t current_capacity = static_cast<index_t>(universe.capacity());
        if (additional_size + current_size > current_capacity) {
            index_t new_min_capacity = std::max((current_capacity + 1) / 2 * 3, current_size + additional_size);
            universe.reserve(new_min_capacity);
        }

        // adjusting names_to_indices capacity
        current_size = static_cast<index_t>(names_to_indices.size());
        if (additional_size + current_size > current_capacity) {
            index_t new_min_capacity = std::max((current_capacity + 1) / 2 * 3, current_size + additional_size);
            names_to_indices.reserve(new_min_capacity);
        }

        for (std::size_t i = 0; i < names.size(); i++) {
            add_object(names[i], weights[i]);
        }
    }

    /// @brief Adds objects to the union-find structure
    /// @param names of objects
    /// @param weights of objects
    /// @param memories of objects
    void add_object(const std::vector<T> &names, const std::vector<unsigned> &weights,
                    const std::vector<memw_t> &memories) {
        if (names.size() != weights.size()) {
            throw std::runtime_error("Vectors of names and weights must be of equal length.");
        }

        // adjusting universe capacity
        index_t additional_size = static_cast<index_t>(names.size());
        index_t current_size = static_cast<index_t>(universe.size());
        index_t current_capacity = static_cast<index_t>(universe.capacity());
        if (additional_size + current_size > current_capacity) {
            unsigned new_min_capacity = std::max((current_capacity + 1) / 2 * 3, current_size + additional_size);
            universe.reserve(new_min_capacity);
        }

        // adjusting names_to_indices capacity
        current_size = static_cast<index_t>(names_to_indices.size());
        if (additional_size + current_size > current_capacity) {
            index_t new_min_capacity = std::max((current_capacity + 1) / 2 * 3, current_size + additional_size);
            names_to_indices.reserve(new_min_capacity);
        }

        for (size_t i = 0; i < names.size(); i++) {
            add_object(names[i], weights[i], memories[i]);
        }
    }

    /// @brief Initiates a union-find structure
    explicit Union_Find_Universe() {}

    /// @brief Initiates a union-find structure
    /// @param names of objects
    explicit Union_Find_Universe(const std::vector<T> &names) { add_object(names); }

    /// @brief Initiates a union-find structure
    /// @param names of objects
    /// @param weights of objects
    explicit Union_Find_Universe(const std::vector<T> &names, const std::vector<workw_t> &weights) {
        add_object(names, weights);
    }

    /// @brief Initiates a union-find structure
    /// @param names of objects
    /// @param weights of objects
    explicit Union_Find_Universe(const std::vector<T> &names, const std::vector<workw_t> &weights,
                                 const std::vector<memw_t> &memories) {
        add_object(names, weights, memories);
    }
};

template<typename Graph_t>
using union_find_universe_t = Union_Find_Universe<vertex_idx_t<Graph_t>, vertex_idx_t<Graph_t>, v_workw_t<Graph_t>,
                                               v_memw_t<Graph_t>>;


} // namespace osp  