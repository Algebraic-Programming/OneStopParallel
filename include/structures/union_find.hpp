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
#include <unordered_map>
#include <vector>
#include <stdexcept>

/// @brief Structure to execute a union-find algorithm
template<typename T>
struct union_find_object {
    const T name; // unique identifier
    unsigned parent_index;
    unsigned rank;
    unsigned weight;
    unsigned memory;

    explicit union_find_object(const T &name_, int parent_index_, unsigned weight_ = 0, unsigned memory_ = 0)
        : name(name_), parent_index(parent_index_), weight(weight_), memory(memory_) {
        rank = 1;
    }
};

/// @brief Class to execute a union-find algorithm
template<typename T>
class Union_Find_Universe {
  private:
    std::vector<union_find_object<T>> universe;
    std::unordered_map<T, unsigned> names_to_indices;
    std::set<unsigned> component_indices;

    unsigned find_origin(unsigned index) {
        while (index != universe[index].parent_index) {
            universe[index].parent_index = universe[universe[index].parent_index].parent_index;
            index = universe[index].parent_index;
        }
        return index;
    }

    int join(unsigned index, unsigned other_index) {
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

    unsigned get_index_from_name(const T &name) const { return names_to_indices.at(name); }

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
    unsigned get_number_of_connected_components() const { return component_indices.size(); }

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
    std::vector<std::pair<T, unsigned>> get_component_names_and_weights() const {
        std::vector<std::pair<T, unsigned>> component_names_and_weights;
        component_names_and_weights.reserve(component_indices.size());
        for (auto &indx : component_indices) {
            component_names_and_weights.emplace_back({universe[indx].name, universe[indx].weight});
        }
        return component_names_and_weights;
    }

    /// @brief Retrieves the (current) names of components together with their weight and memory
    std::vector<std::tuple<T, unsigned, unsigned>> get_component_names_weights_and_memory() const {
        std::vector<std::tuple<T, unsigned, unsigned>> component_names_weights_and_memory;
        component_names_weights_and_memory.reserve(component_indices.size());
        for (auto &indx : component_indices) {
            component_names_weights_and_memory.emplace_back({universe[indx].name, universe[indx].weight, universe[indx].memory});
        }
        return component_names_weights_and_memory;
    }

    /// @brief Retrieves the weight of the component containing the given object
    /// @param name of object
    unsigned get_weight_of_component_by_name(const T &name) {
        unsigned index = get_index_from_name(name);
        index = find_origin(index);
        return universe[index].weight;
    }

    /// @brief Retrieves the memory of the component containing the given object
    /// @param name of object
    unsigned get_memory_of_component_by_name(const T &name) {
        unsigned index = get_index_from_name(name);
        index = find_origin(index);
        return universe[index].memory;
    }

    /// @brief Retrieves the connected components
    /// @return Partition of the names of objects according to the connected components
    std::vector<std::vector<T>> get_connected_components() {
        std::vector<std::vector<unsigned>> connected_components_by_index;
        connected_components_by_index.resize(universe.size());
        for (unsigned i = 0; i < (unsigned) universe.size(); i++) {
            connected_components_by_index[find_origin(i)].emplace_back(i);
        }

        std::vector<std::vector<T>> connected_components_by_name;
        for (auto &comp : connected_components_by_index) {
            if (comp.empty()) {
                continue;
            }
            std::vector<T> names_in_comp;
            names_in_comp.reserve(comp.size());
            for (auto &indx : comp) {
                names_in_comp.emplace_back(universe[indx].name);
            }
            connected_components_by_name.emplace_back(names_in_comp);
        }

        return connected_components_by_name;
    }

    /// @brief Retrieves the connected components and their respective weights
    /// @return Partition of the names of objects according to the connected components together with their respective
    /// weight
    std::vector<std::pair<std::vector<T>, unsigned>> get_connected_components_and_weights() {
        std::vector<std::vector<unsigned>> connected_components_by_index;
        connected_components_by_index.resize(universe.size());
        for (unsigned i = 0; i < (unsigned) universe.size(); i++) {
            connected_components_by_index[find_origin(i)].emplace_back(i);
        }

        std::vector<std::pair<std::vector<T>, unsigned>> connected_components_by_name_incl_weight;
        for (auto &comp : connected_components_by_index) {
            if (comp.empty()) {
                continue;
            }

            unsigned comp_weight = universe[find_origin(comp[0])].weight;

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
    std::vector<std::tuple<std::vector<T>, unsigned, unsigned>> get_connected_components_weights_and_memories() {
        std::vector<std::vector<unsigned>> connected_components_by_index;
        connected_components_by_index.resize(universe.size());
        for (unsigned i = 0; i < (unsigned) universe.size(); i++) {
            connected_components_by_index[find_origin(i)].emplace_back(i);
        }

        std::vector<std::tuple<std::vector<T>, unsigned, unsigned>> connected_components_by_name_incl_weight_memory;
        for (auto &comp : connected_components_by_index) {
            if (comp.empty()) {
                continue;
            }

            unsigned comp_weight = universe[find_origin(comp[0])].weight;
            unsigned comp_memory = universe[find_origin(comp[0])].memory;

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
        unsigned new_index = universe.size();
        universe.emplace_back(name, new_index);
        names_to_indices[name] = new_index;
        component_indices.emplace(new_index);
    }

    /// @brief Adds object to the union-find structure with given weight
    /// @param name of object
    /// @param weight of object
    void add_object(const T &name, const unsigned weight) {
        if (names_to_indices.find(name) != names_to_indices.end()) {
            throw std::runtime_error("This name already exists in the universe.");
        }
        unsigned new_index = universe.size();
        universe.emplace_back(name, new_index, weight);
        names_to_indices[name] = new_index;
        component_indices.emplace(new_index);
    }

    /// @brief Adds object to the union-find structure with given weight and memory
    /// @param name of object
    /// @param weight of object
    /// @param memory of object
    void add_object(const T &name, const unsigned weight, const unsigned memory) {
        if (names_to_indices.find(name) != names_to_indices.end()) {
            throw std::runtime_error("This name already exists in the universe.");
        }
        unsigned new_index = universe.size();
        universe.emplace_back(name, new_index, weight, memory);
        names_to_indices[name] = new_index;
        component_indices.emplace(new_index);
    }

    /// @brief Adds objects to the union-find structure
    /// @param names of objects
    void add_object(const std::vector<T> &names) {
        // adjusting universe capacity
        unsigned additional_size = names.size();
        unsigned current_size = universe.size();
        unsigned current_capacity = universe.capacity();
        if (additional_size + current_size > current_capacity) {
            unsigned new_min_capacity = std::max((current_capacity + 1) / 2 * 3, current_size + additional_size);
            universe.reserve(new_min_capacity);
        }

        // adjusting names_to_indices capacity
        current_size = names_to_indices.size();
        if (additional_size + current_size > current_capacity) {
            unsigned new_min_capacity = std::max((current_capacity + 1) / 2 * 3, current_size + additional_size);
            names_to_indices.reserve(new_min_capacity);
        }

        for (auto &name : names) {
            add_object(name);
        }
    }

    /// @brief Adds objects to the union-find structure
    /// @param names of objects
    /// @param weights of objects
    void add_object(const std::vector<T> &names, const std::vector<unsigned> &weights) {
        if (names.size() != weights.size()) {
            throw std::runtime_error("Vectors of names and weights must be of equal length.");
        }

        // adjusting universe capacity
        unsigned additional_size = names.size();
        unsigned current_size = universe.size();
        unsigned current_capacity = universe.capacity();
        if (additional_size + current_size > current_capacity) {
            unsigned new_min_capacity = std::max((current_capacity + 1) / 2 * 3, current_size + additional_size);
            universe.reserve(new_min_capacity);
        }

        // adjusting names_to_indices capacity
        current_size = names_to_indices.size();
        if (additional_size + current_size > current_capacity) {
            unsigned new_min_capacity = std::max((current_capacity + 1) / 2 * 3, current_size + additional_size);
            names_to_indices.reserve(new_min_capacity);
        }

        for (size_t i = 0; i < names.size(); i++) {
            add_object(names[i], weights[i]);
        }
    }

    /// @brief Adds objects to the union-find structure
    /// @param names of objects
    /// @param weights of objects
    /// @param memories of objects
    void add_object(const std::vector<T> &names, const std::vector<unsigned> &weights, const std::vector<unsigned> &memories) {
        if (names.size() != weights.size()) {
            throw std::runtime_error("Vectors of names and weights must be of equal length.");
        }

        // adjusting universe capacity
        unsigned additional_size = names.size();
        unsigned current_size = universe.size();
        unsigned current_capacity = universe.capacity();
        if (additional_size + current_size > current_capacity) {
            unsigned new_min_capacity = std::max((current_capacity + 1) / 2 * 3, current_size + additional_size);
            universe.reserve(new_min_capacity);
        }

        // adjusting names_to_indices capacity
        current_size = names_to_indices.size();
        if (additional_size + current_size > current_capacity) {
            unsigned new_min_capacity = std::max((current_capacity + 1) / 2 * 3, current_size + additional_size);
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
    explicit Union_Find_Universe(const std::vector<T> &names, const std::vector<unsigned> &weights) {
        add_object(names, weights);
    }

    /// @brief Initiates a union-find structure
    /// @param names of objects
    /// @param weights of objects
    explicit Union_Find_Universe(const std::vector<T> &names, const std::vector<unsigned> &weights, const std::vector<unsigned> &memories) {
        add_object(names, weights, memories);
    }
};