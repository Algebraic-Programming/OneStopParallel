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

#include "dag_divider/WavefrontParallelismDivider.hpp"

bool WavefrontParallelismDivider::compute_split(const std::vector<double> &parallelism, size_t &split) {

    double mean = 0.0;
    double variance = 0.0;
    compute_variance(parallelism, mean, variance);

    std::cout << "Mean: " << mean << ", Variance: " << variance << std::endl;

    size_t i = 1;
    double obj = std::numeric_limits<double>::max();
    split = 0;

    while (i < parallelism.size()) {

        std::vector<double> left(parallelism.begin(), parallelism.begin() + i);
        std::vector<double> right(parallelism.begin() + i, parallelism.end());

        double left_mean = 0;
        double left_variance = 0;
        compute_variance(left, left_mean, left_variance);

        double right_mean = 0;
        double right_variance = 0;
        compute_variance(right, right_mean, right_variance);

        if ((i > min_number_wavefronts) && (i < parallelism.size() - min_number_wavefronts)) {

            if (left_variance + right_variance < obj) {
                split = i;
                obj = left_variance + right_variance;
            }
        }
        // std::cout << "Split at " << i << " left variance: " << left_variance << " right variance: " << right_variance
        //           << " score: " << left_variance + right_variance << std::endl;

        i++;
    }

    if (obj < var_mult * variance) {
        std::cout << "Objective: " << obj << " Split at " << split << std::endl;
        return true;
    } else {
        std::cout << "No split found" << std::endl;
        return false;
    }
}

void WavefrontParallelismDivider::split_sequence(const std::vector<double> &seq, std::vector<size_t> &splits,  size_t offset, unsigned depth) {

    double mean = 0.0;
    double variance = 0.0;

    compute_variance(seq, mean, variance);

    if ((variance > var_threshold) && (depth < max_depth)) {
       
        size_t split = 0;
        if (compute_split(seq, split)) {

            splits.push_back(split + offset);

            std::vector<double> left(seq.begin(), seq.begin() + split);
            std::vector<double> right(seq.begin() + split, seq.end());

            depth++;

            split_sequence(left, splits, offset, depth);
            split_sequence(right, splits, split + offset, depth);
        }
    }
}

std::vector<std::vector<std::vector<unsigned>>> WavefrontParallelismDivider::divide(const ComputationalDag &dag_) {

    forward_statistics.clear();
 
    dag = &dag_;

    const std::vector<unsigned> bot_distance = dag->get_top_node_distance();

    std::vector<std::vector<unsigned>> level_sets(1);

    for (VertexType v = 0; v < bot_distance.size(); v++) {
        if (bot_distance[v] - 1 >= level_sets.size()) {
            level_sets.resize(bot_distance[v]);
        }
        level_sets[bot_distance[v] - 1].emplace_back(v);
    }

    compute_forward_statistics(level_sets, *dag);
    print_wavefront_statistics(forward_statistics);

    std::vector<double> forward_parallelism(forward_statistics.size());
    size_t i = 0;
    for (const auto &stat : forward_statistics) {
        forward_parallelism[i++] = stat.parallelism;
    }

    std::vector<size_t> cut_levels;
    split_sequence(forward_parallelism, cut_levels);

    std::sort(cut_levels.begin(), cut_levels.end());

    std::cout << "Cut levels: ";
    for (const auto level : cut_levels) {
        std::cout << level << " ";
    }
    std::cout << std::endl;

    std::vector<std::vector<std::vector<unsigned>>> vertex_maps;

    if (cut_levels.size() > 0) {

        vertex_maps = std::vector<std::vector<std::vector<unsigned>>>(cut_levels.size() + 1);

        unsigned level_set_idx = 0;
        for (unsigned i = 0; i < cut_levels.size(); i++) {

            Union_Find_Universe<unsigned> uf;
            for (; level_set_idx < cut_levels[i]; level_set_idx++) {
                for (const auto vertex : level_sets[level_set_idx]) {
                    uf.add_object(vertex, dag->nodeWorkWeight(vertex), dag->nodeMemoryWeight(vertex));
                }

                for (const auto &node : level_sets[level_set_idx]) {
                    for (const auto &child : dag->children(node)) {

                        if (uf.is_in_universe(child))
                            uf.join_by_name(node, child);
                    }

                    for (const auto &parent : dag->parents(node)) {
                        if (uf.is_in_universe(parent)) {
                            uf.join_by_name(parent, node);
                        }
                    }
                }
            }
            vertex_maps[i] = uf.get_connected_components();
        }

        Union_Find_Universe<unsigned> uf;
        for (; level_set_idx < level_sets.size(); level_set_idx++) {
            for (const auto vertex : level_sets[level_set_idx]) {
                uf.add_object(vertex, dag->nodeWorkWeight(vertex), dag->nodeMemoryWeight(vertex));
            }

            for (const auto &node : level_sets[level_set_idx]) {
                for (const auto &child : dag->children(node)) {

                    if (uf.is_in_universe(child))
                        uf.join_by_name(node, child);
                }

                for (const auto &parent : dag->parents(node)) {
                    if (uf.is_in_universe(parent)) {
                        uf.join_by_name(parent, node);
                    }
                }
            }
        }

        vertex_maps.back() = uf.get_connected_components();

    } else if (forward_statistics.back().number_of_connected_components != 1) {

        std::cout << "No cut levels found, but the graph has "
                  << forward_statistics.back().number_of_connected_components << " connected cmponents." << std::endl;

        vertex_maps = std::vector<std::vector<std::vector<unsigned>>>(1);
        vertex_maps[0] = forward_statistics.back().connected_components_vertices;
    }
    return vertex_maps;
}

void WavefrontParallelismDivider::compute_forward_statistics(const std::vector<std::vector<unsigned>> &level_sets,
                                                             const ComputationalDag &dag) {

    forward_statistics.resize(level_sets.size());

    unsigned level_set_idx = 0;

    Union_Find_Universe<unsigned> uf;
    for (const auto vertex : level_sets[level_set_idx]) {
        uf.add_object(vertex, dag.nodeWorkWeight(vertex), dag.nodeMemoryWeight(vertex));
    }

    const auto components = uf.get_connected_components_weights_and_memories();

    forward_statistics[level_set_idx].number_of_connected_components = components.size();

    forward_statistics[level_set_idx].total_weight = 0;
    forward_statistics[level_set_idx].max_weight = 0;
    for (unsigned i = 0; i < components.size(); i++) {

        forward_statistics[level_set_idx].total_weight += std::get<1>(components[i]);

        forward_statistics[level_set_idx].connected_components_weights.emplace_back(std::get<1>(components[i]));
        forward_statistics[level_set_idx].connected_components_memories.emplace_back(std::get<2>(components[i]));
        forward_statistics[level_set_idx].connected_components_vertices.emplace_back(std::get<0>(components[i]));

        for (const auto &vertex : forward_statistics[level_set_idx].connected_components_vertices[i]) {
            forward_statistics[level_set_idx].max_weight =
                std::max(forward_statistics[level_set_idx].max_weight, (double)dag.nodeWorkWeight(vertex));
        }
    }

    forward_statistics[level_set_idx].total_acc_weight = forward_statistics[level_set_idx].total_weight;
    forward_statistics[level_set_idx].max_acc_weight = forward_statistics[level_set_idx].max_weight;

    forward_statistics[level_set_idx].parallelism =
        forward_statistics[level_set_idx].total_acc_weight / forward_statistics[level_set_idx].max_acc_weight;

    bool components_stable = false;

    level_set_idx++;

    while (level_set_idx < level_sets.size()) {

        for (const auto vertex : level_sets[level_set_idx]) {
            uf.add_object(vertex, dag.nodeWorkWeight(vertex), dag.nodeMemoryWeight(vertex));
        }

        for (const auto &node : level_sets[level_set_idx]) {
            for (const auto &child : dag.children(node)) {

                if (uf.is_in_universe(child))
                    uf.join_by_name(node, child);
            }

            for (const auto &parent : dag.parents(node)) {
                if (uf.is_in_universe(parent)) {
                    uf.join_by_name(parent, node);
                }
            }
        }

        const auto components = uf.get_connected_components_weights_and_memories();

        forward_statistics[level_set_idx].number_of_connected_components = components.size();

        forward_statistics[level_set_idx].total_weight = 0;
        forward_statistics[level_set_idx].max_weight = 0;

        for (unsigned i = 0; i < components.size(); i++) {

            forward_statistics[level_set_idx].total_weight += std::get<1>(components[i]);

            forward_statistics[level_set_idx].connected_components_weights.emplace_back(std::get<1>(components[i]));
            forward_statistics[level_set_idx].connected_components_memories.emplace_back(std::get<2>(components[i]));
            forward_statistics[level_set_idx].connected_components_vertices.emplace_back(std::get<0>(components[i]));

            for (const auto &vertex : forward_statistics[level_set_idx].connected_components_vertices[i]) {
                forward_statistics[level_set_idx].max_weight =
                    std::max(forward_statistics[level_set_idx].max_weight, (double)dag.nodeWorkWeight(vertex));
            }
        }

        forward_statistics[level_set_idx].total_acc_weight =
            forward_statistics[level_set_idx - 1].total_acc_weight + forward_statistics[level_set_idx].total_weight;
        forward_statistics[level_set_idx].max_acc_weight =
            forward_statistics[level_set_idx - 1].max_acc_weight + forward_statistics[level_set_idx].max_weight;

        forward_statistics[level_set_idx].parallelism =
            forward_statistics[level_set_idx].total_acc_weight / forward_statistics[level_set_idx].max_acc_weight;

 
        level_set_idx++;
    }
}

void WavefrontParallelismDivider::print_wavefront_statistics(const std::vector<wavefron_statistics> &statistics,
                                                             bool reverse) {
    if (reverse) {

        for (size_t i = 0; i < statistics.size(); i++) {

            std::cout << "Level " << i << " has " << statistics[i].number_of_connected_components
                      << " connected components, and parallelism " << statistics[i].parallelism << std::endl;
            // for (size_t j = 0; j < statistics[i].connected_components_vertices.size(); j++) {

            //     std::cout << "Component " << j << " has " <<
            //     statistics[i].connected_components_vertices[j].size()
            //               << " vertice(s), weight: " << statistics[i].connected_components_weights[j]
            //               << ", memory: " << statistics[i].connected_components_memories[j] << std::endl;
            // }
        }

    } else {
        for (size_t i = 0; i < statistics.size(); i++) {
            std::cout << "Level " << i << " has " << statistics[i].number_of_connected_components
                      << " connected components, and parallelism " << statistics[i].parallelism << std::endl;
            // for (size_t j = 0; j < statistics[i].connected_components_vertices.size(); j++) {

            //     std::cout << "Component " << j << " has " <<
            //     statistics[i].connected_components_vertices[j].size()
            //               << " vertice(s), weight: " << statistics[i].connected_components_weights[j]
            //               << ", memory: " << statistics[i].connected_components_memories[j] << std::endl;
            // }
        }
    }
}

void WavefrontParallelismDivider::compute_variance(const std::vector<double> &data, double &mean, double &variance) {

    mean = 0;
    variance = 0;
    for (const auto &d : data) {
        mean += d;
    }
    mean /= data.size();

    for (const auto &d : data) {
        variance += (d - mean) * (d - mean);
    }

    variance /= data.size();
}