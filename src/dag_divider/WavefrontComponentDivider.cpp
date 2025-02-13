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

#include "dag_divider/WavefrontComponentDivider.hpp"

bool WavefrontComponentDivider::compute_split_var(const std::vector<double> &parallelism, size_t &split) {

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

        if (((i > 1) && (i < parallelism.size() - 1)) ||
            ((*(left.end()) == 1 && *(right.begin()) != 1) || (*(left.end()) != 1 && *(right.begin()) == 1))) {

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

void WavefrontComponentDivider::split_sequence_var(const std::vector<double> &seq, std::vector<size_t> &splits,
                                                   size_t offset) {

    double mean = 0.0;
    double variance = 0.0;

    compute_variance(seq, mean, variance);

    if (variance > var_threshold) {

        size_t split = 0;
        if (compute_split_var(seq, split)) {

            splits.push_back(split + offset);

            std::vector<double> left(seq.begin(), seq.begin() + split);
            std::vector<double> right(seq.begin() + split, seq.end());

            split_sequence_var(left, splits, offset);
            split_sequence_var(right, splits, split + offset);
        }
    }
}

bool WavefrontComponentDivider::compute_split_min_diff(const std::vector<double> &sequence, size_t &split,
                                                       bool reverse) {

    if (reverse) {
        for (size_t i = sequence.size() - 3; i > min_subseq_len - 2; i--) {

            if (sequence[i] > sequence[i + 1]) {

                split = i + 1;
                return true;
            }
        }
    } else {
        for (size_t i = 1; i < sequence.size() - (min_subseq_len + 1); i++) {

            if (sequence[i] < sequence[i + 1]) {

                split = i + 1;
                return true;
            }
        }
    }

    return false;
}

void WavefrontComponentDivider::split_sequence_min_diff_fwd(const std::vector<double> &seq, std::vector<size_t> &splits,
                                                            size_t offset) {

    if (seq[0] > diff_threshold + seq.back() && seq.size() > 2 * min_subseq_len) {

        size_t split = 0;
        if (compute_split_min_diff(seq, split, true)) {

            splits.push_back(split + offset);

            std::vector<double> left(seq.begin(), seq.begin() + split);
            split_sequence_min_diff_fwd(left, splits, offset);
        }
    }
}

void WavefrontComponentDivider::split_sequence_min_diff_bwd(const std::vector<double> &seq, std::vector<size_t> &splits,
                                                            size_t offset) {

    if (seq[0] + diff_threshold < seq.back() && seq.size() > 2 * min_subseq_len) {

        size_t split = 0;
        if (compute_split_min_diff(seq, split, false)) {
     
            splits.push_back(split + offset);

            std::vector<double> left(seq.begin() + split, seq.end());
            split_sequence_min_diff_bwd(left, splits, split + offset);
        }
    }
}

std::vector<size_t> WavefrontComponentDivider::combine_split_sequences(std::vector<size_t> &fwd_splits,
                                                                       std::vector<size_t> &bwd_splits) {

    std::sort(bwd_splits.begin(), bwd_splits.end());
    std::sort(fwd_splits.begin(), fwd_splits.end());

    std::cout << "FWD Splits: ";
    for (const auto split : fwd_splits) {
        std::cout << split << " ";
    }
    std::cout << std::endl;

    std::cout << "BWD Splits: ";
    for (const auto split : bwd_splits) {
        std::cout << split << " ";
    }
    std::cout << std::endl;

    std::vector<size_t> cut_levels;

    if (fwd_splits.size() > 0 && bwd_splits.size() > 0) {

        unsigned fwd = 0;
        unsigned bwd = 0;

        while (fwd < fwd_splits.size() && bwd < bwd_splits.size()) {

            if (fwd_splits[fwd] <= bwd_splits[bwd]) {

                if (((fwd + bwd) % 2) == 0) {
                    cut_levels.emplace_back(fwd_splits[fwd++]);
                } else {
                    cut_levels.emplace_back(bwd_splits[bwd++]);
                }

            } else {
                break;
            }
        }

        if (bwd < bwd_splits.size()) {
            while (bwd < bwd_splits.size()) {

                if (bwd_splits[bwd] > cut_levels.back()) {
                    cut_levels.emplace_back(bwd_splits[bwd++]);
                } else {
                    bwd++;
                }
            }
        }

    } else if (fwd_splits.size() > 0) {
        cut_levels = fwd_splits;
    } else if (bwd_splits.size() > 0) {
        cut_levels = bwd_splits;
    }

    std::sort(cut_levels.begin(), cut_levels.end());

    std::cout << "Cut levels: ";
    for (const auto level : cut_levels) {
        std::cout << level << " ";
    }
    std::cout << std::endl;

    return cut_levels;
}

std::vector<size_t>
WavefrontComponentDivider::compute_cut_levels_fwd_bwd_var(std::vector<std::vector<unsigned>> &level_sets) {

    compute_forward_statistics(level_sets, *dag);
    print_wavefront_statistics(forward_statistics);

    std::cout << " ------------------- " << std::endl;

    compute_backward_statistics(level_sets, *dag);
    print_wavefront_statistics(backward_statistics, true);

    std::vector<double> forward_parallelism(forward_statistics.size());
    size_t i = 0;
    for (const auto &stat : forward_statistics) {
        forward_parallelism[i++] = stat.number_of_connected_components;
    }

    std::vector<size_t> fwd_splits;
    split_sequence_var(forward_parallelism, fwd_splits);

    std::vector<double> backward_parallelism(backward_statistics.size());
    i = 0;
    for (const auto &stat : backward_statistics) {
        backward_parallelism[i++] = stat.number_of_connected_components;
    }

    double min_components = std::numeric_limits<double>::max();
    for (auto iter = backward_parallelism.rbegin(); iter != backward_parallelism.rend(); ++iter) {
        min_components = std::min(*iter, min_components);
        *iter = min_components;
    }

    std::vector<size_t> bwd_splits;
    split_sequence_var(backward_parallelism, bwd_splits);

    std::sort(bwd_splits.begin(), bwd_splits.end(), std::greater<>());

    return combine_split_sequences(fwd_splits, bwd_splits);
}

std::vector<size_t>
WavefrontComponentDivider::compute_cut_levels_fwd_bwd_min_diff(std::vector<std::vector<unsigned>> &level_sets) {

    compute_forward_statistics(level_sets, *dag);
    print_wavefront_statistics(forward_statistics);

    std::cout << " ------------------- " << std::endl;

    compute_backward_statistics(level_sets, *dag);
    print_wavefront_statistics(backward_statistics, true);

    std::vector<double> forward_parallelism(forward_statistics.size());
    size_t i = 0;
    for (const auto &stat : forward_statistics) {
        forward_parallelism[i++] = stat.number_of_connected_components;
    }

    std::vector<size_t> fwd_splits;

    split_sequence_min_diff_fwd(forward_parallelism, fwd_splits);

    std::vector<double> backward_parallelism(backward_statistics.size());
    i = 0;
    for (const auto &stat : backward_statistics) {
        backward_parallelism[i++] = stat.number_of_connected_components;
    }

    double min_components = std::numeric_limits<double>::max();
    for (auto iter = backward_parallelism.rbegin(); iter != backward_parallelism.rend(); ++iter) {
        min_components = std::min(*iter, min_components);
        *iter = min_components;
    }

    std::vector<size_t> bwd_splits;

    split_sequence_min_diff_bwd(backward_parallelism, bwd_splits);

    std::sort(bwd_splits.begin(), bwd_splits.end(), std::greater<>());

    return combine_split_sequences(fwd_splits, bwd_splits);
}

std::vector<std::vector<std::vector<unsigned>>> WavefrontComponentDivider::divide(const ComputationalDag &dag_) {

    forward_statistics.clear();
    backward_statistics.clear();

    dag = &dag_;

    const std::vector<unsigned> bot_distance = dag->get_top_node_distance();

    std::vector<std::vector<unsigned>> level_sets(1);

    for (VertexType v = 0; v < bot_distance.size(); v++) {
        if (bot_distance[v] - 1 >= level_sets.size()) {
            level_sets.resize(bot_distance[v]);
        }
        level_sets[bot_distance[v] - 1].emplace_back(v);
    }

    std::vector<size_t> cut_levels;

    switch (split_method) {

    case SplitMethod::MIN_DIFF:
        cut_levels = compute_cut_levels_fwd_bwd_min_diff(level_sets);
        break;

    case SplitMethod::VARIANCE:
        cut_levels = compute_cut_levels_fwd_bwd_var(level_sets);
        break;

    default:
        cut_levels = compute_cut_levels_fwd_bwd_min_diff(level_sets);
        break;
    }

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
    //
}

void WavefrontComponentDivider::compute_forward_statistics(const std::vector<std::vector<unsigned>> &level_sets,
                                                           const ComputationalDag &dag) {

    forward_statistics.resize(level_sets.size());

    unsigned level_set_idx = 0;

    Union_Find_Universe<unsigned> uf;
    for (const auto vertex : level_sets[level_set_idx]) {
        uf.add_object(vertex, dag.nodeWorkWeight(vertex), dag.nodeMemoryWeight(vertex));
    }

    const auto components = uf.get_connected_components_weights_and_memories();

    forward_statistics[level_set_idx].number_of_connected_components = components.size();

    for (unsigned i = 0; i < components.size(); i++) {
        forward_statistics[level_set_idx].connected_components_weights.emplace_back(std::get<1>(components[i]));
        forward_statistics[level_set_idx].connected_components_memories.emplace_back(std::get<2>(components[i]));
        forward_statistics[level_set_idx].connected_components_vertices.emplace_back(std::get<0>(components[i]));
    }

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

        for (unsigned i = 0; i < components.size(); i++) {
            forward_statistics[level_set_idx].connected_components_weights.emplace_back(std::get<1>(components[i]));
            forward_statistics[level_set_idx].connected_components_memories.emplace_back(std::get<2>(components[i]));
            forward_statistics[level_set_idx].connected_components_vertices.emplace_back(std::get<0>(components[i]));
        }

        level_set_idx++;
    }
}

void WavefrontComponentDivider::compute_backward_statistics(const std::vector<std::vector<unsigned>> &level_sets,
                                                            const ComputationalDag &dag) {

    backward_statistics.resize(level_sets.size());

    unsigned level_set_idx = level_sets.size() - 1;

    Union_Find_Universe<unsigned> uf;
    for (const auto vertex : level_sets[level_set_idx]) {
        uf.add_object(vertex, dag.nodeWorkWeight(vertex), dag.nodeMemoryWeight(vertex));
    }

    const auto components = uf.get_connected_components_weights_and_memories();

    backward_statistics[level_set_idx].number_of_connected_components = components.size();

    for (unsigned i = 0; i < components.size(); i++) {
        backward_statistics[level_set_idx].connected_components_weights.emplace_back(std::get<1>(components[i]));
        backward_statistics[level_set_idx].connected_components_memories.emplace_back(std::get<2>(components[i]));
        backward_statistics[level_set_idx].connected_components_vertices.emplace_back(std::get<0>(components[i]));
    }

    size_t min_number_of_components = dag.numberOfVertices();

    while (level_set_idx > 0) {

        level_set_idx--;
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

        backward_statistics[level_set_idx].number_of_connected_components = components.size();

        for (unsigned i = 0; i < components.size(); i++) {
            backward_statistics[level_set_idx].connected_components_weights.emplace_back(std::get<1>(components[i]));
            backward_statistics[level_set_idx].connected_components_memories.emplace_back(std::get<2>(components[i]));
            backward_statistics[level_set_idx].connected_components_vertices.emplace_back(std::get<0>(components[i]));
        }
    }
}

void WavefrontComponentDivider::print_wavefront_statistics(const std::vector<wavefron_statistics> &statistics,
                                                           bool reverse) {
    if (reverse) {

        for (size_t i = 0; i < statistics.size(); i++) {

            std::cout << "Level " << i << " has " << statistics[i].number_of_connected_components
                      << " connected components" << std::endl;
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
                      << " connected components" << std::endl;
            // for (size_t j = 0; j < statistics[i].connected_components_vertices.size(); j++) {

            //     std::cout << "Component " << j << " has " <<
            //     statistics[i].connected_components_vertices[j].size()
            //               << " vertice(s), weight: " << statistics[i].connected_components_weights[j]
            //               << ", memory: " << statistics[i].connected_components_memories[j] << std::endl;
            // }
        }
    }
}

void WavefrontComponentDivider::compute_variance(const std::vector<double> &data, double &mean, double &variance) {

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