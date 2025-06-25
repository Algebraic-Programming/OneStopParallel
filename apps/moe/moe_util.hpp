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

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <iostream> // Added for std::cout
#include <utility>  // for std::pair
#include <map>      // for std::map
#include <tuple>    // for std::tuple
#include <vector>

struct edge {

    unsigned layer;
    unsigned expert_from; // expert in layer
    unsigned expert_to;   // expert in layer + 1

    unsigned weight;
};

struct hyper_edge {

    std::vector<std::pair<unsigned, unsigned>> net; // paris of layer + expert
};


struct active_experts {
    unsigned layer_id;
    std::vector<int> experts;
};


struct token_pass {

    std::vector<active_experts> layer_experts;

};


struct moe_ilp_params {

    moe_ilp_params() {}
    moe_ilp_params(const std::vector<std::vector<unsigned>> &expert_weights_, const std::vector<edge> &edges_,
                   unsigned num_gpus_)
        : num_gpus(num_gpus_), expert_weights(expert_weights_), edges(edges_) {

        if (expert_weights_.empty()) {
            num_experts = 0;
            num_layers = 0;
        } else {
            num_experts = static_cast<unsigned>(expert_weights_.size());
            num_layers = static_cast<unsigned>(expert_weights_[0].size());

            for (unsigned expert = 0; expert < num_experts; expert++) {
                if (expert_weights[expert].size() != num_layers)
                    throw std::invalid_argument("Invalid Argument while constructing moe_ilp_params: expert_weights "
                                                "must be a rectangular matrix!");
            }
        }
    }

    unsigned num_gpus = 0;

    std::vector<std::vector<unsigned>> expert_weights;
    std::vector<edge> edges;

    unsigned num_layers = 0;
    unsigned num_experts = 0;
};




inline std::vector<token_pass> read_token_passes_from_file(const std::string &filename) {
    std::vector<token_pass> all_token_passes;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    token_pass current_pass;
    std::string line;

    while (std::getline(infile, line)) {
        const auto first = line.find_first_not_of(" \t\n\v\f\r");
        if (std::string::npos == first) {
            continue;
        }
        const auto last = line.find_last_not_of(" \t\n\v\f\r");
        line = line.substr(first, (last - first + 1));

        if (line.rfind("%%", 0) == 0) { // starts with %%
            if (!current_pass.layer_experts.empty()) {
                all_token_passes.push_back(current_pass);
                current_pass.layer_experts.clear();
            }
        } else {
            std::stringstream line_stream(line);

            unsigned layer_id;
            char colon = 0;

            line_stream >> layer_id >> colon;

            if (line_stream.fail() || colon != ':') {
                continue; // Skip malformed line
            }

            active_experts current_active_experts;
            current_active_experts.layer_id = layer_id;

            int expert_id;
            char comma;

            while (line_stream >> expert_id) {
                current_active_experts.experts.push_back(expert_id);
                line_stream >> comma; // consume the comma, if present
            }

            current_pass.layer_experts.push_back(current_active_experts);
        }
    }

    if (!current_pass.layer_experts.empty()) {
        all_token_passes.push_back(current_pass);
    }

    return all_token_passes;
}

inline void print_token_pass(const token_pass& tp) {
    for (const auto& ae : tp.layer_experts) {
        std::cout << ae.layer_id << ": ";
        for (size_t i = 0; i < ae.experts.size(); ++i) {
            std::cout << ae.experts[i];
            if (i < ae.experts.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << std::endl;
    }
}

inline void print_token_passes(const std::vector<token_pass>& tps) {
    for (size_t i = 0; i < tps.size(); ++i) {
        std::cout << "--- Token Pass " << i + 1 << " ---" << std::endl;
        print_token_pass(tps[i]);
        std::cout << "--------------------" << std::endl;
    }
}

inline std::vector<edge> generate_edges_from_token_pass(const token_pass &tp, unsigned default_weight = 1) {
    std::vector<edge> edges;

    if (tp.layer_experts.size() < 2) {
        return edges; // Not enough layers to create edges between them
    }

    // Iterate through layers up to the second to last one
    for (size_t i = 0; i < tp.layer_experts.size() - 1; ++i) {
        const auto &from_layer = tp.layer_experts[i];
        const auto &to_layer = tp.layer_experts[i + 1];

        // Check if the 'to' layer should be ignored (e.g., contains only -1)
        if (!to_layer.experts.empty() && to_layer.experts[0] == -1) {
            continue; // Skip this pair of layers
        }

        // Create all combinations of edges between experts of the two layers
        for (int expert_from : from_layer.experts) {
            if (expert_from < 0)
                continue; // Ignore invalid expert IDs

            for (int expert_to : to_layer.experts) {
                if (expert_to < 0)
                    continue; // Ignore invalid expert IDs

                edges.push_back({from_layer.layer_id - 1, static_cast<unsigned>(expert_from),
                                 static_cast<unsigned>(expert_to), default_weight});
            }
        }
    }

    return edges;
}


inline std::vector<edge> generate_weighted_edges_from_token_passes(const std::vector<token_pass> &passes) {
    std::map<std::tuple<unsigned, unsigned, unsigned>, unsigned> edge_counts;

    for (const auto &pass : passes) {
        // Generate edges for the current pass (each with a temporary weight of 1)
        std::vector<edge> current_edges = generate_edges_from_token_pass(pass);
        for (const auto &e : current_edges) {
            // Use a tuple to uniquely identify the edge and increment its count
            edge_counts[{e.layer, e.expert_from, e.expert_to}]++;
        }
    }

    std::vector<edge> weighted_edges;
    weighted_edges.reserve(edge_counts.size());
    for (const auto &[edge_key, weight] : edge_counts) {
        const auto &[layer, from, to] = edge_key;
        weighted_edges.push_back({layer, from, to, weight});
    }

    return weighted_edges;
}


inline std::vector<std::vector<unsigned>> calculate_expert_weights(
    const std::vector<token_pass>& passes,
    unsigned total_num_experts,
    unsigned total_num_layers) {

    // Initialize expert_weights matrix with zeros
    std::vector<std::vector<unsigned>> expert_weights(
        total_num_experts, std::vector<unsigned>(total_num_layers, 0));

    for (const auto& tp : passes) {
        for (const auto& ae : tp.layer_experts) {
            // Convert 1-based layer_id from file to 0-based index for array access
            unsigned layer_idx = ae.layer_id - 1;

            // Ensure layer_idx is within bounds and process experts
            if (layer_idx < total_num_layers) {
                for (int expert_id : ae.experts) {
                    if (expert_id >= 0 && static_cast<unsigned>(expert_id) < total_num_experts) {
                        expert_weights[static_cast<unsigned>(expert_id)][layer_idx]++;
                    }
                }
            }
        }
    }

    return expert_weights;
}



inline moe_ilp_params
create_moe_ilp_params_from_token_passes(const std::vector<token_pass> &passes, unsigned num_gpus) {
    if (passes.empty()) {
        throw std::invalid_argument("Cannot create MoE ILP params from an empty vector of token passes.");
    }

    unsigned max_layer_id = 0;
    int max_expert_id = -1;

    for (const auto &pass : passes) {
        for (const auto &layer_expert_info : pass.layer_experts) {
            if (layer_expert_info.layer_id > max_layer_id) {
                max_layer_id = layer_expert_info.layer_id;
            }
            for (int expert_id : layer_expert_info.experts) {
                if (expert_id > max_expert_id) {
                    max_expert_id = expert_id;
                }
            }
        }
    }

    unsigned num_layers = max_layer_id;
    unsigned num_experts = (max_expert_id >= 0) ? (static_cast<unsigned>(max_expert_id) + 1) : 0;

    if (num_layers == 0 || num_experts == 0) {
        throw std::runtime_error("Could not deduce a valid number of layers or experts from the provided token passes.");
    }

    return moe_ilp_params(calculate_expert_weights(passes, num_experts, num_layers),
                          generate_weighted_edges_from_token_passes(passes), num_gpus);
}


inline void print_layer_details(const moe_ilp_params &params, unsigned layer_idx_to_print) {
    if (layer_idx_to_print >= params.num_layers) {
        std::cout << "Error: Layer ID " << layer_idx_to_print << " is out of bounds (max layer is "
                  << params.num_layers - 1 << ")." << std::endl;
        return;
    }

    std::cout << "\n--- Details for Layer " << layer_idx_to_print << " ---" << std::endl;

    bool found_active_expert = false;
    for (unsigned expert_id = 0; expert_id < params.num_experts; ++expert_id) {
        unsigned expert_weight = params.expert_weights[expert_id][layer_idx_to_print];

        if (expert_weight > 0) {
            found_active_expert = true;
            std::cout << "  Expert " << expert_id << " (Weight: " << expert_weight << "):" << std::endl;

            for (const auto &edge : params.edges) {
                if (edge.layer == layer_idx_to_print && edge.expert_from == expert_id) {
                    std::cout << "    -> To Expert " << edge.expert_to << " (Edge Weight: " << edge.weight << ")"
                              << std::endl;
                }
            }
        }
    }
    if (!found_active_expert) {
        std::cout << "  No active experts found in this layer." << std::endl;
    }
    std::cout << "-----------------------------------" << std::endl;
}