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
#include <iostream>
#include <string>
#include <vector>
#include <sstream> // Required for std::stringstream
#include <utility>

#include "osp/concepts/computational_dag_concept.hpp"

namespace osp { namespace file_reader {

// Helper function to split a string by a delimiter
std::vector<std::string> split(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::string removeLeadingAndTrailingQuotes(const std::string &str) {
    if (str.empty()) {
        return str;
    }

    size_t start = 0;
    size_t end = str.length();

    if (end > 0 && (str.front() == '"' || str.front() == '\'')) {
        start = 1;
    }

    if (end > start && (str.back() == '"' || str.back() == '\'')) {
        end--;
    }

    return str.substr(start, end - start);
}

template<typename Graph_t>
void parseDotNode(std::string line, Graph_t &G) {
    // Extract node id and properties
    std::size_t pos = line.find('[');
    if (pos == std::string::npos) return;
    std::size_t end_pos = line.find(']');
    if (end_pos == std::string::npos) return;

    std::string properties = line.substr(pos + 1, end_pos - pos - 1);

    // Split properties into key-value pairs
    std::vector<std::string> keyValuePairs = split(properties, ';');

    // Create node with properties
    v_workw_t<Graph_t> work_weight = 0;
    v_memw_t<Graph_t> mem_weight = 0;
    v_commw_t<Graph_t> comm_weight = 0;
    v_type_t<Graph_t> type = 0;

    for (const std::string &keyValuePair : keyValuePairs) {
        std::vector<std::string> keyValue = split(keyValuePair, '=');
        if (keyValue.size() != 2) continue;

        std::string key = keyValue[0];
        // trim leading/trailing whitespace from key
        key.erase(0, key.find_first_not_of(" \t\n\r\f\v"));
        key.erase(key.find_last_not_of(" \t\n\r\f\v") + 1);

        if (key.empty()) {
            continue;
        }
        std::string value = removeLeadingAndTrailingQuotes(keyValue[1]);

        if (key == "work_weight") {
            work_weight = static_cast<v_workw_t<Graph_t>>(std::stoll(value));
        } else if (key == "mem_weight") {
            mem_weight = static_cast<v_memw_t<Graph_t>>(std::stoll(value));
        } else if (key == "comm_weight") {
            comm_weight = static_cast<v_commw_t<Graph_t>>(std::stoll(value));
        } else if (key == "type") {
            type = static_cast<v_type_t<Graph_t>>(std::stoll(value));
        }
    }

    if constexpr (is_constructable_cdag_typed_vertex_v<Graph_t>) {
        G.add_vertex(work_weight, comm_weight, mem_weight, type);
    } else {
        G.add_vertex(work_weight, comm_weight, mem_weight);
    }
}

template<typename Graph_t>
void parseDotEdge(std::string line, Graph_t &G) {
    using edge_commw_t_or_default = std::conditional_t<has_edge_weights_v<Graph_t>, e_commw_t<Graph_t>, v_commw_t<Graph_t>>;

    std::size_t arrow_pos = line.find("->");
    if (arrow_pos == std::string::npos) return;

    std::string source_str = line.substr(0, arrow_pos);
    source_str.erase(source_str.find_last_not_of(" \t\n\r\f\v") + 1);


    std::size_t bracket_pos = line.find('[');
    std::string target_str;
    if (bracket_pos != std::string::npos) {
        target_str = line.substr(arrow_pos + 2, bracket_pos - (arrow_pos + 2));
    } else {
        target_str = line.substr(arrow_pos + 2);
    }
    target_str.erase(0, target_str.find_first_not_of(" \t\n\r\f\v"));
    target_str.erase(target_str.find_last_not_of(" \t\n\r\f\v") + 1);


    vertex_idx_t<Graph_t> source_node = static_cast<vertex_idx_t<Graph_t>>(std::stoll(source_str));
    vertex_idx_t<Graph_t> target_node = static_cast<vertex_idx_t<Graph_t>>(std::stoll(target_str));

    edge_commw_t_or_default comm_weight = 0;

    if (bracket_pos != std::string::npos) {
        std::size_t end_bracket_pos = line.find(']', bracket_pos);
        if (end_bracket_pos != std::string::npos) {
            std::string properties = line.substr(bracket_pos + 1, end_bracket_pos - bracket_pos - 1);

            // Split properties into key-value pairs
            std::vector<std::string> keyValuePairs = split(properties, ';');

            for (const std::string &keyValuePair : keyValuePairs) {
                std::vector<std::string> keyValue = split(keyValuePair, '=');
                if (keyValue.size() != 2) continue;
                
                std::string key = keyValue[0];
                key.erase(0, key.find_first_not_of(" \t\n\r\f\v"));
                key.erase(key.find_last_not_of(" \t\n\r\f\v") + 1);

                if (key.empty()) {
                    continue;
                }
                std::string value = removeLeadingAndTrailingQuotes(keyValue[1]);

                if (key == "comm_weight") {
                    comm_weight = static_cast<edge_commw_t_or_default>(std::stoll(value));
                }
            }
        }
    }

    if constexpr (is_constructable_cdag_comm_edge_v<Graph_t>) {
        G.add_edge(source_node, target_node, comm_weight);
    } else {
        G.add_edge(source_node, target_node);
    }
}

template<typename Graph_t>
bool readComputationalDagDotFormat(std::ifstream &infile, Graph_t &graph) {
    std::string line;
    while (std::getline(infile, line)) {
        // Trim leading whitespace
        line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
        // Skip empty or comment lines
        if (line.empty() || line.rfind("digraph", 0) == 0 || line.rfind("}", 0) == 0) {
            continue;
        }

        // Check if the line represents a node or an edge
        if (line.find("->") != std::string::npos) {
            // This is an edge
            parseDotEdge(line, graph);
        } else if (line.find('[') != std::string::npos) {
            // This is a node
            parseDotNode(line, graph);
        }
    }
    return true;
}

template<typename Graph_t>
bool readComputationalDagDotFormat(const std::string &filename, Graph_t &graph) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Unable to find/open input dag file: " << filename << std::endl;
        return false;
    }
    return readComputationalDagDotFormat(infile, graph);
}

}} // namespace osp::file_reader
