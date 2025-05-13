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
#include <utility>
#include <vector>
#include <boost/algorithm/string.hpp>

#include "concepts/computational_dag_concept.hpp"

namespace osp { namespace file_reader {

std::string removeLeadingAndTrailingQuotes(const std::string &str) {
    if (str.empty() || str == "") {
        return str;
    }

    size_t start = 0;
    size_t end = str.size();

    if (str[0] == '"' || str[0] == '\'') {
        start = 1;
    }

    if (str[end - 1] == '"' || str[end - 1] == '\'') {
        end -= 1;
    }

    return str.substr(start, end - start);
}

template<typename Graph_t>
void parseDotNode(std::string line, Graph_t &G) {

    // Extract node id and properties
    std::size_t pos = line.find('[');
    std::string properties = line.substr(pos + 1, line.find(']') - pos - 1);

    // Split properties into key-value pairs
    std::vector<std::string> keyValuePairs;
    boost::split(keyValuePairs, properties, boost::is_any_of(";"));

    // Create node with properties
    v_workw_t<Graph_t> work_weight = 0;
    v_memw_t<Graph_t> mem_weight = 0;
    v_commw_t<Graph_t> comm_weight = 0;
    v_type_t<Graph_t> type = 0;

    for (const std::string &keyValuePair : keyValuePairs) {
        std::vector<std::string> keyValue;
        boost::split(keyValue, keyValuePair, boost::is_any_of("="));

        std::string key = keyValue[0];
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

    std::size_t pos = line.find('[');
    std::string nodes = line.substr(0, pos);
    std::string properties = line.substr(pos + 1, line.find(']') - pos - 1);

    std::vector<std::string> sourceTarget;
    boost::split(sourceTarget, nodes, boost::is_any_of("-"));

    vertex_idx_t<Graph_t> source = static_cast<vertex_idx_t<Graph_t>>(std::stoll(sourceTarget[0]));
    vertex_idx_t<Graph_t> target = static_cast<vertex_idx_t<Graph_t>>(std::stoll(sourceTarget[1].substr(1)));

    // Split properties into key-value pairs
    std::vector<std::string> keyValuePairs;
    boost::split(keyValuePairs, properties, boost::is_any_of(" "));

    // Create edge with properties
    e_commw_t<Graph_t> comm_weight = 0;

    for (const std::string &keyValuePair : keyValuePairs) {
        std::vector<std::string> keyValue;
        boost::split(keyValue, keyValuePair, boost::is_any_of("="));

        std::string key = keyValue[0];
        if (key.empty()) {
            continue;
        }
        std::string value = removeLeadingAndTrailingQuotes(keyValue[1]);

        if (key == "comm_weight") {
            comm_weight = static_cast<e_commw_t<Graph_t>>(std::stoll(value));
        }
    }

    if constexpr (is_constructable_cdag_comm_edge_v<Graph_t>) {
        G.add_edge(source, target, comm_weight);
    } else {
        G.add_edge(source, target);
    }
}

template<typename Graph_t>
bool readComputationalDagDotFormat(std::ifstream &infile, Graph_t &graph) {

    std::string line;
    while (std::getline(infile, line)) {
        // Skip lines that do not contain opening or closing brackets
        if (line.find('{') != std::string::npos || line.find('}') != std::string::npos) {
            continue;
        }

        // Check if the line represents a node or an edge
        if (line.find("->") != std::string::npos) {
            // This is an edge
            parseDotEdge(line, graph);
            // Add the edge to the graph
        } else {
            // This is a node
            parseDotNode(line, graph);
            // Add the node to the graph
        }
    }

    return true;
}

template<typename Graph_t>
bool readComputationalDagDotFormat(const std::string &filename, Graph_t &graph) {

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cout << "Unable to find/open input dag file.\n";

        return false;
    }

    return readComputationalDagDotFormat(infile, graph);
}

}} // namespace osp::file_reader