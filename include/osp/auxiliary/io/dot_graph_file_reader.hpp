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

@author Toni Boehnlein, Christos Matzoros, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner
*/

#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "osp/auxiliary/io/filepath_checker.hpp"
#include "osp/concepts/constructable_computational_dag_concept.hpp"

namespace osp {
namespace file_reader {

std::vector<std::string> Split(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::string RemoveLeadingAndTrailingQuotes(const std::string &str) {
    if (str.empty()) {
        return str;
    }

    std::size_t start = 0;
    std::size_t end = str.length();

    if (end > 0 && (str.front() == '"' || str.front() == '\'')) {
        start = 1;
    }

    if (end > start && (str.back() == '"' || str.back() == '\'')) {
        end--;
    }

    return str.substr(start, end - start);
}

template <typename GraphT>
void ParseDotNode(const std::string &line, GraphT &g) {
    std::size_t pos = line.find('[');
    if (pos == std::string::npos) {
        return;
    }
    std::size_t endPos = line.find(']');
    if (endPos == std::string::npos) {
        return;
    }

    std::string properties = line.substr(pos + 1, endPos - pos - 1);
    std::vector<std::string> keyValuePairs = Split(properties, ';');

    VWorkwT<GraphT> workWeight = 0;
    VMemwT<GraphT> memWeight = 0;
    VCommwT<GraphT> commWeight = 0;
    VTypeT<GraphT> type = 0;

    for (const std::string &keyValuePair : keyValuePairs) {
        std::vector<std::string> keyValue = Split(keyValuePair, '=');
        if (keyValue.size() != 2) {
            continue;
        }

        std::string key = keyValue[0];
        // trim leading/trailing whitespace from key
        key.erase(0, key.find_first_not_of(" \t\n\r\f\v"));
        key.erase(key.find_last_not_of(" \t\n\r\f\v") + 1);

        if (key.empty()) {
            continue;
        }

        std::string value = RemoveLeadingAndTrailingQuotes(keyValue[1]);

        try {
            if (key == "work_weight") {
                workWeight = static_cast<VWorkwT<GraphT>>(std::stoll(value));
            } else if (key == "mem_weight") {
                memWeight = static_cast<VMemwT<GraphT>>(std::stoll(value));
            } else if (key == "comm_weight") {
                commWeight = static_cast<VCommwT<GraphT>>(std::stoll(value));
            } else if (key == "type") {
                type = static_cast<VTypeT<GraphT>>(std::stoll(value));
            }
        } catch (...) {
            std::cerr << "Warning: Failed to parse property value: " << value << "\n";
        }
    }

    if constexpr (IsConstructableCdagTypedVertexV<GraphT>) {
        g.AddVertex(workWeight, commWeight, memWeight, type);
    } else {
        g.add_vertex(workWeight, commWeight, memWeight);
    }
}

template <typename GraphT>
void ParseDotEdge(const std::string &line, GraphT &g) {
    using EdgeCommwTOrDefault = std::conditional_t<HasEdgeWeightsV<GraphT>, ECommwT<GraphT>, VCommwT<GraphT>>;

    std::size_t arrowPos = line.find("->");
    if (arrowPos == std::string::npos) {
        return;
    }

    std::string sourceStr = line.substr(0, arrowPos);
    sourceStr.erase(sourceStr.find_last_not_of(" \t\n\r\f\v") + 1);

    std::string targetStr;
    std::size_t bracketPos = line.find('[', arrowPos);
    if (bracketPos != std::string::npos) {
        targetStr = line.substr(arrowPos + 2, bracketPos - (arrowPos + 2));
    } else {
        targetStr = line.substr(arrowPos + 2);
    }

    targetStr.erase(0, targetStr.find_first_not_of(" \t\n\r\f\v"));
    targetStr.erase(targetStr.find_last_not_of(" \t\n\r\f\v") + 1);

    try {
        VertexIdxT<GraphT> sourceNode = static_cast<VertexIdxT<GraphT>>(std::stoll(sourceStr));
        VertexIdxT<GraphT> targetNode = static_cast<VertexIdxT<GraphT>>(std::stoll(targetStr));

        if constexpr (IsConstructableCdagCommEdgeV<GraphT>) {
            EdgeCommwTOrDefault commWeight = 0;

            if (bracketPos != std::string::npos) {
                std::size_t endBracketPos = line.find(']', bracketPos);
                if (endBracketPos != std::string::npos) {
                    std::string properties = line.substr(bracketPos + 1, endBracketPos - bracketPos - 1);
                    std::vector<std::string> keyValuePairs = Split(properties, ';');

                    for (const auto &keyValuePair : keyValuePairs) {
                        std::vector<std::string> keyValue = Split(keyValuePair, '=');
                        if (keyValue.size() != 2) {
                            continue;
                        }

                        std::string key = keyValue[0];
                        key.erase(0, key.find_first_not_of(" \t\n\r\f\v"));
                        key.erase(key.find_last_not_of(" \t\n\r\f\v") + 1);
                        if (key.empty()) {
                            continue;
                        }

                        std::string value = RemoveLeadingAndTrailingQuotes(keyValue[1]);

                        if (key == "comm_weight") {
                            commWeight = static_cast<EdgeCommwTOrDefault>(std::stoll(value));
                        }
                    }
                }
            }

            g.AddEdge(sourceNode, targetNode, commWeight);
        } else {
            g.add_edge(sourceNode, targetNode);
        }
    } catch (...) {
        std::cerr << "Warning: Failed to parse edge nodes from line: " << line << "\n";
    }
}

template <typename GraphT>
bool ReadComputationalDagDotFormat(std::ifstream &infile, GraphT &graph) {
    std::string line;
    while (std::getline(infile, line)) {
        if (line.length() > maxLineLength) {
            std::cerr << "Warning: Skipping overly long line.\n";
            continue;
        }

        line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));

        if (line.empty() || line.rfind("digraph", 0) == 0 || line.rfind("}", 0) == 0) {
            continue;
        }

        if (line.find("->") != std::string::npos) {
            // This is an edge
            ParseDotEdge(line, graph);
        } else if (line.find('[') != std::string::npos) {
            // This is a node
            ParseDotNode(line, graph);
        }
    }

    return true;
}

template <typename GraphT>
bool ReadComputationalDagDotFormat(const std::string &filename, GraphT &graph) {
    if (std::filesystem::path(filename).extension() != ".dot") {
        std::cerr << "Error: Only .dot files are accepted.\n";
        return false;
    }

    if (!IsPathSafe(filename)) {
        std::cerr << "Error: Unsafe file path.\n";
        return false;
    }

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Unable to find/open input dag file: " << filename << std::endl;
        return false;
    }

    return readComputationalDagDotFormat(infile, graph);
}

}    // namespace file_reader
}    // namespace osp
