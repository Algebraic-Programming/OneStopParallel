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

#include <fstream>
#include <iostream>
#include <sstream>

#include "osp/bsp/model/BspArchitecture.hpp"

namespace osp {
namespace file_reader {

template <typename Graph_t>
bool readBspArchitecture(std::ifstream &infile, BspArchitecture<Graph_t> &architecture) {
    std::string line;

    // Skip comment lines
    while (std::getline(infile, line)) {
        if (!line.empty() && line[0] != '%') {
            break;
        }
    }

    // Parse architecture parameters
    unsigned p = 0;
    int g = 0, L = 0;
    int mem_type = -1;
    int M = 0;

    std::istringstream iss(line);
    if (!(iss >> p >> g >> L)) {
        std::cerr << "Error: Failed to parse p, g, L.\n";
        return false;
    }

    // Try to read optional mem_type and M
    if (!(iss >> mem_type >> M)) {
        mem_type = -1;    // Memory info not present
    }

    architecture.setNumberOfProcessors(p);
    architecture.setCommunicationCosts(static_cast<v_commw_t<Graph_t>>(g));
    architecture.setSynchronisationCosts(static_cast<v_commw_t<Graph_t>>(L));

    if (0 <= mem_type && mem_type <= 3) {
        using memw_t = v_memw_t<Graph_t>;
        switch (mem_type) {
            case 0:
                architecture.setMemoryConstraintType(MEMORY_CONSTRAINT_TYPE::NONE);
                break;
            case 1:
                architecture.setMemoryConstraintType(MEMORY_CONSTRAINT_TYPE::LOCAL);
                architecture.setMemoryBound(static_cast<memw_t>(M));
                break;
            case 2:
                architecture.setMemoryConstraintType(MEMORY_CONSTRAINT_TYPE::GLOBAL);
                architecture.setMemoryBound(static_cast<memw_t>(M));
                break;
            case 3:
                architecture.setMemoryConstraintType(MEMORY_CONSTRAINT_TYPE::PERSISTENT_AND_TRANSIENT);
                architecture.setMemoryBound(static_cast<memw_t>(M));
                break;
            default:
                std::cerr << "Invalid memory type.\n";
                return false;
        }
    } else if (mem_type == -1) {
        std::cout << "No memory type specified. Assuming \"NONE\".\n";
        architecture.setMemoryConstraintType(MEMORY_CONSTRAINT_TYPE::NONE);
    } else {
        std::cerr << "Invalid memory type.\n";
        return false;
    }

    // Parse NUMA matrix (p x p entries)
    for (unsigned i = 0; i < p * p; ++i) {
        do {
            if (!std::getline(infile, line)) {
                std::cerr << "Error: File ended before NUMA matrix was fully read.\n";
                architecture.SetUniformSendCost();
                return false;
            }
        } while (!line.empty() && line[0] == '%');

        unsigned fromProc, toProc;
        int value;
        std::istringstream matrixStream(line);
        if (!(matrixStream >> fromProc >> toProc >> value)) {
            std::cerr << "Error: Failed to parse NUMA matrix line.\n";
            architecture.SetUniformSendCost();
            return false;
        }

        if (fromProc >= p || toProc >= p) {
            std::cerr << "Error: NUMA index out of range.\n";
            architecture.SetUniformSendCost();
            return false;
        }

        if (fromProc == toProc && value != 0) {
            std::cerr << "Error: Diagonal value in NUMA matrix must be zero.\n";
            architecture.SetUniformSendCost();
            return false;
        }

        architecture.SetSendCosts(fromProc, toProc, static_cast<v_commw_t<Graph_t>>(value));
    }

    // Ensure there are no remaining non-comment lines
    while (std::getline(infile, line)) {
        if (!line.empty() && line[0] != '%') {
            std::cerr << "Error: Unexpected extra line after NUMA matrix.\n";
            return false;
        }
    }

    return true;
}

template <typename Graph_t>
bool readBspArchitecture(const std::string &filename, BspArchitecture<Graph_t> &architecture) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Unable to open machine parameter file: " << filename << '\n';
        return false;
    }

    return readBspArchitecture(infile, architecture);
}

}    // namespace file_reader
}    // namespace osp
