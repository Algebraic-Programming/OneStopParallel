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

#include "bsp/model/BspArchitecture.hpp"

namespace osp { namespace file_reader {

template<typename Graph_t>
bool readBspArchitecture(std::ifstream &infile, BspArchitecture<Graph_t> &architecture) {

    std::string line;
    getline(infile, line);
    while (!infile.eof() && line.at(0) == '%')
        getline(infile, line);

    unsigned p;
    int g, L;
    int M;
    int mem_type = -1;
    sscanf(line.c_str(), "%d %d %d %d %d", &p, &g, &L, &mem_type, &M);

    architecture.setNumberOfProcessors(p);
    architecture.setCommunicationCosts(static_cast<v_commw_t<Graph_t>>(g));
    architecture.setSynchronisationCosts(static_cast<v_commw_t<Graph_t>>(L));

    if (0 <= mem_type && mem_type <= 3) {

        if (mem_type == 0) {
            architecture.setMemoryConstraintType(NONE);
        } else if (mem_type == 1) {
            architecture.setMemoryConstraintType(LOCAL);
            architecture.setMemoryBound(static_cast<v_memw_t<Graph_t>>(M));
        } else if (mem_type == 2) {
            architecture.setMemoryConstraintType(GLOBAL);
            architecture.setMemoryBound(static_cast<v_memw_t<Graph_t>>(M));
        } else if (mem_type == 3) {
            architecture.setMemoryConstraintType(PERSISTENT_AND_TRANSIENT);
            architecture.setMemoryBound(static_cast<v_memw_t<Graph_t>>(M));
        }
    } else if (mem_type == -1) {
        std::cout << "No memory type specified. Assuming \"NONE\".\n";
        architecture.setMemoryConstraintType(NONE);
    } else {
        std::cout << "Invalid memory type.\n";
        return false;
    }

    for (unsigned i = 0; i < p * p; ++i) {
        if (infile.eof()) {
            std::cout << "Incorrect input file format (file terminated too early).\n";

            architecture.SetUniformSendCost();
            return false;
        }
        getline(infile, line);
        while (!infile.eof() && line.at(0) == '%')
            getline(infile, line);

        unsigned fromProc, toProc;
        int value;
        sscanf(line.c_str(), "%d %d %d", &fromProc, &toProc, &value);

        if (fromProc >= p || toProc >= p) {
            std::cout << "Incorrect input file format (index out of range or "
                         "negative NUMA value).\n";

            architecture.SetUniformSendCost();
            return false;
        }
        if (fromProc == toProc && value != 0) {
            std::cout << "Incorrect input file format (main diagonal of NUMA cost "
                         "matrix must be 0).\n";

            architecture.SetUniformSendCost();
            return false;
        }
        architecture.setSendCosts(fromProc, toProc, static_cast<v_commw_t<Graph_t>>(value));
    }

    getline(infile, line);
    while (!infile.eof() && line.at(0) == '%')
        getline(infile, line);
    if (!infile.eof()) {
        std::cout << "Incorrect input file format (file has remaining lines).\n";
        return false;
    }

    architecture.computeCommAverage();

    return true;
};

template<typename Graph_t>
bool readBspArchitecture(const std::string &filename, BspArchitecture<Graph_t> &architecture) {

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cout << "Unable to find/open input machine parameter file.\n";

        return false;
    }

    return file_reader::readBspArchitecture(infile, architecture);
};

}} // namespace osp::file_reader