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

#include "auxiliary/auxiliary.hpp"
#include "model/BspArchitecture.hpp"
#include <fstream>
#include <vector>

// TODO rename the class to "machine params", "machine model", "BSP params" or
// sth like this

struct BSPproblem {
    int p, g, L;
    std::vector<std::vector<int>> sendCost;
    unsigned memory_bound = 0;
    int avgComm{};

    explicit BSPproblem(const int a = 0, const int b = 0, const int c = 0) : p(a), g(b), L(c) { SetUniformCost(); };

    void SetUniformCost();

    void SetExpCost(int base);

    // read machine parameters from file
    bool read(std::ifstream &infile, bool NoNUMA = false);

    // write machine parameters to file
    void write(std::ofstream &outfile, bool NoNUMA = false) const;

    // compute average comm. coefficient between a pair of processors
    int computeCommAverage();

    BspArchitecture ConvertToNewBspParam() const;
    void ConvertFromNewBspParam(const BspArchitecture& new_bsp);
};
