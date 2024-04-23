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

#include "structures/bsp.hpp"
#include "model/BspArchitecture.hpp"

#include <auxiliary/auxiliary.hpp>
#include <cmath>
#include <iostream>

void BSPproblem::SetUniformCost() {
    sendCost.clear();
    sendCost.resize(p, std::vector<int>(p, 1));
    for (int i = 0; i < p; ++i)
        sendCost[i][i] = 0;
};

void BSPproblem::SetExpCost(const int base) {
    sendCost.clear();
    sendCost.resize(p, std::vector<int>(p, 0));
    int maxPos = 1;
    for (; intpow(2, maxPos + 1) <= p - 1; ++maxPos) {
    }
    for (int i = 0; i < p; ++i)
        for (int j = i + 1; j < p; ++j)
            for (int pos = maxPos; pos >= 0; --pos)
                if (((1 << pos) & i) != ((1 << pos) & j)) {
                    sendCost[i][j] = sendCost[j][i] = intpow(base, pos);
                    break;
                }
};

// write machine parameters to file
void BSPproblem::write(std::ofstream &outfile, const bool NoNUMA) const {
    outfile << p << " " << g << " " << L << std::endl;

    if (!NoNUMA)
        for (int i = 0; i < p; ++i)
            for (int j = 0; j < p; ++j)
                outfile << i << " " << j << " " << sendCost[i][j] << std::endl;
};

// compute average comm. coefficient between a pair of processors
int BSPproblem::computeCommAverage() {
    double avg = 0;
    for (int i = 0; i < p; ++i)
        for (int j = 0; j < p; ++j)
            avg += sendCost[i][j];
    avg = avg * (double)g / (double)p / (double)p;
    avgComm = static_cast<int>(round(avg));
    return avgComm;
};

// read problem parameters from file
bool BSPproblem::read(std::ifstream &infile, bool NoNUMA) {
    std::string line;
    getline(infile, line);
    while (!infile.eof() && line.at(0) == '%')
        getline(infile, line);

    sscanf(line.c_str(), "%d %d %d", &p, &g, &L);

    SetUniformCost();

    if (!NoNUMA) {
        for (int i = 0; i < p * p; ++i) {
            if (infile.eof()) {
                std::cout << "Incorrect input file format (file terminated too early).\n";
                return false;
            }
            getline(infile, line);
            while (!infile.eof() && line.at(0) == '%')
                getline(infile, line);

            int fromProc, toProc, value;
            sscanf(line.c_str(), "%d %d %d", &fromProc, &toProc, &value);

            if (fromProc < 0 || toProc < 0 || fromProc >= p || toProc >= p || value < 0) {
                std::cout << "Incorrect input file format (index out of range or "
                             "negative NUMA value).\n";
                return false;
            }
            if (fromProc == toProc && value != 0) {
                std::cout << "Incorrect input file format (main diagonal of NUMA cost "
                             "matrix must be 0).\n";
                return false;
            }
            sendCost[fromProc][toProc] = value;
        }
    }
    computeCommAverage();

    return true;
};

BspArchitecture BSPproblem::ConvertToNewBspParam() const
{
    BspArchitecture new_bsp(p, g, L);
    for(int from_proc = 0; from_proc < p; ++from_proc)
        for(int to_proc = 0; to_proc < p; ++to_proc)
            new_bsp.setSendCosts(from_proc, to_proc, sendCost[from_proc][to_proc]);

               
    return new_bsp;
};

void BSPproblem::ConvertFromNewBspParam(const BspArchitecture& new_bsp)
{
    p = new_bsp.numberOfProcessors();
    g = new_bsp.communicationCosts();
    L = new_bsp.synchronisationCosts();
    sendCost.resize(p, std::vector<int>(p, 0));
    for(int from_proc = 0; from_proc < p; ++from_proc)
        for(int to_proc = 0; to_proc < p; ++to_proc)
            sendCost[from_proc][to_proc] = new_bsp.sendCosts(from_proc, to_proc);
    computeCommAverage();
};
