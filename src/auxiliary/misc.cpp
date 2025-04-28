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

#include "auxiliary/misc.hpp"

using namespace osp;

// unbiased random int generator
int osp::randInt(int lim) {
    int rnd = std::rand();
    while (rnd >= RAND_MAX - RAND_MAX % lim)
        rnd = std::rand();

    return rnd % lim;
};

bool osp::isDisjoint(std::vector<intPair> &intervals) {

    sort(intervals.begin(), intervals.end());
    for (size_t i = 0; i + 1 < intervals.size(); ++i)
        if (intervals[i].b > intervals[i + 1].a)
            return false;

    return true;
};



// modify problem filename by adding substring at the right place
std::string osp::editFilename(const std::string &filename, const std::string &toInsert) {
    auto pos = filename.find("_coarse");
    if (pos == std::string::npos)
        pos = filename.find("_instance");
    if (pos == std::string::npos)
        return toInsert + filename;

    return filename.substr(0, pos) + toInsert + filename.substr(pos, filename.length() - pos);
}

// checks if a vector is rearrangement of 0... N-1
bool osp::check_vector_is_rearrangement_of_0_to_N(const std::vector<size_t> &a) {
    std::vector<bool> contained(a.size(), false);
    for (auto &val : a) {
        if (val >= a.size()) {
            return false;
        } else if (contained[val]) {
            return false;
        } else {
            contained[val] = true;
        }
    }
    return true;
}

// Print int vector
void osp::print_int_vector(const std::vector<int> &vec) {
    std::cout << "Vector: ";
    for (auto &i : vec) {
        std::cout << i << ", ";
    }
    std::cout << std::endl;
}
