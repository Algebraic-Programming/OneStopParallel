/*
Copyright 2025 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Christos K. Matzoros, Pal Andras Papp, Raphael S. Steiner
*/

#pragma once

#include <type_traits>
#include <utility>
#include <vector>

namespace osp {

template<typename T, typename Ind>
void permute_inplace(std::vector<T> &vec, std::vector<Ind> &perm) {
    static_assert(std::is_integral_v<Ind>);
    static_assert(std::is_unsigned_v<Ind>);

    assert(vec.size() == perm.size());
    assert([&]() -> bool{
        std::vector<bool> found(perm.size(), false);
        for (const Ind &val : perm) {
            if (val < 0) return false;
            if (val >= perm.size()) return false;
            if (found[val]) return false;
            found[val] = true;
        }
        return true;
    }());
    assert(reinterpret_cast<void*>(&vec) != reinterpret_cast<void*>(&perm));

    for (Ind i = 0; i < perm.size(); ++i) {
        while (perm[i] != i) {
            std::swap(vec[i], vec[perm[i]]);
            std::swap(perm[i], perm[perm[i]]);
        }
    }
}

template<typename T, typename Ind>
void inverse_permute_inplace(std::vector<T> &vec, std::vector<Ind> &perm) {
    static_assert(std::is_integral_v<Ind>);
    static_assert(std::is_unsigned_v<Ind>);

    assert(vec.size() == perm.size());
    assert([&]() -> bool{
        std::vector<bool> found(perm.size(), false);
        for (const Ind &val : perm) {
            if (val < 0) return false;
            if (val >= perm.size()) return false;
            if (found[val]) return false;
            found[val] = true;
        }
        return true;
    }());
    assert(reinterpret_cast<void*>(&vec) != reinterpret_cast<void*>(&perm));

    for (Ind i = 0; i < perm.size(); ++i) {

        Ind j = i;
        while (i != perm[i]) {
            std::swap(vec[j], vec[perm[i]]);
            j = perm[i];
            std::swap(perm[j], perm[i]);
        }
    }
}

} // namespace osp