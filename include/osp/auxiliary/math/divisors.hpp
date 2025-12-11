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

@author Toni Boehnlein, Christos Matzoros, Pal Andras Papp, Raphael S. Steiner
*/
#pragma once

#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace osp {

template <typename integral_type>
integral_type intSqrtFloor(integral_type num) {
    static_assert(std::is_integral_v<integral_type>);
    assert(num > 0);

    integral_type sqrt = 1;
    integral_type numCopy = num;
    while (numCopy >= 4) {
        sqrt *= 2;
        numCopy /= 4;
    }
    integral_type power2 = sqrt / 2;
    while (power2 > 0) {
        integral_type sum = sqrt + power2;
        if (sum * sum <= num) {
            sqrt = sum;
        }
        power2 /= 2;
    }

    return sqrt;
}

template <typename integral_type>
std::vector<integral_type> divisorsList(integral_type num) {
    static_assert(std::is_integral_v<integral_type>);
    assert(num > 0);

    std::vector<integral_type> divs;

    integral_type ub = intSqrtFloor<integral_type>(num);
    for (integral_type div = 1; div <= ub; ++div) {
        if (num % div == 0) {
            divs.emplace_back(div);
        }
    }
    for (std::size_t indx = divs.back() * divs.back() == num ? divs.size() - 2U : divs.size() - 1U;
         indx != std::numeric_limits<std::size_t>::max();
         --indx) {
        divs.emplace_back(num / divs[indx]);
    }

    return divs;
}

}    // end namespace osp
