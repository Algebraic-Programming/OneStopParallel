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

#include <cmath>

namespace osp {

template<typename float_type>
float_type log_sum_exp(float_type lhs, float_type rhs) {
    static_assert(std::is_floating_point_v<float_type>);

    float_type max = std::max(lhs, rhs);
    
    float_type result = max;
    result += std::log2( std::exp2(lhs - max) + std::exp2(rhs - max) );
    return result;
}

} // end namespace osp
