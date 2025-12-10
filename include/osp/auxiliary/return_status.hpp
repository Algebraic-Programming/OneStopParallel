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

#include <iostream>

namespace osp {

enum class RETURN_STATUS { OSP_SUCCESS,
                           BEST_FOUND,
                           TIMEOUT,
                           ERROR };

/**
 * @brief Converts the enum to a string literal.
 * Returns const char* to avoid std::string allocation overhead.
 */
inline const char *to_string(const RETURN_STATUS status) {
    switch (status) {
    case RETURN_STATUS::OSP_SUCCESS:
        return "SUCCESS";
    case RETURN_STATUS::BEST_FOUND:
        return "BEST FOUND";
    case RETURN_STATUS::TIMEOUT:
        return "TIMEOUT";
    case RETURN_STATUS::ERROR:
        return "ERROR";
    default:
        return "UNKNOWN";
    }
}

/**
 * @brief Stream operator overload using the helper function.
 */
inline std::ostream &operator<<(std::ostream &os, RETURN_STATUS status) {
    return os << to_string(status);
}

} // namespace osp