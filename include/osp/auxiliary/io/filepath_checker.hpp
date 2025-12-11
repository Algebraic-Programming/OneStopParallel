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
#include <vector>

namespace osp {
namespace file_reader {

constexpr std::size_t MAX_LINE_LENGTH = 1 << 14;    // 16 KB

// Path safety to avoid symlink, traversal or malicious file types
inline bool isPathSafe(const std::string &path) {
    try {
        std::filesystem::path resolved = std::filesystem::weakly_canonical(path);
        if (std::filesystem::is_symlink(resolved)) { return false; }
        if (!std::filesystem::is_regular_file(resolved)) { return false; }
        if (resolved.string().find('\0') != std::string::npos) { return false; }
        return true;
    } catch (...) { return false; }
}

}    // namespace file_reader
}    // namespace osp
