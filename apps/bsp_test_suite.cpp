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

#include "graph_implementations/boost_graphs/boost_graph.hpp"
#include "util/BspScheduleTestSuiteRunner.hpp"
#include <filesystem>
#include <iostream>
#include <string>

using graph_t = osp::boost_graph_int_t;

int main(int argc, char *argv[]) {

    std::string main_config_location =
        std::filesystem::canonical("/proc/self/exe").remove_filename().string() + "main_config.json";

    osp::BspScheduleTestSuiteRunner<graph_t> runner(argc, argv, main_config_location);
    return runner.run(argc, argv);

    return 0;
}
