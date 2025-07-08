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

@author Toni Boehnlein, Christos Matzoros, Pal Andras Papp, Raphael S. Steiner
*/

#include "test_suite_runner/BspScheduleTestSuiteRunner.hpp"
#include <filesystem>
#include <iostream>
#include <string>
#include "graph_implementations/eigen_matrix_adapter/sparse_matrix.hpp"

#ifdef EIGEN_FOUND

using graph_t = osp::sparse_matrix_graph_int32_t;

int main(int argc, char *argv[]) {

    osp::BspScheduleTestSuiteRunner<graph_t> runner;
    return runner.run(argc, argv);

    return 0;
}

#endif