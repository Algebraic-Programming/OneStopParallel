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

#include <filesystem>
#include <iostream>
#include <string>

#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "test_suite_runner/BspScheduleRecompTestSuiteRunner.hpp"

using graph_t = osp::computational_dag_edge_idx_vector_impl_def_int_t;

int main(int argc, char *argv[]) {

    osp::BspScheduleRecompTestSuiteRunner<graph_t> runner;
    return runner.run(argc, argv);

    return 0;
}
