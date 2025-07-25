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

#define BOOST_TEST_MODULE BSP_SCHEDULE_RECOMP
#include <boost/test/unit_test.hpp>

#include "osp/dag_divider/WavefrontComponentDivider.hpp"
#include "osp/dag_divider/WavefrontMerkleDivider.hpp"
#include "osp/dag_divider/MerkleHashComputer.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/auxiliary/io/dot_graph_file_reader.hpp"

#include <filesystem>
#include <iostream>

using namespace osp;

BOOST_AUTO_TEST_CASE(BspScheduleRecomp_test)
{

    using graph_t = computational_dag_vector_impl_def_t;

    graph_t graph;

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    file_reader::readComputationalDagDotFormat("/home/toni/work/data/ast/deepseekAttention_s_1.dot", graph);

    WavefrontMerkleDivider<graph_t> divider; 


    divider.divide(graph);

    WavefrontComponentDivider<graph_t> wf_divider;
    wf_divider.divide(graph);


};