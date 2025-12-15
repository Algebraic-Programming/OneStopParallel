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

#define BOOST_TEST_MODULE coarse_refine_scheduler
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <unordered_set>
#include <vector>

#include "osp/auxiliary/random_graph_generator/Erdos_Renyi_graph.hpp"
#include "osp/auxiliary/random_graph_generator/near_diagonal_random_graph.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(ErdosRenyiGraphTest) {
    std::vector<size_t> graphSizes({100, 500, 500});
    std::vector<double> graphChances({10, 8, 20});

    for (size_t i = 0; i < graphSizes.size(); i++) {
        ComputationalDagVectorImplDefIntT graph;
        erdos_renyi_graph_gen(graph, graphSizes[i], graphChances[i]);

        BOOST_CHECK_EQUAL(graph.NumVertices(), graphSizes[i]);
        BOOST_CHECK_EQUAL(is_acyclic(graph), true);
    }
}

BOOST_AUTO_TEST_CASE(NearDiagRandomGraphTest) {
    std::vector<size_t> graphSizes({100, 500, 500});
    std::vector<double> graphBw({10, 20, 30});
    std::vector<double> graphProb({0.14, 0.02, 0.07});

    for (size_t i = 0; i < graphSizes.size(); i++) {
        ComputationalDagVectorImplDefIntT graph;
        near_diag_random_graph(graph, graphSizes[i], graphBw[i], graphProb[i]);

        BOOST_CHECK_EQUAL(graph.NumVertices(), graphSizes[i]);
        BOOST_CHECK_EQUAL(is_acyclic(graph), true);
    }
}
