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

#include "auxiliary/random_graph_generator/Erdos_Renyi_graph.hpp"
#include "auxiliary/random_graph_generator/near_diagonal_random_graph.hpp"
#include "graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "graph_algorithms/directed_graph_util.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(Erdos_Renyi_graph_test) {

    std::vector<size_t> graph_sizes({100, 500, 500});
    std::vector<double> graph_chances({10, 8, 20});

    for (size_t i = 0; i < graph_sizes.size(); i++) {
        computational_dag_vector_impl_def_int_t graph;
        erdos_renyi_graph_gen(graph, graph_sizes[i], graph_chances[i]);

        BOOST_CHECK_EQUAL(graph.num_vertices(), graph_sizes[i]);
        BOOST_CHECK_EQUAL(is_acyclic(graph), true);
    }
};

BOOST_AUTO_TEST_CASE(near_diag_random_graph_test) {

    std::vector<size_t> graph_sizes({100, 500, 500});
    std::vector<double> graph_bw({1, 2, 3});
    std::vector<double> graph_prob({10, 8, 20});

    for (size_t i = 0; i < graph_sizes.size(); i++) {
        computational_dag_vector_impl_def_int_t graph;
        near_diag_random_graph(graph, graph_sizes[i], graph_bw[i], graph_prob[i]);

        BOOST_CHECK_EQUAL(graph.num_vertices(), graph_sizes[i]);
        BOOST_CHECK_EQUAL(is_acyclic(graph), true);
    }
};
