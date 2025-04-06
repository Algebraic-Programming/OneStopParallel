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

#define BOOST_TEST_MODULE Bsp_Architecture
#include <boost/test/unit_test.hpp>

#include "model/bsp/BspInstance.hpp"
#include "model/bsp/BspSchedule.hpp"
#include "graph_implementations/computational_dag_vector_impl.hpp"


using namespace osp;

BOOST_AUTO_TEST_CASE(test_1)
{
    BspArchitecture architecture(4, 2, 3);
    computational_dag_vector_impl_def_t graph;
    
    BspInstance instance(graph, architecture);

    BOOST_CHECK_EQUAL(instance.numberOfVertices(), 0);
    BOOST_CHECK_EQUAL(instance.numberOfProcessors(), 4);

    BspSchedule<computational_dag_vector_impl_def_t> schedule(instance);

}

