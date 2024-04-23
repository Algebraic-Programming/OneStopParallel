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

#include "model/VectorSchedule.hpp"

void VectorSchedule::mergeSupersteps(unsigned start_step, unsigned end_step) {

    assert(start_step < end_step);

    number_of_supersteps = 0;

    for (const auto& vertex : getInstance().getComputationalDag().vertices()) {
    
        if (node_to_superstep_assignment[vertex] > start_step && node_to_superstep_assignment[vertex] <= end_step) {

            node_to_superstep_assignment[vertex] = start_step;
        } else if (node_to_superstep_assignment[vertex] > end_step) {
            node_to_superstep_assignment[vertex] -= end_step - start_step;
        }


        if (node_to_superstep_assignment[vertex] >= number_of_supersteps) {
            number_of_supersteps = node_to_superstep_assignment[vertex] + 1;

        }
    }

}

void VectorSchedule::insertSupersteps(unsigned step_before, unsigned num_new_steps) {

    number_of_supersteps += num_new_steps;

    for (const auto& vertex : getInstance().getComputationalDag().vertices()) {

        if (node_to_superstep_assignment[vertex] > step_before) {
            node_to_superstep_assignment[vertex] += num_new_steps;
        }
    }
}