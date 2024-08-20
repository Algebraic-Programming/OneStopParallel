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

#include "scheduler/PebblingILP/PebblingStrategy.hpp"

#include <algorithm>
#include <iostream>


void PebblingStrategy::setNumberOfSuperstepsFromAssignment() {

    number_of_timesteps = 0;

    for (unsigned i = 0; i < instance->numberOfVertices(); ++i)

        if (node_to_timestep_assignment[i] >= number_of_timesteps)
            number_of_timesteps = node_to_timestep_assignment[i] + 1;
}


bool PebblingStrategy::satisfiesPrecedenceConstraints() const {

    if (node_to_processor_assignment.size() != instance->numberOfVertices() ||
        node_to_timestep_assignment.size() != instance->numberOfVertices()) {
        return false;
    }

    // bool comm_edge_found = false;

    for (const auto &ep : instance->getComputationalDag().edges()) {
        const unsigned &source = instance->getComputationalDag().source(ep);
        const unsigned &target = instance->getComputationalDag().target(ep);

        const int different_processors =
            (node_to_processor_assignment[source] == node_to_processor_assignment[target]) ? 0 : 1;

        if (node_to_timestep_assignment[source] + different_processors > node_to_timestep_assignment[target]) {
            // std::cout << "This is not a valid scheduling (problems with nodes " << source << " and " << target <<
            // ")."
            //           << std::endl; // todo should be removed
            return false;
        }
    }

    return true;
};
