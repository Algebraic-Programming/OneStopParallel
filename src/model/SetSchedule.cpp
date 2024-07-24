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

#include "model/SetSchedule.hpp"

BspSchedule SetSchedule::buildBspSchedule(std::map<KeyTriple, unsigned> cs) {

    std::vector<unsigned> node_to_processor_assignment(instance->numberOfVertices(), 0);
    std::vector<unsigned> node_to_superstep_assignment(instance->numberOfVertices(), 0);

    for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

        for (unsigned step = 0; step < number_of_supersteps; step++) {

            for (auto &node : step_processor_vertices[step][proc]) {
                node_to_processor_assignment[node] = proc;
                node_to_superstep_assignment[node] = step;
            }
        }
    }

    BspSchedule bsp_schedule(*instance, node_to_processor_assignment, node_to_superstep_assignment, cs);

    assert(bsp_schedule.satisfiesPrecedenceConstraints());
    assert(bsp_schedule.hasValidCommSchedule());

    return bsp_schedule;
}

BspSchedule SetSchedule::buildBspSchedule(ICommunicationScheduler &comm_scheduler) {

    std::vector<unsigned> node_to_processor_assignment(instance->numberOfVertices(), 0);
    std::vector<unsigned> node_to_superstep_assignment(instance->numberOfVertices(), 0);

    for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

        for (unsigned step = 0; step < number_of_supersteps; step++) {

            for (auto &node : step_processor_vertices[step][proc]) {
                node_to_processor_assignment[node] = proc;
                node_to_superstep_assignment[node] = step;
            }
        }
    }

    auto cs = comm_scheduler.computeCommunicationSchedule(*this);
    BspSchedule bsp_schedule(*instance, node_to_processor_assignment, node_to_superstep_assignment, cs);

    assert(bsp_schedule.satisfiesPrecedenceConstraints());
    assert(bsp_schedule.hasValidCommSchedule());

    return bsp_schedule;
}

void SetSchedule::mergeSupersteps(unsigned start_step, unsigned end_step) {

    unsigned step = start_step + 1;
    for (; step <= end_step; step++) {

        for (unsigned proc = 0; proc < getInstance().numberOfProcessors(); proc++) {

            step_processor_vertices[start_step][proc].merge(step_processor_vertices[step][proc]);
        }
    }

    for (; step < number_of_supersteps; step++) {

        for (unsigned proc = 0; proc < getInstance().numberOfProcessors(); proc++) {

            step_processor_vertices[step - (end_step - start_step)][proc] = std::move(step_processor_vertices[step][proc]);
        }
    }
}

void SetSchedule::insertSupersteps(unsigned step_before, unsigned num_new_steps) {

    number_of_supersteps += num_new_steps;
    
    for (unsigned step = step_before + 1; step < number_of_supersteps; step++) {

        step_processor_vertices.push_back(step_processor_vertices[step]);
        step_processor_vertices[step] = std::vector<std::unordered_set<VertexType>>(getInstance().numberOfProcessors());

    }

}

