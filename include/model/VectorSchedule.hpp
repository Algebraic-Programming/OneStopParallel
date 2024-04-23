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

#pragma once

#include "BspSchedule.hpp"
#include "IBspSchedule.hpp"
#include "algorithms/CommunicationScheduler.hpp"

class VectorSchedule : public IBspSchedule {

  private:
    const BspInstance *instance;

  public:
    unsigned int number_of_supersteps;

    std::vector<unsigned> node_to_processor_assignment;
    std::vector<unsigned> node_to_superstep_assignment;

    VectorSchedule() = default;

    VectorSchedule(const BspInstance &inst) : instance(&inst), number_of_supersteps(0) {
        node_to_processor_assignment = std::vector<unsigned>(inst.numberOfVertices(), instance->numberOfProcessors());
        node_to_superstep_assignment = std::vector<unsigned>(inst.numberOfVertices(), 0);
    }

    VectorSchedule(const IBspSchedule &schedule)
        : instance(&schedule.getInstance()), number_of_supersteps(schedule.numberOfSupersteps()) {

        node_to_processor_assignment = std::vector<unsigned>(schedule.getInstance().numberOfVertices(), instance->numberOfProcessors());
        node_to_superstep_assignment = std::vector<unsigned>(schedule.getInstance().numberOfVertices(), schedule.numberOfSupersteps());

        for (unsigned i = 0; i < schedule.getInstance().numberOfVertices(); i++) {

            node_to_processor_assignment[i] = schedule.assignedProcessor(i);
            node_to_superstep_assignment[i] = schedule.assignedSuperstep(i);
        }
    }

    virtual ~VectorSchedule() = default;

    const BspInstance &getInstance() const override { return *instance; }

    void setAssignedSuperstep(unsigned int node, unsigned int superstep) override {
        node_to_superstep_assignment[node] = superstep;
    };
    void setAssignedProcessor(unsigned int node, unsigned int processor) override {
        node_to_processor_assignment[node] = processor;
    };

    unsigned numberOfSupersteps() const override { return number_of_supersteps; }

    unsigned assignedSuperstep(unsigned int node) const override { return node_to_superstep_assignment[node]; }
    unsigned assignedProcessor(unsigned int node) const override { return node_to_processor_assignment[node]; }

    BspSchedule buildBspSchedule(ICommunicationScheduler &comm_scheduler) {

        BspSchedule bsp_schedule(*instance, node_to_processor_assignment,
                                 node_to_superstep_assignment, comm_scheduler.computeCommunicationSchedule(*this));

        assert(bsp_schedule.satisfiesPrecedenceConstraints());
        assert(bsp_schedule.hasValidCommSchedule());

        return bsp_schedule;
    }

    BspSchedule buildBspSchedule(std::map<KeyTriple, unsigned> cs) {

        BspSchedule bsp_schedule(*instance, node_to_processor_assignment,
                                 node_to_superstep_assignment, cs);
        
        assert(bsp_schedule.satisfiesPrecedenceConstraints());

        return bsp_schedule;
    }

    void mergeSupersteps(unsigned start_step, unsigned end_step);
    void insertSupersteps(unsigned step_before, unsigned num_new_steps);



};