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

#include "model/IBspSchedule.hpp"

#include <list>
#include <map>
#include <stdexcept>
#include <vector>

class PebblingStrategy : public IBspSchedule {

  private:
    const BspInstance *instance;

    unsigned int number_of_timesteps;

    std::vector<unsigned> node_to_processor_assignment;
    std::vector<unsigned> node_to_timestep_assignment;

    std::vector<std::vector<std::vector<bool>>> has_red_pebble_processor;
    std::vector<std::vector<bool>> has_blue_pebble;

    // alternative description of the pebbling strategy
    std::vector<std::vector<VertexType>> red_pebbles_timestep;
    std::vector<VertexType> blue_pebbles_timestep;


  public:
    PebblingStrategy() : instance(0), number_of_timesteps(0) {}

    PebblingStrategy(const BspInstance &inst) : instance(&inst), number_of_timesteps(0) {}

    PebblingStrategy(const BspInstance &inst, unsigned num_timesteps)
        : instance(&inst), number_of_timesteps(num_timesteps),
          node_to_processor_assignment(std::vector<unsigned>(inst.numberOfVertices(), 0)),
          node_to_timestep_assignment(std::vector<unsigned>(inst.numberOfVertices(), 0)) {}

    PebblingStrategy(const BspInstance &inst, unsigned num_timesteps, std::vector<unsigned> processor_assignment_,
                     std::vector<unsigned> timestep_assignment_)
        : instance(&inst), number_of_timesteps(num_timesteps), node_to_processor_assignment(processor_assignment_),
          node_to_timestep_assignment(timestep_assignment_) {}

    virtual ~PebblingStrategy() = default;

    const BspInstance &getInstance() const override { return *instance; }

    void setAssignedSuperstep(unsigned node, unsigned superstep) override { node_to_timestep_assignment[node] = superstep;}
    void setAssignedProcessor(unsigned node, unsigned processor) override { node_to_processor_assignment[node] = processor;}

    unsigned assignedSuperstep(unsigned node) const override { return node_to_timestep_assignment[node]; }
    unsigned assignedProcessor(unsigned node) const override { return node_to_processor_assignment[node]; }

    unsigned numberOfSupersteps() const override { return number_of_timesteps; }


    void setAssignedSuperstep(const std::vector<unsigned> &vec) { node_to_timestep_assignment = vec;}

    void setAssignedProcessors(const std::vector<unsigned> &vec) { node_to_processor_assignment = vec;}

    inline const std::vector<unsigned> &assignedTimesteps() const { return node_to_timestep_assignment; }
    inline const std::vector<unsigned> &assignedProcessors() const { return node_to_processor_assignment; }

    /// @brief Checks validity with graph topology
    /// @return true iff schedule is valid with instance
    bool satisfiesPrecedenceConstraints() const;

    void setNumberOfTimesteps(unsigned num_steps) { number_of_timesteps = num_steps;}

    void setNumberOfSuperstepsFromAssignment();
};
