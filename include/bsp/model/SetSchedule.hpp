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

#include "IBspSchedule.hpp"
#include "concepts/computational_dag_concept.hpp"

namespace osp {

/**
 * @class SetSchedule
 * @brief Represents a working schedule set for the BSP scheduling algorithm.
 *
 * This class implements the `IBspSchedule` interface and provides functionality to manage the assignment of nodes to
 * processors and supersteps. It stores the assignment information in a data structure called `processor_step_vertices`,
 * which is a 2D vector of unordered sets. Each element in the `processor_step_vertices` vector represents a processor
 * and a superstep, and contains a set of nodes assigned to that processor and superstep.
 *
 * The `SetSchedule` class provides methods to set and retrieve the assigned processor and superstep for a given
 * node, as well as to build a `BspSchedule` object based on the current assignment.
 *
 * @note This class assumes that the `BspInstance` and `ICommunicationScheduler` classes are defined and accessible.
 */
template<typename Graph_t>
class SetSchedule : public IBspSchedule<Graph_t> {

    static_assert(is_computational_dag_v<Graph_t>,
        "BspSchedule can only be used with computational DAGs.");

  private:
    using vertex_idx = vertex_idx_t<Graph_t>;

    const BspInstance<Graph_t> *instance;

  public:
    unsigned number_of_supersteps;

    std::vector<std::vector<std::unordered_set<vertex_idx>>> step_processor_vertices;

    SetSchedule() = default;

    SetSchedule(const BspInstance<Graph_t> &inst, unsigned num_supersteps)
        : instance(&inst), number_of_supersteps(num_supersteps) {

        step_processor_vertices = std::vector<std::vector<std::unordered_set<vertex_idx>>>(
            num_supersteps, std::vector<std::unordered_set<vertex_idx>>(inst.numberOfProcessors()));
    }

    SetSchedule(const IBspSchedule<Graph_t> &schedule)
        : instance(&schedule.getInstance()), number_of_supersteps(schedule.numberOfSupersteps()) {

        step_processor_vertices = std::vector<std::vector<std::unordered_set<vertex_idx>>>(
            schedule.numberOfSupersteps(),
            std::vector<std::unordered_set<vertex_idx>>(schedule.getInstance().numberOfProcessors()));

        for (unsigned i = 0; i < schedule.getInstance().numberOfVertices(); i++) {

            step_processor_vertices[schedule.assignedSuperstep(i)][schedule.assignedProcessor(i)].insert(i);
        }
    }

    virtual ~SetSchedule() = default;

    const BspInstance<Graph_t> &getInstance() const override { return *instance; }

    unsigned numberOfSupersteps() const override { return number_of_supersteps; }

    void setAssignedSuperstep(vertex_idx node, unsigned superstep) override {

        unsigned assigned_processor = 0;
        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

            for (unsigned step = 0; step < number_of_supersteps; step++) {

                if (step_processor_vertices[step][proc].find(node) != step_processor_vertices[step][proc].end()) {
                    assigned_processor = proc;
                    step_processor_vertices[step][proc].erase(node);
                }
            }
        }

        step_processor_vertices[superstep][assigned_processor].insert(node);
    }

    void setAssignedProcessor(vertex_idx node, unsigned processor) override {

        unsigned assigned_step = 0;
        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

            for (unsigned step = 0; step < number_of_supersteps; step++) {

                if (step_processor_vertices[step][proc].find(node) != step_processor_vertices[step][proc].end()) {
                    assigned_step = step;
                    step_processor_vertices[step][proc].erase(node);
                }
            }
        }

        step_processor_vertices[assigned_step][processor].insert(node);
    }

    /// @brief returns number of supersteps if the node is not assigned
    /// @param node
    /// @return the assigned superstep
    unsigned assignedSuperstep(vertex_idx node) const override {

        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

            for (unsigned step = 0; step < number_of_supersteps; step++) {

                if (step_processor_vertices[step][proc].find(node) != step_processor_vertices[step][proc].end())
                    return step;
            }
        }

        return number_of_supersteps;
    }

    /// @brief returns number of processors if node is not assigned
    /// @param node
    /// @return the assigned processor
    unsigned assignedProcessor(vertex_idx node) const override {

        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

            for (unsigned step = 0; step < number_of_supersteps; step++) {

                if (step_processor_vertices[step][proc].find(node) != step_processor_vertices[step][proc].end())
                    return proc;
            }
        }

        return instance->numberOfProcessors();
    }

    void mergeSupersteps(unsigned start_step, unsigned end_step) {

        unsigned step = start_step + 1;
        for (; step <= end_step; step++) {

            for (unsigned proc = 0; proc < getInstance().numberOfProcessors(); proc++) {

                step_processor_vertices[start_step][proc].merge(step_processor_vertices[step][proc]);
            }
        }

        for (; step < number_of_supersteps; step++) {

            for (unsigned proc = 0; proc < getInstance().numberOfProcessors(); proc++) {

                step_processor_vertices[step - (end_step - start_step)][proc] =
                    std::move(step_processor_vertices[step][proc]);
            }
        }
    }

    void insertSupersteps(unsigned step_before, unsigned num_new_steps) {

        number_of_supersteps += num_new_steps;

        for (unsigned step = step_before + 1; step < number_of_supersteps; step++) {

            step_processor_vertices.push_back(step_processor_vertices[step]);
            step_processor_vertices[step] =
                std::vector<std::unordered_set<vertex_idx>>(getInstance().numberOfProcessors());
        }
    }
};

} // namespace osp