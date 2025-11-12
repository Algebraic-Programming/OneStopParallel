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
#include "osp/concepts/computational_dag_concept.hpp"
#include <vector>

namespace osp {

template<typename Graph_t>
class VectorSchedule : public IBspSchedule<Graph_t> {

    static_assert(is_computational_dag_v<Graph_t>,
        "BspSchedule can only be used with computational DAGs.");

  private:
    const BspInstance<Graph_t> *instance;

  public:
    unsigned int number_of_supersteps;

    std::vector<unsigned> node_to_processor_assignment;
    std::vector<unsigned> node_to_superstep_assignment;


    /**
     * @brief Default constructor for VectorSchedule.
     */
    VectorSchedule() : instance(nullptr), number_of_supersteps(0) {}

    VectorSchedule(const BspInstance<Graph_t> &inst) : instance(&inst), number_of_supersteps(0) {
        node_to_processor_assignment = std::vector<unsigned>(inst.numberOfVertices(), instance->numberOfProcessors());
        node_to_superstep_assignment = std::vector<unsigned>(inst.numberOfVertices(), 0);
    }

    VectorSchedule(const IBspSchedule<Graph_t> &schedule)
        : instance(&schedule.getInstance()), number_of_supersteps(schedule.numberOfSupersteps()) {

        node_to_processor_assignment =
            std::vector<unsigned>(schedule.getInstance().numberOfVertices(), instance->numberOfProcessors());
        node_to_superstep_assignment =
            std::vector<unsigned>(schedule.getInstance().numberOfVertices(), schedule.numberOfSupersteps());

        for (vertex_idx_t<Graph_t> i = 0; i < schedule.getInstance().numberOfVertices(); i++) {

            node_to_processor_assignment[i] = schedule.assignedProcessor(i);
            node_to_superstep_assignment[i] = schedule.assignedSuperstep(i);
        }
    }

    VectorSchedule(const VectorSchedule &other)
        : instance(other.instance), number_of_supersteps(other.number_of_supersteps),
          node_to_processor_assignment(other.node_to_processor_assignment),
          node_to_superstep_assignment(other.node_to_superstep_assignment) {}

    VectorSchedule &operator=(const IBspSchedule<Graph_t> &other) {
        if (this != &other) {
            instance = &other.getInstance();
            number_of_supersteps = other.numberOfSupersteps();
            node_to_processor_assignment =
                std::vector<unsigned>(instance->numberOfVertices(), instance->numberOfProcessors());
            node_to_superstep_assignment = std::vector<unsigned>(instance->numberOfVertices(), number_of_supersteps);

            for (vertex_idx_t<Graph_t> i = 0; i < instance->numberOfVertices(); i++) {
                node_to_processor_assignment[i] = other.assignedProcessor(i);
                node_to_superstep_assignment[i] = other.assignedSuperstep(i);
            }
        }
        return *this;
    }

    VectorSchedule &operator=(const VectorSchedule &other) {
        if (this != &other) {
            instance = other.instance;
            number_of_supersteps = other.number_of_supersteps;
            node_to_processor_assignment = other.node_to_processor_assignment;
            node_to_superstep_assignment = other.node_to_superstep_assignment;
        }
        return *this;
    }

    VectorSchedule(VectorSchedule &&other) noexcept
        : instance(other.instance), number_of_supersteps(other.number_of_supersteps),
          node_to_processor_assignment(std::move(other.node_to_processor_assignment)),
          node_to_superstep_assignment(std::move(other.node_to_superstep_assignment)) {}

    virtual ~VectorSchedule() = default;

    void clear() {
        node_to_processor_assignment.clear();
        node_to_superstep_assignment.clear();
        number_of_supersteps = 0;
    }

    const BspInstance<Graph_t> &getInstance() const override { return *instance; }

    void setAssignedSuperstep(vertex_idx_t<Graph_t> vertex, unsigned superstep) override {
        node_to_superstep_assignment[vertex] = superstep;
    };
    void setAssignedProcessor(vertex_idx_t<Graph_t> vertex, unsigned processor) override {
        node_to_processor_assignment[vertex] = processor;
    };

    unsigned numberOfSupersteps() const override { return number_of_supersteps; }

    unsigned assignedSuperstep(vertex_idx_t<Graph_t> vertex) const override {
        return node_to_superstep_assignment[vertex];
    }
    unsigned assignedProcessor(vertex_idx_t<Graph_t> vertex) const override {
        return node_to_processor_assignment[vertex];
    }

    void mergeSupersteps(unsigned start_step, unsigned end_step) {

        number_of_supersteps = 0;

        for (const auto &vertex : getInstance().vertices()) {

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

    void insertSupersteps(const unsigned step_before, const unsigned num_new_steps) {

        number_of_supersteps += num_new_steps;

        for (const auto &vertex : getInstance().vertices()) {

            if (node_to_superstep_assignment[vertex] > step_before) {
                node_to_superstep_assignment[vertex] += num_new_steps;
            }
        }
    }
};

} // namespace osp