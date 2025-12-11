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

#include <vector>

#include "osp/bsp/model/IBspSchedule.hpp"
#include "osp/concepts/computational_dag_concept.hpp"

namespace osp {

template <typename GraphT>
class VectorSchedule : public IBspSchedule<GraphT> {
    static_assert(isComputationalDagV<GraphT>, "BspSchedule can only be used with computational DAGs.");

  private:
    const BspInstance<GraphT> *instance_;

  public:
    unsigned int numberOfSupersteps;

    std::vector<unsigned> nodeToProcessorAssignment;
    std::vector<unsigned> nodeToSuperstepAssignment;

    /**
     * @brief Default constructor for VectorSchedule.
     */
    VectorSchedule() : instance_(nullptr), numberOfSupersteps(0) {}

    VectorSchedule(const BspInstance<GraphT> &inst) : instance_(&inst), numberOfSupersteps(0) {
        nodeToProcessorAssignment = std::vector<unsigned>(inst.numberOfVertices(), instance_->numberOfProcessors());
        nodeToSuperstepAssignment = std::vector<unsigned>(inst.numberOfVertices(), 0);
    }

    VectorSchedule(const IBspSchedule<GraphT> &schedule)
        : instance_(&schedule.GetInstance()), numberOfSupersteps(schedule.NumberOfSupersteps()) {
        nodeToProcessorAssignment
            = std::vector<unsigned>(schedule.GetInstance().NumberOfVertices(), instance_->NumberOfProcessors());
        nodeToSuperstepAssignment = std::vector<unsigned>(schedule.GetInstance().NumberOfVertices(), schedule.NumberOfSupersteps());

        for (VertexIdxT<GraphT> i = 0; i < schedule.GetInstance().NumberOfVertices(); i++) {
            nodeToProcessorAssignment[i] = schedule.AssignedProcessor(i);
            nodeToSuperstepAssignment[i] = schedule.AssignedSuperstep(i);
        }
    }

    VectorSchedule(const VectorSchedule &other)
        : instance_(other.instance_),
          numberOfSupersteps(other.numberOfSupersteps),
          nodeToProcessorAssignment(other.nodeToProcessorAssignment),
          nodeToSuperstepAssignment(other.nodeToSuperstepAssignment) {}

    VectorSchedule &operator=(const IBspSchedule<GraphT> &other) {
        if (this != &other) {
            instance_ = &other.getInstance();
            numberOfSupersteps = other.numberOfSupersteps();
            nodeToProcessorAssignment = std::vector<unsigned>(instance_->numberOfVertices(), instance_->numberOfProcessors());
            nodeToSuperstepAssignment = std::vector<unsigned>(instance_->numberOfVertices(), numberOfSupersteps);

            for (VertexIdxT<GraphT> i = 0; i < instance_->numberOfVertices(); i++) {
                nodeToProcessorAssignment[i] = other.assignedProcessor(i);
                nodeToSuperstepAssignment[i] = other.assignedSuperstep(i);
            }
        }
        return *this;
    }

    VectorSchedule &operator=(const VectorSchedule &other) {
        if (this != &other) {
            instance_ = other.instance_;
            numberOfSupersteps = other.numberOfSupersteps;
            nodeToProcessorAssignment = other.nodeToProcessorAssignment;
            nodeToSuperstepAssignment = other.nodeToSuperstepAssignment;
        }
        return *this;
    }

    VectorSchedule(VectorSchedule &&other) noexcept
        : instance_(other.instance_),
          numberOfSupersteps(other.numberOfSupersteps),
          nodeToProcessorAssignment(std::move(other.nodeToProcessorAssignment)),
          nodeToSuperstepAssignment(std::move(other.nodeToSuperstepAssignment)) {}

    virtual ~VectorSchedule() = default;

    void Clear() {
        nodeToProcessorAssignment.clear();
        nodeToSuperstepAssignment.clear();
        numberOfSupersteps = 0;
    }

    const BspInstance<GraphT> &GetInstance() const override { return *instance_; }

    void SetAssignedSuperstep(VertexIdxT<GraphT> vertex, unsigned superstep) override {
        nodeToSuperstepAssignment[vertex] = superstep;
    };

    void SetAssignedProcessor(VertexIdxT<GraphT> vertex, unsigned processor) override {
        nodeToProcessorAssignment[vertex] = processor;
    };

    unsigned NumberOfSupersteps() const override { return numberOfSupersteps; }

    unsigned AssignedSuperstep(VertexIdxT<GraphT> vertex) const override { return nodeToSuperstepAssignment[vertex]; }

    unsigned AssignedProcessor(VertexIdxT<GraphT> vertex) const override { return nodeToProcessorAssignment[vertex]; }

    void MergeSupersteps(unsigned startStep, unsigned endStep) {
        numberOfSupersteps = 0;

        for (const auto &vertex : getInstance().Vertices()) {
            if (nodeToSuperstepAssignment[vertex] > startStep && nodeToSuperstepAssignment[vertex] <= endStep) {
                nodeToSuperstepAssignment[vertex] = startStep;
            } else if (nodeToSuperstepAssignment[vertex] > endStep) {
                nodeToSuperstepAssignment[vertex] -= endStep - startStep;
            }

            if (nodeToSuperstepAssignment[vertex] >= numberOfSupersteps) {
                numberOfSupersteps = nodeToSuperstepAssignment[vertex] + 1;
            }
        }
    }

    void InsertSupersteps(const unsigned stepBefore, const unsigned numNewSteps) {
        numberOfSupersteps += numNewSteps;

        for (const auto &vertex : getInstance().Vertices()) {
            if (nodeToSuperstepAssignment[vertex] > stepBefore) {
                nodeToSuperstepAssignment[vertex] += numNewSteps;
            }
        }
    }
};

}    // namespace osp
