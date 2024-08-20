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

#include "dag_partitioners/IListPartitioner.hpp"

class VariancePartitioner : public IListPartitioner {

    private:

    /// @brief threshold percentage of idle processors as to when a new superstep should be introduced
    float max_percent_idle_processors;

    /// @brief the power in the power mean average of the variance scheduler
    double variance_power;

    /// @brief the multiplier by which the memory bound should increase if it the scheduler determines it should be breached
    double memory_capacity_increase;

    /// @brief percentage of the average workload by which the processor priorities may diverge
    float max_priority_difference_percent;

    /// @brief Computes a power mean average of the bottom node distance
    /// @param graph graph
    /// @param power the power in the power mean average
    /// @return vector of the logarithm of power mean averaged bottom node distance
    std::vector<double> compute_work_variance(const ComputationalDag& graph, double power = 2) const;


    struct VarianceCompare
    {
        bool operator()(const std::pair<VertexType, double>& lhs, const std::pair<VertexType, double>& rhs) const {
            return ((lhs.second > rhs.second) || ((lhs.second == rhs.second) && (lhs.first < rhs.first)));
        }
    };


    public:

    VariancePartitioner(ProcessorPriorityMethod proc_priority_method_ = FLATSPLINE, bool use_memory_constraint_ = false, float max_percent_idle_processors_ = 0.2, double variance_power_ = 2, double memory_capacity_increase_ = 1.1, float max_priority_difference_percent_ = 0.34, unsigned timelimit = 3600) : IListPartitioner(proc_priority_method_, timelimit, use_memory_constraint_), max_percent_idle_processors(max_percent_idle_processors_), variance_power(variance_power_), memory_capacity_increase(memory_capacity_increase_), max_priority_difference_percent(max_priority_difference_percent_) { };

    virtual ~VariancePartitioner() = default;

    std::pair<RETURN_STATUS, DAGPartition> computePartition(const BspInstance &instance) override;

    std::string getPartitionerName() const override {
        if (use_memory_constraint) {
            return "VarianceMemoryPartitioner";
        } else {
            return "VariancePartitioner";
        }
    };
};