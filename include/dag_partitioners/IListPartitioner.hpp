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

#include <cmath>
#include <functional>

#include "auxiliary/auxiliary.hpp"
#include "dag_partitioners/Partitioner.hpp"

/// @brief A common interface for all List Scheduler partitioners
class IListPartitioner : public Partitioner {

    public:
    
    /// @brief Processor choosing priority method 
    enum ProcessorPriorityMethod { LINEAR, FLATSPLINE };

    protected:

    ProcessorPriorityMethod proc_priority_method;

    /// @brief linear function for interpolation
    /// @param alpha in [0,1] 
    /// @return alpha
    float linear_interpolation(float alpha);

    /// @brief cubic function for interpolation
    /// @param alpha
    /// @return (-2)*alpha^3 + 3*alpha^2
    float flat_spline_interpolation(float alpha);

    /// @brief Computes the interpolated priorities
    /// @param superstep_partition_work vector with current work distribution in current superstep
    /// @param total_partition_work vector with current work distribution overall
    /// @param total_work total work weight of all nodes of the graph
    /// @param instance bsp instance
    /// @return vector with the interpolated priorities
    std::vector<float> computeProcessorPrioritiesInterpolation(const std::vector<long unsigned>& superstep_partition_work, const std::vector<long unsigned>& total_partition_work, const long unsigned& total_work, const BspInstance &instance);
    
    /// @brief Computes processor priorities
    /// @param superstep_partition_work vector with current work distribution in current superstep
    /// @param total_partition_work vector with current work distribution overall
    /// @param total_work total work weight of all nodes of the graph
    /// @param instance bsp instance
    /// @return vector with the processor priorities
    std::vector<float> computeProcessorPriorities(const std::vector<long unsigned>& superstep_partition_work, const std::vector<long unsigned>& total_partition_work, const long unsigned& total_work, const BspInstance &instance);
    
    /// @brief Computes processor priorities
    /// @param superstep_partition_work vector with current work distribution in current superstep
    /// @param total_partition_work vector with current work distribution overall
    /// @param total_work total work weight of all nodes of the graph
    /// @param instance bsp instance
    /// @return vector with the processors in order of priority
    std::vector<unsigned> computeProcessorPriority(const std::vector<long unsigned>& superstep_partition_work, const std::vector<long unsigned>& total_partition_work, const long unsigned& total_work, const BspInstance &instance);


    public:

    /// @brief Constructor for the IListPartitioner class
    IListPartitioner(ProcessorPriorityMethod proc_priority_method_ = FLATSPLINE, unsigned timelimit = 3600, bool use_memory_constraint_ = false) : Partitioner(timelimit, use_memory_constraint_), proc_priority_method(proc_priority_method_) { };

    /// @brief Deconstructor for the IListPartitioner class
    virtual ~IListPartitioner() = default;

    /// @brief Sets or changes the processor priority method 
    void setProcessorPriorityMethod(ProcessorPriorityMethod priority) { proc_priority_method = priority; };

    virtual std::pair<RETURN_STATUS, DAGPartition> computePartition(const BspInstance &instance) override = 0;

    virtual std::string getPartitionerName() const override = 0;
};