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

#include "dag_partitioners/IListPartitioner.hpp"

float IListPartitioner::linear_interpolation(float alpha, const float slack) {
    return (1.0 - slack) * alpha;
}

float IListPartitioner::flat_spline_interpolation(float alpha, const float slack) {
    return (1.0 - slack) *  ((-2) * pow(alpha, 3)) + (3 * pow(alpha, 2));
}

std::vector<float> IListPartitioner::computeProcessorPrioritiesInterpolation(const std::vector<long unsigned>& superstep_partition_work, const std::vector<long unsigned>& total_partition_work, const long unsigned& total_work, const BspInstance &instance, const float slack) {
    long unsigned work_till_now = 0;
    for (const auto& part_work : total_partition_work) {
        work_till_now += part_work;
    }

    float percentage_complete = (float) work_till_now / (float) total_work;

    float value = percentage_complete;

    switch (proc_priority_method)
    {
    case LINEAR:
        value = linear_interpolation(percentage_complete, slack);
        break;

    case FLATSPLINE:
        value = flat_spline_interpolation(percentage_complete, slack);
        break;

    case SUPERSTEP_ONLY:
        value = superstep_only_interpolation();
        break;

    case GLOBAL_ONLY:
        value = global_only_interpolation();
        break;
    
    default:
        value = flat_spline_interpolation(percentage_complete, slack);
        break;
    }

    std::vector<float> proc_prio(instance.numberOfProcessors());
    for (size_t i = 0; i < proc_prio.size(); i++) {
        proc_prio[i] = ((1 - value) * superstep_partition_work[i]) + (value * total_partition_work[i]);
    }

    return proc_prio;
}

std::vector<float> IListPartitioner::computeProcessorPriorities(const std::vector<long unsigned>& superstep_partition_work, const std::vector<long unsigned>& total_partition_work, const long unsigned& total_work, const BspInstance &instance, const float slack) {
    switch (proc_priority_method)
    {
    case LINEAR:
        return computeProcessorPrioritiesInterpolation(superstep_partition_work, total_partition_work, total_work, instance, slack);

    case FLATSPLINE:
        return computeProcessorPrioritiesInterpolation(superstep_partition_work, total_partition_work, total_work, instance, slack);
    
    case SUPERSTEP_ONLY:
        return computeProcessorPrioritiesInterpolation(superstep_partition_work, total_partition_work, total_work, instance, slack);

    case GLOBAL_ONLY:
        return computeProcessorPrioritiesInterpolation(superstep_partition_work, total_partition_work, total_work, instance, slack);
    
    default:
        return computeProcessorPrioritiesInterpolation(superstep_partition_work, total_partition_work, total_work, instance, slack);
    }
}

std::vector<unsigned> IListPartitioner::computeProcessorPriority(const std::vector<long unsigned>& superstep_partition_work, const std::vector<long unsigned>& total_partition_work, const long unsigned& total_work, const BspInstance &instance, const float slack) {
    std::vector<size_t> temp = sorting_arrangement(computeProcessorPriorities(superstep_partition_work, total_partition_work, total_work, instance, slack));
    std::vector<unsigned> output(temp.size());
    for (size_t i = 0 ; i < output.size(); i++) {
        output[i] = temp[i];
    }

    return output;
}