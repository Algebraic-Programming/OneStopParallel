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

#include "osp/bsp/scheduler/Scheduler.hpp"

namespace osp {

struct LinearInterpolation {
    float operator()(float alpha, const float slack = 0.0f) { return (1.0f - slack) * alpha; }
};

struct FlatSplineInterpolation {
    float operator()(float alpha, const float slack = 0.0f) {
        return (1.0f - slack) * static_cast<float>(((-2.0) * pow(alpha, 3.0)) + (3.0 * pow(alpha, 2.0)));
    }
};

struct SuperstepOnlyInterpolation {
    float operator()(float, const float) { return 0.0f; };
};

struct GlobalOnlyInterpolation {
    float operator()(float, const float) { return 1.0f; };
};

template <typename GraphT, typename InterpolationT = FlatSplineInterpolation>
class LoadBalancerBase : public Scheduler<GraphT> {
    static_assert(std::is_invocable_r<float, InterpolationT, float, float>::value,
                  "Interpolation_t must be invocable with two float arguments and return a float.");

  protected:
    /// @brief Computes the interpolated priorities
    /// @param superstep_partition_work vector with current work distribution in current superstep
    /// @param total_partition_work vector with current work distribution overall
    /// @param total_work total work weight of all nodes of the graph
    /// @param instance bsp instance
    /// @param slack how much to ignore global balance
    /// @return vector with the interpolated priorities
    std::vector<float> ComputeProcessorPrioritiesInterpolation(const std::vector<VWorkwT<GraphT>> &superstepPartitionWork,
                                                               const std::vector<VWorkwT<GraphT>> &totalPartitionWork,
                                                               const VWorkwT<GraphT> &totalWork,
                                                               const BspInstance<GraphT> &instance,
                                                               const float slack = 0.0) {
        VWorkwT<GraphT> workTillNow = 0;
        for (const auto &partWork : totalPartitionWork) {
            workTillNow += partWork;
        }

        float percentageComplete = static_cast<float>(workTillNow) / static_cast<float>(totalWork);

        float value = InterpolationT()(percentageComplete, slack);

        std::vector<float> procPrio(instance.NumberOfProcessors());
        for (size_t i = 0; i < procPrio.size(); i++) {
            assert(static_cast<double>(totalPartitionWork[i]) < std::numeric_limits<float>::max()
                   && static_cast<double>(superstepPartitionWork[i]) < std::numeric_limits<float>::max());
            procPrio[i] = ((1 - value) * static_cast<float>(superstepPartitionWork[i]))
                          + (value * static_cast<float>(totalPartitionWork[i]));
        }

        return procPrio;
    }

    /// @brief Computes processor priorities
    /// @param superstepPartitionWork vector with current work distribution in current superstep
    /// @param totalPartitionWork vector with current work distribution overall
    /// @param totalWork total work weight of all nodes of the graph
    /// @param instance bsp instance
    /// @param slack how much to ignore global balance
    /// @return vector with the processors in order of priority
    std::vector<unsigned> ComputeProcessorPriority(const std::vector<VWorkwT<GraphT>> &superstepPartitionWork,
                                                   const std::vector<VWorkwT<GraphT>> &totalPartitionWork,
                                                   const VWorkwT<GraphT> &totalWork,
                                                   const BspInstance<GraphT> &instance,
                                                   const float slack = 0.0) {
        return SortingArrangement<float, unsigned>(
            ComputeProcessorPrioritiesInterpolation(superstepPartitionWork, totalPartitionWork, totalWork, instance, slack));
    }

  public:
    /// @brief Deconstructor for the IListPartitioner class
    virtual ~LoadBalancerBase() = default;
};

}    // namespace osp
