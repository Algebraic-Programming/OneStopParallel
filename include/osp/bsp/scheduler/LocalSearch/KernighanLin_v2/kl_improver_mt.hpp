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

#include <omp.h>

#include "kl_improver.hpp"

namespace osp {

template <typename GraphT,
          typename CommCostFunctionT,
          typename MemoryConstraintT = NoLocalSearchMemoryConstraint,
          unsigned windowSize = 1,
          typename CostT = double>
class KlImproverMt : public KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT> {
  protected:
    unsigned maxNumThreads_ = std::numeric_limits<unsigned>::max();

    void SetThreadBoundaries(const unsigned numThreads, const unsigned numSteps, bool lastThreadLargeRange) {
        if (numThreads == 1) {
            this->SetStartStep(0, this->threadDataVec_[0]);
            this->threadDataVec_[0].end_step = (numSteps > 0) ? numSteps - 1 : 0;
            this->threadDataVec_[0].original_end_step = this->threadDataVec_[0].end_step;
            return;
        } else {
            const unsigned totalGapSize = (numThreads - 1) * this->parameters_.thread_range_gap;
            const unsigned bonus = this->parameters_.thread_min_range;
            const unsigned stepsToDistribute = numSteps - totalGapSize - bonus;
            const unsigned baseRange = stepsToDistribute / numThreads;
            const unsigned remainder = stepsToDistribute % numThreads;
            const unsigned largeRangeThreadIdx = lastThreadLargeRange ? numThreads - 1 : 0;

            unsigned currentStartStep = 0;
            for (unsigned i = 0; i < numThreads; ++i) {
                this->threadFinishedVec_[i] = false;
                this->SetStartStep(currentStartStep, this->threadDataVec_[i]);
                unsigned currentRange = baseRange + (i < remainder ? 1 : 0);
                if (i == largeRangeThreadIdx) {
                    currentRange += bonus;
                }

                const unsigned endStep = currentStartStep + currentRange - 1;
                this->threadDataVec_[i].end_step = endStep;
                this->threadDataVec_[i].original_end_step = this->threadDataVec_[i].end_step;
                currentStartStep = endStep + 1 + this->parameters_.thread_range_gap;
#ifdef KL_DEBUG_1
                std::cout << "thread " << i << ": start_step=" << this->thread_data_vec[i].start_step
                          << ", end_step=" << this->thread_data_vec[i].end_step << std::endl;
#endif
            }
        }
    }

    void SetNumThreads(unsigned &numThreads, const unsigned numSteps) {
        unsigned maxAllowedThreads = 0;
        if (numSteps >= this->parameters_.thread_min_range + this->parameters_.thread_range_gap) {
            const unsigned divisor = this->parameters_.thread_min_range + this->parameters_.thread_range_gap;
            if (divisor > 0) {
                // This calculation is based on the constraint that one thread's range is
                // 'min_range' larger than the others, and all ranges are at least 'min_range'.
                maxAllowedThreads = (numSteps + this->parameters_.thread_range_gap - this->parameters_.thread_min_range) / divisor;
            } else {
                maxAllowedThreads = numSteps;
            }
        } else if (numSteps >= this->parameters_.thread_min_range) {
            maxAllowedThreads = 1;
        }

        if (numThreads > maxAllowedThreads) {
            numThreads = maxAllowedThreads;
        }

        if (numThreads == 0) {
            numThreads = 1;
        }
#ifdef KL_DEBUG_1
        std::cout << "num threads: " << num_threads << " number of supersteps: " << num_steps
                  << ", max allowed threads: " << max_allowed_threads << std::endl;
#endif
    }

  public:
    KlImproverMt() : KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>() {}

    explicit KlImproverMt(unsigned seed) : KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>(seed) {}

    virtual ~KlImproverMt() = default;

    void SetMaxNumThreads(const unsigned numThreads) { maxNumThreads_ = numThreads; }

    virtual ReturnStatus improveSchedule(BspSchedule<GraphT> &schedule) override {
        if (schedule.GetInstance().NumberOfProcessors() < 2) {
            return ReturnStatus::BEST_FOUND;
        }

        unsigned numThreads = std::min(maxNumThreads_, static_cast<unsigned>(omp_get_max_threads()));
        SetNumThreads(numThreads, schedule.NumberOfSupersteps());

        this->threadDataVec_.resize(numThreads);
        this->threadFinishedVec_.assign(numThreads, true);

        if (numThreads == 1) {
            this->parameters_.num_parallel_loops
                = 1;    // no parallelization with one thread. Affects parameters.max_out_iteration calculation in set_parameters()
        }

        this->SetParameters(schedule.GetInstance().NumberOfVertices());
        this->InitializeDatastructures(schedule);
        const CostT initialCost = this->activeSchedule_.get_cost();

        for (size_t i = 0; i < this->parameters_.num_parallel_loops; ++i) {
            SetThreadBoundaries(numThreads, schedule.NumberOfSupersteps(), i % 2 == 0);

#pragma omp parallel num_threads(numThreads)
            {
                const size_t threadId = static_cast<size_t>(omp_get_thread_num());
                auto &threadData = this->threadDataVec_[threadId];
                threadData.active_schedule_data.initialize_cost(this->activeSchedule_.get_cost());
                threadData.selection_strategy.setup(threadData.start_step, threadData.end_step);
                this->RunLocalSearch(threadData);
            }

            this->SynchronizeActiveSchedule(numThreads);
            if (numThreads > 1) {
                this->activeSchedule_.set_cost(this->commCostF_.compute_schedule_cost());
                SetNumThreads(numThreads, schedule.NumberOfSupersteps());
                this->threadFinishedVec_.resize(numThreads);
            }
        }

        if (initialCost > this->activeSchedule_.get_cost()) {
            this->activeSchedule_.write_schedule(schedule);
            this->CleanupDatastructures();
            return ReturnStatus::OSP_SUCCESS;
        } else {
            this->CleanupDatastructures();
            return ReturnStatus::BEST_FOUND;
        }
    }
};

}    // namespace osp
