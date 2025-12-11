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
          unsigned WindowSize = 1,
          typename CostT = double>
class KlImproverMt : public KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT> {
  protected:
    unsigned maxNumThreads_ = std::numeric_limits<unsigned>::max();

    void SetThreadBoundaries(const unsigned numThreads, const unsigned numSteps, bool lastThreadLargeRange) {
        if (numThreads == 1) {
            this->SetStartStep(0, this->threadDataVec_[0]);
            this->threadDataVec_[0].endStep = (numSteps > 0) ? numSteps - 1 : 0;
            this->threadDataVec_[0].originalEndStep = this->threadDataVec_[0].endStep;
            return;
        } else {
            const unsigned totalGapSize = (numThreads - 1) * this->parameters_.threadRangeGap;
            const unsigned bonus = this->parameters_.threadMinRange;
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
                this->threadDataVec_[i].endStep = endStep;
                this->threadDataVec_[i].originalEndStep = this->threadDataVec_[i].endStep;
                currentStartStep = endStep + 1 + this->parameters_.threadRangeGap;
#ifdef KL_DEBUG_1
                std::cout << "thread " << i << ": start_step=" << this->thread_data_vec[i].start_step
                          << ", end_step=" << this->thread_data_vec[i].end_step << std::endl;
#endif
            }
        }
    }

    void SetNumThreads(unsigned &numThreads, const unsigned numSteps) {
        unsigned maxAllowedThreads = 0;
        if (numSteps >= this->parameters_.threadMinRange + this->parameters_.threadRangeGap) {
            const unsigned divisor = this->parameters_.threadMinRange + this->parameters_.threadRangeGap;
            if (divisor > 0) {
                // This calculation is based on the constraint that one thread's range is
                // 'min_range' larger than the others, and all ranges are at least 'min_range'.
                maxAllowedThreads = (numSteps + this->parameters_.threadRangeGap - this->parameters_.threadMinRange) / divisor;
            } else {
                maxAllowedThreads = numSteps;
            }
        } else if (numSteps >= this->parameters_.threadMinRange) {
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
    KlImproverMt() : KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT>() {}

    explicit KlImproverMt(unsigned seed) : KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT>(seed) {}

    virtual ~KlImproverMt() = default;

    void SetMaxNumThreads(const unsigned numThreads) { maxNumThreads_ = numThreads; }

    virtual RETURN_STATUS ImproveSchedule(BspSchedule<GraphT> &schedule) override {
        if (schedule.getInstance().NumberOfProcessors() < 2) {
            return RETURN_STATUS::BEST_FOUND;
        }

        unsigned numThreads = std::min(maxNumThreads_, static_cast<unsigned>(omp_get_max_threads()));
        SetNumThreads(numThreads, schedule.numberOfSupersteps());

        this->threadDataVec_.resize(numThreads);
        this->threadFinishedVec_.assign(numThreads, true);

        if (numThreads == 1) {
            this->parameters_.numParallelLoops
                = 1;    // no parallelization with one thread. Affects parameters.max_out_iteration calculation in set_parameters()
        }

        this->set_parameters(schedule.getInstance().NumberOfVertices());
        this->initialize_datastructures(schedule);
        const CostT initialCost = this->activeSchedule_.GetCost();

        for (size_t i = 0; i < this->parameters_.numParallelLoops; ++i) {
            SetThreadBoundaries(numThreads, schedule.numberOfSupersteps(), i % 2 == 0);

#pragma omp parallel num_threads(numThreads)
            {
                const size_t threadId = static_cast<size_t>(omp_get_thread_num());
                auto &threadData = this->threadDataVec_[threadId];
                threadData.activeScheduleData.InitializeCost(this->activeSchedule_.GetCost());
                threadData.selectionStrategy.Setup(threadData.startStep, threadData.endStep);
                this->RunLocalSearch(threadData);
            }

            this->SynchronizeActiveSchedule(numThreads);
            if (numThreads > 1) {
                this->activeSchedule_.SetCost(this->commCostF_.ComputeScheduleCost());
                SetNumThreads(numThreads, schedule.numberOfSupersteps());
                this->threadFinishedVec_.resize(numThreads);
            }
        }

        if (initialCost > this->activeSchedule_.GetCost()) {
            this->activeSchedule_.WriteSchedule(schedule);
            this->CleanupDatastructures();
            return RETURN_STATUS::OSP_SUCCESS;
        } else {
            this->CleanupDatastructures();
            return RETURN_STATUS::BEST_FOUND;
        }
    }
};

}    // namespace osp
