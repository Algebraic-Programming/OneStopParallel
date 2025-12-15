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
            this->threadDataVec_[0].endStep_ = (numSteps > 0) ? numSteps - 1 : 0;
            this->threadDataVec_[0].originalEndStep_ = this->threadDataVec_[0].endStep_;
            return;
        } else {
            const unsigned totalGapSize = (numThreads - 1) * this->parameters_.threadRangeGap_;
            const unsigned bonus = this->parameters_.threadMinRange_;
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
                this->threadDataVec_[i].endStep_ = endStep;
                this->threadDataVec_[i].originalEndStep_ = this->threadDataVec_[i].endStep_;
                currentStartStep = endStep + 1 + this->parameters_.threadRangeGap_;
#ifdef KL_DEBUG_1
                std::cout << "thread " << i << ": start_step=" << this->threadDataVec_[i].startStep_
                          << ", end_step=" << this->threadDataVec_[i].endStep_ << std::endl;
#endif
            }
        }
    }

    void SetNumThreads(unsigned &numThreads, const unsigned numSteps) {
        unsigned maxAllowedThreads = 0;
        if (numSteps >= this->parameters_.threadMinRange_ + this->parameters_.threadRangeGap_) {
            const unsigned divisor = this->parameters_.threadMinRange_ + this->parameters_.threadRangeGap_;
            if (divisor > 0) {
                // This calculation is based on the constraint that one thread's range is
                // 'min_range' larger than the others, and all ranges are at least 'min_range'.
                maxAllowedThreads = (numSteps + this->parameters_.threadRangeGap_ - this->parameters_.threadMinRange_) / divisor;
            } else {
                maxAllowedThreads = numSteps;
            }
        } else if (numSteps >= this->parameters_.threadMinRange_) {
            maxAllowedThreads = 1;
        }

        if (numThreads > maxAllowedThreads) {
            numThreads = maxAllowedThreads;
        }

        if (numThreads == 0) {
            numThreads = 1;
        }
#ifdef KL_DEBUG_1
        std::cout << "num threads: " << numThreads << " number of supersteps: " << numSteps
                  << ", max allowed threads: " << maxAllowedThreads << std::endl;
#endif
    }

  public:
    KlImproverMt() : KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT>() {}

    explicit KlImproverMt(unsigned seed) : KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT>(seed) {}

    virtual ~KlImproverMt() = default;

    void SetMaxNumThreads(const unsigned numThreads) { maxNumThreads_ = numThreads; }

    virtual ReturnStatus ImproveSchedule(BspSchedule<GraphT> &schedule) override {
        if (schedule.GetInstance().NumberOfProcessors() < 2) {
            return ReturnStatus::BEST_FOUND;
        }

        unsigned numThreads = std::min(maxNumThreads_, static_cast<unsigned>(omp_get_max_threads()));
        SetNumThreads(numThreads, schedule.NumberOfSupersteps());

        this->threadDataVec_.resize(numThreads);
        this->threadFinishedVec_.assign(numThreads, true);

        if (numThreads == 1) {
            this->parameters_.numParallelLoops_
                = 1;    // no parallelization with one thread. Affects parameters.max_out_iteration calculation in set_parameters()
        }

        this->SetParameters(schedule.GetInstance().NumberOfVertices());
        this->InitializeDatastructures(schedule);
        const CostT initialCost = this->activeSchedule_.GetCost();

        for (size_t i = 0; i < this->parameters_.numParallelLoops_; ++i) {
            SetThreadBoundaries(numThreads, schedule.NumberOfSupersteps(), i % 2 == 0);

#pragma omp parallel num_threads(numThreads)
            {
                const size_t threadId = static_cast<size_t>(omp_get_thread_num());
                auto &threadData = this->threadDataVec_[threadId];
                threadData.activeScheduleData_.InitializeCost(this->activeSchedule_.GetCost());
                threadData.selectionStrategy_.Setup(threadData.startStep_, threadData.endStep_);
                this->RunLocalSearch(threadData);
            }

            this->SynchronizeActiveSchedule(numThreads);
            if (numThreads > 1) {
                this->activeSchedule_.SetCost(this->commCostF_.ComputeScheduleCost());
                SetNumThreads(numThreads, schedule.NumberOfSupersteps());
                this->threadFinishedVec_.resize(numThreads);
            }
        }

        if (initialCost > this->activeSchedule_.GetCost()) {
            this->activeSchedule_.WriteSchedule(schedule);
            this->CleanupDatastructures();
            return ReturnStatus::OSP_SUCCESS;
        } else {
            this->CleanupDatastructures();
            return ReturnStatus::BEST_FOUND;
        }
    }
};

}    // namespace osp
