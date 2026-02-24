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

#include <thread>
#include <vector>

#include "kl_improver.hpp"

namespace osp {

/// Synchronized (round-based) parallel KL improver.
///
/// Splits the superstep range across worker threads.  Each thread
/// runs an independent local search on its partition, then a barrier
/// synchronizes and stitches results together.
///
/// Thread-safety requirements on CommCostFunctionT:
///   - ComputeNodeAffinity must only read shared comm data for nodes
///     within the calling thread's [startStep, endStep] range.
///   - UpdateDatastructureAfterMove must only mutate comm data for
///     steps within [startStep, endStep].
///
/// All four cost functions satisfy these requirements:
///   - KlTotalCommCostFunction      (UpdateDatastructureAfterMove is no-op)
///   - KlHyperTotalCommCostFunction (mutations scoped to thread range)
///   - KlBspCommCostFunction        (comm deltas and updates scoped via
///                                    StepRangeProxy; parents/children outside
///                                    thread range are skipped)
///   - KlMaxBspCommCostFunction     (same scoping mechanism as BSP)
///
/// After each parallel round, SynchronizeActiveSchedule recomputes
/// the full schedule cost from scratch, correcting any cross-boundary
/// approximations from the scoped per-thread cost model.
///
template <typename GraphT,
          typename CommCostFunctionT,
          typename MemoryConstraintT = NoLocalSearchMemoryConstraint,
          unsigned windowSize = 1,
          typename CostT = double>
class KlSyncParallelImprover : public KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT> {
  protected:
    unsigned maxNumThreads_ = std::numeric_limits<unsigned>::max();

    void SetThreadBoundaries(const unsigned numThreads, const unsigned numSteps, bool lastThreadLargeRange) {
        if (numThreads == 1) {
            this->SetStartStep(0, this->threadDataVec_[0]);
            this->threadDataVec_[0].endStep_ = (numSteps > 0) ? numSteps - 1 : 0;
            this->threadDataVec_[0].originalEndStep_ = this->threadDataVec_[0].endStep_;
            return;
        } else {
            this->parameters_.threadRangeGap_ = std::max(this->parameters_.threadRangeGap_, this->activeSchedule_.GetStaleness());
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
    KlSyncParallelImprover() : KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>() {}

    explicit KlSyncParallelImprover(unsigned seed)
        : KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>(seed) {}

    virtual ~KlSyncParallelImprover() = default;

    void SetMaxNumThreads(const unsigned numThreads) { maxNumThreads_ = numThreads; }

    virtual ReturnStatus ImproveSchedule(BspSchedule<GraphT> &schedule) override {
        if (schedule.GetInstance().NumberOfProcessors() < 2) {
            return ReturnStatus::BEST_FOUND;
        }

        const unsigned hwThreads = std::max(1u, std::thread::hardware_concurrency());
        unsigned numThreads = std::min(maxNumThreads_, hwThreads);
        SetNumThreads(numThreads, schedule.NumberOfSupersteps());

        this->threadDataVec_.resize(numThreads);
        this->threadFinishedVec_.assign(numThreads, true);

        if (numThreads == 1) {
            this->parameters_.numParallelLoops_ = 1;
        }

        this->SetParameters(schedule.GetInstance().NumberOfVertices());
        this->InitializeDatastructures(schedule);
        const CostT initialCost = this->activeSchedule_.GetCost();

        // Track the global best across all parallel iterations.
        // Each thread optimizes its local step range independently, so
        // the stitched-together result may be worse than before.
        //
        // We use `schedule` itself as the best-state store: WriteSchedule
        // only overwrites proc/step assignments while preserving the
        // dynamic type (important for MaxBspSchedule::GetStaleness).
        CostT bestCost = initialCost;
        bool improved = false;

        for (size_t i = 0; i < this->parameters_.numParallelLoops_; ++i) {
            SetThreadBoundaries(numThreads, this->activeSchedule_.NumSteps(), i % 2 == 0);

            if (numThreads == 1) {
                // Single-thread fast path: no thread creation overhead
                auto &threadData = this->threadDataVec_[0];
                threadData.activeScheduleData_.InitializeCost(this->activeSchedule_.GetCost());
                threadData.selectionStrategy_.Setup(threadData.startStep_, threadData.endStep_);
                this->RunLocalSearch(threadData);
            } else {
                std::vector<std::thread> workers;
                workers.reserve(numThreads);

                for (unsigned t = 0; t < numThreads; ++t) {
                    auto &threadData = this->threadDataVec_[t];
                    threadData.activeScheduleData_.InitializeCost(this->activeSchedule_.GetCost());
                    threadData.selectionStrategy_.Setup(threadData.startStep_, threadData.endStep_);

                    std::cout << "Thread " << t << " processing steps " << threadData.startStep_ << " to " << threadData.endStep_
                              << std::endl;

                    workers.emplace_back([this, &threadData]() { this->RunLocalSearch(threadData); });
                }

                for (auto &w : workers) {
                    w.join();
                }
            }

            this->SynchronizeActiveSchedule(numThreads);
            const CostT currentCost = this->activeSchedule_.GetCost();

            if (currentCost < bestCost) {
                // Improved: save to schedule (preserves dynamic type)
                this->activeSchedule_.WriteSchedule(schedule);
                bestCost = currentCost;
                improved = true;
            } else if (numThreads > 1 && currentCost > bestCost) {
                // Regressed: revert to the best known schedule stored
                // in `schedule`. Re-initialization is heavyweight but
                // regression should be rare; correctness takes priority.
                this->activeSchedule_.Initialize(schedule);
                this->commCostF_.Initialize(this->activeSchedule_, this->procRange_);
            }

            if (numThreads > 1) {
                SetNumThreads(numThreads, this->activeSchedule_.NumSteps());
                this->threadFinishedVec_.resize(numThreads);
            }
        }

        this->CleanupDatastructures();
        return improved ? ReturnStatus::OSP_SUCCESS : ReturnStatus::BEST_FOUND;
    }
};

}    // namespace osp
