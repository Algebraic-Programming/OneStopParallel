/*
Copyright 2024 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner
*/

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <vector>

#include "osp/bsp/model/BspInstance.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"

namespace osp {

/**
 * @brief Represents the result of a subgraph scheduling operation.
 *
 * Contains the makespan of the schedule and the assignment of workers to nodes.
 */
struct SubgraphSchedule {
    /** @brief The total duration of the schedule. */
    double makespan_;

    /** @brief The number of workers of each type assigned to each node. */
    std::vector<std::vector<unsigned>> nodeAssignedWorkerPerType_;

    /** @brief Indicates which nodes were trimmed (if applicable). */
    std::vector<bool> wasTrimmed_;
};

/**
 * @class EftSubgraphScheduler
 * @brief A scheduler for subgraph tasks in a BSP model, optimizing for Earliest Finish Time (EFT).
 *
 * This class implements a scheduling algorithm that assigns processors to subgraph tasks
 * based on their computational requirements and the available resources, aiming to minimize
 * the overall makespan.
 *
 * @tparam GraphT The type of the graph representing the computational DAG.
 */
template <typename GraphT>
class EftSubgraphScheduler {
  public:
    EftSubgraphScheduler() = default;

    /**
     * @brief Runs the scheduling algorithm.
     *
     * @param instance The BSP instance containing the computational DAG and architecture.
     * @param multiplicities The multiplicity (parallelism) allowed for each node.
     * @param requiredProcTypes The required processor types for each node.
     * @param maxNumProcs The maximum number of processors allowed for each node.
     * @return The resulting schedule.
     */
    SubgraphSchedule Run(const BspInstance<GraphT> &instance,
                         const std::vector<unsigned> &multiplicities,
                         const std::vector<std::vector<VWorkwT<GraphT>>> &requiredProcTypes,
                         const std::vector<unsigned> &maxNumProcs) {
        PrepareForScheduling(instance, multiplicities, requiredProcTypes, maxNumProcs);
        return ExecuteSchedule(instance);
    }

    /**
     * @brief Sets the minimum work per processor threshold.
     *
     * @param minWorkPerProcessor The minimum work weight to justify assigning a processor.
     */
    void SetMinWorkPerProcessor(const VWorkwT<GraphT> minWorkPerProcessor) { minWorkPerProcessor_ = minWorkPerProcessor; }

  private:
    using JobIdT = VertexIdxT<GraphT>;

    VWorkwT<GraphT> minWorkPerProcessor_ = 2000;

    enum class JobStatus { WAITING, READY, RUNNING, COMPLETED };

    struct Job {
        JobIdT id_;

        std::vector<VWorkwT<GraphT>> requiredProcTypes_;
        VWorkwT<GraphT> totalWork_;
        unsigned multiplicity_ = 1;
        unsigned maxNumProcs_ = 1;

        JobIdT inDegreeCurrent_ = 0;

        JobStatus status_ = JobStatus::WAITING;
        VWorkwT<GraphT> upwardRank_ = 0.0;

        // --- Execution Tracking Members ---
        std::vector<unsigned> assignedWorkers_;
        double startTime_ = -1.0;
        double finishTime_ = -1.0;
    };

    struct JobPtrCompare {
        bool operator()(const Job *lhs, const Job *rhs) const {
            if (lhs->upwardRank_ != rhs->upwardRank_) {
                return lhs->upwardRank_ > rhs->upwardRank_;
            }
            return lhs->id_ > rhs->id_;
        }
    };

    std::vector<Job> jobs_;
    std::set<const Job *, JobPtrCompare> readyJobs_;

    void PrepareForScheduling(const BspInstance<GraphT> &instance,
                              const std::vector<unsigned> &multiplicities,
                              const std::vector<std::vector<VWorkwT<GraphT>>> &requiredProcTypes,
                              const std::vector<unsigned> &maxNumProcs) {
        jobs_.resize(instance.NumberOfVertices());
        const auto &graph = instance.GetComputationalDag();
        const size_t numWorkerTypes = instance.GetArchitecture().GetProcessorTypeCount().size();

        CalculateUpwardRanks(graph);

        JobIdT idx = 0;
        for (auto &job : jobs_) {
            job.id_ = idx;
            job.inDegreeCurrent_ = graph.InDegree(idx);
            if (job.inDegreeCurrent_ == 0) {
                job.status_ = JobStatus::READY;
                readyJobs_.insert(&job);
            } else {
                job.status_ = JobStatus::WAITING;
            }
            job.totalWork_ = graph.VertexWorkWeight(idx);
            job.maxNumProcs_ = std::min(
                maxNumProcs[idx], static_cast<unsigned>((job.totalWork_ + minWorkPerProcessor_ - 1) / minWorkPerProcessor_));
            job.multiplicity_ = std::min(multiplicities[idx], job.maxNumProcs_);
            job.requiredProcTypes_ = requiredProcTypes[idx];
            job.assignedWorkers_.resize(numWorkerTypes, 0);
            job.startTime_ = -1.0;
            job.finishTime_ = -1.0;
            idx++;
        }
    }

    void CalculateUpwardRanks(const GraphT &graph) {
        const auto reverseTopOrder = GetTopOrderReverse(graph);

        for (const auto &vertex : reverseTopOrder) {
            VWorkwT<GraphT> maxSuccessorRank = 0.0;
            for (const auto &child : graph.Children(vertex)) {
                maxSuccessorRank = std::max(maxSuccessorRank, jobs_.at(child).upwardRank_);
            }

            Job &job = jobs_.at(vertex);
            job.upwardRank_ = graph.VertexWorkWeight(vertex) + maxSuccessorRank;
        }
    }

    SubgraphSchedule ExecuteSchedule(const BspInstance<GraphT> &instance) {
        double currentTime = 0.0;
        std::vector<unsigned> availableWorkers = instance.GetArchitecture().GetProcessorTypeCount();
        std::vector<JobIdT> runningJobs;
        unsigned completedCount = 0;
        const auto &graph = instance.GetComputationalDag();

        while (completedCount < jobs_.size()) {
            AssignAndStartJobs(availableWorkers, runningJobs, currentTime);

            if (runningJobs.empty() && completedCount < jobs_.size()) {
                // Deadlock detected
                SubgraphSchedule result;
                result.makespan_ = -1.0;
                return result;
            }
            if (runningJobs.empty()) {
                break;
            }

            double nextEventTime = std::numeric_limits<double>::max();
            for (JobIdT id : runningJobs) {
                nextEventTime = std::min(nextEventTime, jobs_.at(id).finishTime_);
            }
            currentTime = nextEventTime;

            ProcessCompletedJobs(runningJobs, availableWorkers, completedCount, currentTime, graph);
        }

        SubgraphSchedule result;
        result.makespan_ = currentTime;
        result.nodeAssignedWorkerPerType_.resize(jobs_.size());
        const size_t numWorkerTypes = instance.GetArchitecture().GetProcessorTypeCount().size();
        for (const auto &job : jobs_) {
            result.nodeAssignedWorkerPerType_[job.id_].resize(numWorkerTypes);
            for (size_t i = 0; i < numWorkerTypes; ++i) {
                result.nodeAssignedWorkerPerType_[job.id_][i]
                    = (job.assignedWorkers_[i] + job.multiplicity_ - 1) / job.multiplicity_;
            }
        }
        return result;
    }

    void AssignAndStartJobs(std::vector<unsigned> &availableWorkers, std::vector<JobIdT> &runningJobs, double currentTime) {
        const size_t numWorkerTypes = availableWorkers.size();
        std::vector<Job *> jobsToStart;
        VWorkwT<GraphT> totalRunnablePriority = 0.0;

        // Collect startable jobs and assign minimal resources
        for (const Job *jobPtr : readyJobs_) {
            Job &job = jobs_[jobPtr->id_];
            bool canStart = true;
            for (size_t typeIdx = 0; typeIdx < numWorkerTypes; ++typeIdx) {
                if (job.requiredProcTypes_[typeIdx] > 0 && availableWorkers[typeIdx] < job.multiplicity_) {
                    canStart = false;
                    break;
                }
            }

            if (canStart) {
                jobsToStart.push_back(&job);
                totalRunnablePriority += job.upwardRank_;
                for (size_t typeIdx = 0; typeIdx < numWorkerTypes; ++typeIdx) {
                    if (job.requiredProcTypes_[typeIdx] > 0) {
                        job.assignedWorkers_[typeIdx] = job.multiplicity_;
                        availableWorkers[typeIdx] -= job.multiplicity_;
                    }
                }
            }
        }

        if (jobsToStart.empty()) return;

        // Distribute extra resources
        DistributeProportionalWorkers(jobsToStart, availableWorkers, totalRunnablePriority);
        DistributeGreedyWorkers(jobsToStart, availableWorkers);

        // Start jobs
        for (Job *jobPtr : jobsToStart) {
            Job &job = *jobPtr;
            job.status_ = JobStatus::RUNNING;
            job.startTime_ = currentTime;

            unsigned totalAssignedWorkers = std::accumulate(job.assignedWorkers_.begin(), job.assignedWorkers_.end(), 0u);
            double execTime = (totalAssignedWorkers > 0)
                                  ? static_cast<double>(job.totalWork_) / static_cast<double>(totalAssignedWorkers)
                                  : 0.0;
            job.finishTime_ = currentTime + execTime;

            runningJobs.push_back(job.id_);
            readyJobs_.erase(&job);
        }
    }

    void DistributeProportionalWorkers(const std::vector<Job *> &jobsToStart, std::vector<unsigned> &availableWorkers,
                                       VWorkwT<GraphT> totalRunnablePriority) {
        const size_t numWorkerTypes = availableWorkers.size();
        const std::vector<unsigned> remainingWorkersPool = availableWorkers;    // Snapshot for calculation

        for (Job *jobPtr : jobsToStart) {
            Job &job = *jobPtr;
            for (size_t typeIdx = 0; typeIdx < numWorkerTypes; ++typeIdx) {
                if (job.requiredProcTypes_[typeIdx] > 0) {
                    const unsigned currentTotalAssigned
                        = std::accumulate(job.assignedWorkers_.begin(), job.assignedWorkers_.end(), 0u);
                    const unsigned maxAdditionalWorkers
                        = (job.maxNumProcs_ > currentTotalAssigned) ? (job.maxNumProcs_ - currentTotalAssigned) : 0;

                    const double proportion
                        = (totalRunnablePriority > 0)
                              ? (static_cast<double>(job.upwardRank_) / static_cast<double>(totalRunnablePriority))
                              : (1.0 / static_cast<double>(jobsToStart.size()));
                    const unsigned proportionalShare
                        = static_cast<unsigned>(static_cast<double>(remainingWorkersPool[typeIdx]) * proportion);
                    const unsigned numProportionalChunks = (job.multiplicity_ > 0) ? proportionalShare / job.multiplicity_ : 0;
                    const unsigned numAvailableChunks
                        = (job.multiplicity_ > 0) ? availableWorkers[typeIdx] / job.multiplicity_ : 0;
                    const unsigned numChunksAllowedByMax
                        = (job.multiplicity_ > 0) ? maxAdditionalWorkers / job.multiplicity_ : 0;
                    const unsigned numChunksToAssign
                        = std::min({numProportionalChunks, numAvailableChunks, numChunksAllowedByMax});
                    const unsigned assigned = numChunksToAssign * job.multiplicity_;
                    job.assignedWorkers_[typeIdx] += assigned;
                    availableWorkers[typeIdx] -= assigned;
                }
            }
        }
    }

    void DistributeGreedyWorkers(const std::vector<Job *> &jobsToStart, std::vector<unsigned> &availableWorkers) {
        const size_t numWorkerTypes = availableWorkers.size();
        for (Job *jobPtr : jobsToStart) {
            Job &job = *jobPtr;
            for (size_t typeIdx = 0; typeIdx < numWorkerTypes; ++typeIdx) {
                if (job.requiredProcTypes_[typeIdx] > 0 && job.multiplicity_ > 0) {
                    const unsigned currentTotalAssigned
                        = std::accumulate(job.assignedWorkers_.begin(), job.assignedWorkers_.end(), 0u);
                    const unsigned maxAdditionalWorkers
                        = (job.maxNumProcs_ > currentTotalAssigned) ? (job.maxNumProcs_ - currentTotalAssigned) : 0;
                    const unsigned numAvailableChunks = availableWorkers[typeIdx] / job.multiplicity_;
                    const unsigned numChunksAllowedByMax = maxAdditionalWorkers / job.multiplicity_;
                    const unsigned assigned = std::min(numAvailableChunks, numChunksAllowedByMax) * job.multiplicity_;
                    job.assignedWorkers_[typeIdx] += assigned;
                    availableWorkers[typeIdx] -= assigned;
                }
            }
        }
    }

    void ProcessCompletedJobs(std::vector<JobIdT> &runningJobs, std::vector<unsigned> &availableWorkers, unsigned &completedCount,
                              double currentTime, const GraphT &graph) {
        const size_t numWorkerTypes = availableWorkers.size();

        // Optimize removal loop
        for (size_t i = 0; i < runningJobs.size();) {
            Job &job = jobs_.at(runningJobs[i]);
            if (job.finishTime_ <= currentTime) {
                job.status_ = JobStatus::COMPLETED;

                // Release workers
                for (size_t typeIdx = 0; typeIdx < numWorkerTypes; ++typeIdx) {
                    availableWorkers[typeIdx] += job.assignedWorkers_[typeIdx];
                }
                completedCount++;

                // Update successors
                for (const auto &successorId : graph.Children(job.id_)) {
                    Job &successorJob = jobs_.at(successorId);
                    successorJob.inDegreeCurrent_--;
                    if (successorJob.inDegreeCurrent_ == 0) {
                        successorJob.status_ = JobStatus::READY;
                        readyJobs_.insert(&successorJob);
                    }
                }

                // Fast removal: swap with last and pop
                runningJobs[i] = runningJobs.back();
                runningJobs.pop_back();
            } else {
                ++i;
            }
        }
    }
};

}    // namespace osp
