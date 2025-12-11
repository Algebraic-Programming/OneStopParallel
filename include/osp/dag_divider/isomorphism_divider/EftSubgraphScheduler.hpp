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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "osp/bsp/model/BspInstance.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"

namespace osp {

struct SubgraphSchedule {
    double makespan;
    std::vector<std::vector<unsigned>> nodeAssignedWorkerPerType;
    std::vector<bool> wasTrimmed;
};

template <typename GraphT>
class EftSubgraphScheduler {
  public:
    EftSubgraphScheduler() = default;

    SubgraphSchedule Run(const BspInstance<GraphT> &instance,
                         const std::vector<unsigned> &multiplicities,
                         const std::vector<std::vector<VWorkwT<GraphT>>> &requiredProcTypes,
                         const std::vector<unsigned> &maxNumProcs) {
        PrepareForScheduling(instance, multiplicities, requiredProcTypes, maxNumProcs);
        return ExecuteSchedule(instance);
    }

    void SetMinWorkPerProcessor(const VWorkwT<GraphT> minWorkPerProcessor) { minWorkPerProcessor_ = minWorkPerProcessor; }

  private:
    static constexpr bool verbose = false;

    using JobIdT = VertexIdxT<GraphT>;

    VWorkwT<GraphT> minWorkPerProcessor_ = 2000;

    enum class JobStatus { WAITING, READY, RUNNING, COMPLETED };

    struct Job {
        JobIdT id;

        std::vector<VWorkwT<GraphT>> requiredProcTypes;
        VWorkwT<GraphT> totalWork;
        unsigned multiplicity = 1;
        unsigned maxNumProcs = 1;

        JobIdT inDegreeCurrent = 0;

        JobStatus status = JobStatus::WAITING;
        VWorkwT<GraphT> upwardRank = 0.0;

        // --- Execution Tracking Members ---
        std::vector<unsigned> assignedWorkers;
        double startTime = -1.0;
        double finishTime = -1.0;
    };

    // Custom comparator for storing Job pointers in the ready set, sorted by rank.
    struct JobPtrCompare {
        bool operator()(const Job *lhs, const Job *rhs) const {
            if (lhs->upwardRank != rhs->upwardRank) {
                return lhs->upwardRank > rhs->upwardRank;
            }
            return lhs->id > rhs->id;    // Tie-breaking
        }
    };

    std::vector<Job> jobs_;
    std::set<const Job *, JobPtrCompare> readyJobs_;

    void PrepareForScheduling(const BspInstance<GraphT> &instance,
                              const std::vector<unsigned> &multiplicities,
                              const std::vector<std::vector<VWorkwT<GraphT>>> &requiredProcTypes,
                              const std::vector<unsigned> &maxNumProcs) {
        jobs_.resize(instance.NumberOfVertices());
        if constexpr (verbose) {
            std::cout << "--- Preparing for Subgraph Scheduling ---" << std::endl;
        }
        const auto &graph = instance.GetComputationalDag();
        const size_t numWorkerTypes = instance.GetArchitecture().GetProcessorTypeCount().size();

        CalculateUpwardRanks(graph);

        if constexpr (verbose) {
            std::cout << "Initializing jobs..." << std::endl;
        }
        JobIdT idx = 0;
        for (auto &job : jobs_) {
            job.id = idx;
            job.inDegreeCurrent = graph.InDegree(idx);
            if (job.inDegreeCurrent == 0) {
                job.status = JobStatus::READY;
                readyJobs_.insert(&job);
            } else {
                job.status = JobStatus::WAITING;
            }
            job.totalWork = graph.VertexWorkWeight(idx);
            job.maxNumProcs = std::min(maxNumProcs[idx],
                                       static_cast<unsigned>((job.totalWork + minWorkPerProcessor_ - 1) / minWorkPerProcessor_));
            job.multiplicity = std::min(multiplicities[idx], job.maxNumProcs);
            job.requiredProcTypes = requiredProcTypes[idx];
            job.assignedWorkers.resize(numWorkerTypes, 0);
            job.startTime = -1.0;
            job.finishTime = -1.0;

            if constexpr (verbose) {
                std::cout << "  - Job " << idx << ": rank=" << job.upward_rank << ", mult=" << job.multiplicity
                          << ", max_procs=" << job.max_num_procs << ", work=" << job.total_work
                          << ", status=" << (job.status == JobStatus::READY ? "READY" : "WAITING") << std::endl;
            }
            idx++;
        }
    }

    void CalculateUpwardRanks(const GraphT &graph) {
        const auto reverseTopOrder = GetTopOrderReverse(graph);

        for (const auto &vertex : reverseTopOrder) {
            VWorkwT<GraphT> maxSuccessorRank = 0.0;
            for (const auto &child : graph.Children(vertex)) {
                maxSuccessorRank = std::max(maxSuccessorRank, jobs_.at(child).upwardRank);
            }

            Job &job = jobs_.at(vertex);
            job.upwardRank = graph.VertexWorkWeight(vertex) + maxSuccessorRank;
        }
    }

    SubgraphSchedule ExecuteSchedule(const BspInstance<GraphT> &instance) {
        double currentTime = 0.0;
        std::vector<unsigned> availableWorkers = instance.GetArchitecture().GetProcessorTypeCount();
        const size_t numWorkerTypes = availableWorkers.size();
        std::vector<JobIdT> runningJobs;
        unsigned completedCount = 0;
        const auto &graph = instance.GetComputationalDag();

        if constexpr (verbose) {
            std::cout << "\n--- Subgraph Scheduling Execution Started ---" << std::endl;
            std::cout << "Total jobs: " << jobs_.size() << std::endl;
            std::cout << "Initial available workers: ";
            for (size_t i = 0; i < numWorkerTypes; ++i) {
                std::cout << "T" << i << ":" << availableWorkers[i] << " ";
            }
            std::cout << std::endl;
        }

        while (completedCount < jobs_.size()) {
            if constexpr (verbose) {
                std::cout << "\n[T=" << currentTime << "] --- New Scheduling Step ---" << std::endl;
                std::cout << "Completed jobs: " << completedCount << "/" << jobs_.size() << std::endl;
                std::cout << "Available workers: ";
                for (size_t i = 0; i < numWorkerTypes; ++i) {
                    std::cout << "T" << i << ":" << availableWorkers[i] << " ";
                }
                std::cout << std::endl;
                std::cout << "Ready queue size: " << readyJobs_.size() << ". Running jobs: " << runningJobs.size() << std::endl;
            }

            std::vector<Job *> jobsToStart;
            VWorkwT<GraphT> totalRunnablePriority = 0.0;

            // Iterate through ready jobs and assign minimum resources if available.
            for (const Job *jobPtr : readyJobs_) {
                Job &job = jobs_[jobPtr->id];
                bool canStart = true;
                for (size_t typeIdx = 0; typeIdx < numWorkerTypes; ++typeIdx) {
                    if (job.requiredProcTypes[typeIdx] > 0 && availableWorkers[typeIdx] < job.multiplicity) {
                        canStart = false;
                        break;
                    }
                }

                if (canStart) {
                    jobsToStart.push_back(&job);
                    totalRunnablePriority += job.upwardRank;
                    for (size_t typeIdx = 0; typeIdx < numWorkerTypes; ++typeIdx) {
                        if (job.requiredProcTypes[typeIdx] > 0) {
                            job.assignedWorkers[typeIdx] = job.multiplicity;
                            availableWorkers[typeIdx] -= job.multiplicity;
                        }
                    }
                }
            }

            if (!jobsToStart.empty()) {
                if constexpr (verbose) {
                    std::cout << "Allocating workers to " << jobsToStart.size() << " runnable jobs..." << std::endl;
                }

                // Distribute remaining workers proportionally among the jobs that just started.
                const std::vector<unsigned> remainingWorkersPool = availableWorkers;
                for (Job *jobPtr : jobsToStart) {
                    Job &job = *jobPtr;
                    for (size_t typeIdx = 0; typeIdx < numWorkerTypes; ++typeIdx) {
                        if (job.requiredProcTypes[typeIdx] > 0) {
                            const unsigned currentTotalAssigned
                                = std::accumulate(job.assignedWorkers.begin(), job.assignedWorkers.end(), 0u);
                            const unsigned maxAdditionalWorkers
                                = (job.maxNumProcs > currentTotalAssigned) ? (job.maxNumProcs - currentTotalAssigned) : 0;

                            const double proportion
                                = (totalRunnablePriority > 0)
                                      ? (static_cast<double>(job.upwardRank) / static_cast<double>(totalRunnablePriority))
                                      : (1.0 / static_cast<double>(jobsToStart.size()));
                            const unsigned proportionalShare
                                = static_cast<unsigned>(static_cast<double>(remainingWorkersPool[typeIdx]) * proportion);
                            const unsigned numProportionalChunks = (job.multiplicity > 0) ? proportionalShare / job.multiplicity
                                                                                          : 0;
                            const unsigned numAvailableChunks
                                = (job.multiplicity > 0) ? availableWorkers[typeIdx] / job.multiplicity : 0;
                            const unsigned numChunksAllowedByMax
                                = (job.multiplicity > 0) ? maxAdditionalWorkers / job.multiplicity : 0;
                            const unsigned numChunksToAssign
                                = std::min({numProportionalChunks, numAvailableChunks, numChunksAllowedByMax});
                            const unsigned assigned = numChunksToAssign * job.multiplicity;
                            job.assignedWorkers[typeIdx] += assigned;
                            availableWorkers[typeIdx] -= assigned;
                        }
                    }
                }

                // Greedily assign any remaining workers to the highest-rank jobs that can take them.
                for (Job *jobPtr : jobsToStart) {
                    Job &job = *jobPtr;
                    for (size_t typeIdx = 0; typeIdx < numWorkerTypes; ++typeIdx) {
                        if (job.requiredProcTypes[typeIdx] > 0 && job.multiplicity > 0) {
                            const unsigned currentTotalAssigned
                                = std::accumulate(job.assignedWorkers.begin(), job.assignedWorkers.end(), 0u);
                            const unsigned maxAdditionalWorkers
                                = (job.maxNumProcs > currentTotalAssigned) ? (job.maxNumProcs - currentTotalAssigned) : 0;
                            const unsigned numAvailableChunks = availableWorkers[typeIdx] / job.multiplicity;
                            const unsigned numChunksAllowedByMax = maxAdditionalWorkers / job.multiplicity;
                            const unsigned assigned = std::min(numAvailableChunks, numChunksAllowedByMax) * job.multiplicity;
                            job.assignedWorkers[typeIdx] += assigned;
                            availableWorkers[typeIdx] -= assigned;
                        }
                    }
                }

                for (Job *jobPtr : jobsToStart) {
                    Job &job = *jobPtr;

                    job.status = JobStatus::RUNNING;
                    job.startTime = currentTime;

                    // Calculate finish time based on total work and total assigned workers.
                    unsigned totalAssignedWorkers = std::accumulate(job.assignedWorkers.begin(), job.assignedWorkers.end(), 0u);
                    double execTime = (totalAssignedWorkers > 0)
                                          ? static_cast<double>(job.totalWork) / static_cast<double>(totalAssignedWorkers)
                                          : 0.0;
                    job.finishTime = currentTime + execTime;

                    runningJobs.push_back(job.id);
                    readyJobs_.erase(&job);
                }
            }

            // 2. ADVANCE TIME
            if (runningJobs.empty() && completedCount < jobs_.size()) {
                std::cerr << "Error: Deadlock detected. No running jobs and " << jobs_.size() - completedCount
                          << " jobs incomplete." << std::endl;
                if constexpr (verbose) {
                    std::cout << "Deadlock! Ready queue:" << std::endl;
                    for (const auto *readyJobPtr : readyJobs_) {
                        const Job &job = *readyJobPtr;
                        std::cout << "  - Job " << job.id << " (mult " << job.multiplicity << ") needs workers: ";
                        for (size_t typeIdx = 0; typeIdx < numWorkerTypes; ++typeIdx) {
                            if (job.requiredProcTypes[typeIdx] > 0) {
                                std::cout << "T" << typeIdx << ":" << job.multiplicity << " ";
                            }
                        }
                        std::cout << std::endl;
                    }
                    std::cout << "Available workers: ";
                    for (size_t i = 0; i < numWorkerTypes; ++i) {
                        std::cout << "T" << i << ":" << availableWorkers[i] << " ";
                    }
                    std::cout << std::endl;
                }
                SubgraphSchedule result;
                result.makespan = -1.0;
                return result;
            }
            if (runningJobs.empty()) {
                break;    // All jobs are done
            }

            double nextEventTime = std::numeric_limits<double>::max();
            for (JobIdT id : runningJobs) {
                nextEventTime = std::min(nextEventTime, jobs_.at(id).finishTime);
            }
            if constexpr (verbose) {
                std::cout << "Advancing time from " << currentTime << " to " << nextEventTime << std::endl;
            }
            currentTime = nextEventTime;

            // 3. PROCESS COMPLETED JOBS
            auto it = runningJobs.begin();
            while (it != runningJobs.end()) {
                Job &job = jobs_.at(*it);
                if (job.finishTime <= currentTime) {
                    job.status = JobStatus::COMPLETED;
                    if constexpr (verbose) {
                        std::cout << "Job " << job.id << " finished at T=" << currentTime << std::endl;
                    }
                    // Release workers
                    for (size_t typeIdx = 0; typeIdx < numWorkerTypes; ++typeIdx) {
                        availableWorkers[typeIdx] += job.assignedWorkers[typeIdx];
                    }
                    completedCount++;

                    // Update successors
                    if constexpr (verbose) {
                        std::cout << "  - Updating successors..." << std::endl;
                    }
                    for (const auto &successorId : graph.Children(job.id)) {
                        Job &successorJob = jobs_.at(successorId);
                        successorJob.inDegreeCurrent--;
                        if (successorJob.inDegreeCurrent == 0) {
                            successorJob.status = JobStatus::READY;
                            readyJobs_.insert(&successorJob);
                            if constexpr (verbose) {
                                std::cout << "    - Successor " << successorJob.id << " is now READY." << std::endl;
                            }
                        }
                    }
                    it = runningJobs.erase(it);    // Remove from running list
                } else {
                    ++it;
                }
            }
        }

        if constexpr (verbose) {
            std::cout << "\n--- Subgraph Scheduling Finished ---" << std::endl;
            std::cout << "Final Makespan: " << currentTime << std::endl;
            std::cout << "Job Summary:" << std::endl;
            for (const auto &job : jobs_) {
                std::cout << "  - Job " << job.id << ": Multiplicity=" << job.multiplicity << ", Max Procs=" << job.max_num_procs
                          << ", Work=" << job.total_work << ", Start=" << job.start_time << ", Finish=" << job.finish_time
                          << ", Workers=[";
                for (size_t i = 0; i < job.assigned_workers.size(); ++i) {
                    std::cout << "T" << i << ":" << job.assigned_workers[i] << (i == job.assigned_workers.size() - 1 ? "" : ", ");
                }
                std::cout << "]" << std::endl;
            }
        }

        SubgraphSchedule result;
        result.makespan = currentTime;
        result.nodeAssignedWorkerPerType.resize(jobs_.size());
        for (const auto &job : jobs_) {
            result.nodeAssignedWorkerPerType[job.id].resize(numWorkerTypes);
            for (size_t i = 0; i < numWorkerTypes; ++i) {
                result.nodeAssignedWorkerPerType[job.id][i] = (job.assignedWorkers[i] + job.multiplicity - 1) / job.multiplicity;
            }
        }
        return result;
    }
};

}    // namespace osp
