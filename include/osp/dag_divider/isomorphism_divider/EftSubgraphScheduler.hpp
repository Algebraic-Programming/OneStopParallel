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
    double makespan_;
    std::vector<std::vector<unsigned>> nodeAssignedWorkerPerType_;
    std::vector<bool> wasTrimmed_;
};

template <typename GraphT>
class EftSubgraphScheduler {
  public:
    EftSubgraphScheduler() = default;

    SubgraphSchedule Run(const BspInstance<GraphT> &instance,
                         const std::vector<unsigned> &multiplicities,
                         const std::vector<std::vector<VWorkwT<GraphT>>> &requiredProcTypes,
                         const std::vector<unsigned> &maxNumProcs) {
        prepare_for_scheduling(instance, multiplicities, required_proc_types, max_num_procs);
        return ExecuteSchedule(instance);
    }

    void SetMinWorkPerProcessor(const VWorkwT<GraphT> minWorkPerProcessor) { min_work_per_processor_ = min_work_per_processor; }

  private:
    static constexpr bool verbose_ = false;

    using job_id_t = VertexIdxT<GraphT>;

    VWorkwT<GraphT> minWorkPerProcessor_ = 2000;

    enum class JobStatus { WAITING, READY, RUNNING, COMPLETED };

    struct Job {
        job_id_t id_;

        std::vector<VWorkwT<GraphT>> requiredProcTypes_;
        VWorkwT<GraphT> totalWork_;
        unsigned multiplicity_ = 1;
        unsigned maxNumProcs_ = 1;

        job_id_t inDegreeCurrent_ = 0;

        JobStatus status_ = JobStatus::WAITING;
        VWorkwT<GraphT> upwardRank_ = 0.0;

        // --- Execution Tracking Members ---
        std::vector<unsigned> assignedWorkers_;
        double startTime_ = -1.0;
        double finishTime_ = -1.0;
    };

    // Custom comparator for storing Job pointers in the ready set, sorted by rank.
    struct JobPtrCompare {
        bool operator()(const Job *lhs, const Job *rhs) const {
            if (lhs->upwardRank_ != rhs->upwardRank_) {
                return lhs->upwardRank_ > rhs->upwardRank_;
            }
            return lhs->id_ > rhs->id_;    // Tie-breaking
        }
    };

    std::vector<Job> jobs_;
    std::set<const Job *, JobPtrCompare> readyJobs_;

    void PrepareForScheduling(const BspInstance<GraphT> &instance,
                              const std::vector<unsigned> &multiplicities,
                              const std::vector<std::vector<VWorkwT<GraphT>>> &requiredProcTypes,
                              const std::vector<unsigned> &maxNumProcs) {
        jobs_.resize(instance.NumberOfVertices());
        if constexpr (verbose_) {
            std::cout << "--- Preparing for Subgraph Scheduling ---" << std::endl;
        }
        const auto &graph = instance.GetComputationalDag();
        const size_t numWorkerTypes = instance.GetArchitecture().getProcessorTypeCount().size();

        CalculateUpwardRanks(graph);

        if constexpr (verbose_) {
            std::cout << "Initializing jobs..." << std::endl;
        }
        job_id_t idx = 0;
        for (auto &job : jobs_) {
            job.id = idx;
            job.in_degree_current = graph.InDegree(idx);
            if (job.in_degree_current == 0) {
                job.status = JobStatus::READY;
                readyJobs_.insert(&job);
            } else {
                job.status = JobStatus::WAITING;
            }
            job.total_work = graph.VertexWorkWeight(idx);
            job.max_num_procs
                = std::min(max_num_procs[idx],
                           static_cast<unsigned>((job.total_work + min_work_per_processor_ - 1) / min_work_per_processor_));
            job.multiplicity = std::min(multiplicities[idx], job.max_num_procs);
            job.required_proc_types = required_proc_types[idx];
            job.assigned_workers.resize(numWorkerTypes, 0);
            job.start_time = -1.0;
            job.finish_time = -1.0;

            if constexpr (verbose_) {
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
                maxSuccessorRank = std::max(max_successor_rank, jobs_.at(child).upward_rank);
            }

            Job &job = jobs_.at(vertex);
            job.upwardRank_ = graph.VertexWorkWeight(vertex) + max_successor_rank;
        }
    }

    SubgraphSchedule ExecuteSchedule(const BspInstance<GraphT> &instance) {
        double currentTime = 0.0;
        std::vector<unsigned> availableWorkers = instance.GetArchitecture().getProcessorTypeCount();
        const size_t numWorkerTypes = availableWorkers.size();
        std::vector<job_id_t> runningJobs;
        unsigned completedCount = 0;
        const auto &graph = instance.GetComputationalDag();

        if constexpr (verbose_) {
            std::cout << "\n--- Subgraph Scheduling Execution Started ---" << std::endl;
            std::cout << "Total jobs: " << jobs_.size() << std::endl;
            std::cout << "Initial available workers: ";
            for (size_t i = 0; i < numWorkerTypes; ++i) {
                std::cout << "T" << i << ":" << availableWorkers[i] << " ";
            }
            std::cout << std::endl;
        }

        while (completedCount < jobs_.size()) {
            if constexpr (verbose_) {
                std::cout << "\n[T=" << currentTime << "] --- New Scheduling Step ---" << std::endl;
                std::cout << "Completed jobs: " << completedCount << "/" << jobs_.size() << std::endl;
                std::cout << "Available workers: ";
                for (size_t i = 0; i < numWorkerTypes; ++i) {
                    std::cout << "T" << i << ":" << availableWorkers[i] << " ";
                }
                std::cout << std::endl;
                std::cout << "Ready queue size: " << readyJobs_.size() << ". Running jobs: " << running_jobs.size() << std::endl;
            }

            std::vector<Job *> jobsToStart;
            VWorkwT<GraphT> totalRunnablePriority = 0.0;

            // Iterate through ready jobs and assign minimum resources if available.
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

            if (!jobsToStart.empty()) {
                if constexpr (verbose_) {
                    std::cout << "Allocating workers to " << jobsToStart.size() << " runnable jobs..." << std::endl;
                }

                // Distribute remaining workers proportionally among the jobs that just started.
                const std::vector<unsigned> remainingWorkersPool = availableWorkers;
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
                                      ? (static_cast<double>(job.upwardRank_) / static_cast<double>(total_runnable_priority))
                                      : (1.0 / static_cast<double>(jobsToStart.size()));
                            const unsigned proportionalShare
                                = static_cast<unsigned>(static_cast<double>(remainingWorkersPool[typeIdx]) * proportion);
                            const unsigned numProportionalChunks = (job.multiplicity_ > 0) ? proportionalShare / job.multiplicity_
                                                                                           : 0;
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

                // Greedily assign any remaining workers to the highest-rank jobs that can take them.
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

                for (Job *jobPtr : jobsToStart) {
                    Job &job = *jobPtr;

                    job.status_ = JobStatus::RUNNING;
                    job.startTime_ = currentTime;

                    // Calculate finish time based on total work and total assigned workers.
                    unsigned totalAssignedWorkers = std::accumulate(job.assignedWorkers_.begin(), job.assignedWorkers_.end(), 0u);
                    double execTime = (totalAssignedWorkers > 0)
                                          ? static_cast<double>(job.totalWork_) / static_cast<double>(totalAssignedWorkers)
                                          : 0.0;
                    job.finishTime_ = currentTime + execTime;

                    runningJobs.push_back(job.id_);
                    readyJobs_.erase(&job);
                }
            }

            // 2. ADVANCE TIME
            if (runningJobs.empty() && completedCount < jobs_.size()) {
                std::cerr << "Error: Deadlock detected. No running jobs and " << jobs_.size() - completedCount
                          << " jobs incomplete." << std::endl;
                if constexpr (verbose_) {
                    std::cout << "Deadlock! Ready queue:" << std::endl;
                    for (const auto *readyJobPtr : readyJobs_) {
                        const Job &job = *readyJobPtr;
                        std::cout << "  - Job " << job.id_ << " (mult " << job.multiplicity_ << ") needs workers: ";
                        for (size_t typeIdx = 0; typeIdx < numWorkerTypes; ++typeIdx) {
                            if (job.requiredProcTypes_[typeIdx] > 0) {
                                std::cout << "T" << typeIdx << ":" << job.multiplicity_ << " ";
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
                result.makespan_ = -1.0;
                return result;
            }
            if (runningJobs.empty()) {
                break;    // All jobs are done
            }

            double nextEventTime = std::numeric_limits<double>::max();
            for (job_id_t id : running_jobs) {
                next_event_time = std::min(next_event_time, jobs_.at(id).finish_time);
            }
            if constexpr (verbose_) {
                std::cout << "Advancing time from " << currentTime << " to " << nextEventTime << std::endl;
            }
            currentTime = nextEventTime;

            // 3. PROCESS COMPLETED JOBS
            auto it = running_jobs.begin();
            while (it != running_jobs.end()) {
                Job &job = jobs_.at(*it);
                if (job.finishTime_ <= currentTime) {
                    job.status_ = JobStatus::COMPLETED;
                    if constexpr (verbose_) {
                        std::cout << "Job " << job.id_ << " finished at T=" << currentTime << std::endl;
                    }
                    // Release workers
                    for (size_t typeIdx = 0; typeIdx < numWorkerTypes; ++typeIdx) {
                        availableWorkers[typeIdx] += job.assignedWorkers_[typeIdx];
                    }
                    completedCount++;

                    // Update successors
                    if constexpr (verbose_) {
                        std::cout << "  - Updating successors..." << std::endl;
                    }
                    for (const auto &successor_id : graph.Children(job.id)) {
                        Job &successor_job = jobs_.at(successor_id);
                        successor_job.in_degree_current--;
                        if (successor_job.in_degree_current == 0) {
                            successor_job.status = JobStatus::READY;
                            ready_jobs_.insert(&successor_job);
                            if constexpr (verbose) {
                                std::cout << "    - Successor " << successor_job.id << " is now READY." << std::endl;
                            }
                        }
                    }
                    it = running_jobs.erase(it);    // Remove from running list
                } else {
                    ++it;
                }
            }
        }

        if constexpr (verbose_) {
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
        result.makespan_ = currentTime;
        result.nodeAssignedWorkerPerType_.resize(jobs_.size());
        for (const auto &job : jobs_) {
            result.nodeAssignedWorkerPerType_[job.id].resize(numWorkerTypes);
            for (size_t i = 0; i < numWorkerTypes; ++i) {
                result.nodeAssignedWorkerPerType_[job.id][i] = (job.assigned_workers[i] + job.multiplicity - 1) / job.multiplicity;
            }
        }
        return result;
    }
};

}    // namespace osp
