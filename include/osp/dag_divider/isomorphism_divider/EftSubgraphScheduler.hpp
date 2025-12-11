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
    std::vector<std::vector<unsigned>> node_assigned_worker_per_type;
    std::vector<bool> was_trimmed;
};

template <typename Graph_t>
class EftSubgraphScheduler {
  public:
    EftSubgraphScheduler() = default;

    SubgraphSchedule run(const BspInstance<Graph_t> &instance,
                         const std::vector<unsigned> &multiplicities,
                         const std::vector<std::vector<v_workw_t<Graph_t>>> &required_proc_types,
                         const std::vector<unsigned> &max_num_procs) {
        prepare_for_scheduling(instance, multiplicities, required_proc_types, max_num_procs);
        return execute_schedule(instance);
    }

    void setMinWorkPerProcessor(const v_workw_t<Graph_t> min_work_per_processor) {
        min_work_per_processor_ = min_work_per_processor;
    }

  private:
    static constexpr bool verbose = false;

    using job_id_t = vertex_idx_t<Graph_t>;

    v_workw_t<Graph_t> min_work_per_processor_ = 2000;

    enum class JobStatus { WAITING, READY, RUNNING, COMPLETED };

    struct Job {
        job_id_t id;

        std::vector<v_workw_t<Graph_t>> required_proc_types;
        v_workw_t<Graph_t> total_work;
        unsigned multiplicity = 1;
        unsigned max_num_procs = 1;

        job_id_t in_degree_current = 0;

        JobStatus status = JobStatus::WAITING;
        v_workw_t<Graph_t> upward_rank = 0.0;

        // --- Execution Tracking Members ---
        std::vector<unsigned> assigned_workers;
        double start_time = -1.0;
        double finish_time = -1.0;
    };

    // Custom comparator for storing Job pointers in the ready set, sorted by rank.
    struct JobPtrCompare {
        bool operator()(const Job *lhs, const Job *rhs) const {
            if (lhs->upward_rank != rhs->upward_rank) {
                return lhs->upward_rank > rhs->upward_rank;
            }
            return lhs->id > rhs->id;    // Tie-breaking
        }
    };

    std::vector<Job> jobs_;
    std::set<const Job *, JobPtrCompare> ready_jobs_;

    void prepare_for_scheduling(const BspInstance<Graph_t> &instance,
                                const std::vector<unsigned> &multiplicities,
                                const std::vector<std::vector<v_workw_t<Graph_t>>> &required_proc_types,
                                const std::vector<unsigned> &max_num_procs) {
        jobs_.resize(instance.numberOfVertices());
        if constexpr (verbose) {
            std::cout << "--- Preparing for Subgraph Scheduling ---" << std::endl;
        }
        const auto &graph = instance.getComputationalDag();
        const size_t num_worker_types = instance.getArchitecture().getProcessorTypeCount().size();

        calculate_upward_ranks(graph);

        if constexpr (verbose) {
            std::cout << "Initializing jobs..." << std::endl;
        }
        job_id_t idx = 0;
        for (auto &job : jobs_) {
            job.id = idx;
            job.in_degree_current = graph.in_degree(idx);
            if (job.in_degree_current == 0) {
                job.status = JobStatus::READY;
                ready_jobs_.insert(&job);
            } else {
                job.status = JobStatus::WAITING;
            }
            job.total_work = graph.vertex_work_weight(idx);
            job.max_num_procs
                = std::min(max_num_procs[idx],
                           static_cast<unsigned>((job.total_work + min_work_per_processor_ - 1) / min_work_per_processor_));
            job.multiplicity = std::min(multiplicities[idx], job.max_num_procs);
            job.required_proc_types = required_proc_types[idx];
            job.assigned_workers.resize(num_worker_types, 0);
            job.start_time = -1.0;
            job.finish_time = -1.0;

            if constexpr (verbose) {
                std::cout << "  - Job " << idx << ": rank=" << job.upward_rank << ", mult=" << job.multiplicity
                          << ", max_procs=" << job.max_num_procs << ", work=" << job.total_work
                          << ", status=" << (job.status == JobStatus::READY ? "READY" : "WAITING") << std::endl;
            }
            idx++;
        }
    }

    void calculate_upward_ranks(const Graph_t &graph) {
        const auto reverse_top_order = GetTopOrderReverse(graph);

        for (const auto &vertex : reverse_top_order) {
            v_workw_t<Graph_t> max_successor_rank = 0.0;
            for (const auto &child : graph.children(vertex)) {
                max_successor_rank = std::max(max_successor_rank, jobs_.at(child).upward_rank);
            }

            Job &job = jobs_.at(vertex);
            job.upward_rank = graph.vertex_work_weight(vertex) + max_successor_rank;
        }
    }

    SubgraphSchedule execute_schedule(const BspInstance<Graph_t> &instance) {
        double current_time = 0.0;
        std::vector<unsigned> available_workers = instance.getArchitecture().getProcessorTypeCount();
        const size_t num_worker_types = available_workers.size();
        std::vector<job_id_t> running_jobs;
        unsigned completed_count = 0;
        const auto &graph = instance.getComputationalDag();

        if constexpr (verbose) {
            std::cout << "\n--- Subgraph Scheduling Execution Started ---" << std::endl;
            std::cout << "Total jobs: " << jobs_.size() << std::endl;
            std::cout << "Initial available workers: ";
            for (size_t i = 0; i < num_worker_types; ++i) {
                std::cout << "T" << i << ":" << available_workers[i] << " ";
            }
            std::cout << std::endl;
        }

        while (completed_count < jobs_.size()) {
            if constexpr (verbose) {
                std::cout << "\n[T=" << current_time << "] --- New Scheduling Step ---" << std::endl;
                std::cout << "Completed jobs: " << completed_count << "/" << jobs_.size() << std::endl;
                std::cout << "Available workers: ";
                for (size_t i = 0; i < num_worker_types; ++i) {
                    std::cout << "T" << i << ":" << available_workers[i] << " ";
                }
                std::cout << std::endl;
                std::cout << "Ready queue size: " << ready_jobs_.size() << ". Running jobs: " << running_jobs.size() << std::endl;
            }

            std::vector<Job *> jobs_to_start;
            v_workw_t<Graph_t> total_runnable_priority = 0.0;

            // Iterate through ready jobs and assign minimum resources if available.
            for (const Job *job_ptr : ready_jobs_) {
                Job &job = jobs_[job_ptr->id];
                bool can_start = true;
                for (size_t type_idx = 0; type_idx < num_worker_types; ++type_idx) {
                    if (job.required_proc_types[type_idx] > 0 && available_workers[type_idx] < job.multiplicity) {
                        can_start = false;
                        break;
                    }
                }

                if (can_start) {
                    jobs_to_start.push_back(&job);
                    total_runnable_priority += job.upward_rank;
                    for (size_t type_idx = 0; type_idx < num_worker_types; ++type_idx) {
                        if (job.required_proc_types[type_idx] > 0) {
                            job.assigned_workers[type_idx] = job.multiplicity;
                            available_workers[type_idx] -= job.multiplicity;
                        }
                    }
                }
            }

            if (!jobs_to_start.empty()) {
                if constexpr (verbose) {
                    std::cout << "Allocating workers to " << jobs_to_start.size() << " runnable jobs..." << std::endl;
                }

                // Distribute remaining workers proportionally among the jobs that just started.
                const std::vector<unsigned> remaining_workers_pool = available_workers;
                for (Job *job_ptr : jobs_to_start) {
                    Job &job = *job_ptr;
                    for (size_t type_idx = 0; type_idx < num_worker_types; ++type_idx) {
                        if (job.required_proc_types[type_idx] > 0) {
                            const unsigned current_total_assigned
                                = std::accumulate(job.assigned_workers.begin(), job.assigned_workers.end(), 0u);
                            const unsigned max_additional_workers
                                = (job.max_num_procs > current_total_assigned) ? (job.max_num_procs - current_total_assigned) : 0;

                            const double proportion
                                = (total_runnable_priority > 0)
                                      ? (static_cast<double>(job.upward_rank) / static_cast<double>(total_runnable_priority))
                                      : (1.0 / static_cast<double>(jobs_to_start.size()));
                            const unsigned proportional_share
                                = static_cast<unsigned>(static_cast<double>(remaining_workers_pool[type_idx]) * proportion);
                            const unsigned num_proportional_chunks
                                = (job.multiplicity > 0) ? proportional_share / job.multiplicity : 0;
                            const unsigned num_available_chunks
                                = (job.multiplicity > 0) ? available_workers[type_idx] / job.multiplicity : 0;
                            const unsigned num_chunks_allowed_by_max
                                = (job.multiplicity > 0) ? max_additional_workers / job.multiplicity : 0;
                            const unsigned num_chunks_to_assign
                                = std::min({num_proportional_chunks, num_available_chunks, num_chunks_allowed_by_max});
                            const unsigned assigned = num_chunks_to_assign * job.multiplicity;
                            job.assigned_workers[type_idx] += assigned;
                            available_workers[type_idx] -= assigned;
                        }
                    }
                }

                // Greedily assign any remaining workers to the highest-rank jobs that can take them.
                for (Job *job_ptr : jobs_to_start) {
                    Job &job = *job_ptr;
                    for (size_t type_idx = 0; type_idx < num_worker_types; ++type_idx) {
                        if (job.required_proc_types[type_idx] > 0 && job.multiplicity > 0) {
                            const unsigned current_total_assigned
                                = std::accumulate(job.assigned_workers.begin(), job.assigned_workers.end(), 0u);
                            const unsigned max_additional_workers
                                = (job.max_num_procs > current_total_assigned) ? (job.max_num_procs - current_total_assigned) : 0;
                            const unsigned num_available_chunks = available_workers[type_idx] / job.multiplicity;
                            const unsigned num_chunks_allowed_by_max = max_additional_workers / job.multiplicity;
                            const unsigned assigned = std::min(num_available_chunks, num_chunks_allowed_by_max) * job.multiplicity;
                            job.assigned_workers[type_idx] += assigned;
                            available_workers[type_idx] -= assigned;
                        }
                    }
                }

                for (Job *job_ptr : jobs_to_start) {
                    Job &job = *job_ptr;

                    job.status = JobStatus::RUNNING;
                    job.start_time = current_time;

                    // Calculate finish time based on total work and total assigned workers.
                    unsigned total_assigned_workers = std::accumulate(job.assigned_workers.begin(), job.assigned_workers.end(), 0u);
                    double exec_time = (total_assigned_workers > 0)
                                           ? static_cast<double>(job.total_work) / static_cast<double>(total_assigned_workers)
                                           : 0.0;
                    job.finish_time = current_time + exec_time;

                    running_jobs.push_back(job.id);
                    ready_jobs_.erase(&job);
                }
            }

            // 2. ADVANCE TIME
            if (running_jobs.empty() && completed_count < jobs_.size()) {
                std::cerr << "Error: Deadlock detected. No running jobs and " << jobs_.size() - completed_count
                          << " jobs incomplete." << std::endl;
                if constexpr (verbose) {
                    std::cout << "Deadlock! Ready queue:" << std::endl;
                    for (const auto *ready_job_ptr : ready_jobs_) {
                        const Job &job = *ready_job_ptr;
                        std::cout << "  - Job " << job.id << " (mult " << job.multiplicity << ") needs workers: ";
                        for (size_t type_idx = 0; type_idx < num_worker_types; ++type_idx) {
                            if (job.required_proc_types[type_idx] > 0) {
                                std::cout << "T" << type_idx << ":" << job.multiplicity << " ";
                            }
                        }
                        std::cout << std::endl;
                    }
                    std::cout << "Available workers: ";
                    for (size_t i = 0; i < num_worker_types; ++i) {
                        std::cout << "T" << i << ":" << available_workers[i] << " ";
                    }
                    std::cout << std::endl;
                }
                SubgraphSchedule result;
                result.makespan = -1.0;
                return result;
            }
            if (running_jobs.empty()) {
                break;    // All jobs are done
            }

            double next_event_time = std::numeric_limits<double>::max();
            for (job_id_t id : running_jobs) {
                next_event_time = std::min(next_event_time, jobs_.at(id).finish_time);
            }
            if constexpr (verbose) {
                std::cout << "Advancing time from " << current_time << " to " << next_event_time << std::endl;
            }
            current_time = next_event_time;

            // 3. PROCESS COMPLETED JOBS
            auto it = running_jobs.begin();
            while (it != running_jobs.end()) {
                Job &job = jobs_.at(*it);
                if (job.finish_time <= current_time) {
                    job.status = JobStatus::COMPLETED;
                    if constexpr (verbose) {
                        std::cout << "Job " << job.id << " finished at T=" << current_time << std::endl;
                    }
                    // Release workers
                    for (size_t type_idx = 0; type_idx < num_worker_types; ++type_idx) {
                        available_workers[type_idx] += job.assigned_workers[type_idx];
                    }
                    completed_count++;

                    // Update successors
                    if constexpr (verbose) {
                        std::cout << "  - Updating successors..." << std::endl;
                    }
                    for (const auto &successor_id : graph.children(job.id)) {
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

        if constexpr (verbose) {
            std::cout << "\n--- Subgraph Scheduling Finished ---" << std::endl;
            std::cout << "Final Makespan: " << current_time << std::endl;
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
        result.makespan = current_time;
        result.node_assigned_worker_per_type.resize(jobs_.size());
        for (const auto &job : jobs_) {
            result.node_assigned_worker_per_type[job.id].resize(num_worker_types);
            for (size_t i = 0; i < num_worker_types; ++i) {
                result.node_assigned_worker_per_type[job.id][i]
                    = (job.assigned_workers[i] + job.multiplicity - 1) / job.multiplicity;
            }
        }
        return result;
    }
};

}    // namespace osp
