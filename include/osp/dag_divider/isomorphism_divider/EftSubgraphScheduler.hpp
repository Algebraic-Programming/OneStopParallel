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


#include "osp/graph_algorithms/directed_graph_util.hpp"
#include "osp/bsp/model/BspInstance.hpp"
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace osp {

struct SubgraphSchedule {
    double makespan;
    std::vector<std::vector<unsigned>> node_assigned_worker_per_type;   
};

template<typename Graph_t>
class EftSubgraphScheduler {
public:
    
    EftSubgraphScheduler() = default;

    SubgraphSchedule run(const BspInstance<Graph_t>& instance, const std::vector<unsigned>& multiplicities, const std::vector<std::vector<v_workw_t<Graph_t>>> & required_proc_types) {
        prepare_for_scheduling(instance, multiplicities, required_proc_types);
        return execute_schedule(instance);
    }

private:

    static constexpr bool verbose = false;

    using job_id_t = vertex_idx_t<Graph_t>;

    enum class JobStatus {
        WAITING,
        READY,
        RUNNING,
        COMPLETED
    };


    struct Job {
        job_id_t id;

        std::vector<v_workw_t<Graph_t>> required_proc_types;
        v_workw_t<Graph_t> total_work;
        unsigned multiplicity = 1;

        job_id_t in_degree_current = 0;
        
        JobStatus status = JobStatus::WAITING;
        double upward_rank = 0.0;

        // --- Execution Tracking Members ---
        std::vector<unsigned> assigned_workers;
        double start_time = -1.0;
        double finish_time = -1.0;

        bool operator<(Job const &rhs) const {
            return (upward_rank > rhs.upward_rank) || (upward_rank == rhs.upward_rank and id > rhs.id);
        }

    };

    std::vector<Job> jobs_;
    std::set<Job> ready_jobs_;

    void prepare_for_scheduling(const BspInstance<Graph_t>& instance, const std::vector<unsigned>& multiplicities, const std::vector<std::vector<v_workw_t<Graph_t>>> & required_proc_types) {        
        jobs_.resize(instance.numberOfVertices());
        if constexpr (verbose) {
            std::cout << "--- Preparing for Subgraph Scheduling ---" << std::endl;
        }
        const auto & graph = instance.getComputationalDag();
        const size_t num_worker_types = instance.getArchitecture().getProcessorTypeCount().size();

        calculate_upward_ranks(graph);

        if constexpr (verbose) std::cout << "Initializing jobs..." << std::endl;
        job_id_t idx = 0;
        for (auto& job : jobs_) {
            job.id = idx;
            job.in_degree_current = graph.in_degree(idx);
            if (job.in_degree_current == 0) {
                job.status = JobStatus::READY;
                ready_jobs_.insert(job);
            } else {
                job.status = JobStatus::WAITING;
            }
            job.multiplicity = multiplicities[idx];
            job.required_proc_types = required_proc_types[idx];           
            job.assigned_workers.resize(num_worker_types, 0);
            job.total_work = graph.vertex_work_weight(idx);
            job.start_time = -1.0;
            job.finish_time = -1.0;

            if constexpr (verbose) {
                std::cout << "  - Job " << idx << ": rank=" << job.upward_rank << ", mult=" << job.multiplicity 
                          << ", work=" << job.total_work << ", status=" << (job.status == JobStatus::READY ? "READY" : "WAITING") << std::endl;
            }
            idx++;
        }        
    }

    void calculate_upward_ranks(const Graph_t & graph) {
        for (const auto & vertex : graph.vertices()) {
            if (is_sink(vertex, graph)) {
                calculate_rank_recursive(vertex, graph);
            }
        }  
    }
    
    double calculate_rank_recursive(vertex_idx_t<Graph_t> vertex, const Graph_t & graph) {
        Job& job = jobs_.at(vertex);
        if (job.upward_rank > 0.0) {
            return job.upward_rank; // Memoization
        }

        double max_successor_rank = 0.0;
        for (const auto& child : graph.children(vertex)) {
            max_successor_rank = std::max(max_successor_rank, calculate_rank_recursive(child, graph));
        }

        
        job.upward_rank = static_cast<double>(graph.vertex_work_weight(vertex)) + max_successor_rank;
        return job.upward_rank;
    }

    SubgraphSchedule execute_schedule(const BspInstance<Graph_t>& instance) {
        double current_time = 0.0; 
        std::vector<unsigned> available_workers = instance.getArchitecture().getProcessorTypeCount();
        const size_t num_worker_types = available_workers.size();
        std::vector<job_id_t> running_jobs;
        unsigned completed_count = 0;
        const auto& graph = instance.getComputationalDag();

        if constexpr (verbose) {
            std::cout << "\n--- Subgraph Scheduling Execution Started ---" << std::endl;
            std::cout << "Total jobs: " << jobs_.size() << std::endl;
            std::cout << "Initial available workers: ";
            for(size_t i=0; i<num_worker_types; ++i) std::cout << "T" << i << ":" << available_workers[i] << " ";
            std::cout << std::endl;
        }

        while (completed_count < jobs_.size()) {       

            if constexpr (verbose) {
                std::cout << "\n[T=" << current_time << "] --- New Scheduling Step ---" << std::endl;
                std::cout << "Completed jobs: " << completed_count << "/" << jobs_.size() << std::endl;
                std::cout << "Available workers: ";
                for(size_t i=0; i<num_worker_types; ++i) std::cout << "T" << i << ":" << available_workers[i] << " ";
                std::cout << std::endl;
                std::cout << "Ready queue size: " << ready_jobs_.size() << ". Running jobs: " << running_jobs.size() << std::endl;
            }

            // 1a. Find all ready jobs that could potentially start.
            std::vector<job_id_t> candidate_ids;
            if constexpr (verbose) std::cout << "Finding candidate jobs from ready set:" << std::endl;
            for (const auto& ready_job_template : ready_jobs_) {
                const Job& job = jobs_[ready_job_template.id];
                bool can_potentially_start = true;
                for (size_t type_idx = 0; type_idx < num_worker_types; ++type_idx) {
                    if (job.required_proc_types[type_idx] > 0 && available_workers[type_idx] < job.multiplicity) {
                        can_potentially_start = false;
                        break;
                    }
                }
                if (can_potentially_start) {
                    candidate_ids.push_back(job.id);
                }
            }

            // 1b. From the candidates, find which can actually run in this step and calculate their total priority.
            // This is a greedy approach: we check candidates in priority order and provisionally "assign"
            // one chunk of resources to see who else fits.
            std::vector<job_id_t> runnable_ids;
            double total_runnable_priority = 0.0;
            std::vector<unsigned> temp_available_workers = available_workers;
            for (const job_id_t job_id : candidate_ids) {
                const Job& job = jobs_[job_id];
                bool can_run_now = true;
                for (size_t type_idx = 0; type_idx < num_worker_types; ++type_idx) {
                    if (job.required_proc_types[type_idx] > 0 && temp_available_workers[type_idx] < job.multiplicity) {
                        can_run_now = false;
                        break;
                    }
                }
                if (can_run_now) {
                    runnable_ids.push_back(job_id);
                    total_runnable_priority += job.upward_rank;
                    for (size_t type_idx = 0; type_idx < num_worker_types; ++type_idx) {
                        if (job.required_proc_types[type_idx] > 0) {
                            temp_available_workers[type_idx] -= job.multiplicity;
                        }
                    }
                }
            }

            std::vector<Job> newly_started_jobs;
            if (!runnable_ids.empty()) {
                if constexpr (verbose) {
                    std::cout << "Allocating workers to " << runnable_ids.size() << " runnable jobs..." << std::endl;
                }
                const std::vector<unsigned> initial_available_workers = available_workers;

                for (const job_id_t job_id : runnable_ids) {
                    Job& job = jobs_[job_id];
                    
                    // 1c. Double-check if it can still start, as higher-priority jobs might have taken workers.
                    bool can_start = true;
                    for (size_t type_idx = 0; type_idx < num_worker_types; ++type_idx) {
                        if (job.required_proc_types[type_idx] > 0 && available_workers[type_idx] < job.multiplicity) {
                            can_start = false;
                            break;
                        }
                    }

                    if (can_start) {
                        if constexpr (verbose) std::cout << "  - Starting Job " << job.id << ":" << std::endl;
                        // 1d. This job will start. Allocate workers to it, respecting the proportional cap.
                        for (size_t type_idx = 0; type_idx < num_worker_types; ++type_idx) {
                            if (job.required_proc_types[type_idx] > 0) {
                                // Greedily determine how many chunks are available right now.
                                unsigned num_available_chunks = available_workers[type_idx] / job.multiplicity;

                                // Calculate the max number of chunks this job is allowed based on its priority-weighted share
                                // of the *initial* pool of workers for this time step.
                                double proportion = (total_runnable_priority > 0) ? (job.upward_rank / total_runnable_priority) : (1.0 / static_cast<double>(runnable_ids.size()));
                                unsigned cap = static_cast<unsigned>(initial_available_workers[type_idx] * proportion);
                                unsigned num_capped_chunks = cap / job.multiplicity;

                                // Assign the minimum of what's available and what the cap allows.
                                unsigned num_chunks_to_assign = std::min(num_available_chunks, num_capped_chunks);
                                
                                // We must assign at least one chunk for the job to start. The can_start check ensures this is safe.
                                num_chunks_to_assign = std::max(1u, num_chunks_to_assign);

                                unsigned assigned = num_chunks_to_assign * job.multiplicity;
                                job.assigned_workers[type_idx] = assigned;
                                available_workers[type_idx] -= assigned;
                                if constexpr (verbose) {
                                    std::cout << "    - Type " << type_idx << ": assigned " << assigned << " workers (" << num_chunks_to_assign << " chunks). "
                                              << "Available now: " << available_workers[type_idx] << std::endl;
                                }
                            }
                        }

                        // 1e. Finalize starting the job.
                        job.status = JobStatus::RUNNING;
                        job.start_time = current_time;

                        // Calculate finish time based on the bottleneck worker type.
                        double max_exec_time = 0.0;
                        for (size_t type_idx = 0; type_idx < num_worker_types; ++type_idx) {
                            if (job.required_proc_types[type_idx] > 0) {
                                double exec_time = static_cast<double>(job.required_proc_types[type_idx]) / job.assigned_workers[type_idx];
                                max_exec_time = std::max(max_exec_time, exec_time);
                            }
                        }
                        job.finish_time = current_time + max_exec_time;
                        
                        if constexpr (verbose) {
                            std::cout << "    - Job " << job.id << " started at " << job.start_time << ", will finish at " << job.finish_time << std::endl;
                        }
                        
                        running_jobs.push_back(job.id);
                        newly_started_jobs.push_back(job);
                    }
                }
            }

            // Remove newly started jobs from the ready set.
            for(const auto& started_job : newly_started_jobs) {
                ready_jobs_.erase(started_job);
            }

            // 2. ADVANCE TIME
            if (running_jobs.empty()) {
                if (completed_count < jobs_.size()) {
                     std::cerr << "Error: Deadlock detected. No running jobs and " 
                               << jobs_.size() - completed_count << " jobs incomplete." << std::endl;
                    if constexpr (verbose) {
                        std::cout << "Deadlock! Ready queue:" << std::endl;
                        for (const auto& ready_job_template : ready_jobs_) {
                            const Job& job = jobs_[ready_job_template.id];
                            std::cout << "  - Job " << job.id << " (mult " << job.multiplicity << ") needs workers: ";
                            for(size_t type_idx = 0; type_idx < num_worker_types; ++type_idx) {
                                if (job.required_proc_types[type_idx] > 0) {
                                    std::cout << "T" << type_idx << ":" << job.multiplicity << " ";
                                }
                            }
                            std::cout << std::endl;
                        }
                        std::cout << "Available workers: ";
                        for(size_t i=0; i<num_worker_types; ++i) std::cout << "T" << i << ":" << available_workers[i] << " ";
                        std::cout << std::endl;
                    }
                     SubgraphSchedule result;
                     result.makespan = -1.0;
                     return result;
                }
                break; // All jobs are done
            }
            
            double next_event_time = std::numeric_limits<double>::max();
            for (job_id_t id : running_jobs) {
                next_event_time = std::min(next_event_time, jobs_.at(id).finish_time);
            }
            if constexpr (verbose) std::cout << "Advancing time from " << current_time << " to " << next_event_time << std::endl;
            current_time = next_event_time;

            // 3. PROCESS COMPLETED JOBS
            auto it = running_jobs.begin();
            while (it != running_jobs.end()) {
                Job& job = jobs_.at(*it);
                if (job.finish_time <= current_time) {
                    job.status = JobStatus::COMPLETED;
                    if constexpr (verbose) std::cout << "Job " << job.id << " finished at T=" << current_time << std::endl;
                    // Release workers
                    for(size_t type_idx = 0; type_idx < num_worker_types; ++type_idx) {
                        available_workers[type_idx] += job.assigned_workers[type_idx];
                    }
                    completed_count++;

                    // Update successors
                    if constexpr (verbose) std::cout << "  - Updating successors..." << std::endl;
                    for (const auto& successor_id : graph.children(job.id)) {
                        Job& successor_job = jobs_.at(successor_id);
                        successor_job.in_degree_current--;
                        if (successor_job.in_degree_current == 0) {
                            successor_job.status = JobStatus::READY;
                            ready_jobs_.insert(successor_job);
                            if constexpr (verbose) std::cout << "    - Successor " << successor_job.id << " is now READY." << std::endl;
                        }
                    }
                    it = running_jobs.erase(it); // Remove from running list
                } else {
                    ++it;
                }
            }
        }

        if constexpr (verbose) {
            std::cout << "\n--- Subgraph Scheduling Finished ---" << std::endl;
            std::cout << "Final Makespan: " << current_time << std::endl;
            std::cout << "Job Summary:" << std::endl;
            for(const auto& job : jobs_) {
                std::cout << "  - Job " << job.id << ": Multiplicity=" << job.multiplicity <<  ", Start=" << job.start_time << ", Finish=" << job.finish_time << ", Workers=[";
                for(size_t i=0; i<job.assigned_workers.size(); ++i) {
                    std::cout << "T" << i << ":" << job.assigned_workers[i] << (i == job.assigned_workers.size()-1 ? "" : ", ");
                }
                std::cout << "]" << std::endl;
            }
        }

        SubgraphSchedule result;
        result.makespan = current_time;
        result.node_assigned_worker_per_type.resize(jobs_.size());
        for(const auto& job : jobs_) {
            result.node_assigned_worker_per_type[job.id].resize(num_worker_types);
            for (size_t i = 0; i < num_worker_types; ++i) {
                result.node_assigned_worker_per_type[job.id][i] = job.assigned_workers[i] > 0 ? job.assigned_workers[i] / job.multiplicity : 0;
            }
        }

        return result;
    }
};

} // namespace osp