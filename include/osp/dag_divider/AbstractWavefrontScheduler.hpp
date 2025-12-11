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
#include <cassert>
#include <iostream>
#include <numeric>

#include "DagDivider.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/computational_dag_util.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"

namespace osp {

/**
 * @class AbstractWavefrontScheduler
 * @brief Base class for schedulers that operate on wavefronts of a DAG.
 */
template <typename Graph_t, typename constr_graph_t>
class AbstractWavefrontScheduler : public Scheduler<Graph_t> {
  protected:
    IDagDivider<Graph_t> *divider;
    Scheduler<constr_graph_t> *scheduler;
    static constexpr bool enable_debug_prints = true;

    /**
     * @brief Distributes processors proportionally, ensuring active components get at least one if possible.
     * @param allocation A reference to the vector that will be filled with the processor allocation.
     * @return True if the scarcity case was hit (fewer processors than active components), false otherwise.
     */
    bool distributeProcessors(unsigned total_processors_of_type,
                              const std::vector<double> &work_weights,
                              std::vector<unsigned> &allocation) const {
        allocation.assign(work_weights.size(), 0);
        double total_work = std::accumulate(work_weights.begin(), work_weights.end(), 0.0);
        if (total_work <= 1e-9 || total_processors_of_type == 0) {
            return false;
        }

        std::vector<size_t> active_indices;
        for (size_t i = 0; i < work_weights.size(); ++i) {
            if (work_weights[i] > 1e-9) {
                active_indices.push_back(i);
            }
        }

        if (active_indices.empty()) {
            return false;
        }

        size_t num_active_components = active_indices.size();
        unsigned remaining_procs = total_processors_of_type;

        // --- Stage 1: Guarantee at least one processor if possible (anti-starvation) ---
        if (total_processors_of_type >= num_active_components) {
            // Abundance case: Give one processor to each active component first.
            for (size_t idx : active_indices) {
                allocation[idx] = 1;
            }
            remaining_procs -= static_cast<unsigned>(num_active_components);
        } else {
            // Scarcity case: Not enough processors for each active component.
            std::vector<std::pair<double, size_t>> sorted_work;
            for (size_t idx : active_indices) {
                sorted_work.push_back({work_weights[idx], idx});
            }
            std::sort(sorted_work.rbegin(), sorted_work.rend());
            for (unsigned i = 0; i < remaining_procs; ++i) {
                allocation[sorted_work[i].second]++;
            }
            return true;    // Scarcity case was hit.
        }

        // --- Stage 2: Proportional Distribution of Remaining Processors ---
        if (remaining_procs > 0) {
            std::vector<double> adjusted_work_weights;
            double adjusted_total_work = 0;

            double work_per_proc = total_work / static_cast<double>(total_processors_of_type);

            for (size_t idx : active_indices) {
                double adjusted_work = std::max(0.0, work_weights[idx] - work_per_proc);
                adjusted_work_weights.push_back(adjusted_work);
                adjusted_total_work += adjusted_work;
            }

            if (adjusted_total_work > 1e-9) {
                std::vector<std::pair<double, size_t>> remainders;
                unsigned allocated_count = 0;

                for (size_t i = 0; i < active_indices.size(); ++i) {
                    double exact_share = (adjusted_work_weights[i] / adjusted_total_work) * remaining_procs;
                    unsigned additional_alloc = static_cast<unsigned>(std::floor(exact_share));
                    allocation[active_indices[i]] += additional_alloc;    // Add to the base allocation of 1
                    remainders.push_back({exact_share - additional_alloc, active_indices[i]});
                    allocated_count += additional_alloc;
                }

                std::sort(remainders.rbegin(), remainders.rend());

                unsigned remainder_processors = remaining_procs - allocated_count;
                for (unsigned i = 0; i < remainder_processors; ++i) {
                    if (i < remainders.size()) {
                        allocation[remainders[i].second]++;
                    }
                }
            }
        }
        return false;    // Scarcity case was not hit.
    }

    BspArchitecture<constr_graph_t> createSubArchitecture(const BspArchitecture<Graph_t> &original_arch,
                                                          const std::vector<unsigned> &sub_dag_proc_types) const {
        // The calculation is now inside the assert, so it only happens in debug builds.
        assert(std::accumulate(sub_dag_proc_types.begin(), sub_dag_proc_types.end(), 0u) > 0
               && "Attempted to create a sub-architecture with zero processors.");

        BspArchitecture<constr_graph_t> sub_architecture(original_arch);
        std::vector<v_memw_t<Graph_t>> sub_dag_processor_memory(original_arch.getProcessorTypeCount().size(),
                                                                std::numeric_limits<v_memw_t<Graph_t>>::max());
        for (unsigned i = 0; i < original_arch.numberOfProcessors(); ++i) {
            sub_dag_processor_memory[original_arch.processorType(i)]
                = std::min(original_arch.memoryBound(i), sub_dag_processor_memory[original_arch.processorType(i)]);
        }
        sub_architecture.SetProcessorsConsequTypes(sub_dag_proc_types, sub_dag_processor_memory);
        return sub_architecture;
    }

    bool validateWorkDistribution(const std::vector<constr_graph_t> &sub_dags, const BspInstance<Graph_t> &instance) const {
        const auto &original_arch = instance.getArchitecture();
        for (const auto &rep_sub_dag : sub_dags) {
            const double total_rep_work = sumOfVerticesWorkWeights(rep_sub_dag);

            double sum_of_compatible_works_for_rep = 0.0;
            for (unsigned type_idx = 0; type_idx < original_arch.getNumberOfProcessorTypes(); ++type_idx) {
                sum_of_compatible_works_for_rep += sumOfCompatibleWorkWeights(rep_sub_dag, instance, type_idx);
            }

            if (sum_of_compatible_works_for_rep > total_rep_work + 1e-9) {
                if constexpr (enable_debug_prints) {
                    std::cerr << "ERROR: Sum of compatible work (" << sum_of_compatible_works_for_rep << ") exceeds total work ("
                              << total_rep_work << ") for a sub-dag. Aborting." << std::endl;
                }
                return false;
            }
        }
        return true;
    }

  public:
    AbstractWavefrontScheduler(IDagDivider<Graph_t> &div, Scheduler<constr_graph_t> &sched) : divider(&div), scheduler(&sched) {}
};

}    // namespace osp
