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
#include <deque>
#include <list>
#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>

#include "algorithms/Minimal_matching/Hungarian_algorithm.hpp"
#include "coarsen/coarsen_history.hpp"
#include "model/BspInstance.hpp"
#include "model/BspSchedule.hpp"
#include "refine/superstep_clumps.hpp"
#include "structures/dag.hpp"

struct CoarseRefineScheduler_parameters {
    const int number_of_partitions;
    const CoarsenParams coarsen_param;
    const Coarse_Scheduler_Params coarse_schedule_params_initial;
    const Coarse_Scheduler_Params coarse_schedule_params_final;
    const int min_nodes_after_coarsen;
    const int number_of_final_no_change_reps;

    CoarseRefineScheduler_parameters(const Coarse_Scheduler_Params &coarse_schedule_params_,
                                     const CoarsenParams coarsen_param_ = CoarsenParams(),
                                     const int min_nodes_after_coarsen_per_partition_ = 1000,
                                     const int number_of_final_no_change_reps_ = 4)
        : number_of_partitions(coarse_schedule_params_.number_of_partitions), coarsen_param(coarsen_param_),
          coarse_schedule_params_initial(coarse_schedule_params_),
          coarse_schedule_params_final(coarse_schedule_params_),
          min_nodes_after_coarsen(min_nodes_after_coarsen_per_partition_ * number_of_partitions),
          number_of_final_no_change_reps(number_of_final_no_change_reps_) {
        if (coarse_schedule_params_initial.number_of_partitions != coarse_schedule_params_final.number_of_partitions) {
            throw std::logic_error("Number of parititions/processors needs to agree in both Coarse_Schedule_Params");
        }
    };

    CoarseRefineScheduler_parameters(const Coarse_Scheduler_Params &coarse_schedule_params_initial_,
                                     const Coarse_Scheduler_Params &coarse_schedule_params_final_,
                                     const CoarsenParams coarsen_param_ = CoarsenParams(),
                                     const int min_nodes_after_coarsen_per_partition_ = 1000,
                                     const int number_of_final_no_change_reps_ = 4)
        : number_of_partitions(coarse_schedule_params_initial_.number_of_partitions), coarsen_param(coarsen_param_),
          coarse_schedule_params_initial(coarse_schedule_params_initial_),
          coarse_schedule_params_final(coarse_schedule_params_final_),
          min_nodes_after_coarsen(min_nodes_after_coarsen_per_partition_ * number_of_partitions),
          number_of_final_no_change_reps(number_of_final_no_change_reps_) {
        if (coarse_schedule_params_initial.number_of_partitions != coarse_schedule_params_final.number_of_partitions) {
            throw std::logic_error("Number of parititions/processors needs to agree in both Coarse_Schedule_Params");
        }
    };
};

class CoarseRefineScheduler {
  private:
    friend class CoBalDMixR;
    const DAG &original_graph;
    CoarseRefineScheduler_parameters params;

    CoarsenHistory dag_evolution;

    int active_subdag;
    std::vector<std::unique_ptr<SubDAG>> subdag_conversion;

    std::unique_ptr<LooseSchedule> active_loose_schedule;

  public:
    CoarseRefineScheduler(const DAG &graph, const CoarseRefineScheduler_parameters params_)
        : original_graph(graph), params(params_), dag_evolution(graph, params_.coarsen_param),  active_subdag(-1){};

    void run_coarsen();

    void run_schedule_initialise();
    bool run_schedule_refine();
    void run_schedule_evolve();
    void run_schedule_evolution(const bool only_above_thresh = true, const unsigned comm_cost_multiplier = 0,
                                const unsigned com_cost_addition = 0);
    bool run_schedule_superstep_joinings(const bool parity, const bool only_above_thresh = true,
                                         const unsigned comm_cost_multiplier = 0, const unsigned com_cost_addition = 0);

    void run_all(const bool only_above_thresh_initial = true, const bool only_above_thresh_final = false,
                 const unsigned comm_cost_multiplier = 0, const unsigned com_cost_addition = 0);

    /// @brief Gets a possible schedule (Processors at each stage can still be permuted)
    /// @return Superstep vector containing processor vector containing node vector
    std::vector<std::vector<std::vector<int>>> get_loose_schedule() const;

    /// @brief Gets a possible allocation of each node to superstep and processor. (Processors at each stage can still
    /// be permuted)
    /// @return Map: Nodes -> Supersteps x Processor
    std::unordered_map<int, std::pair<unsigned, unsigned>> get_loose_node_schedule_allocation() const;

    /// @brief prints loose schedule
    void print_loose_schedule() const;

    /// @brief Produces a computing schedule
    /// @return Map: Nodes -> Supersteps x Processor
    std::vector<std::pair<unsigned, unsigned>> produce_node_computing_schedule() const;

    /// @brief Produces a computing schedule for non-homogenous processor to processor communication costs
    /// @return Map: Nodes -> Supersteps x Processor
    std::vector<std::pair<unsigned, unsigned>>
    produce_node_computing_schedule(const std::vector<std::vector<unsigned>> &processsor_comm_costs) const;

    /// @brief Produces a computing schedule
    /// @return Superstep[ Processor[ Vector of Nodes ] ]
    std::vector<std::vector<std::vector<unsigned>>> get_computing_schedule() const;

    /// @brief prints computing schedule
    void print_computing_schedule() const;

    /// @brief Produces a computing schedule for non-homogenous processor to processor communication costs
    /// @return Superstep[ Processor[ Vector of Nodes ] ]
    std::vector<std::vector<std::vector<unsigned>>>
    get_computing_schedule(const std::vector<std::vector<unsigned>> &processsor_comm_costs) const;

    /// @brief prints computing schedule for non-homogenous processor to processor communication costs
    void print_computing_schedule(const std::vector<std::vector<unsigned>> &processsor_comm_costs) const;

    BspSchedule produce_bsp_schedule(const BspInstance &bsp_instance) const;
};