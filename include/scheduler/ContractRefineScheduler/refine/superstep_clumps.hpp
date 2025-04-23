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
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "scheduler/Partitioners/partitioners.hpp"
#include "auxiliary/Balanced_Coin_Flips.hpp"
#include "scheduler/ContractRefineScheduler/refine/new_superstep.hpp"
#include "structures/dag.hpp"
#include "scheduler/Minimal_matching/Hungarian_algorithm.hpp"

/**
 * @brief Contains all algorithm parameters
 * @param number_of_partitions The number of partitions or processors
 * @param balance_threshhold Upper bound of max_weight/avg_weight of a superstep as to when to consider it balanced
 * @param part_algo Which basic partitioner to use to try an balance a superstep
 * @param nodes_per_clump Number of nodes per clump required to initiate a superstep split (either this or next)
 * -- approx 2 / max_weight_for_flag 
 * @param nodes_per_partition Number of nodes per partition required to initiate a superstep split (either this or
 * previous) -- approx 2 * (balance_thresh-1)^{-1}
 * @param clumps_per_partition Number of Clumps per partition in a superstep for it to NOT be considered starving for
 * clumps. If starving, then the superstep is flagged for shaving cut
 * @param max_weight_for_flag Multiplicative factor x, such that x.avg_weight is an upper bound on the heaviest clump.
 * If violated, then the superstep is flagged for balanced cut
 * @param balanced_cut_ratio Relative upper limit on clumps being split in a balanced cut
 * @param min_weight_for_split Multiplicative factor x, such that x.avg_weight is a minimum requirement for splitting a
 * clump in balanced cut
 * @param hill_climb_simple_improvement_attemps Number of times hill climbing is
 * @param min_comp_generation_when_shaving Attempts to generate said number of clumps when shaving
 * @param coin_type Type of balanced coin to make balanced pseudo random decisions
 *
 */
struct Coarse_Scheduler_Params {
    const unsigned number_of_partitions;
    const float balance_threshhold;

    const PartitionAlgorithm part_algo;
    const float nodes_per_clump;
    const float nodes_per_partition;
    const float clumps_per_partition;
    const float max_weight_for_flag;
    const float balanced_cut_ratio;
    const float min_weight_for_split;
    const unsigned hill_climb_simple_improvement_attemps;
    const int min_comp_generation_when_shaving;

    const CoinType coin_type;

    Coarse_Scheduler_Params(const unsigned number_of_partitions_, const float balance_threshhold_ = 1.2,
                            const PartitionAlgorithm part_algo_ = Greedy, const CoinType coin_type_ = Thue_Morse,
                            const float clumps_per_partition_ = 6, const float nodes_per_clump_ = 4,
                            const float nodes_per_partition_ = 15, const float max_weight_for_flag_ = 1 / 3,
                            const float balanced_cut_ratio_ = 1 / 3, const float min_weight_for_split_ = 1 / 48,
                            const unsigned hill_climb_simple_improvement_attemps_ = 10,
                            const int min_comp_generation_when_shaving_ = 3)
        : number_of_partitions(number_of_partitions_), balance_threshhold(balance_threshhold_), part_algo(part_algo_),
          nodes_per_clump(nodes_per_clump_), nodes_per_partition(nodes_per_partition_),
          clumps_per_partition(clumps_per_partition_), max_weight_for_flag(max_weight_for_flag_),
          balanced_cut_ratio(balanced_cut_ratio_), min_weight_for_split(min_weight_for_split_),
          hill_climb_simple_improvement_attemps(hill_climb_simple_improvement_attemps_),
          min_comp_generation_when_shaving(min_comp_generation_when_shaving_), coin_type(coin_type_){};

    Coarse_Scheduler_Params( const Coarse_Scheduler_Params& other )
        : number_of_partitions(other.number_of_partitions), balance_threshhold(other.balance_threshhold), part_algo(other.part_algo),
          nodes_per_clump(other.nodes_per_clump), nodes_per_partition(other.nodes_per_partition),
          clumps_per_partition(other.clumps_per_partition),  max_weight_for_flag(other.max_weight_for_flag),
          balanced_cut_ratio(other.balanced_cut_ratio), min_weight_for_split(other.min_weight_for_split),
          hill_climb_simple_improvement_attemps(other.hill_climb_simple_improvement_attemps),
          min_comp_generation_when_shaving(other.min_comp_generation_when_shaving),
          coin_type(other.coin_type) {};

    /**
     * @brief Returns a linear combination of the two Coarse_Scheduler_Params with given ratio 
     */
    static Coarse_Scheduler_Params lin_comb( const Coarse_Scheduler_Params& first, const Coarse_Scheduler_Params& second, const std::pair<unsigned, unsigned>& ratio );
};

struct Clump {
    std::unordered_set<int> node_set;
    int total_weight;

    Clump();
    Clump(const DAG &graph);
    Clump(const DAG &graph, const std::unordered_set<int> &node_set_);
    Clump(const SubDAG &graph);
    Clump(const SubDAG &graph, const std::unordered_set<int> &node_set_);

    struct Comparator {
        constexpr bool operator()(const Clump &a, const Clump &b) const { return (a.total_weight > b.total_weight); };
    };

    ~Clump() = default;
};


template<typename T>
class LooseSuperStep {
  private:
    const Coarse_Scheduler_Params params;

    float imbalance = std::numeric_limits<float>::infinity(); // max/avg
    std::vector<unsigned> allocation;

    // Flags
    bool too_few_nodes_in_clumps;
    bool too_few_clumps_superstep;
    bool fat_clump_sizes;

  public:
    const unsigned id;
    const std::multiset<T, typename T::Comparator> collection;

    std::vector<std::vector<int>> get_current_processors_with_nodes() const {
        std::vector<std::vector<int>> output;
        output.resize(params.number_of_partitions);

        unsigned i = 0;
        for (auto &clump : collection) {
            output[allocation[i]].insert(output[allocation[i]].end(), clump.node_set.begin(), clump.node_set.end());

            i++;
        }

        return output;
    };

    inline std::vector<unsigned> get_current_allocation() const { return allocation; };

    void permute_allocation( std::vector<unsigned> permutation) {
        // checking whether permutation
        assert( permutation.size() == params.number_of_partitions );
        std::vector<bool> check(params.number_of_partitions, false);
        for (size_t i = 0; i < permutation.size(); i++) {
            assert( permutation[i] < params.number_of_partitions );
            assert( check[ permutation[i] ] == false );
            check[permutation[i]] = true;
        }


        for (size_t i = 0; i < allocation.size(); i++) {
            allocation[i] = permutation[ allocation[i] ];
        }
    };

    int get_avg_weight_of_partition() const {
        int avg_weight = 0;
        int avg_weight_remainder = 0;

        for (auto &elem : collection) {
            avg_weight_remainder += elem.total_weight % params.number_of_partitions;
            avg_weight += elem.total_weight / params.number_of_partitions;
        }
        avg_weight += avg_weight_remainder / params.number_of_partitions;

        return avg_weight;
    };

    unsigned get_curretn_max_weight_of_partition() const {
        std::vector<unsigned> bins( params.number_of_partitions ,0);
        unsigned clmp_ind = 0;
        for (auto& clmp : collection ) {
            bins[ allocation[ clmp_ind ] ] += clmp.total_weight;
            clmp_ind++;
        }

        unsigned maximum_ = 0;
        for (long unsigned bin_ind = 0; bin_ind<bins.size(); bin_ind++) {
            maximum_ = std::max(maximum_, bins[bin_ind]);
        }

        return maximum_;
    };

    std::multiset<int, std::greater<int>> get_collection_weights() const {
        std::multiset<int, std::greater<int>> collection_weights;
        for (auto &item : collection) {
            collection_weights.emplace(item.total_weight);
        }

        return collection_weights;
    };

    bool work_distribution_attempt(const PartitionAlgorithm algo) {
        std::vector<unsigned> new_allocation;
        std::multiset<int, std::greater<int>> coll_weights = get_collection_weights();

        // std::cout << "Chosen Algorithm: " << algo <<std::endl;
        switch (algo) {
        case Greedy:
            new_allocation = greedy_partitioner(params.number_of_partitions, coll_weights);
            break;

        case KarmarkarKarp:
            if (is_power_of(params.number_of_partitions,2)) {
                new_allocation = kk_partitioner(params.number_of_partitions, coll_weights);
            }
            else {
                std::cout <<  "Karmarkar Karp algorithm only supports powers of two." << std::endl;
                std::cout << "Running Greedy partitioner instead." << std::endl;
                new_allocation = greedy_partitioner(params.number_of_partitions, coll_weights);
            }
            break;

        case BinPacking:
            try {
                new_allocation = binpacking_partitioner(params.number_of_partitions, coll_weights);
            }
            catch (...) {
                std::cout << "Bin packing not yet implemented." << std::endl;
                std::cout << "Running Greedy partitioner instead." << std::endl;
                new_allocation = greedy_partitioner(params.number_of_partitions, coll_weights);
            }
            break;
#ifdef COPT
        case ILP:
            try {
                new_allocation = ilp_partitioner(params.number_of_partitions, coll_weights);
            }
            catch (...) {
                std::cout << "ILP failed or reached time limit." << std::endl;
                std::cout << "Running Greedy partitioner instead." << std::endl;
                new_allocation = greedy_partitioner(params.number_of_partitions, coll_weights);
            }
            break;
#endif
        default:
            new_allocation = greedy_partitioner(params.number_of_partitions, coll_weights);
            break;
        }

        bool improvement = false;
        float new_imbalance = calculate_imbalance(params.number_of_partitions, coll_weights, new_allocation);
        if (new_imbalance < imbalance) {
            allocation = new_allocation;
            imbalance = new_imbalance;
            improvement = true;
        }

        return improvement;
    }

    bool work_distribution_attempt() { return work_distribution_attempt(params.part_algo); }

    bool run_allocation_improvement(const int runs) {
        std::multiset<int, std::greater<int>> coll_weights = get_collection_weights();

        std::pair<float, std::vector<unsigned>> new_imbalance_n_allocation =
            hill_climb_weight_balance_single_superstep(runs, params.number_of_partitions, coll_weights, allocation);

        bool output = (new_imbalance_n_allocation.first * 1.0001 < imbalance);
        if (output) {
            allocation = new_imbalance_n_allocation.second;
            imbalance = new_imbalance_n_allocation.first;
        }

        return output;
    }

    bool run_allocation_improvement() {
        return run_allocation_improvement(params.hill_climb_simple_improvement_attemps);
    }

    unsigned get_number_of_clumps() const { return collection.size(); };

    unsigned get_number_of_nodes() const {
        unsigned num_nodes = 0;

        for (auto &elem : collection) {
            num_nodes += elem.node_set.size();
        }

        return num_nodes;
    }

    inline float get_imbalance() const { return imbalance; };

    inline bool too_few_nodes() const { return too_few_nodes_in_clumps; };
    inline bool too__few_clumps() const { return too_few_clumps_superstep; };
    inline bool fat_nodes() const { return fat_clump_sizes; };

    void update_flags() {
        if ((get_number_of_nodes() < collection.size() * params.nodes_per_clump) &&
            (get_number_of_nodes() < params.number_of_partitions * params.nodes_per_partition)) {
            too_few_nodes_in_clumps = true;
        } else {
            too_few_nodes_in_clumps = false;
        }

        too_few_clumps_superstep =
            collection.size() < params.number_of_partitions * params.clumps_per_partition ? true : false;
        fat_clump_sizes = collection.cbegin()->total_weight > get_avg_weight_of_partition() * params.max_weight_for_flag
                              ? true
                              : false;
    };

    LooseSuperStep<T>(unsigned id_, std::vector<T> collection_, const Coarse_Scheduler_Params &params_)
        : params(params_), id(id_), collection(collection_.cbegin(), collection_.cend()) {
        if (collection.empty()) {
            throw std::runtime_error("Empty Loose Superstep.");
        }
        work_distribution_attempt();
        update_flags();
    };

    LooseSuperStep<T>(unsigned id_, std::multiset<T, typename T::Comparator> &collection_,
                      std::vector<unsigned> allocation_, float imbalance_, const Coarse_Scheduler_Params &params_)
        : params(params_), imbalance(imbalance_), allocation(allocation_), id(id_), collection(collection_) {
        if (collection.empty()) {
            throw std::runtime_error("Empty Loose Superstep.");
        }
        update_flags();
    };

    LooseSuperStep<T>(unsigned id_, LooseSuperStep<T> superstep_)
        : params(superstep_.params), imbalance(superstep_.imbalance), allocation(superstep_.allocation), id(id_), collection(superstep_.collection) {
        if (collection.empty()) {
            throw std::runtime_error("Empty Loose Superstep.");
        }
        update_flags();
    };

    struct Comparator {
        constexpr bool operator()(const LooseSuperStep<T> &a, const LooseSuperStep<T> &b) const {
            return ((a.get_imbalance() > b.get_imbalance()) || ((a.get_imbalance() == b.get_imbalance()) && (a.id > b.id )) );
        };
    };

    ~LooseSuperStep() = default;
};

class LooseSchedule {
  private:
    friend class CoarseRefineScheduler;
    friend class BalDMixR;
    friend class CoBalDMixR;
    const SubDAG &graph;
    const Coarse_Scheduler_Params params;
    std::unique_ptr<BalancedCoinFlips> coin;
    unsigned step_id_counter;

    // id of supersteps in topological order
    std::vector<unsigned> superstep_ordered_ids;
    std::multiset<LooseSuperStep<Clump>, typename LooseSuperStep<Clump>::Comparator> supersteps;

  public:
    /// @brief Adds superstep in position, where position = 0 is in the beginning and position =
    /// superstep_ordered_ids.size() is at the end
    void add_loose_superstep(unsigned position, std::vector<std::unordered_set<int>> &vec_node_sets);

    /// @brief Adds superstep in position, where position = 0 is in the beginning and position =
    /// superstep_ordered_ids.size() is at the end
    void add_loose_superstep_with_allocation(unsigned position, LooseSuperStep<Clump> superstep);

    /// @brief Splits a superstep into two parts by passed method returns true if non-trivial split was achieved
    bool split_into_two_supersteps(
        std::multiset<LooseSuperStep<Clump>, typename LooseSuperStep<Clump>::Comparator>::iterator supersteps_iterator,
        const CutType cut_type);

    /// @brief Runs a hill climb on passed superstep
    bool run_hill_climb(
        std::multiset<LooseSuperStep<Clump>, typename LooseSuperStep<Clump>::Comparator>::iterator supersteps_iterator,
        int hill_climb_iterations);
    bool run_hill_climb(std::multiset<LooseSuperStep<Clump>, typename LooseSuperStep<Clump>::Comparator>::iterator
                            supersteps_iterator) {
        return run_hill_climb(supersteps_iterator, params.hill_climb_simple_improvement_attemps);
    }

    /// @brief Introduces new supersteps by splitting non-optimal ones and runs hill climb on ones without flag
    /// @return Returns true if at least one superstep has been split
    bool run_superstep_improvement_iteration();

    /**
     * @brief returns iterator position of superstep for a given superstep id. Returns superstep.end() if not found.
     * 
     * @param id Superstep id
     */
    std::multiset<LooseSuperStep<Clump>, typename LooseSuperStep<Clump>::Comparator>::iterator find_id( const unsigned id ) const;

    /**
     * @brief attempts to combine supersteps with id superstep_ordered_ids[superstep_position] and superstep_ordered_ids[superstep_position+1]
     * 
     * @param superstep_position 
     * @param comm_cost_multiplier takes communication into account with said multiplier - this over estimates as it does not remove void comm cost if same processor
     * @param com_cost_addition takes an additive com cost into account
     * @return true if succeeded
     * @return false if failed
     */
    bool combine_superstep_attempt( const unsigned superstep_position, const unsigned comm_cost_multiplier = 0, const unsigned com_cost_addition = 0, const bool true_costs = false);


    /**
     * @brief Check superstep with parity parity if they can be combined with the subsequent superstep
     * 
     * @param parity parity
     * @param only_above_thresh only superstep above the balance threshhold should be considered
     * @param comm_cost_multiplier takes communication into account with said multiplier - this over estimates as it does not remove void comm cost if same processor
     * @param com_cost_addition takes an additive com cost into account
     * @return true if any succeeded
     * @return false if no change
     */
    bool run_joining_supersteps_improvements(const bool parity, const bool only_above_thresh = true, const unsigned comm_cost_multiplier = 0, const unsigned com_cost_addition = 0, const bool true_costs = false);

    void run_processor_assignment(const std::vector<std::vector<unsigned>>& processsor_comm_costs);
    void run_processor_assignment();

    /// @brief Gets a possible allocation of each node to superstep and processor. (Processors at each stage can still
    /// be permuted)
    /// @return Map: Nodes -> Supersteps x Processor
    std::unordered_map<int, std::pair<unsigned, unsigned>> get_current_node_schedule_allocation() const;

    /// @brief Gets a possible schedule (Processors at each stage can still be permuted)
    /// @return Superstep vector containing processor vector containing node vector
    std::vector<std::vector<std::vector<int>>> get_current_schedule() const;

    /// @brief Gets the current superstep imbalances for the current schedule
    std::vector<float> get_current_superstep_imbalances_in_order() const;

    /// @brief Permutes allocation of given superstep
    void permute_allocation_of_superstep(const unsigned index, std::vector<unsigned> permutation);

    /// @brief prints current schedule
    void print_current_schedule() const;

    LooseSchedule(const SubDAG &graph_, const Coarse_Scheduler_Params &params_)
        : graph(graph_), params(params_), step_id_counter(0) {
        switch (params.coin_type) {
        case Thue_Morse:
            coin = std::make_unique<Thue_Morse_Sequence>();
            break;

        case Biased_Randomly:
            coin = std::make_unique<Biased_Random>();
            break;

        default:
            coin = std::make_unique<Thue_Morse_Sequence>();
            break;
        }
    };

    LooseSchedule(const SubDAG &graph_, const Coarse_Scheduler_Params &params_,
                  std::vector<unsigned> superstep_ordered_ids_,
                  std::multiset<LooseSuperStep<Clump>, typename LooseSuperStep<Clump>::Comparator> supersteps_)
        : graph(graph_), params(params_), superstep_ordered_ids(superstep_ordered_ids_), supersteps(supersteps_)
    {
        switch (params.coin_type) {
        case Thue_Morse:
            coin = std::make_unique<Thue_Morse_Sequence>();
            break;

        case Biased_Randomly:
            coin = std::make_unique<Biased_Random>();
            break;

        default:
            coin = std::make_unique<Thue_Morse_Sequence>();
            break;
        }

        // checks
        for (auto& sstep : supersteps_) {
            assert( std::any_of( superstep_ordered_ids_.cbegin(), superstep_ordered_ids_.cend(), [&sstep](auto & i){ return (i == sstep.id); } ));
        }
        for (auto& id : superstep_ordered_ids_) {
            assert( std::any_of(supersteps_.cbegin(), supersteps_.cend(), [&id](auto & sstep){ return (id == sstep.id); } ) );
        }

        step_id_counter = 0;
        for (auto &id : superstep_ordered_ids) {
            step_id_counter = std::max(id, step_id_counter);
        }
        step_id_counter++;
    };

    // for testing
    std::pair<std::multiset<LooseSuperStep<Clump>, typename LooseSuperStep<Clump>::Comparator>::iterator,
              std::multiset<LooseSuperStep<Clump>, typename LooseSuperStep<Clump>::Comparator>::iterator>
    get_supersteps_delimits() {
        return {supersteps.begin(), supersteps.end()};
    };
};