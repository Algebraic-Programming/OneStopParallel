#define BOOST_TEST_MODULE coarse_refine_scheduler
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <unordered_set>
#include <vector>


#include "scheduler/ContractRefineScheduler/contract_refine_scheduler.hpp"
#include "scheduler/Partitioners/partitioners.hpp"
#include "auxiliary/Balanced_Coin_Flips.hpp"
#include "auxiliary/Erdos_Renyi_graph.hpp"
#include "structures/dag.hpp"
#include "scheduler/ContractRefineScheduler/refine/superstep_clumps.hpp"   

BOOST_AUTO_TEST_CASE(Coarse_Refine_test_small) {
    std::vector<unsigned> graph_sizes({100, 500, 500});
    std::vector<unsigned> graph_chances({10, 8, 20});

    assert( graph_chances.size() == graph_sizes.size() );

    const unsigned number_of_partitions_ = 2;
    const float balance_threshhold_ = 1.2;
    const PartitionAlgorithm part_algo_ = Greedy;
    const CoinType coin_type_ = Thue_Morse;
    const float clumps_per_partition_ = 6;
    const float nodes_per_clump_ = 8;
    const float nodes_per_partition_refine = 15; // approx 2 * (balance_thresh-1)^{-1}
    const float nodes_per_partition_final = 8; // approx 2 * (balance_thresh-1)^{-1}
    const float max_weight_for_flag_ = 1/3; // approx 2 / clumps_per_partition
    const float balanced_cut_ratio_ = 1/3;
    const float min_weight_for_split_ = 1/48;
    const int hill_climb_simple_improvement_attemps_ = 5;
    const int min_comp_generation_when_shaving_ = 3;
    
    Coarse_Scheduler_Params params_init(
        number_of_partitions_,
        balance_threshhold_,
        part_algo_,
        coin_type_,
        clumps_per_partition_,
        nodes_per_clump_,
        nodes_per_partition_refine,
        max_weight_for_flag_,
        balanced_cut_ratio_,
        min_weight_for_split_,
        hill_climb_simple_improvement_attemps_,
        min_comp_generation_when_shaving_
    );

    Coarse_Scheduler_Params params_final(
        number_of_partitions_,
        balance_threshhold_,
        part_algo_,
        coin_type_,
        clumps_per_partition_,
        nodes_per_clump_,
        nodes_per_partition_final,
        max_weight_for_flag_,
        balanced_cut_ratio_,
        min_weight_for_split_,
        hill_climb_simple_improvement_attemps_,
        min_comp_generation_when_shaving_
    );

    CoarseRefineScheduler_parameters params(params_init, params_final);


    for (int i = 0; i< graph_sizes.size(); i++ ) {
        const DAG graph = erdos_renyi_graph_gen(graph_sizes[i], graph_chances[i] );

        CoarseRefineScheduler schedule_alg(graph, params);
        schedule_alg.run_all();

        std::unordered_map<int, std::pair<unsigned, unsigned>> node_allocation = schedule_alg.get_loose_node_schedule_allocation();

        for (int i = 0 ; i < graph.n ; i++) {
            BOOST_CHECK( node_allocation.find(i) != node_allocation.cend() );
            BOOST_CHECK( node_allocation.at(i).second >= 0 || node_allocation.at(i).second < params.number_of_partitions );
            for (auto node : graph.descendants(i)) {
                BOOST_CHECK_LE( node_allocation.at(i).first, node_allocation.at(node).first );
                if (node_allocation.at(i).first == node_allocation.at(node).first) {
                    BOOST_CHECK_EQUAL( node_allocation.at(i).second, node_allocation.at(node).second );
                }
            }
        }

        schedule_alg.print_computing_schedule();

        std::cout << "End graph test" << std::endl << std::endl;
    }
};


BOOST_AUTO_TEST_CASE(Coarse_Refine_test_medium) {
    std::vector<unsigned> graph_sizes({1000, 5000, 5000});
    std::vector<unsigned> graph_chances({10, 6, 50});

    assert( graph_chances.size() == graph_sizes.size() );

    const unsigned number_of_partitions_ = 4;
    const float balance_threshhold_ = 1.1;
    const PartitionAlgorithm part_algo_ = Greedy;
    const CoinType coin_type_ = Thue_Morse;
    const float clumps_per_partition_ = 6;
    const float nodes_per_clump_ = 4;
    const float nodes_per_partition_refine = 30; // approx 2 * (balance_thresh-1)^{-1}
    const float nodes_per_partition_final = 20; // approx 2 * (balance_thresh-1)^{-1}
    const float max_weight_for_flag_ = 1/3; // approx 2 / clumps_per_partition
    const float balanced_cut_ratio_ = 1/3;
    const float min_weight_for_split_ = 1/48;
    const int hill_climb_simple_improvement_attemps_ = 10;
    const int min_comp_generation_when_shaving_ = 6;
    
    Coarse_Scheduler_Params params_init(
        number_of_partitions_,
        balance_threshhold_,
        part_algo_,
        coin_type_,
        clumps_per_partition_,
        nodes_per_clump_,
        nodes_per_partition_refine,
        max_weight_for_flag_,
        balanced_cut_ratio_,
        min_weight_for_split_,
        hill_climb_simple_improvement_attemps_,
        min_comp_generation_when_shaving_
    );

    Coarse_Scheduler_Params params_final(
        number_of_partitions_,
        balance_threshhold_,
        part_algo_,
        coin_type_,
        clumps_per_partition_,
        nodes_per_clump_,
        nodes_per_partition_final,
        max_weight_for_flag_,
        balanced_cut_ratio_,
        min_weight_for_split_,
        hill_climb_simple_improvement_attemps_,
        min_comp_generation_when_shaving_
    );

    CoarseRefineScheduler_parameters params(params_init, params_final);


    for (int i = 0; i< graph_sizes.size(); i++ ) {
        const DAG graph = erdos_renyi_graph_gen(graph_sizes[i], graph_chances[i] );

        CoarseRefineScheduler schedule_alg(graph, params);
        
        schedule_alg.run_all();

        std::unordered_map<int, std::pair<unsigned, unsigned>> node_allocation = schedule_alg.get_loose_node_schedule_allocation();
        
        schedule_alg.print_computing_schedule();

        for (int i = 0 ; i < graph.n ; i++) {
            assert(node_allocation.find(i) != node_allocation.cend());
            BOOST_CHECK( node_allocation.find(i) != node_allocation.cend() );
            BOOST_CHECK( node_allocation.at(i).second >= 0 || node_allocation.at(i).second < params.number_of_partitions );
            for (auto node : graph.descendants(i)) {
                BOOST_CHECK_LE( node_allocation.at(i).first, node_allocation.at(node).first );
                if (node_allocation.at(i).first == node_allocation.at(node).first) {
                    BOOST_CHECK_EQUAL( node_allocation.at(i).second, node_allocation.at(node).second );
                }
            }
        }


        std::cout << "End graph test" << std::endl << std::endl;
    }
};



// BOOST_AUTO_TEST_CASE(Coarse_Refine_test_large) {
//     std::vector<unsigned> graph_sizes({10000, 50000, 100000});
//     std::vector<unsigned> graph_chances({200, 50, 10});

//     assert( graph_chances.size() == graph_sizes.size() );

//     const unsigned number_of_partitions_ = 16;
//     const float balance_threshhold_initial_ = 1.15;
//     const float balance_threshhold_final = 1.5;
//     const PartitionAlgorithm part_algo_ = Greedy;
//     const CoinType coin_type_ = Thue_Morse;
//     const float clumps_per_partition_ = 6;
//     const float nodes_per_clump_ = 4;
//     const float nodes_per_partition_refine = 30; // approx 2 * (balance_thresh-1)^{-1}
//     const float nodes_per_partition_final = 20; // approx 2 * (balance_thresh-1)^{-1}
//     const float max_weight_for_flag_ = 1/3; // approx 2 / clumps_per_partition
//     const float balanced_cut_ratio_ = 1/3;
//     const float min_weight_for_split_ = 1/48;
//     const int hill_climb_simple_improvement_attemps_ = 5;
//     const int min_comp_generation_when_shaving_ = 12;
    
//     Coarse_Scheduler_Params params_init(
//         number_of_partitions_,
//         balance_threshhold_initial_,
//         part_algo_,
//         coin_type_,
//         clumps_per_partition_,
//         nodes_per_clump_,
//         nodes_per_partition_refine,
//         max_weight_for_flag_,
//         balanced_cut_ratio_,
//         min_weight_for_split_,
//         hill_climb_simple_improvement_attemps_,
//         min_comp_generation_when_shaving_
//     );

//     Coarse_Scheduler_Params params_final(
//         number_of_partitions_,
//         balance_threshhold_final,
//         part_algo_,
//         coin_type_,
//         clumps_per_partition_,
//         nodes_per_clump_,
//         nodes_per_partition_final,
//         max_weight_for_flag_,
//         balanced_cut_ratio_,
//         min_weight_for_split_,
//         hill_climb_simple_improvement_attemps_,
//         min_comp_generation_when_shaving_
//     );

//     CoarseRefineScheduler_parameters params(params_init, params_final);


//     for (int i = 0; i< graph_sizes.size(); i++ ) {
//         const DAG graph = erdos_renyi_graph_gen(graph_sizes[i], graph_chances[i] );

//         CoarseRefineScheduler schedule_alg(graph, params);
        
//         schedule_alg.run_all();

//         std::unordered_map<int, std::pair<unsigned, unsigned>> node_allocation = schedule_alg.get_loose_node_schedule_allocation();

//         // for (int i = 0 ; i < graph.n ; i++) {
//         //     assert(node_allocation.find(i) != node_allocation.cend());
//         //     BOOST_CHECK( node_allocation.find(i) != node_allocation.cend() );
//         //     BOOST_CHECK( node_allocation.at(i).second >= 0 || node_allocation.at(i).second < params.number_of_partitions );
//         //     for (auto node : graph.descendants(i)) {
//         //         BOOST_CHECK_LE( node_allocation.at(i).first, node_allocation.at(node).first );
//         //         if (node_allocation.at(i).first == node_allocation.at(node).first) {
//         //             BOOST_CHECK_EQUAL( node_allocation.at(i).second, node_allocation.at(node).second );
//         //         }
//         //     }
//         // }

//         schedule_alg.print_computing_schedule();

//         std::cout << "End graph test" << std::endl << std::endl;
//     }
// };


// BOOST_AUTO_TEST_CASE(Coarse_Refine_test_very_dense_medium) {
//     std::vector<unsigned> graph_sizes({10000});
//     std::vector<unsigned> graph_chances({200});

//     assert( graph_chances.size() == graph_sizes.size() );

//     const unsigned number_of_partitions_ = 4;
//     const float balance_threshhold_initial = 1.5;
//     const float balance_threshhold_final = 2;
//     const PartitionAlgorithm part_algo_ = Greedy;
//     const CoinType coin_type_ = Thue_Morse;
//     const float clumps_per_partition_ = 6;
//     const float nodes_per_clump_ = 10;
//     const float nodes_per_partition_refine = 30; // approx 2 * (balance_thresh-1)^{-1}
//     const float nodes_per_partition_final = 20; // approx 2 * (balance_thresh-1)^{-1}
//     const float max_weight_for_flag_ = 1/3; // approx 2 / clumps_per_partition
//     const float balanced_cut_ratio_ = 1/3;
//     const float min_weight_for_split_ = 1/48;
//     const int hill_climb_simple_improvement_attemps_ = 5;
    
//     Coarse_Scheduler_Params params_init(
//         number_of_partitions_,
//         balance_threshhold_initial,
//         part_algo_,
//         coin_type_,
//         clumps_per_partition_,
//         nodes_per_clump_,
//         nodes_per_partition_refine,
//         max_weight_for_flag_,
//         balanced_cut_ratio_,
//         min_weight_for_split_,
//         hill_climb_simple_improvement_attemps_
//     );

//     Coarse_Scheduler_Params params_final(
//         number_of_partitions_,
//         balance_threshhold_final,
//         part_algo_,
//         coin_type_,
//         clumps_per_partition_,
//         nodes_per_clump_,
//         nodes_per_partition_final,
//         max_weight_for_flag_,
//         balanced_cut_ratio_,
//         min_weight_for_split_,
//         hill_climb_simple_improvement_attemps_
//     );

//     CoarseRefineScheduler_parameters params(params_init, params_final);


//     for (int i = 0; i< graph_sizes.size(); i++ ) {
//         const DAG graph = erdos_renyi_graph_gen(graph_sizes[i], graph_chances[i] );

//         CoarseRefineScheduler schedule_alg(graph, params);
        
//         schedule_alg.run_all();

//         std::unordered_map<int, std::pair<unsigned, unsigned>> node_allocation = schedule_alg.get_loose_node_schedule_allocation();

//         // for (int i = 0 ; i < graph.n ; i++) {
//         //     assert(node_allocation.find(i) != node_allocation.cend());
//         //     BOOST_CHECK( node_allocation.find(i) != node_allocation.cend() );
//         //     BOOST_CHECK( node_allocation.at(i).second >= 0 || node_allocation.at(i).second < params.number_of_partitions );
//         //     for (auto node : graph.descendants(i)) {
//         //         BOOST_CHECK_LE( node_allocation.at(i).first, node_allocation.at(node).first );
//         //         if (node_allocation.at(i).first == node_allocation.at(node).first) {
//         //             BOOST_CHECK_EQUAL( node_allocation.at(i).second, node_allocation.at(node).second );
//         //         }
//         //     }
//         // }

//         schedule_alg.print_computing_schedule();

//         std::cout << "End graph test" << std::endl << std::endl;
//     }
// };


// BOOST_AUTO_TEST_CASE(Coarse_Refine_test_dense_large) {
//     std::vector<unsigned> graph_sizes({100000});
//     std::vector<unsigned> graph_chances({200});

//     assert( graph_chances.size() == graph_sizes.size() );

//     const unsigned number_of_partitions_ = 16;
//     const float balance_threshhold_ = 1.1;
//     const PartitionAlgorithm part_algo_ = Greedy;
//     const CoinType coin_type_ = Thue_Morse;
//     const float clumps_per_partition_ = 6;
//     const float nodes_per_clump_ = 10;
//     const float nodes_per_partition_refine = 30; // approx 2 * (balance_thresh-1)^{-1}
//     const float nodes_per_partition_final = 20; // approx 2 * (balance_thresh-1)^{-1}
//     const float max_weight_for_flag_ = 1/3; // approx 2 / clumps_per_partition
//     const float balanced_cut_ratio_ = 1/3;
//     const float min_weight_for_split_ = 1/48;
//     const int hill_climb_simple_improvement_attemps_ = 5;
    
//     Coarse_Scheduler_Params params_init(
//         number_of_partitions_,
//         balance_threshhold_,
//         part_algo_,
//         coin_type_,
//         clumps_per_partition_,
//         nodes_per_clump_,
//         nodes_per_partition_refine,
//         max_weight_for_flag_,
//         balanced_cut_ratio_,
//         min_weight_for_split_,
//         hill_climb_simple_improvement_attemps_
//     );

//     Coarse_Scheduler_Params params_final(
//         number_of_partitions_,
//         balance_threshhold_,
//         part_algo_,
//         coin_type_,
//         clumps_per_partition_,
//         nodes_per_clump_,
//         nodes_per_partition_final,
//         max_weight_for_flag_,
//         balanced_cut_ratio_,
//         min_weight_for_split_,
//         hill_climb_simple_improvement_attemps_
//     );

//     CoarseRefineScheduler_parameters params(params_init, params_final);


//     for (int i = 0; i< graph_sizes.size(); i++ ) {
//         const DAG graph = erdos_renyi_graph_gen(graph_sizes[i], graph_chances[i] );

//         CoarseRefineScheduler schedule_alg(graph, params);
        
//         schedule_alg.run_all();

//         std::unordered_map<int, std::pair<unsigned, unsigned>> node_allocation = schedule_alg.get_loose_node_schedule_allocation();

//         // for (int i = 0 ; i < graph.n ; i++) {
//         //     assert(node_allocation.find(i) != node_allocation.cend());
//         //     BOOST_CHECK( node_allocation.find(i) != node_allocation.cend() );
//         //     BOOST_CHECK( node_allocation.at(i).second >= 0 || node_allocation.at(i).second < params.number_of_partitions );
//         //     for (auto node : graph.descendants(i)) {
//         //         BOOST_CHECK_LE( node_allocation.at(i).first, node_allocation.at(node).first );
//         //         if (node_allocation.at(i).first == node_allocation.at(node).first) {
//         //             BOOST_CHECK_EQUAL( node_allocation.at(i).second, node_allocation.at(node).second );
//         //         }
//         //     }
//         // }

//         schedule_alg.print_loose_schedule();

//         std::cout << "End graph test" << std::endl << std::endl;
//     }
// };



// BOOST_AUTO_TEST_CASE(Coarse_Refine_test_massive) {
//     std::vector<unsigned> graph_sizes({100000});
//     std::vector<unsigned> graph_chances({100});

//     assert( graph_chances.size() == graph_sizes.size() );

//     const unsigned number_of_partitions_ = 8;
//     const float balance_threshhold_initial = 1.35;
//     const float balance_threshhold_final = 2;
//     const PartitionAlgorithm part_algo_ = Greedy;
//     const CoinType coin_type_ = Thue_Morse;
//     const float clumps_per_partition_ = 6;
//     const float nodes_per_clump_ = 10;
//     const float nodes_per_partition_refine = 50; // approx 2 * (balance_thresh-1)^{-1}
//     const float nodes_per_partition_final = 30; // approx 2 * (balance_thresh-1)^{-1}
//     const float max_weight_for_flag_ = 1/3; // approx 2 / clumps_per_partition
//     const float balanced_cut_ratio_ = 1/3;
//     const float min_weight_for_split_ = 1/48;
//     const int hill_climb_simple_improvement_attemps_ = 5;

//     Coarse_Scheduler_Params params_init(
//         number_of_partitions_,
//         balance_threshhold_initial,
//         part_algo_,
//         coin_type_,
//         clumps_per_partition_,
//         nodes_per_clump_,
//         nodes_per_partition_refine,
//         max_weight_for_flag_,
//         balanced_cut_ratio_,
//         min_weight_for_split_,
//         hill_climb_simple_improvement_attemps_
//     );

//     Coarse_Scheduler_Params params_final(
//         number_of_partitions_,
//         balance_threshhold_final,
//         part_algo_,
//         coin_type_,
//         clumps_per_partition_,
//         nodes_per_clump_,
//         nodes_per_partition_final,
//         max_weight_for_flag_,
//         balanced_cut_ratio_,
//         min_weight_for_split_,
//         hill_climb_simple_improvement_attemps_
//     );

//     CoarseRefineScheduler_parameters params(params_init, params_final, CoarsenParams(), 150);


//     for (int i = 0; i< graph_sizes.size(); i++ ) {
//         const DAG graph = erdos_renyi_graph_gen(graph_sizes[i], graph_chances[i] );

//         CoarseRefineScheduler schedule_alg(graph, params);
        
//         schedule_alg.run_all();

//         std::unordered_map<int, std::pair<unsigned, unsigned>> node_allocation = schedule_alg.get_loose_node_schedule_allocation();

//         // for (int i = 0 ; i < graph.n ; i++) {
//         //     assert(node_allocation.find(i) != node_allocation.cend());
//         //     BOOST_CHECK( node_allocation.find(i) != node_allocation.cend() );
//         //     BOOST_CHECK( node_allocation.at(i).second >= 0 || node_allocation.at(i).second < params.number_of_partitions );
//         //     for (auto node : graph.descendants(i)) {
//         //         BOOST_CHECK_LE( node_allocation.at(i).first, node_allocation.at(node).first );
//         //         if (node_allocation.at(i).first == node_allocation.at(node).first) {
//         //             BOOST_CHECK_EQUAL( node_allocation.at(i).second, node_allocation.at(node).second );
//         //         }
//         //     }
//         // }

//         schedule_alg.print_computing_schedule();

//         std::cout << "End graph test" << std::endl << std::endl;
//     }
// };