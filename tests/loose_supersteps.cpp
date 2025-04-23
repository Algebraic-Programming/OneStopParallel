#define BOOST_TEST_MODULE loose_supersteps_test
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <unordered_set>
#include <vector>

#include "scheduler/ContractRefineScheduler/refine/superstep_clumps.hpp"
#include "scheduler/Partitioners/partitioners.hpp"
#include "auxiliary/Balanced_Coin_Flips.hpp"
#include "auxiliary/Erdos_Renyi_graph.hpp"
#include "structures/dag.hpp"

BOOST_AUTO_TEST_CASE(DAG1_Clumps) {
    const DAG graph(
        {// In edes
         {6, 2, 5},
         {6, 9},
         {6, 3, 5},
         {},
         {8},
         {9},
         {9},
         {0},
         {},
         {}},
        {// Out edges
         {7},
         {},
         {0},
         {2},
         {},
         {2, 0},
         {1, 2, 0},
         {},
         {4},
         {6, 1, 5}},
        // Work weights
        {1, 1, 1, 1, 2, 3, 2, 1, 1, 1},
        // Communication weights
        {1, 1, 1, 1, 2, 3, 2, 1, 1, 1});

    std::unordered_set<int> a({0,1,2}); // wt 3
    std::unordered_set<int> b({3,4,5}); // wt 6
    std::unordered_set<int> c({6,7}); // wt 3
    std::unordered_set<int> d({8}); // wt 1
    std::unordered_set<int> e({9}); // wt 1

    Clump aa(graph, a);
    Clump bb(graph, b);
    Clump cc(graph, c);
    Clump dd(graph, d);
    Clump ee(graph, e);

    std::vector<int> wts_ans({6,3,3,1,1});

    LooseSuperStep<Clump> one_ss(0, {aa,bb,cc,dd,ee}, Coarse_Scheduler_Params(2));

    int i = 0;
    for (auto it = one_ss.collection.cbegin(); it != one_ss.collection.cend(); it++ ) {
        BOOST_CHECK_EQUAL( (*it).total_weight, wts_ans[i] );
        i++;
    }

    std::multiset<int, std::greater<int>> ordered_wts = one_ss.get_collection_weights();

    i = 0;
    for (auto it = ordered_wts.cbegin(); it != ordered_wts.cend(); it++ ) {
        BOOST_CHECK_EQUAL( (*it), wts_ans[i] );
        i++;
    }
};


BOOST_AUTO_TEST_CASE(DAG1_LooseSchedule_balanced) {
    const DAG graph(
        {// In edes
         {6, 2, 5},
         {6, 9},
         {6, 3, 5},
         {},
         {8},
         {9},
         {9},
         {0},
         {},
         {}},
        {// Out edges
         {7},
         {},
         {0},
         {2},
         {},
         {2, 0},
         {1, 2, 0},
         {},
         {4},
         {6, 1, 5}},
        // Work weights
        {1, 1, 1, 1, 2, 3, 2, 1, 1, 1},
        // Communication weights
        {1, 1, 1, 1, 2, 3, 2, 1, 1, 1});

    SubDAG graph_0 = graph.toSubDAG();

    std::vector<std::unordered_set<int>> superstep_node_sets( {{8,4},{3,9,6,1,2,5,0,7}} );
    Coarse_Scheduler_Params params(2);

    LooseSchedule sched(graph_0, params);
    sched.add_loose_superstep(0, superstep_node_sets);

    std::multiset< LooseSuperStep<Clump>, typename LooseSuperStep<Clump>::Comparator>::iterator superstep_begin = sched.get_supersteps_delimits().first;

    sched.split_into_two_supersteps( superstep_begin, Balanced );

    std::unordered_map<int, std::pair<unsigned, unsigned>> node_allocation = sched.get_current_node_schedule_allocation();

    for (int i = 0 ; i < graph_0.n ; i++) {
        BOOST_CHECK( node_allocation.find(i) != node_allocation.cend() );
        BOOST_CHECK( node_allocation.at(i).first == 0 || node_allocation.at(i).first == 1 );
        BOOST_CHECK( node_allocation.at(i).second == 0 || node_allocation.at(i).second == 1 );
        for (auto node : graph_0.descendants(i)) {
            BOOST_CHECK_LE( node_allocation.at(i).first, node_allocation.at(node).first );
            if (node_allocation.at(i).first == node_allocation.at(node).first) {
                BOOST_CHECK_EQUAL( node_allocation.at(i).second, node_allocation.at(node).second );
            }
        }
    }

    sched.print_current_schedule();

};


BOOST_AUTO_TEST_CASE(DAG1_LooseSchedule_Shaving) {
    const DAG graph(
        {// In edes
         {6, 2, 5},
         {6, 9},
         {6, 3, 5},
         {},
         {8},
         {9},
         {9},
         {0},
         {},
         {}},
        {// Out edges
         {7},
         {},
         {0},
         {2},
         {},
         {2, 0},
         {1, 2, 0},
         {},
         {4},
         {6, 1, 5}},
        // Work weights
        {1, 1, 1, 1, 2, 3, 2, 1, 1, 1},
        // Communication weights
        {1, 1, 1, 1, 2, 3, 2, 1, 1, 1});

    SubDAG graph_0 = graph.toSubDAG();

    std::vector<std::unordered_set<int>> superstep_node_sets( {{8,4},{3,9,6,1,2,5,0,7}} );
    Coarse_Scheduler_Params params(2);

    LooseSchedule sched(graph_0, params);
    sched.add_loose_superstep(0, superstep_node_sets);

    std::multiset< LooseSuperStep<Clump>, typename LooseSuperStep<Clump>::Comparator>::iterator superstep_begin = sched.get_supersteps_delimits().first;

    sched.split_into_two_supersteps( superstep_begin, Shaving );

    std::unordered_map<int, std::pair<unsigned, unsigned>> node_allocation = sched.get_current_node_schedule_allocation();

    for (int i = 0 ; i < graph_0.n ; i++) {
        BOOST_CHECK( node_allocation.find(i) != node_allocation.cend() );
        BOOST_CHECK( node_allocation.at(i).first == 0 || node_allocation.at(i).first == 1 );
        BOOST_CHECK( node_allocation.at(i).second == 0 || node_allocation.at(i).second == 1 );
        for (auto node : graph_0.descendants(i)) {
            BOOST_CHECK_LE( node_allocation.at(i).first, node_allocation.at(node).first );
            if (node_allocation.at(i).first == node_allocation.at(node).first) {
                BOOST_CHECK_EQUAL( node_allocation.at(i).second, node_allocation.at(node).second );
            }
        }
    }

    sched.print_current_schedule();

};


BOOST_AUTO_TEST_CASE(DAG1_LooseSchedule_run_hill_climb) {
    const DAG graph(
        {// In edes
         {6, 2, 5},
         {6, 9},
         {6, 3, 5},
         {},
         {8},
         {9},
         {9},
         {0},
         {},
         {}},
        {// Out edges
         {7},
         {},
         {0},
         {2},
         {},
         {2, 0},
         {1, 2, 0},
         {},
         {4},
         {6, 1, 5}},
        // Work weights
        {1, 1, 1, 1, 2, 3, 2, 1, 1, 1},
        // Communication weights
        {1, 1, 1, 1, 2, 3, 2, 1, 1, 1});

    SubDAG graph_0 = graph.toSubDAG();

    std::vector<std::unordered_set<int>> superstep_node_sets( {{8,4},{3,9,6,1,2,5,0,7}} );
    Coarse_Scheduler_Params params(2);

    LooseSchedule sched(graph_0, params);
    sched.add_loose_superstep(0, superstep_node_sets);

    std::multiset< LooseSuperStep<Clump>, typename LooseSuperStep<Clump>::Comparator>::iterator superstep_begin = sched.get_supersteps_delimits().first;

    sched.run_hill_climb(superstep_begin, 100);

    std::unordered_map<int, std::pair<unsigned, unsigned>> node_allocation = sched.get_current_node_schedule_allocation();

    for (int i = 0 ; i < graph_0.n ; i++) {
        BOOST_CHECK( node_allocation.find(i) != node_allocation.cend() );
        BOOST_CHECK( node_allocation.at(i).first == 0 );
        BOOST_CHECK( node_allocation.at(i).second == 0 || node_allocation.at(i).second == 1 );
        for (auto node : graph_0.descendants(i)) {
            BOOST_CHECK_LE( node_allocation.at(i).first, node_allocation.at(node).first );
            if (node_allocation.at(i).first == node_allocation.at(node).first) {
                    BOOST_CHECK_EQUAL( node_allocation.at(i).second, node_allocation.at(node).second );
            }
        }
    }

    sched.print_current_schedule();

};


BOOST_AUTO_TEST_CASE(DAG1_LooseSchedule_run_improvements1) {
    const DAG graph(
        {// In edes
         {6, 2, 5},
         {6, 9},
         {6, 3, 5},
         {},
         {8},
         {9},
         {9},
         {0},
         {},
         {}},
        {// Out edges
         {7},
         {},
         {0},
         {2},
         {},
         {2, 0},
         {1, 2, 0},
         {},
         {4},
         {6, 1, 5}},
        // Work weights
        {1, 1, 1, 1, 2, 3, 2, 1, 1, 1},
        // Communication weights
        {1, 1, 1, 1, 2, 3, 2, 1, 1, 1});

    SubDAG graph_0 = graph.toSubDAG();

    std::vector<std::unordered_set<int>> superstep_node_sets( {{8,4},{3,9,6,1,2,5,0,7}} );
    
    const unsigned number_of_partitions_ = 2;
    const float balance_threshhold_ = 1.1;
    const PartitionAlgorithm part_algo_ = Greedy;
    const CoinType coin_type_ = Thue_Morse;
    const float clumps_per_partition_ = 2;
    const float nodes_per_clump_ = 2;
    const float nodes_per_partition_ = 2;
    const float max_weight_for_flag_ = 1/2;
    const float balanced_cut_ratio_ = 1/3;
    const float min_weight_for_split_ = 1/24;
    const int hill_climb_simple_improvement_attemps_ = 20;
    
    Coarse_Scheduler_Params params(
        number_of_partitions_,
        balance_threshhold_,
        part_algo_,
        coin_type_,
        clumps_per_partition_,
        nodes_per_clump_,
        nodes_per_partition_,
        max_weight_for_flag_,
        balanced_cut_ratio_,
        min_weight_for_split_,
        hill_climb_simple_improvement_attemps_
    );

    LooseSchedule sched(graph_0, params);
    sched.add_loose_superstep(0, superstep_node_sets);

    bool change = true;
    while (change) {
        
        change = sched.run_superstep_improvement_iteration();

        std::unordered_map<int, std::pair<unsigned, unsigned>> node_allocation = sched.get_current_node_schedule_allocation();

        for (int i = 0 ; i < graph_0.n ; i++) {
            BOOST_CHECK( node_allocation.find(i) != node_allocation.cend() );
            BOOST_CHECK( node_allocation.at(i).second == 0 || node_allocation.at(i).second == 1 );
            for (auto node : graph_0.descendants(i)) {
                BOOST_CHECK_LE( node_allocation.at(i).first, node_allocation.at(node).first );
                if (node_allocation.at(i).first == node_allocation.at(node).first) {
                    BOOST_CHECK_EQUAL( node_allocation.at(i).second, node_allocation.at(node).second );
                }
            }
        }

        sched.print_current_schedule();

    }

};


BOOST_AUTO_TEST_CASE(DAG1_LooseSchedule_run_improvements2) {
    std::vector<unsigned> graph_sizes({100, 500, 1000, 1000}); //, 5000, 5000});
    std::vector<unsigned> graph_chances({10, 5, 8, 20}); //, 3, 14});

    assert( graph_chances.size() == graph_sizes.size() );

    const unsigned number_of_partitions_ = 10;
    const float balance_threshhold_ = 1.1;
    const PartitionAlgorithm part_algo_ = Greedy;
    const CoinType coin_type_ = Thue_Morse;
    const float clumps_per_partition_ = 6;
    const float nodes_per_clump_ = 4;
    const float nodes_per_partition_ = 4;
    const float max_weight_for_flag_ = 1/3;
    const float balanced_cut_ratio_ = 1/3;
    const float min_weight_for_split_ = 1/48;
    const int hill_climb_simple_improvement_attemps_ = 5;
    
    Coarse_Scheduler_Params params(
        number_of_partitions_,
        balance_threshhold_,
        part_algo_,
        coin_type_,
        clumps_per_partition_,
        nodes_per_clump_,
        nodes_per_partition_,
        max_weight_for_flag_,
        balanced_cut_ratio_,
        min_weight_for_split_,
        hill_climb_simple_improvement_attemps_
    );


    for (size_t i = 0; i< graph_sizes.size(); i++ ) {
        DAG graph = erdos_renyi_graph_gen(graph_sizes[i], graph_chances[i] );
        SubDAG graph_0 = graph.toSubDAG();

        LooseSchedule sched(graph_0, params);
        std::vector<std::unordered_set<int>> connected_comp = graph_0.weakly_connected_components();

        sched.add_loose_superstep(0, connected_comp);

        int nochanges = 0;
        while (nochanges < 3) {

            bool change = sched.run_superstep_improvement_iteration();
            if (change)
                nochanges = 0;
            else
                nochanges++;

            std::unordered_map<int, std::pair<unsigned, unsigned>> node_allocation = sched.get_current_node_schedule_allocation();

            for (int i = 0 ; i < graph_0.n ; i++) {
                BOOST_CHECK( node_allocation.find(i) != node_allocation.cend() );
                BOOST_CHECK( node_allocation.at(i).second >= 0 || node_allocation.at(i).second < params.number_of_partitions );
                for (auto node : graph_0.descendants(i)) {
                    BOOST_CHECK_LE( node_allocation.at(i).first, node_allocation.at(node).first );
                    if (node_allocation.at(i).first == node_allocation.at(node).first) {
                        BOOST_CHECK_EQUAL( node_allocation.at(i).second, node_allocation.at(node).second );
                    }
                }
            }
        }
        sched.print_current_schedule();
    }
};