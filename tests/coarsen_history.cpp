#define BOOST_TEST_MODULE Coarsen_History
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <chrono>

#include "auxiliary/auxiliary.hpp"
#include "auxiliary/Erdos_Renyi_graph.hpp"
#include "scheduler/ContractRefineScheduler/coarsen/coarsen_history.hpp"

BOOST_AUTO_TEST_CASE(Coarsen_History1) {
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

    CoarsenHistory contraction_chain(graph);

    const contract_edge_sort edge_sort = Contract_Edge_Decrease;

    int no_change = 0;
    while( no_change < 3  ) {
        int diff = contraction_chain.run_and_add_contraction(edge_sort);
        if (diff == 0)
            no_change++;
    }



    std::cout << "First Graph Size: " << contraction_chain.retrieve_dag_evolution()[0].n << " Final Graph Size: " << contraction_chain.retrieve_dag_evolution().back().n << std::endl;
        
    for (auto& dag : contraction_chain.retrieve_dag_evolution()) {
        BOOST_CHECK( dag.is_acyclic() );
    }

    std::vector<DAG> dag_seq = contraction_chain.retrieve_dag_evolution();
    std::vector<std::unordered_map<int, int>> contr_seq = contraction_chain.retrieve_contr_maps();
    std::vector<std::unordered_map<int, std::unordered_set<int>>> exp_seq = contraction_chain.retrieve_expansion_maps();

    BOOST_CHECK_EQUAL( dag_seq.size()-1, exp_seq.size() );
    BOOST_CHECK_EQUAL( contr_seq.size(), exp_seq.size() );

    for (long unsigned i = 0; i < dag_seq.size(); i++) {
        for (int j = 0; j < dag_seq[i].n; j++) {
            if (i != dag_seq.size()-1) {
                BOOST_CHECK( contr_seq[i].find(j) != contr_seq[i].cend() );
                int post_j = contr_seq[i].at(j);
                BOOST_CHECK( exp_seq[i].find(post_j) != exp_seq[i].cend() );
                BOOST_CHECK( exp_seq[i].at(post_j).find(j) != exp_seq[i].at(post_j).cend() );
            }
            if ( i != 0 ) {
                BOOST_CHECK( exp_seq[i-1].find(j) != exp_seq[i-1].cend() );
                BOOST_CHECK( exp_seq[i-1].size() != 0 );
                for (auto& node : exp_seq[i-1].at(j)) {
                    BOOST_CHECK_EQUAL( contr_seq[i-1].at(node), j );
                }

                BOOST_CHECK_EQUAL( dag_seq[i].workW[j], dag_seq[i-1].workW_of_node_set( exp_seq[i-1].at(j) ) );
            }
        }
    }

    // std::cout << "Contraction maps:" << std::endl;
    // for (auto& [pre,post] : contr_seq[0]) {
    //     std::cout << pre << " -> " << post <<std::endl;
    // }
    // std::cout << "Expansion maps:" << std::endl;
    // for (auto& [pre,post] : exp_seq[0]) {
    //     std::cout << pre << " -> ";
    //     for (auto& node : exp_seq[0].at(pre)) {
    //         std::cout << node << ", ";
    //     }
    //     std::cout<< std::endl;
    // }
}


BOOST_AUTO_TEST_CASE(Coarsen_History2) {
    
    std::vector<unsigned> graph_sizes({100, 500, 1000, 5000, 10000}); //  50000, 50000, 100000, 100000});
    std::vector<unsigned> graph_chances({10, 5, 8, 8, 2}); // 3, 10, 3, 10});

    assert( graph_chances.size() == graph_sizes.size() );

    for (size_t i = 0; i< graph_sizes.size(); i++ ) {
        DAG graph = erdos_renyi_graph_gen(graph_sizes[i], graph_chances[i] );
    
        CoarsenHistory contraction_chain(graph);
        const contract_edge_sort edge_sort = Contract_Edge_Decrease;

        auto start = std::chrono::high_resolution_clock::now();
        int no_change_in_a_row = 0;
        while( no_change_in_a_row < 5  ) {
            int diff = contraction_chain.run_and_add_contraction(edge_sort);
            if (diff == 0)
                no_change_in_a_row++;
            else
                no_change_in_a_row = 0;
        }
        auto end = std::chrono::high_resolution_clock::now();
        double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        time_taken *= 1e-9;

        for (auto& dag : contraction_chain.retrieve_dag_evolution()) {
            BOOST_CHECK( dag.is_acyclic() );
        }

        std::cout << "Initial Size: " << contraction_chain.retrieve_dag_evolution()[0].n
            << ", Final Size: " << contraction_chain.retrieve_dag_evolution().back().n
            << ", Components: " << contraction_chain.retrieve_dag_evolution().back().weakly_connected_components().size() 
            << ", Chain length: " << contraction_chain.retrieve_dag_evolution().size()
            << ", Time: " << time_taken << "s" << std::endl;
    }        
}

BOOST_AUTO_TEST_CASE(Coarsen_History3) {
    
    std::vector<unsigned> graph_sizes({100, 500, 1000, 5000, 10000}); //  50000, 50000, 100000, 100000});
    std::vector<unsigned> graph_chances({10, 5, 8, 8, 2}); // 3, 10, 3, 10});

    assert( graph_chances.size() == graph_sizes.size() );

    for (size_t i = 0; i< graph_sizes.size(); i++ ) {
        DAG graph = erdos_renyi_graph_gen(graph_sizes[i], graph_chances[i] );
    
        CoarsenHistory contraction_chain(graph);

        auto start = std::chrono::high_resolution_clock::now();
        contraction_chain.run_dag_evolution();
        auto end = std::chrono::high_resolution_clock::now();
        double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        time_taken *= 1e-9;

        for (auto& dag : contraction_chain.retrieve_dag_evolution()) {
            BOOST_CHECK( dag.is_acyclic() );
        }

        std::cout << "Initial Size: " << contraction_chain.retrieve_dag_evolution()[0].n
            << ", Final Size: " << contraction_chain.retrieve_dag_evolution().back().n
            << ", Components: " << contraction_chain.retrieve_dag_evolution().back().weakly_connected_components().size() 
            << ", Chain length: " << contraction_chain.retrieve_dag_evolution().size()
            << ", Time: " << time_taken << "s" << std::endl;

        std::vector<DAG> dag_seq = contraction_chain.retrieve_dag_evolution();
        std::vector<std::unordered_map<int, int>> contr_seq = contraction_chain.retrieve_contr_maps();
        std::vector<std::unordered_map<int, std::unordered_set<int>>> exp_seq = contraction_chain.retrieve_expansion_maps();

        BOOST_CHECK_EQUAL( dag_seq.size()-1, exp_seq.size() );
        BOOST_CHECK_EQUAL( contr_seq.size(), exp_seq.size() );

        for (long unsigned i = 0; i < dag_seq.size(); i++) {
            for (int j = 0; j < dag_seq[i].n; j++) {
                if (i != dag_seq.size()-1) {
                    BOOST_CHECK( contr_seq[i].find(j) != contr_seq[i].cend() );
                    int post_j = contr_seq[i].at(j);
                    BOOST_CHECK( exp_seq[i].find(post_j) != exp_seq[i].cend() );
                    BOOST_CHECK( exp_seq[i].at(post_j).find(j) != exp_seq[i].at(post_j).cend() );
                }
                if ( i != 0 ) {
                    BOOST_CHECK( exp_seq[i-1].find(j) != exp_seq[i-1].cend() );
                    BOOST_CHECK( exp_seq[i-1].size() != 0 );
                    for (auto& node : exp_seq[i-1].at(j)) {
                        BOOST_CHECK_EQUAL( contr_seq[i-1].at(node), j );
                    }

                    BOOST_CHECK_EQUAL( dag_seq[i].workW[j], dag_seq[i-1].workW_of_node_set( exp_seq[i-1].at(j) ) );
                }
            }
        }
    }        
}