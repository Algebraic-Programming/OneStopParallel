#define BOOST_TEST_MODULE simple_partitioners
#include <boost/test/unit_test.hpp>

#include "scheduler/Partitioners/partitioners.hpp"


BOOST_AUTO_TEST_CASE(KarmarkarKarp_test_0) {
    KK_Tracking wt0( 8, 0 );
    KK_Tracking wt1( 7, 1 );
    KK_Tracking wt2( 8, 2 );

    KK_Tracking wt3 = KK_Tracking::take_difference(wt1, wt0);
    KK_Tracking wt4 = KK_Tracking::take_difference(wt0, wt1);

    KK_Tracking wt5 = KK_Tracking::take_difference(wt3, wt2);

    BOOST_CHECK( wt0 > wt1 );
    BOOST_CHECK( ! (wt0 > wt2) );

    BOOST_CHECK_EQUAL( wt3.weight, 1 );
    BOOST_CHECK( wt3.positive == std::vector<unsigned>({0}) );
    BOOST_CHECK( wt3.negative == std::vector<unsigned>({1}) );
    BOOST_CHECK( wt3.weight == wt4.weight );
    BOOST_CHECK( wt3.positive == wt4.positive );
    BOOST_CHECK( wt3.negative == wt4.negative );

    BOOST_CHECK_EQUAL( wt5.weight, 7 );
    BOOST_CHECK( wt5.positive == std::vector<unsigned>({2,1}) );
    BOOST_CHECK( wt5.negative == std::vector<unsigned>({0}) );
}


BOOST_AUTO_TEST_CASE(KarmarkarKarp_test_1) {
    std::multiset<int, std::greater<int>> weights({20,10,6,5,2});

    std::vector<unsigned> allocation1 = kk_partitioner(1, weights);
    std::vector<unsigned> allocation_ans1(weights.size(), 0);
    BOOST_CHECK( allocation1 == allocation_ans1 );

    std::vector<unsigned> allocation2 = kk_partitioner(2, weights);
    std::vector<unsigned> allocation_ans2({1,0,0,0,1});
    BOOST_CHECK( allocation2 == allocation_ans2 );

    BOOST_CHECK_THROW( kk_partitioner(3, weights), std::logic_error  );

    std::vector<unsigned> allocation4 = kk_partitioner(4, weights);
    std::vector<unsigned> allocation_ans4({3,0,1,1,2});
    BOOST_CHECK( allocation4 == allocation_ans4 );
    
}



BOOST_AUTO_TEST_CASE(Greedy_partitioner_1) {
    std::multiset<int, std::greater<int>> weights({5,3,6,3,1,2});

    std::vector<unsigned> allocation1 = greedy_partitioner(1, weights);
    std::vector<unsigned> allocation_ans1(weights.size(), 0);
    BOOST_CHECK( allocation1 == allocation_ans1 );

    std::vector<unsigned> allocation2 = greedy_partitioner(2, weights);
    std::vector<unsigned> allocation_ans2({0,1,1,0,1,0});
    BOOST_CHECK( allocation2 == allocation_ans2 );

    std::vector<unsigned> allocation3 = greedy_partitioner(3, weights);
    std::vector<unsigned> allocation_ans3({0,1,2,2,1,0});
    BOOST_CHECK( allocation3 == allocation_ans3 );
}

BOOST_AUTO_TEST_CASE(Greedy_partitioner_2) {
    std::multiset<int, std::greater<int>> weights({5,23,7,3,1,4});

    std::vector<unsigned> allocation1 = greedy_partitioner(1, weights);
    std::vector<unsigned> allocation_ans1(weights.size(), 0);
    BOOST_CHECK( allocation1 == allocation_ans1 );

    std::vector<unsigned> allocation2 = greedy_partitioner(2, weights);
    std::vector<unsigned> allocation_ans2({0,1,1,1,1,1});
    BOOST_CHECK( allocation2 == allocation_ans2 );

    std::vector<unsigned> allocation3 = greedy_partitioner(3, weights);
    std::vector<unsigned> allocation_ans3({0,1,2,2,1,2});
    BOOST_CHECK( allocation3 == allocation_ans3 );
}

BOOST_AUTO_TEST_CASE(Greedy_partitioner_3) {
    std::vector<unsigned> sizes({9846, 34, 354, 73, 2438, 2302});

    for (auto& n : sizes) {
        std::vector<int> all_ones(n,1);
        std::multiset<int, std::greater<int>> weights(all_ones.begin(), all_ones.end());

        for (unsigned j = 1; j< 25; j++ ) {
            std::vector<unsigned> allocation = greedy_partitioner(j, weights);

            std::vector<unsigned> counter(j,0);
            for (auto& loc : allocation) {
                counter[loc]++;
            }
            
            unsigned div = n/j;
            unsigned remainder = n%j;
            for (unsigned i = 0; i < j ; i++) {
                unsigned additional = i < remainder ? 1 : 0; 
                BOOST_CHECK_EQUAL( counter[i] , div+additional  );
            }
        }
    }
}



BOOST_AUTO_TEST_CASE(Hill_Climbing_1) {
    std::multiset<int, std::greater<int>> weights({5,3,6,3,1,2,3,4,7,2,23,23,2,6,2,4,7,3,7,34,24,4,23,7,1,1,4,5,4,4,4,15,8,5,35});

    for (int j = 1; j < 9; j++) {
        std::vector<unsigned> allocation(weights.size());
        for (size_t i = 0 ; i < allocation.size(); i++) {
            allocation[i] = randInt(j);
        }

        auto answer = hill_climb_weight_balance_single_superstep(weights.size()*100, j, weights, allocation);
        
        for (auto& loc: answer.second) {
            BOOST_CHECK_LE( 0, loc );
            BOOST_CHECK_LE( loc , j-1 );
        }
        BOOST_CHECK_LE( 1, answer.first );
        BOOST_CHECK_LE( answer.first, 2 );
    }
}




BOOST_AUTO_TEST_CASE(Hill_Climbing_2a) {
    std::vector<int> set_sizes({100,500,1000});

    for (auto& quant : set_sizes) {
        std::multiset<int, std::greater<int>> weights;
        for (int i = 0; i < quant ; i++) {
            weights.emplace(1+randInt(11)+randInt(11)+randInt(11));
        }
        // std::cout << "Weightsize: " << weights.size() << " weights: ";
        // for (auto& wt: weights) {
        //     std::cout << wt << " ";
        // }
        // std::cout << stds::endl;
        for (int j = 1; j < 21; j++) {
            std::vector<unsigned> allocation(weights.size());
            for (size_t i = 0 ; i < allocation.size(); i++) {
                allocation[i] = randInt(j);
            }

            auto answer = hill_climb_weight_balance_single_superstep(weights.size()*5, j, weights, allocation);
            
            for (auto& loc: answer.second) {
                BOOST_CHECK_LE( 0, loc );
                BOOST_CHECK_LE( loc , j-1 );
            }
            BOOST_CHECK_LE( 1, answer.first );
            BOOST_CHECK_LE( answer.first, 1.1 );
        }
    }
}


BOOST_AUTO_TEST_CASE(Hill_Climbing_2b) {
    std::vector<int> set_sizes({100,500,1000});

    for (auto& quant : set_sizes) {
        std::multiset<int, std::greater<int>> weights;
        for (int i = 0; i < quant ; i++) {
            weights.emplace(1+randInt(11)+randInt(11)+randInt(11));
        }
        for (int j = 1; j < 21; j++) {
            std::vector<unsigned> allocation(weights.size());
            for (size_t i = 0 ; i < allocation.size(); i++) {
                allocation[i] = 0;
            }

            auto answer = hill_climb_weight_balance_single_superstep(weights.size()*5, j, weights, allocation);
            
            for (auto& loc: answer.second) {
                BOOST_CHECK_LE( 0, loc );
                BOOST_CHECK_LE( loc , j-1 );
            }
            BOOST_CHECK_LE( 1, answer.first );
            BOOST_CHECK_LE( answer.first, 1.1 );
        }
    }
}



BOOST_AUTO_TEST_CASE(Hill_Climbing_3a) {
    std::vector<int> set_sizes({200,500,1000});

    for (auto& quant : set_sizes) {
        std::multiset<int, std::greater<int>> weights;
        for (int i = 0; i < quant ; i++) {
            weights.emplace(randInt(5));
        }
        for (int j = 1; j < 21; j++) {
            std::vector<unsigned> allocation(weights.size());
            for (size_t i = 0 ; i < allocation.size(); i++) {
                allocation[i] = randInt(j);
            }

            auto answer = hill_climb_weight_balance_single_superstep(weights.size()*5, j, weights, allocation);
            
            for (auto& loc: answer.second) {
                BOOST_CHECK_LE( 0, loc );
                BOOST_CHECK_LE( loc , j-1 );
            }
            BOOST_CHECK_LE( 1, answer.first );
            BOOST_CHECK_LE( answer.first, 1.1 );
        }
    }
}




BOOST_AUTO_TEST_CASE(Hill_Climbing_3b) {
    std::vector<int> set_sizes({200,500,1000});

    for (auto& quant : set_sizes) {
        std::multiset<int, std::greater<int>> weights;
        for (int i = 0; i < quant ; i++) {
            weights.emplace(randInt(5));
        }
        for (int j = 1; j < 21; j++) {
            std::vector<unsigned> allocation(weights.size());
            for (size_t i = 0 ; i < allocation.size(); i++) {
                allocation[i] = 0;
            }

            auto answer = hill_climb_weight_balance_single_superstep(weights.size()*5, j, weights, allocation);
            
            for (auto& loc: answer.second) {
                BOOST_CHECK_LE( 0, loc );
                BOOST_CHECK_LE( loc , j-1 );
            }
            BOOST_CHECK_LE( 1, answer.first );
            BOOST_CHECK_LE( answer.first, 1.1 );
        }
    }
}




BOOST_AUTO_TEST_CASE(Hill_Climbing_4) {
    std::vector<unsigned> sizes({9846, 34, 354, 73, 2478, 602});

    for (auto& n : sizes) {
        std::vector<int> all_ones(n,1);
        std::multiset<int, std::greater<int>> weights(all_ones.begin(), all_ones.end());

        for (int j = 1; j< 13; j++ ) {
            std::vector<unsigned> allocation(n,0);
            auto answer = hill_climb_weight_balance_single_superstep(n, j, weights, allocation);
            allocation = answer.second;

            std::vector<unsigned> counter(j,0);
            for (auto& loc : allocation) {
                counter[loc]++;
            }
            
            unsigned div = n/j;
            for (int i = 0; i < j ; i++) { 
                BOOST_CHECK_LE( counter[i] , div+1  );
                BOOST_CHECK_GE( counter[i] , div  );
            }
        }
    }
}


/* //ILP test require a license, cannot be tested on the gitlab server
BOOST_AUTO_TEST_CASE(ILP_0) {
    std::vector<unsigned> sizes({5, 73});

    for (auto& n : sizes) {
        std::vector<int> all_ones(n,1);
        std::multiset<int, std::greater<int>> weights(all_ones.begin(), all_ones.end());

        for (int j = 1; j< 13; j++ ) {
            std::vector<unsigned> allocation = ilp_partitioner(j, weights);

            std::vector<unsigned> counter(j,0);
            for (auto& loc : allocation) {
                counter[loc]++;
            }
            
            unsigned div = n/j;
            for (int i = 0; i < j ; i++) { 
                BOOST_CHECK_LE( counter[i] , div+1  );
            }
        }
    }
}



BOOST_AUTO_TEST_CASE(ILP_1) {
    std::vector<unsigned> sizes({46, 34, 154, 73, 78, 62, 5});

    for (auto& n : sizes) {
        std::vector<int> all_ones(n,1);
        std::multiset<int, std::greater<int>> weights(all_ones.begin(), all_ones.end());

        for (int j = 1; j< 13; j++ ) {
            std::vector<unsigned> allocation = ilp_partitioner(j, weights);

            std::vector<unsigned> counter(j,0);
            for (auto& loc : allocation) {
                counter[loc]++;
            }
            
            unsigned div = n/j;
            for (int i = 0; i < j ; i++) { 
                BOOST_CHECK_LE( counter[i] , div+1  );
            }
        }
    }
}
*/
