#define BOOST_TEST_MODULE Min_sym_sub_sum
#include <boost/test/unit_test.hpp>

#include "scheduler/SubArchitectureSchedulers/SubArchitectures.hpp"

BOOST_AUTO_TEST_CASE(min_sym_sub_sum1) {
    std::vector<std::vector<unsigned>> matrix({{0,1,3},{1,0,3},{3,3,0}});
    std::vector<unsigned> ans = SubArchitectureScheduler::min_symmetric_sub_sum(matrix, 2);
    std::vector<unsigned> sol({0,1});
    // for (auto num : ans) {
    //     std::cout << num << " ";
    // }
    // std::cout << std::endl;
    BOOST_CHECK( std::is_permutation( ans.begin(), ans.end(), sol.begin(), sol.end() ) );
};

BOOST_AUTO_TEST_CASE(min_sym_sub_sum2) {
    std::vector<std::vector<unsigned>> matrix({{0,1,3,1},{1,0,3,1},{3,3,0,3},{1,1,3,0}});
    std::vector<unsigned> ans = SubArchitectureScheduler::min_symmetric_sub_sum(matrix, 3);
    std::vector<unsigned> sol({0,1,3});
    // for (auto num : ans) {
    //     std::cout << num << " ";
    // }
    // std::cout << std::endl;
    BOOST_CHECK( std::is_permutation( ans.begin(), ans.end(), sol.begin(), sol.end() ) );
};

BOOST_AUTO_TEST_CASE(min_sym_sub_sum3) {
    std::vector<std::vector<unsigned>> matrix({{0,3,1,3},{3,0,3,3},{1,3,0,3},{3,3,3,0}});
    std::vector<unsigned> ans = SubArchitectureScheduler::min_symmetric_sub_sum(matrix, 2);
    std::vector<unsigned> sol({0,2});
    // for (auto num : ans) {
    //     std::cout << num << " ";
    // }
    // std::cout << std::endl;
    BOOST_CHECK( std::is_permutation( ans.begin(), ans.end(), sol.begin(), sol.end() ) );
};