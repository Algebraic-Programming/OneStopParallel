#define BOOST_TEST_MODULE DAG
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <unordered_set>
#include <vector>
#include <algorithm>

#include "auxiliary/auxiliary.hpp"
#include "structures/dag.hpp"

BOOST_AUTO_TEST_CASE(DAG1) {
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
    const DAG graph_empty;

    std::cout << "Checking sizes:" << std::endl;

    BOOST_CHECK_EQUAL(graph.n, graph.In.size());
    BOOST_CHECK_EQUAL(graph.n, graph.Out.size());
    BOOST_CHECK_EQUAL(graph.n, graph.workW.size());
    BOOST_CHECK_EQUAL(graph.n, graph.commW.size());

    BOOST_CHECK_EQUAL(graph_empty.n, graph_empty.In.size());
    BOOST_CHECK_EQUAL(graph_empty.n, graph_empty.Out.size());
    BOOST_CHECK_EQUAL(graph_empty.n, graph_empty.workW.size());
    BOOST_CHECK_EQUAL(graph_empty.n, graph_empty.commW.size());

    std::cout << "Checking workW sums:" << std::endl;
    BOOST_CHECK_EQUAL(graph.workW_of_node_set({0, 1}), 2);
    BOOST_CHECK_EQUAL(graph.workW_of_node_set({5, 3}), 4);
    BOOST_CHECK_EQUAL(graph.workW_of_node_set({}), 0);
    BOOST_CHECK_EQUAL(graph.workW_of_node_set({0, 1, 2, 3, 4, 5}), 9);

    BOOST_CHECK_EQUAL(graph_empty.workW_of_node_set({}), 0);

    std::cout << "Checking every in edge is contained in out edge:" << std::endl;

    int num_edges = 0;
    for (unsigned i = 0; i < graph.n; i++) {
        num_edges += graph.In[i].size();
        for (auto &j : graph.In[i]) {
            BOOST_CHECK(std::any_of(graph.Out[j].cbegin(), graph.Out[j].cend(), [i](int k) { return k == i; }));
        }
    }

    std::cout << "Checking every out edge is contained in in edge:" << std::endl;

    for (unsigned i = 0; i < graph.n; i++) {
        for (auto &j : graph.Out[i]) {
            BOOST_CHECK(std::any_of(graph.In[j].cbegin(), graph.In[j].cend(), [i](int k) { return k == i; }));
        }
    }

    std::cout << "Checking every in edge is contained in out edge:" << std::endl;

    for (unsigned i = 0; i < graph_empty.n; i++) {
        for (auto &j : graph_empty.In[i]) {
            BOOST_CHECK(
                std::any_of(graph_empty.Out[j].cbegin(), graph_empty.Out[j].cend(), [i](int k) { return k == i; }));
        }
    }

    std::cout << "Checking every out edge is contained in in edge:" << std::endl;

    for (unsigned i = 0; i < graph_empty.n; i++) {
        for (auto &j : graph_empty.Out[i]) {
            BOOST_CHECK(
                std::any_of(graph_empty.In[j].cbegin(), graph_empty.In[j].cend(), [i](int k) { return k == i; }));
        }
    }

    std::cout << "Checking topological order:" << std::endl;
    std::vector<int> top_order = graph.GetTopOrder();
    BOOST_CHECK(top_order.size() == graph.n);
    BOOST_CHECK(graph_empty.GetTopOrder().size() == graph_empty.n);

    std::vector<size_t> index_in_top_order = sorting_arrangement(top_order);
    for (unsigned i = 0; i < graph.n; i++) {
        for (auto &j : graph.Out[i]) {
            BOOST_CHECK_LT(index_in_top_order[i], index_in_top_order[j]);
        }
    }

    std::set<int> all_nodes;
    for (unsigned i = 0; i < graph.n; i++) {
        all_nodes.emplace(i);
    }

    std::set<int> nodes_a({8, 0});
    std::set<int> nodes_b({6, 2, 5, 3});
    std::set<int> nodes_c({6, 9, 1});

    std::vector<bool> bool_a(graph.n, false);
    std::vector<bool> bool_b(graph.n, false);
    std::vector<bool> bool_c(graph.n, false);

    for (auto &i : nodes_a) {
        bool_a[i] = true;
    }
    for (auto &i : nodes_b) {
        bool_b[i] = true;
    }
    for (auto &i : nodes_c) {
        bool_c[i] = true;
    }

    BOOST_CHECK(graph.GetFilteredTopOrder(bool_a) == std::vector<int>({0, 8}) ||
                graph.GetFilteredTopOrder(bool_a) == std::vector<int>({8, 0}));
    BOOST_CHECK(graph.GetFilteredTopOrder(bool_b)[3] == 2);
    BOOST_CHECK(graph.GetFilteredTopOrder(bool_c) == std::vector<int>({9, 6, 1}));

    std::cout << "Checking longest path:" << std::endl;
    BOOST_CHECK_EQUAL(graph.getLongestPath(all_nodes), 4);
    BOOST_CHECK_EQUAL(graph.getLongestPath(nodes_a), 0);
    BOOST_CHECK_EQUAL(graph.getLongestPath(nodes_b), 1);
    BOOST_CHECK_EQUAL(graph.getLongestPath(nodes_c), 2);

    BOOST_CHECK_EQUAL(graph_empty.getLongestPath({}), 0);

    std::vector<int> longest_path = graph.longest_chain();

    std::vector<int> long_chain1({9, 6, 2, 0, 7});
    std::vector<int> long_chain2({9, 5, 2, 0, 7});

    std::cout << "Longest path: ";
    for (auto &i : longest_path) {
        std::cout << i << ", ";
    }
    std::cout << std::endl;

    BOOST_CHECK_EQUAL(graph.getLongestPath(all_nodes) + 1, graph.longest_chain().size());
    BOOST_CHECK(longest_path == long_chain1 || longest_path == long_chain2);

    BOOST_CHECK(graph_empty.longest_chain() == std::vector<int>({}));

    // std::cout << "Checking contractable edges:" << std::endl;

    // const std::map<intPair, bool> truth_table = graph.GetContractableEdges();

    // BOOST_CHECK(truth_table.size() == num_edges);
    // for (auto const &[edge, _] : truth_table) {
    //     BOOST_CHECK(
    //         std::any_of(graph.Out[edge.a].cbegin(), graph.Out[edge.a].cend(), [edge](int j) { return j == edge.b;
    //         }));
    // }

    // std::cout << "Edge keys in GetContractableEdges: ";
    // for (auto const &[edge, value] : truth_table) {
    //     std::cout << "(" << edge.a << ", " << edge.b << ", " << value << "), ";
    // }
    // std::cout << std::endl;

    // std::vector<intPair> edge_collection;
    // std::vector<bool> contrble;
    // edge_collection.emplace_back(0, 7);
    // contrble.emplace_back(true);

    // edge_collection.emplace_back(6, 1);
    // contrble.emplace_back(true);

    // edge_collection.emplace_back(9, 1);
    // contrble.emplace_back(false);

    // edge_collection.emplace_back(5, 2);
    // contrble.emplace_back(true);

    // edge_collection.emplace_back(6, 0);
    // contrble.emplace_back(false);

    // for (int i = 0; i < edge_collection.size(); i++) {
    //     BOOST_CHECK(truth_table.at(edge_collection[i]) == contrble[i]);
    // }

    std::cout << "Checking ancestors and descendants:" << std::endl;

    BOOST_CHECK(graph.ancestors(9) == std::unordered_set<int>({9}));
    BOOST_CHECK(graph.ancestors(2) == std::unordered_set<int>({2, 6, 9, 5, 3}));
    BOOST_CHECK(graph.ancestors(4) == std::unordered_set<int>({4, 8}));
    BOOST_CHECK(graph.ancestors(5) == std::unordered_set<int>({5, 9}));

    BOOST_CHECK(graph.descendants(9) == std::unordered_set<int>({9, 5, 6, 1, 2, 0, 7}));
    BOOST_CHECK(graph.descendants(3) == std::unordered_set<int>({3, 2, 0, 7}));
    BOOST_CHECK(graph.descendants(0) == std::unordered_set<int>({0, 7}));
    BOOST_CHECK(graph.descendants(8) == std::unordered_set<int>({8, 4}));
    BOOST_CHECK(graph.descendants(4) == std::unordered_set<int>({4}));

    std::cout << "Checking weakly connected components:" << std::endl;

    std::vector<std::unordered_set<int>> weakly_connected = graph.weakly_connected_components();
    std::vector<std::unordered_set<int>> ans1(2, std::unordered_set<int>());
    ans1[0] = {8, 4};
    ans1[1] = {3, 9, 6, 5, 2, 1, 0, 7};
    std::vector<std::unordered_set<int>> ans2(2, std::unordered_set<int>());
    ans2[0] = {3, 9, 6, 5, 2, 1, 0, 7};
    ans2[1] = {8, 4};
    BOOST_CHECK(weakly_connected == ans1 || weakly_connected == ans2);


    weakly_connected = graph.weakly_connected_components({8,4, 6,2,5});
    ans1[0] = {8, 4};
    ans1[1] = {6,2,5};

    ans2[0] = {6,2,5};
    ans2[1] = {8, 4};
    BOOST_CHECK(weakly_connected == ans1 || weakly_connected == ans2);

    weakly_connected = graph.weakly_connected_components({1, 6, 7});
    ans1[0] = {1,6};
    ans1[1] = {7};
    
    ans2[0] = {7};
    ans2[1] = {1,6};
    BOOST_CHECK(weakly_connected == ans1 || weakly_connected == ans2);

    weakly_connected = graph.weakly_connected_components({3, 9, 5, 0, 7, 4});
    std::vector<std::unordered_set<int>> correct_ans(3, std::unordered_set<int>());
    correct_ans[0] = {4};
    correct_ans[1] = {3};
    correct_ans[2] = {9, 5, 0, 7};
    BOOST_CHECK( std::is_permutation(weakly_connected.begin(), weakly_connected.end(), correct_ans.begin()) );


    std::vector<std::unordered_set<int>> ans3({{}});
    BOOST_CHECK(graph_empty.weakly_connected_components() == ans3);

    // WAS DELETED FROM HERE
    std::cout << "Checking Top and Bottom node distance:" << std::endl;

    std::vector<int> top_dist({4, 3, 3, 1, 2, 2, 2, 5, 1, 1});
    std::vector<int> bottom_dist({2, 1, 3, 4, 1, 4, 4, 1, 2, 5});

    // for (auto& top : graph.get_bottom_node_distance() ) {
    //     std::cout << top << " ";
    // }

    BOOST_CHECK(graph.get_top_node_distance() == top_dist);
    BOOST_CHECK(graph.get_bottom_node_distance() == bottom_dist);

    DAG graph_second;

    graph_second.n = 7;
    // Side-question: If n is equal to the size of the vectors, why do we need to store it?
    // Because it is redundant!
    graph_second.In = {
        {}, {0}, {0}, {1}, {1, 2}, {2}, {3, 5},
    };
    graph_second.Out = {
        {1, 2}, {3, 4}, {4, 5}, {6}, {}, {6}, {},
    };
    graph_second.workW = {1, 1, 1, 1, 1, 1, 3};
    graph_second.commW = graph_second.workW;

    std::vector<int> top_dist_second({1, 2, 2, 3, 3, 3, 4});
    std::vector<int> bottom_dist_second({4, 3, 3, 2, 1, 2, 1});

    // for (auto& top : graph.get_bottom_node_distance() ) {
    //     std::cout << top << " ";
    // }

    BOOST_CHECK(graph_second.get_top_node_distance() == top_dist_second);
    BOOST_CHECK(graph_second.get_bottom_node_distance() == bottom_dist_second);


    std::cout << "Checking strict poset integer map:" << std::endl;

    std::vector<double> poisson_params({0, 0.08, 0.1, 0.2, 0.5, 1, 4});

    for (int loops = 0 ; loops < 10; loops++) {
        for (int noise = 0; noise<6; noise++) {
            for (auto& pois_para: poisson_params) {
                std::vector<int> poset_int_map = graph.get_strict_poset_integer_map(noise, pois_para);
                for (unsigned i = 0; i<graph.n ; i++) {
                    for (auto& j : graph.Out[i]) {
                        BOOST_CHECK_LE( poset_int_map[i]+1, poset_int_map[j] );
                    }
                }
            }
        }
    }

    std::cout << "Poset int map:" << std::endl;;
    std::vector<int> poset_int_map_out = graph.get_strict_poset_integer_map(0, 0);
    for (unsigned i = 0; i < graph.n ; i++ ) {
        std::cout<< "Node: " << i << " Value: " << poset_int_map_out[i] << std::endl;
    }


    std::cout << "Checking common parents and children:" << std::endl;

    BOOST_CHECK_EQUAL( graph.count_common_parents_plus_common_children(9,9) , 3 );
    BOOST_CHECK_EQUAL( graph.count_common_parents_plus_common_children(4,4) , 1 );
    BOOST_CHECK_EQUAL( graph.count_common_parents_plus_common_children(2,2) , 4 );
    BOOST_CHECK_EQUAL( graph.count_common_parents_plus_common_children(9,2) , 0 );
    BOOST_CHECK_EQUAL( graph.count_common_parents_plus_common_children(2,3) , 0 );
    BOOST_CHECK_EQUAL( graph.count_common_parents_plus_common_children(2,5) , 1 );
    BOOST_CHECK_EQUAL( graph.count_common_parents_plus_common_children(6,5) , 3 );
    BOOST_CHECK_EQUAL( graph.count_common_parents_plus_common_children(0,1) , 1 );

    for (unsigned i = 0; i<graph.n ; i++) {
        for (unsigned j = 0; j< graph.n; j++) {
            BOOST_CHECK_EQUAL( graph.count_common_parents_plus_common_children(i,j) , graph.count_common_parents_plus_common_children(j, i) );
            BOOST_CHECK_LE( 0,  graph.count_common_parents_plus_common_children(i,j));
            BOOST_CHECK_LE( graph.count_common_parents_plus_common_children(i,j) , std::min( graph.In[i].size()+graph.Out[i].size(), graph.In[j].size()+graph.Out[j].size() ));
        }
    }

    std::cout << "Checking contractable edge list" << std::endl;

    contract_edge_sort edge_sort = Contract_Edge_Decrease;
    std::multiset<Edge_Weighted, Edge_Weighted::Comparator> contr_edges = graph.get_contractable_edges(edge_sort, graph.get_strict_poset_integer_map() );
    std::unordered_multiset<std::pair<int, int>, pair_hash> contr_edge_set;

    for (auto& wtd_edge : contr_edges) {
        contr_edge_set.emplace(wtd_edge.edge_pair);
        std::unordered_set<int> desc = graph.descendants(wtd_edge.edge_pair.first);
        std::unordered_set<int> anc = graph.ancestors(wtd_edge.edge_pair.second);
        std::unordered_set<int> intersec = get_intersection(desc,anc);
        std::unordered_set<int> supposed_intersec( {wtd_edge.edge_pair.first, wtd_edge.edge_pair.second} );
        BOOST_CHECK( intersec == supposed_intersec );
    }

    for (unsigned i = 0; i < graph.n; i++) {
        for (unsigned j = 0; j< graph.n; j++) {
            std::unordered_set<int> desc = graph.descendants(i);
            std::unordered_set<int> anc = graph.ancestors(j);
            std::unordered_set<int> intersec = get_intersection(desc,anc);
            std::unordered_set<int> supposed_intersec( {i,j} );
            std::pair<int, int> edge_pair({i,j});
            if (intersec != supposed_intersec) {
                BOOST_CHECK( contr_edge_set.find(edge_pair) == contr_edge_set.cend() );
            }
        }
    }

    std::cout << "DAG contraction" << std::endl;

    const std::vector<std::unordered_set<int>> contr_partition1({{2,3,0}});
    const std::pair<DAG, std::unordered_map<int, int >> contraction_pair1 = graph.contracted_graph_without_loops(contr_partition1);
    const DAG* con_graph1 = &contraction_pair1.first;
    const std::unordered_map<int, int >* con_map1 = &contraction_pair1.second;

    BOOST_CHECK_EQUAL( (*con_graph1).comm_edge_W.at(std::make_pair(con_map1->at( 5 ), con_map1->at(2))) , 2 );
    BOOST_CHECK_EQUAL( (*con_graph1).comm_edge_W.at(std::make_pair(con_map1->at( 8 ), con_map1->at(4))) , graph.comm_edge_W.at(std::make_pair(8,4)) );
    BOOST_CHECK( (*con_graph1).comm_edge_W.find( std::make_pair(con_map1->at( 9 ), con_map1->at(2))) == (*con_graph1).comm_edge_W.cend()  );
    BOOST_CHECK_EQUAL( (*con_graph1).workW[con_map1->at( 2 )] , 3 );
    BOOST_CHECK_EQUAL( (*con_graph1).workW[con_map1->at( 0 )] , 3 );
    BOOST_CHECK_EQUAL( (*con_graph1).commW[con_map1->at( 3 )] , 3 );
    BOOST_CHECK_EQUAL( (*con_graph1).Out[con_map1->at( 5 )][0] , con_map1->at( 3 ) );
    BOOST_CHECK_EQUAL( (*con_graph1).Out[con_map1->at( 5 )].size() , 1 );
    BOOST_CHECK_EQUAL( (*con_graph1).In[con_map1->at( 0 )].size() , 2 );



    const std::vector<std::unordered_set<int>> contr_partition2({{9,6},{2,5},{8,4}});
    const std::pair<DAG, std::unordered_map<int, int >> contraction_pair2 = graph.contracted_graph_without_loops(contr_partition2);
    const DAG* con_graph2 = &contraction_pair2.first;
    const std::unordered_map<int, int >* con_map2 = &contraction_pair2.second;

    BOOST_CHECK_EQUAL( (*con_graph2).Out[con_map2->at( 9 )].size() , 3 );
    BOOST_CHECK_EQUAL( (*con_graph2).Out[con_map2->at( 8 )].size() , 0 );
    BOOST_CHECK_EQUAL( (*con_graph2).In[con_map2->at( 8 )].size() , 0 );
    BOOST_CHECK_EQUAL( (*con_graph2).In[con_map2->at( 0 )].size() , 2 );
    BOOST_CHECK_EQUAL( (*con_graph2).In[con_map2->at( 2 )].size() , 2 );

    std::vector<int> Out9({con_map2->at( 5 ), con_map2->at( 0), con_map2->at( 1 )});
    BOOST_CHECK( std::is_permutation((*con_graph2).Out[con_map2->at( 9 )].begin(), (*con_graph2).Out[con_map2->at( 9 )].end(), Out9.begin())   );

    std::vector<int> In5({con_map2->at( 3 ), con_map2->at( 9 )});
    BOOST_CHECK( std::is_permutation((*con_graph2).In[con_map2->at( 5 )].begin(), (*con_graph2).In[con_map2->at( 5 )].end(), In5.begin())   );

   

    std::cout << "graph acyclic check" << std::endl;

    BOOST_CHECK( graph.is_acyclic() == true );

    DAG acyclic_test1({},{},{},{});
    BOOST_CHECK( acyclic_test1.is_acyclic() == true );

    DAG acyclic_test2({{}},{{}},{1},{1});
    BOOST_CHECK( acyclic_test2.is_acyclic() == true );

    DAG acyclic_test3({{0}},{{0}},{1},{1});
    BOOST_CHECK( acyclic_test3.is_acyclic() == false );

    DAG acyclic_test4({{1},{0}},{{1},{0}},{1,1},{1,1});
    BOOST_CHECK( acyclic_test4.is_acyclic() == false );

    DAG acyclic_test5({{1},{2},{0},{4},{}},{{2},{0},{1},{},{3}},{1,1,1,1,1},{1,1,1,1,1});
    BOOST_CHECK( acyclic_test5.is_acyclic() == false );

    DAG acyclic_test6({{},{0,2},{0,1}},{{1,2},{2},{1}},{1,1,1},{1,1,1});
    BOOST_CHECK( acyclic_test6.is_acyclic() == false );

    DAG acyclic_test7({{},{0},{0,1}},{{1,2},{2},{}},{1,1,1},{1,1,1});
    BOOST_CHECK( acyclic_test7.is_acyclic() == true );

    DAG acyclic_test8({{},{0},{0,1,2}},{{1,2},{2},{2}},{1,1,1},{1,1,1});
    BOOST_CHECK( acyclic_test8.is_acyclic() == false );




























    // WAS DELETED TO HERE
    std::cout << "Checking SubDAG's and functions:" << std::endl;

    std::unordered_set<int> node_set0({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    std::unordered_set<int> node_set1({9, 5, 6, 2});
    std::unordered_set<int> node_set2({3, 5, 0, 7, 2});
    std::unordered_set<int> node_set3({});
    std::unordered_set<int> node_set4({8, 4, 3, 2});

    SubDAG graph0 = graph.toSubDAG();
    SubDAG graph0_copy(graph, node_set0);
    SubDAG graph1(graph, node_set1);
    SubDAG graph1_copy1(graph0, node_set1);
    SubDAG graph1_copy2(graph1, node_set1);
    SubDAG graph2(graph, node_set2);
    SubDAG graph3(graph0, node_set3);
    SubDAG graph4(graph0, node_set4);

    std::vector<SubDAG *> sub_dag_collection(
        {&graph0, &graph0_copy, &graph1, &graph1_copy1, &graph1_copy2, &graph2, &graph3, &graph4});

    std::cout << "Checking workW sums:" << std::endl;
    BOOST_CHECK_EQUAL(graph0.workW_of_node_set({0, 1}), 2);
    BOOST_CHECK_EQUAL(graph0_copy.workW_of_node_set({0, 1}), 2);

    BOOST_CHECK_EQUAL(graph0.workW_of_node_set({5, 3}), 4);
    BOOST_CHECK_EQUAL(graph0_copy.workW_of_node_set({5, 3}), 4);
    BOOST_CHECK_EQUAL(graph2.workW_of_node_set({5, 3}), 4);

    BOOST_CHECK_EQUAL(graph0.workW_of_node_set({}), 0);
    BOOST_CHECK_EQUAL(graph0_copy.workW_of_node_set({}), 0);
    BOOST_CHECK_EQUAL(graph1.workW_of_node_set({}), 0);
    BOOST_CHECK_EQUAL(graph1_copy1.workW_of_node_set({}), 0);
    BOOST_CHECK_EQUAL(graph1_copy2.workW_of_node_set({}), 0);
    BOOST_CHECK_EQUAL(graph2.workW_of_node_set({}), 0);
    BOOST_CHECK_EQUAL(graph3.workW_of_node_set({}), 0);
    BOOST_CHECK_EQUAL(graph4.workW_of_node_set({}), 0);

    BOOST_CHECK_EQUAL(graph2.workW_of_node_set({3, 2}), 2);
    BOOST_CHECK_EQUAL(graph4.workW_of_node_set({3, 2}), 2);

    for (auto &daggy : sub_dag_collection) {
        std::cout << "Checking every in edge is contained in out edge:" << std::endl;

        int num_edges = 0;
        for (unsigned i = 0; i < (*daggy).n; i++) {
            num_edges += (*daggy).In[i].size();
            for (auto &j : (*daggy).In[i]) {
                BOOST_CHECK(
                    std::any_of((*daggy).Out[j].cbegin(), (*daggy).Out[j].cend(), [i](int k) { return k == i; }));
            }
        }

        std::cout << "Checking every out edge is contained in in edge:" << std::endl;

        for (unsigned i = 0; i < (*daggy).n; i++) {
            for (auto &j : (*daggy).Out[i]) {
                BOOST_CHECK(std::any_of((*daggy).In[j].cbegin(), (*daggy).In[j].cend(), [i](int k) { return k == i; }));
            }
        }
    }

    std::cout << "Checking topological order:" << std::endl;

    for (auto &daggy : sub_dag_collection) {
        std::vector<int> top_order = (*daggy).GetTopOrder();
        BOOST_CHECK(top_order.size() == (*daggy).n);

        const std::vector<size_t> index_in_top_order = sorting_arrangement(top_order);

        for (unsigned i = 0; i < (*daggy).n; i++) {
            for (const auto &j : (*daggy).Out[i]) {
                BOOST_CHECK_LT(index_in_top_order[i], index_in_top_order[j]);
            }
        }
    }

    std::cout << "Checking longest chain:" << std::endl;

    std::vector<int> longest_path0 = graph0.longest_chain();
    std::vector<int> longest_path0_copy = graph0_copy.longest_chain();

    std::vector<int> long_chain01({9, 6, 2, 0, 7});
    std::vector<int> long_chain02({9, 5, 2, 0, 7});

    BOOST_CHECK(longest_path0 == long_chain01 || longest_path0 == long_chain02);
    BOOST_CHECK(longest_path0_copy == long_chain01 || longest_path0_copy == long_chain02);

    std::vector<int> longest_path1 = graph1.longest_chain();
    std::vector<int> longest_path1_copy1 = graph1_copy1.longest_chain();
    std::vector<int> longest_path1_copy2 = graph1_copy2.longest_chain();

    std::vector<int> long_chain11({9, 6, 2});
    std::vector<int> long_chain12({9, 5, 2});

    BOOST_CHECK(longest_path1 == long_chain11 || longest_path1 == long_chain12);
    BOOST_CHECK(longest_path1_copy1 == long_chain11 || longest_path1_copy1 == long_chain12);
    BOOST_CHECK(longest_path1_copy2 == long_chain11 || longest_path1_copy2 == long_chain12);

    std::vector<int> longest_path2 = graph2.longest_chain();

    std::vector<int> long_chain21({3, 2, 0, 7});
    std::vector<int> long_chain22({5, 2, 0, 7});

    BOOST_CHECK(longest_path2 == long_chain21 || longest_path2 == long_chain22);

    std::vector<int> longest_path3 = graph3.longest_chain();

    std::vector<int> long_chain3({});

    BOOST_CHECK(longest_path3 == long_chain3);

    std::vector<int> longest_path4 = graph4.longest_chain();

    std::vector<int> long_chain41({8, 4});
    std::vector<int> long_chain42({3, 2});

    BOOST_CHECK(longest_path4 == long_chain41 || longest_path4 == long_chain42);

    std::cout << "Checking ancestors and descendants:" << std::endl;

    BOOST_CHECK(graph0.ancestors(9) == std::unordered_set<int>({9}));
    BOOST_CHECK(graph0.ancestors(2) == std::unordered_set<int>({2, 6, 9, 5, 3}));
    BOOST_CHECK(graph0.ancestors(4) == std::unordered_set<int>({4, 8}));
    BOOST_CHECK(graph0.ancestors(5) == std::unordered_set<int>({5, 9}));

    BOOST_CHECK(graph0.descendants(9) == std::unordered_set<int>({9, 5, 6, 1, 2, 0, 7}));
    BOOST_CHECK(graph0.descendants(3) == std::unordered_set<int>({3, 2, 0, 7}));
    BOOST_CHECK(graph0.descendants(0) == std::unordered_set<int>({0, 7}));
    BOOST_CHECK(graph0.descendants(8) == std::unordered_set<int>({8, 4}));
    BOOST_CHECK(graph0.descendants(4) == std::unordered_set<int>({4}));

    BOOST_CHECK(graph0_copy.ancestors(9) == std::unordered_set<int>({9}));
    BOOST_CHECK(graph0_copy.ancestors(2) == std::unordered_set<int>({2, 6, 9, 5, 3}));
    BOOST_CHECK(graph0_copy.ancestors(4) == std::unordered_set<int>({4, 8}));
    BOOST_CHECK(graph0_copy.ancestors(5) == std::unordered_set<int>({5, 9}));

    BOOST_CHECK(graph0_copy.descendants(9) == std::unordered_set<int>({9, 5, 6, 1, 2, 0, 7}));
    BOOST_CHECK(graph0_copy.descendants(3) == std::unordered_set<int>({3, 2, 0, 7}));
    BOOST_CHECK(graph0_copy.descendants(0) == std::unordered_set<int>({0, 7}));
    BOOST_CHECK(graph0_copy.descendants(8) == std::unordered_set<int>({8, 4}));
    BOOST_CHECK(graph0_copy.descendants(4) == std::unordered_set<int>({4}));

    BOOST_CHECK(graph1.ancestors(9) == std::unordered_set<int>({9}));
    BOOST_CHECK(graph1.ancestors(2) == std::unordered_set<int>({2, 6, 9, 5}));
    BOOST_CHECK(graph1.ancestors(5) == std::unordered_set<int>({5, 9}));

    BOOST_CHECK(graph1.descendants(9) == std::unordered_set<int>({9, 5, 6, 2}));
    BOOST_CHECK(graph1.descendants(5) == std::unordered_set<int>({5, 2}));

    BOOST_CHECK(graph1_copy1.ancestors(9) == std::unordered_set<int>({9}));
    BOOST_CHECK(graph1_copy1.ancestors(2) == std::unordered_set<int>({2, 6, 9, 5}));
    BOOST_CHECK(graph1_copy1.ancestors(5) == std::unordered_set<int>({5, 9}));

    BOOST_CHECK(graph1_copy1.descendants(9) == std::unordered_set<int>({9, 5, 6, 2}));
    BOOST_CHECK(graph1_copy1.descendants(5) == std::unordered_set<int>({5, 2}));

    BOOST_CHECK(graph1_copy2.ancestors(9) == std::unordered_set<int>({9}));
    BOOST_CHECK(graph1_copy2.ancestors(2) == std::unordered_set<int>({2, 6, 9, 5}));
    BOOST_CHECK(graph1_copy2.ancestors(5) == std::unordered_set<int>({5, 9}));

    BOOST_CHECK(graph1_copy2.descendants(9) == std::unordered_set<int>({9, 5, 6, 2}));
    BOOST_CHECK(graph1_copy2.descendants(5) == std::unordered_set<int>({5, 2}));

    BOOST_CHECK(graph2.ancestors(0) == std::unordered_set<int>({0, 2, 5, 3}));
    BOOST_CHECK(graph2.ancestors(2) == std::unordered_set<int>({2, 3, 5}));
    BOOST_CHECK(graph2.ancestors(5) == std::unordered_set<int>({5}));

    BOOST_CHECK(graph2.descendants(0) == std::unordered_set<int>({0, 7}));
    BOOST_CHECK(graph2.descendants(5) == std::unordered_set<int>({5, 2, 0, 7}));
    BOOST_CHECK(graph2.descendants(7) == std::unordered_set<int>({7}));

    BOOST_CHECK(graph4.ancestors(4) == std::unordered_set<int>({4, 8}));
    BOOST_CHECK(graph4.ancestors(3) == std::unordered_set<int>({3}));

    BOOST_CHECK(graph4.descendants(8) == std::unordered_set<int>({8, 4}));
    BOOST_CHECK(graph4.descendants(2) == std::unordered_set<int>({2}));

    std::cout << "Checking weakly connected components:" << std::endl;

    std::vector<std::unordered_set<int>> weakly_connected0 = graph0.weakly_connected_components();
    std::vector<std::unordered_set<int>> weakly_connected0_copy = graph0_copy.weakly_connected_components();
    std::vector<std::unordered_set<int>> ans01(2, std::unordered_set<int>());
    ans01[0] = {8, 4};
    ans01[1] = {3, 9, 6, 5, 2, 1, 0, 7};
    std::vector<std::unordered_set<int>> ans02(2, std::unordered_set<int>());
    ans02[0] = {3, 9, 6, 5, 2, 1, 0, 7};
    ans02[1] = {8, 4};
    BOOST_CHECK(weakly_connected0 == ans01 || weakly_connected0 == ans02);
    BOOST_CHECK(weakly_connected0_copy == ans01 || weakly_connected0_copy == ans02);

    std::vector<std::unordered_set<int>> weakly_connected1 = graph1.weakly_connected_components();
    std::vector<std::unordered_set<int>> weakly_connected1_copy1 = graph1_copy1.weakly_connected_components();
    std::vector<std::unordered_set<int>> weakly_connected1_copy2 = graph1_copy2.weakly_connected_components();
    std::vector<std::unordered_set<int>> ansi1(1, std::unordered_set<int>());
    ansi1[0] = {9, 5, 6, 2};
    BOOST_CHECK(weakly_connected1 == ansi1);
    BOOST_CHECK(weakly_connected1_copy1 == ansi1);
    BOOST_CHECK(weakly_connected1_copy2 == ansi1);

    std::vector<std::unordered_set<int>> weakly_connected2 = graph2.weakly_connected_components();
    std::vector<std::unordered_set<int>> ansi2(1, std::unordered_set<int>());
    ansi2[0] = {3, 5, 7, 2, 0};
    BOOST_CHECK(weakly_connected2 == ansi2);

    std::vector<std::unordered_set<int>> weakly_connected3 = graph3.weakly_connected_components();
    std::vector<std::unordered_set<int>> ansi3(1, std::unordered_set<int>());
    ansi3[0] = {};
    BOOST_CHECK(weakly_connected3 == ansi3);

    std::vector<std::unordered_set<int>> weakly_connected4 = graph4.weakly_connected_components();
    std::vector<std::unordered_set<int>> ans41(2, std::unordered_set<int>());
    std::vector<std::unordered_set<int>> ans42(2, std::unordered_set<int>());
    ans41[0] = {8, 4};
    ans41[1] = {3, 2};
    ans42[0] = {3, 2};
    ans42[1] = {8, 4};
    BOOST_CHECK(weakly_connected4 == ans41 || weakly_connected4 == ans42);


    weakly_connected = graph0.weakly_connected_components({8,4, 6,2,5});
    ans1[0] = {8, 4};
    ans1[1] = {6,2,5};

    ans2[0] = {6,2,5};
    ans2[1] = {8, 4};
    BOOST_CHECK(weakly_connected == ans1 || weakly_connected == ans2);

    weakly_connected = graph0.weakly_connected_components({1, 6, 7});
    ans1[0] = {1,6};
    ans1[1] = {7};
    
    ans2[0] = {7};
    ans2[1] = {1,6};
    BOOST_CHECK(weakly_connected == ans1 || weakly_connected == ans2);

    weakly_connected = graph0.weakly_connected_components({3, 9, 5, 0, 7, 4});
    correct_ans[0] = {4};
    correct_ans[1] = {3};
    correct_ans[2] = {9, 5, 0, 7};
    BOOST_CHECK( std::is_permutation(weakly_connected.begin(), weakly_connected.end(), correct_ans.begin()) );
}
