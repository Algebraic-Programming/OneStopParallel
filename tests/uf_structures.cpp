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

#define BOOST_TEST_MODULE Union_Find
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "osp/auxiliary/datastructures/union_find.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(UnionFindStructure1) {
    std::vector<std::string> names({"a", "b", "c", "d", "e", "f"});
    Union_Find_Universe<std::string, unsigned, int, int> testUniverse(names);

    for (auto &name : names) {
        BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name(name), name);
    }

    BOOST_CHECK_EQUAL(testUniverse.get_number_of_connected_components(), 6);

    BOOST_CHECK_THROW(testUniverse.add_object("a"), std::runtime_error);
    BOOST_CHECK_THROW(testUniverse.add_object("e"), std::runtime_error);

    BOOST_CHECK_EQUAL(testUniverse.get_number_of_connected_components(), 6);

    testUniverse.join_by_name("a", "b");
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("a"), testUniverse.find_origin_by_name("b"));
    BOOST_CHECK_EQUAL(testUniverse.get_number_of_connected_components(), 5);

    testUniverse.join_by_name("b", "c");
    BOOST_CHECK_EQUAL(testUniverse.get_number_of_connected_components(), 4);
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("a"), testUniverse.find_origin_by_name("b"));
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("a"), testUniverse.find_origin_by_name("c"));
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("b"), testUniverse.find_origin_by_name("c"));

    testUniverse.join_by_name("d", "b");
    BOOST_CHECK_EQUAL(testUniverse.get_number_of_connected_components(), 3);
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("a"), testUniverse.find_origin_by_name("d"));
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("d"), testUniverse.find_origin_by_name("b"));

    testUniverse.join_by_name("a", "c");
    BOOST_CHECK_EQUAL(testUniverse.get_number_of_connected_components(), 3);
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("a"), testUniverse.find_origin_by_name("c"));

    testUniverse.join_by_name("a", "d");
    BOOST_CHECK_EQUAL(testUniverse.get_number_of_connected_components(), 3);
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("a"), testUniverse.find_origin_by_name("d"));

    testUniverse.join_by_name("e", "f");
    BOOST_CHECK_EQUAL(testUniverse.get_number_of_connected_components(), 2);
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("e"), testUniverse.find_origin_by_name("f"));
    BOOST_CHECK_NE(testUniverse.find_origin_by_name("a"), testUniverse.find_origin_by_name("f"));

    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("a"), testUniverse.find_origin_by_name("b"));
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("b"), testUniverse.find_origin_by_name("c"));
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("c"), testUniverse.find_origin_by_name("d"));
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("a"), testUniverse.find_origin_by_name("d"));
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("c"), testUniverse.find_origin_by_name("b"));

    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("e"), testUniverse.find_origin_by_name("f"));

    BOOST_CHECK_NE(testUniverse.find_origin_by_name("a"), testUniverse.find_origin_by_name("f"));
}

BOOST_AUTO_TEST_CASE(UnionFindStructure2) {
    std::vector<std::string> names({"a", "b", "c", "d", "e", "f", "g", "h", "i"});
    Union_Find_Universe<std::string, unsigned, int, int> testUniverse;

    for (auto &name : names) {
        testUniverse.add_object(name);
    }

    for (auto &name : names) {
        BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name(name), name);
    }

    BOOST_CHECK_THROW(testUniverse.add_object("c"), std::runtime_error);
    BOOST_CHECK_THROW(testUniverse.add_object("i"), std::runtime_error);

    for (auto &name : names) {
        BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name(name), name);
    }

    BOOST_CHECK_EQUAL(testUniverse.get_number_of_connected_components(), 9);

    testUniverse.join_by_name("a", "b");
    BOOST_CHECK_EQUAL(testUniverse.get_number_of_connected_components(), 8);
    testUniverse.join_by_name("b", "c");
    BOOST_CHECK_EQUAL(testUniverse.get_number_of_connected_components(), 7);
    testUniverse.join_by_name("c", "d");
    BOOST_CHECK_EQUAL(testUniverse.get_number_of_connected_components(), 6);
    testUniverse.join_by_name("d", "e");
    BOOST_CHECK_EQUAL(testUniverse.get_number_of_connected_components(), 5);
    testUniverse.join_by_name("e", "f");
    BOOST_CHECK_EQUAL(testUniverse.get_number_of_connected_components(), 4);

    testUniverse.join_by_name("c", "f");
    BOOST_CHECK_EQUAL(testUniverse.get_number_of_connected_components(), 4);

    testUniverse.join_by_name("g", "h");
    BOOST_CHECK_EQUAL(testUniverse.get_number_of_connected_components(), 3);
    testUniverse.join_by_name("h", "i");
    BOOST_CHECK_EQUAL(testUniverse.get_number_of_connected_components(), 2);

    testUniverse.join_by_name("b", "h");
    BOOST_CHECK_EQUAL(testUniverse.get_number_of_connected_components(), 1);

    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("a"), testUniverse.find_origin_by_name("b"));
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("b"), testUniverse.find_origin_by_name("c"));
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("c"), testUniverse.find_origin_by_name("d"));
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("a"), testUniverse.find_origin_by_name("h"));
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("c"), testUniverse.find_origin_by_name("i"));
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("f"), testUniverse.find_origin_by_name("g"));
}

BOOST_AUTO_TEST_CASE(UnionFindWeightStructure) {
    std::vector<std::string> names({"a", "b", "c", "d", "e", "f"});
    std::vector<unsigned> weights({1, 2, 1, 3, 1, 1});

    Union_Find_Universe<std::string, unsigned, unsigned, unsigned> testUniverse(names, weights, weights);

    for (size_t i = 0; i < names.size(); i++) {
        BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name(names[i]), names[i]);
        BOOST_CHECK_EQUAL(testUniverse.get_weight_of_component_by_name(names[i]), weights[i]);
        BOOST_CHECK_EQUAL(testUniverse.get_memory_of_component_by_name(names[i]), weights[i]);
    }

    BOOST_CHECK_EQUAL(testUniverse.get_number_of_connected_components(), 6);

    BOOST_CHECK_THROW(testUniverse.add_object("a"), std::runtime_error);
    BOOST_CHECK_THROW(testUniverse.add_object("e"), std::runtime_error);

    testUniverse.join_by_name("a", "b");
    testUniverse.join_by_name("b", "c");
    testUniverse.join_by_name("d", "b");
    testUniverse.join_by_name("a", "c");
    testUniverse.join_by_name("a", "d");

    testUniverse.join_by_name("e", "f");

    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("a"), testUniverse.find_origin_by_name("b"));
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("b"), testUniverse.find_origin_by_name("c"));
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("c"), testUniverse.find_origin_by_name("d"));
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("a"), testUniverse.find_origin_by_name("d"));
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("c"), testUniverse.find_origin_by_name("b"));

    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("e"), testUniverse.find_origin_by_name("f"));

    BOOST_CHECK_NE(testUniverse.find_origin_by_name("a"), testUniverse.find_origin_by_name("f"));

    BOOST_CHECK_EQUAL(testUniverse.get_weight_of_component_by_name("a"), 7);
    BOOST_CHECK_EQUAL(testUniverse.get_weight_of_component_by_name("b"), 7);
    BOOST_CHECK_EQUAL(testUniverse.get_weight_of_component_by_name("e"), 2);

    BOOST_CHECK_EQUAL(testUniverse.get_memory_of_component_by_name("a"), 7);
    BOOST_CHECK_EQUAL(testUniverse.get_memory_of_component_by_name("e"), 2);
    BOOST_CHECK_EQUAL(testUniverse.get_memory_of_component_by_name("b"), 7);

    std::vector<std::pair<std::vector<std::string>, unsigned>> componentsNWeights
        = testUniverse.get_connected_components_and_weights();
    unsigned totalCompWeights = 0;
    unsigned totalElements = 0;
    for (auto &[comp, wt] : componentsNWeights) {
        totalCompWeights += wt;
        totalElements += static_cast<unsigned>(comp.size());
        for (auto &name : comp) {
            BOOST_CHECK(std::any_of(names.cbegin(), names.cend(), [name](std::string otherName) { return name == otherName; }));
        }
    }

    std::vector<std::tuple<std::vector<std::string>, unsigned, unsigned>> componentsNWeightsNMemory
        = testUniverse.get_connected_components_weights_and_memories();
    unsigned totalCompWeights2 = 0;
    unsigned totalCompMemory = 0;
    unsigned totalElements2 = 0;
    for (const auto &[comp, wt, mem] : componentsNWeightsNMemory) {
        totalCompWeights2 += wt;
        totalCompMemory += mem;
        totalElements2 += static_cast<unsigned>(comp.size());
        for (auto &name : comp) {
            BOOST_CHECK(std::any_of(names.cbegin(), names.cend(), [name](std::string otherName) { return name == otherName; }));
        }
    }

    unsigned totalWeight = 0;
    for (auto &wt : weights) {
        totalWeight += wt;
    }

    BOOST_CHECK_EQUAL(totalElements, names.size());
    BOOST_CHECK_EQUAL(totalElements2, names.size());
    BOOST_CHECK_EQUAL(totalWeight, totalCompWeights);
    BOOST_CHECK_EQUAL(totalWeight, totalCompWeights2);
    BOOST_CHECK_EQUAL(totalWeight, totalCompMemory);

    for (auto &name : names) {
        BOOST_CHECK(std::any_of(
            componentsNWeights.cbegin(), componentsNWeights.cend(), [name](std::pair<std::vector<std::string>, unsigned> compPair) {
                return std::any_of(
                    compPair.first.cbegin(), compPair.first.cend(), [name](std::string otherName) { return name == otherName; });
            }));
    }
}

BOOST_AUTO_TEST_CASE(UnionFindStructureWeightCompCount) {
    std::vector<std::string> names({"a", "b", "c", "d", "e", "f"});
    std::vector<unsigned> weights({1, 2, 1, 3, 1, 1});
    Union_Find_Universe<std::string, unsigned, unsigned, unsigned> testUniverse;

    for (size_t i = 0; i < names.size(); i++) {
        testUniverse.add_object(names[i], weights[i], weights[i]);
    }

    for (size_t i = 0; i < names.size(); i++) {
        BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name(names[i]), names[i]);
        BOOST_CHECK_EQUAL(testUniverse.get_weight_of_component_by_name(names[i]), weights[i]);
        BOOST_CHECK_EQUAL(testUniverse.get_memory_of_component_by_name(names[i]), weights[i]);
    }

    testUniverse.join_by_name("a", "b");
    BOOST_CHECK_EQUAL(testUniverse.get_weight_of_component_by_name("a"), 3);
    BOOST_CHECK_EQUAL(testUniverse.get_memory_of_component_by_name("a"), 3);
    testUniverse.join_by_name("a", "b");
    BOOST_CHECK_EQUAL(testUniverse.get_weight_of_component_by_name("a"), 3);
    BOOST_CHECK_EQUAL(testUniverse.get_memory_of_component_by_name("a"), 3);
    testUniverse.join_by_name("b", "a");
    BOOST_CHECK_EQUAL(testUniverse.get_weight_of_component_by_name("a"), 3);
    BOOST_CHECK_EQUAL(testUniverse.get_memory_of_component_by_name("a"), 3);

    testUniverse.join_by_name("a", "c");
    BOOST_CHECK_EQUAL(testUniverse.get_weight_of_component_by_name("c"), 4);
    BOOST_CHECK_EQUAL(testUniverse.get_memory_of_component_by_name("c"), 4);
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("a"), testUniverse.find_origin_by_name("b"));
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("b"), testUniverse.find_origin_by_name("c"));

    testUniverse.join_by_name("d", "e");
    testUniverse.join_by_name("e", "f");
    BOOST_CHECK_EQUAL(testUniverse.get_weight_of_component_by_name("f"), 5);
    BOOST_CHECK_EQUAL(testUniverse.get_memory_of_component_by_name("f"), 5);
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("a"), testUniverse.find_origin_by_name("b"));
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("e"), testUniverse.find_origin_by_name("f"));
    BOOST_CHECK_NE(testUniverse.find_origin_by_name("a"), testUniverse.find_origin_by_name("e"));
    BOOST_CHECK_NE(testUniverse.find_origin_by_name("b"), testUniverse.find_origin_by_name("d"));

    std::vector<std::pair<std::vector<std::string>, unsigned>> compNWeights = testUniverse.get_connected_components_and_weights();
    BOOST_CHECK(compNWeights.size() == 2);
    BOOST_CHECK(compNWeights.size() == testUniverse.get_number_of_connected_components());
    BOOST_CHECK(compNWeights[0].first.size() == 3);
    BOOST_CHECK(compNWeights[1].first.size() == 3);
    BOOST_CHECK((compNWeights[0].second == 4 && compNWeights[1].second == 5)
                || (compNWeights[0].second == 5 && compNWeights[1].second == 4));

    std::vector<std::tuple<std::vector<std::string>, unsigned, unsigned>> compNWeightNMemory
        = testUniverse.get_connected_components_weights_and_memories();
    BOOST_CHECK(compNWeightNMemory.size() == 2);
    BOOST_CHECK(compNWeightNMemory.size() == testUniverse.get_number_of_connected_components());
    BOOST_CHECK(std::get<0>(compNWeightNMemory[0]).size() == 3);
    BOOST_CHECK(std::get<0>(compNWeightNMemory[1]).size() == 3);
    BOOST_CHECK((std::get<1>(compNWeightNMemory[0]) == 4 && std::get<1>(compNWeightNMemory[1]) == 5)
                || (std::get<1>(compNWeightNMemory[0]) == 5 && std::get<1>(compNWeightNMemory[1]) == 4));
    BOOST_CHECK((std::get<2>(compNWeightNMemory[0]) == 4 && std::get<2>(compNWeightNMemory[1]) == 5)
                || (std::get<2>(compNWeightNMemory[0]) == 5 && std::get<2>(compNWeightNMemory[1]) == 4));

    std::vector<std::pair<std::vector<std::string>, unsigned>> componentsNWeights
        = testUniverse.get_connected_components_and_weights();
    unsigned totalCompWeights = 0;
    unsigned totalElements = 0;
    for (auto &[comp, wt] : componentsNWeights) {
        totalCompWeights += wt;
        totalElements += static_cast<unsigned>(comp.size());
        for (auto &name : comp) {
            BOOST_CHECK(std::any_of(names.cbegin(), names.cend(), [name](std::string otherName) { return name == otherName; }));
        }
    }

    unsigned totalWeight = 0;
    for (auto &wt : weights) {
        totalWeight += wt;
    }

    BOOST_CHECK_EQUAL(totalElements, names.size());
    BOOST_CHECK_EQUAL(totalWeight, totalCompWeights);
    for (auto &name : names) {
        BOOST_CHECK(std::any_of(
            componentsNWeights.cbegin(), componentsNWeights.cend(), [name](std::pair<std::vector<std::string>, unsigned> compPair) {
                return std::any_of(
                    compPair.first.cbegin(), compPair.first.cend(), [name](std::string otherName) { return name == otherName; });
            }));
    }
}

BOOST_AUTO_TEST_CASE(UnionFindStructureWeightChainsCompCount) {
    std::vector<std::string> names({"a", "b", "c", "d", "e", "f", "g", "h", "i"});
    std::vector<unsigned> weights({1, 1, 1, 1, 1, 1, 1, 1, 1});
    Union_Find_Universe<std::string, unsigned, unsigned, unsigned> testUniverse(names, weights, weights);

    testUniverse.join_by_name("a", "b");
    BOOST_CHECK_EQUAL(testUniverse.get_weight_of_component_by_name("a"), 2);
    BOOST_CHECK_EQUAL(testUniverse.get_memory_of_component_by_name("a"), 2);
    testUniverse.join_by_name("b", "c");
    BOOST_CHECK_EQUAL(testUniverse.get_weight_of_component_by_name("a"), 3);
    BOOST_CHECK_EQUAL(testUniverse.get_memory_of_component_by_name("a"), 3);
    testUniverse.join_by_name("c", "d");
    BOOST_CHECK_EQUAL(testUniverse.get_weight_of_component_by_name("c"), 4);
    BOOST_CHECK_EQUAL(testUniverse.get_memory_of_component_by_name("c"), 4);
    BOOST_CHECK_EQUAL(testUniverse.get_weight_of_component_by_name("e"), 1);
    BOOST_CHECK_EQUAL(testUniverse.get_memory_of_component_by_name("e"), 1);
    testUniverse.join_by_name("d", "e");
    BOOST_CHECK_EQUAL(testUniverse.get_weight_of_component_by_name("e"), 5);
    BOOST_CHECK_EQUAL(testUniverse.get_memory_of_component_by_name("e"), 5);
    testUniverse.join_by_name("a", "b");
    BOOST_CHECK_EQUAL(testUniverse.get_weight_of_component_by_name("a"), 5);
    BOOST_CHECK_EQUAL(testUniverse.get_memory_of_component_by_name("a"), 5);
    BOOST_CHECK_EQUAL(testUniverse.get_weight_of_component_by_name("e"), 5);
    BOOST_CHECK_EQUAL(testUniverse.get_memory_of_component_by_name("e"), 5);
    testUniverse.join_by_name("e", "f");
    BOOST_CHECK_EQUAL(testUniverse.get_weight_of_component_by_name("a"), 6);
    BOOST_CHECK_EQUAL(testUniverse.get_memory_of_component_by_name("a"), 6);

    testUniverse.join_by_name("c", "f");
    BOOST_CHECK_EQUAL(testUniverse.get_weight_of_component_by_name("a"), 6);
    BOOST_CHECK_EQUAL(testUniverse.get_memory_of_component_by_name("a"), 6);
    BOOST_CHECK_EQUAL(testUniverse.get_number_of_connected_components(), 4);

    testUniverse.join_by_name("g", "h");
    BOOST_CHECK_EQUAL(testUniverse.get_weight_of_component_by_name("g"), 2);
    BOOST_CHECK_EQUAL(testUniverse.get_memory_of_component_by_name("g"), 2);

    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("i"), "i");
    BOOST_CHECK_EQUAL(testUniverse.get_weight_of_component_by_name("i"), 1);
    BOOST_CHECK_EQUAL(testUniverse.get_memory_of_component_by_name("i"), 1);

    testUniverse.join_by_name("h", "i");
    BOOST_CHECK_EQUAL(testUniverse.get_weight_of_component_by_name("i"), 3);
    BOOST_CHECK_EQUAL(testUniverse.get_memory_of_component_by_name("i"), 3);

    testUniverse.join_by_name("b", "h");
    BOOST_CHECK_EQUAL(testUniverse.get_weight_of_component_by_name("a"), 9);
    BOOST_CHECK_EQUAL(testUniverse.get_memory_of_component_by_name("a"), 9);

    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("a"), testUniverse.find_origin_by_name("b"));
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("d"), testUniverse.find_origin_by_name("i"));
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("e"), testUniverse.find_origin_by_name("h"));
    BOOST_CHECK_EQUAL(testUniverse.find_origin_by_name("b"), testUniverse.find_origin_by_name("i"));

    std::vector<std::pair<std::vector<std::string>, unsigned>> componentsNWeights
        = testUniverse.get_connected_components_and_weights();
    unsigned totalCompWeights = 0;
    unsigned totalElements = 0;
    for (auto &[comp, wt] : componentsNWeights) {
        totalCompWeights += wt;
        totalElements += static_cast<unsigned>(comp.size());
        for (auto &name : comp) {
            BOOST_CHECK(std::any_of(names.cbegin(), names.cend(), [name](std::string otherName) { return name == otherName; }));
        }
    }

    unsigned totalWeight = 0;
    for (auto &wt : weights) {
        totalWeight += wt;
    }

    BOOST_CHECK_EQUAL(totalElements, names.size());
    BOOST_CHECK_EQUAL(totalWeight, totalCompWeights);
    for (auto &name : names) {
        BOOST_CHECK(std::any_of(
            componentsNWeights.cbegin(), componentsNWeights.cend(), [name](std::pair<std::vector<std::string>, unsigned> compPair) {
                return std::any_of(
                    compPair.first.cbegin(), compPair.first.cend(), [name](std::string otherName) { return name == otherName; });
            }));
    }
}
