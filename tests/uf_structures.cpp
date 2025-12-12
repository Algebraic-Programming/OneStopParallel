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
    UnionFindUniverse<std::string, unsigned, int, int> testUniverse(names);

    for (auto &name : names) {
        BOOST_CHECK_EQUAL(testUniverse.FindOriginByName(name), name);
    }

    BOOST_CHECK_EQUAL(testUniverse.GetNumberOfConnectedComponents(), 6);

    BOOST_CHECK_THROW(testUniverse.AddObject("a"), std::runtime_error);
    BOOST_CHECK_THROW(testUniverse.AddObject("e"), std::runtime_error);

    BOOST_CHECK_EQUAL(testUniverse.GetNumberOfConnectedComponents(), 6);

    testUniverse.JoinByName("a", "b");
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("b"));
    BOOST_CHECK_EQUAL(testUniverse.GetNumberOfConnectedComponents(), 5);

    testUniverse.JoinByName("b", "c");
    BOOST_CHECK_EQUAL(testUniverse.GetNumberOfConnectedComponents(), 4);
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("b"));
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("c"));
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("b"), testUniverse.FindOriginByName("c"));

    testUniverse.JoinByName("d", "b");
    BOOST_CHECK_EQUAL(testUniverse.GetNumberOfConnectedComponents(), 3);
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("d"));
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("d"), testUniverse.FindOriginByName("b"));

    testUniverse.JoinByName("a", "c");
    BOOST_CHECK_EQUAL(testUniverse.GetNumberOfConnectedComponents(), 3);
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("c"));

    testUniverse.JoinByName("a", "d");
    BOOST_CHECK_EQUAL(testUniverse.GetNumberOfConnectedComponents(), 3);
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("d"));

    testUniverse.JoinByName("e", "f");
    BOOST_CHECK_EQUAL(testUniverse.GetNumberOfConnectedComponents(), 2);
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("e"), testUniverse.FindOriginByName("f"));
    BOOST_CHECK_NE(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("f"));

    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("b"));
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("b"), testUniverse.FindOriginByName("c"));
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("c"), testUniverse.FindOriginByName("d"));
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("d"));
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("c"), testUniverse.FindOriginByName("b"));

    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("e"), testUniverse.FindOriginByName("f"));

    BOOST_CHECK_NE(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("f"));
}

BOOST_AUTO_TEST_CASE(UnionFindStructure2) {
    std::vector<std::string> names({"a", "b", "c", "d", "e", "f", "g", "h", "i"});
    UnionFindUniverse<std::string, unsigned, int, int> testUniverse;

    for (auto &name : names) {
        testUniverse.AddObject(name);
    }

    for (auto &name : names) {
        BOOST_CHECK_EQUAL(testUniverse.FindOriginByName(name), name);
    }

    BOOST_CHECK_THROW(testUniverse.AddObject("c"), std::runtime_error);
    BOOST_CHECK_THROW(testUniverse.AddObject("i"), std::runtime_error);

    for (auto &name : names) {
        BOOST_CHECK_EQUAL(testUniverse.FindOriginByName(name), name);
    }

    BOOST_CHECK_EQUAL(testUniverse.GetNumberOfConnectedComponents(), 9);

    testUniverse.JoinByName("a", "b");
    BOOST_CHECK_EQUAL(testUniverse.GetNumberOfConnectedComponents(), 8);
    testUniverse.JoinByName("b", "c");
    BOOST_CHECK_EQUAL(testUniverse.GetNumberOfConnectedComponents(), 7);
    testUniverse.JoinByName("c", "d");
    BOOST_CHECK_EQUAL(testUniverse.GetNumberOfConnectedComponents(), 6);
    testUniverse.JoinByName("d", "e");
    BOOST_CHECK_EQUAL(testUniverse.GetNumberOfConnectedComponents(), 5);
    testUniverse.JoinByName("e", "f");
    BOOST_CHECK_EQUAL(testUniverse.GetNumberOfConnectedComponents(), 4);

    testUniverse.JoinByName("c", "f");
    BOOST_CHECK_EQUAL(testUniverse.GetNumberOfConnectedComponents(), 4);

    testUniverse.JoinByName("g", "h");
    BOOST_CHECK_EQUAL(testUniverse.GetNumberOfConnectedComponents(), 3);
    testUniverse.JoinByName("h", "i");
    BOOST_CHECK_EQUAL(testUniverse.GetNumberOfConnectedComponents(), 2);

    testUniverse.JoinByName("b", "h");
    BOOST_CHECK_EQUAL(testUniverse.GetNumberOfConnectedComponents(), 1);

    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("b"));
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("b"), testUniverse.FindOriginByName("c"));
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("c"), testUniverse.FindOriginByName("d"));
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("h"));
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("c"), testUniverse.FindOriginByName("i"));
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("f"), testUniverse.FindOriginByName("g"));
}

BOOST_AUTO_TEST_CASE(UnionFindWeightStructure) {
    std::vector<std::string> names({"a", "b", "c", "d", "e", "f"});
    std::vector<unsigned> weights({1, 2, 1, 3, 1, 1});

    UnionFindUniverse<std::string, unsigned, unsigned, unsigned> testUniverse(names, weights, weights);

    for (size_t i = 0; i < names.size(); i++) {
        BOOST_CHECK_EQUAL(testUniverse.FindOriginByName(names[i]), names[i]);
        BOOST_CHECK_EQUAL(testUniverse.GetWeightOfComponentByName(names[i]), weights[i]);
        BOOST_CHECK_EQUAL(testUniverse.GetMemoryOfComponentByName(names[i]), weights[i]);
    }

    BOOST_CHECK_EQUAL(testUniverse.GetNumberOfConnectedComponents(), 6);

    BOOST_CHECK_THROW(testUniverse.AddObject("a"), std::runtime_error);
    BOOST_CHECK_THROW(testUniverse.AddObject("e"), std::runtime_error);

    testUniverse.JoinByName("a", "b");
    testUniverse.JoinByName("b", "c");
    testUniverse.JoinByName("d", "b");
    testUniverse.JoinByName("a", "c");
    testUniverse.JoinByName("a", "d");

    testUniverse.JoinByName("e", "f");

    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("b"));
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("b"), testUniverse.FindOriginByName("c"));
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("c"), testUniverse.FindOriginByName("d"));
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("d"));
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("c"), testUniverse.FindOriginByName("b"));

    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("e"), testUniverse.FindOriginByName("f"));

    BOOST_CHECK_NE(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("f"));

    BOOST_CHECK_EQUAL(testUniverse.GetWeightOfComponentByName("a"), 7);
    BOOST_CHECK_EQUAL(testUniverse.GetWeightOfComponentByName("b"), 7);
    BOOST_CHECK_EQUAL(testUniverse.GetWeightOfComponentByName("e"), 2);

    BOOST_CHECK_EQUAL(testUniverse.GetMemoryOfComponentByName("a"), 7);
    BOOST_CHECK_EQUAL(testUniverse.GetMemoryOfComponentByName("e"), 2);
    BOOST_CHECK_EQUAL(testUniverse.GetMemoryOfComponentByName("b"), 7);

    std::vector<std::pair<std::vector<std::string>, unsigned>> componentsNWeights
        = testUniverse.GetConnectedComponentsAndWeights();
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
        = testUniverse.GetConnectedComponentsWeightsAndMemories();
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
    UnionFindUniverse<std::string, unsigned, unsigned, unsigned> testUniverse;

    for (size_t i = 0; i < names.size(); i++) {
        testUniverse.AddObject(names[i], weights[i], weights[i]);
    }

    for (size_t i = 0; i < names.size(); i++) {
        BOOST_CHECK_EQUAL(testUniverse.FindOriginByName(names[i]), names[i]);
        BOOST_CHECK_EQUAL(testUniverse.GetWeightOfComponentByName(names[i]), weights[i]);
        BOOST_CHECK_EQUAL(testUniverse.GetMemoryOfComponentByName(names[i]), weights[i]);
    }

    testUniverse.JoinByName("a", "b");
    BOOST_CHECK_EQUAL(testUniverse.GetWeightOfComponentByName("a"), 3);
    BOOST_CHECK_EQUAL(testUniverse.GetMemoryOfComponentByName("a"), 3);
    testUniverse.JoinByName("a", "b");
    BOOST_CHECK_EQUAL(testUniverse.GetWeightOfComponentByName("a"), 3);
    BOOST_CHECK_EQUAL(testUniverse.GetMemoryOfComponentByName("a"), 3);
    testUniverse.JoinByName("b", "a");
    BOOST_CHECK_EQUAL(testUniverse.GetWeightOfComponentByName("a"), 3);
    BOOST_CHECK_EQUAL(testUniverse.GetMemoryOfComponentByName("a"), 3);

    testUniverse.JoinByName("a", "c");
    BOOST_CHECK_EQUAL(testUniverse.GetWeightOfComponentByName("c"), 4);
    BOOST_CHECK_EQUAL(testUniverse.GetMemoryOfComponentByName("c"), 4);
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("b"));
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("b"), testUniverse.FindOriginByName("c"));

    testUniverse.JoinByName("d", "e");
    testUniverse.JoinByName("e", "f");
    BOOST_CHECK_EQUAL(testUniverse.GetWeightOfComponentByName("f"), 5);
    BOOST_CHECK_EQUAL(testUniverse.GetMemoryOfComponentByName("f"), 5);
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("b"));
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("e"), testUniverse.FindOriginByName("f"));
    BOOST_CHECK_NE(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("e"));
    BOOST_CHECK_NE(testUniverse.FindOriginByName("b"), testUniverse.FindOriginByName("d"));

    std::vector<std::pair<std::vector<std::string>, unsigned>> compNWeights = testUniverse.GetConnectedComponentsAndWeights();
    BOOST_CHECK(compNWeights.size() == 2);
    BOOST_CHECK(compNWeights.size() == testUniverse.GetNumberOfConnectedComponents());
    BOOST_CHECK(compNWeights[0].first.size() == 3);
    BOOST_CHECK(compNWeights[1].first.size() == 3);
    BOOST_CHECK((compNWeights[0].second == 4 && compNWeights[1].second == 5)
                || (compNWeights[0].second == 5 && compNWeights[1].second == 4));

    std::vector<std::tuple<std::vector<std::string>, unsigned, unsigned>> compNWeightNMemory
        = testUniverse.GetConnectedComponentsWeightsAndMemories();
    BOOST_CHECK(compNWeightNMemory.size() == 2);
    BOOST_CHECK(compNWeightNMemory.size() == testUniverse.GetNumberOfConnectedComponents());
    BOOST_CHECK(std::get<0>(compNWeightNMemory[0]).size() == 3);
    BOOST_CHECK(std::get<0>(compNWeightNMemory[1]).size() == 3);
    BOOST_CHECK((std::get<1>(compNWeightNMemory[0]) == 4 && std::get<1>(compNWeightNMemory[1]) == 5)
                || (std::get<1>(compNWeightNMemory[0]) == 5 && std::get<1>(compNWeightNMemory[1]) == 4));
    BOOST_CHECK((std::get<2>(compNWeightNMemory[0]) == 4 && std::get<2>(compNWeightNMemory[1]) == 5)
                || (std::get<2>(compNWeightNMemory[0]) == 5 && std::get<2>(compNWeightNMemory[1]) == 4));

    std::vector<std::pair<std::vector<std::string>, unsigned>> componentsNWeights
        = testUniverse.GetConnectedComponentsAndWeights();
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
    UnionFindUniverse<std::string, unsigned, unsigned, unsigned> testUniverse(names, weights, weights);

    testUniverse.JoinByName("a", "b");
    BOOST_CHECK_EQUAL(testUniverse.GetWeightOfComponentByName("a"), 2);
    BOOST_CHECK_EQUAL(testUniverse.GetMemoryOfComponentByName("a"), 2);
    testUniverse.JoinByName("b", "c");
    BOOST_CHECK_EQUAL(testUniverse.GetWeightOfComponentByName("a"), 3);
    BOOST_CHECK_EQUAL(testUniverse.GetMemoryOfComponentByName("a"), 3);
    testUniverse.JoinByName("c", "d");
    BOOST_CHECK_EQUAL(testUniverse.GetWeightOfComponentByName("c"), 4);
    BOOST_CHECK_EQUAL(testUniverse.GetMemoryOfComponentByName("c"), 4);
    BOOST_CHECK_EQUAL(testUniverse.GetWeightOfComponentByName("e"), 1);
    BOOST_CHECK_EQUAL(testUniverse.GetMemoryOfComponentByName("e"), 1);
    testUniverse.JoinByName("d", "e");
    BOOST_CHECK_EQUAL(testUniverse.GetWeightOfComponentByName("e"), 5);
    BOOST_CHECK_EQUAL(testUniverse.GetMemoryOfComponentByName("e"), 5);
    testUniverse.JoinByName("a", "b");
    BOOST_CHECK_EQUAL(testUniverse.GetWeightOfComponentByName("a"), 5);
    BOOST_CHECK_EQUAL(testUniverse.GetMemoryOfComponentByName("a"), 5);
    BOOST_CHECK_EQUAL(testUniverse.GetWeightOfComponentByName("e"), 5);
    BOOST_CHECK_EQUAL(testUniverse.GetMemoryOfComponentByName("e"), 5);
    testUniverse.JoinByName("e", "f");
    BOOST_CHECK_EQUAL(testUniverse.GetWeightOfComponentByName("a"), 6);
    BOOST_CHECK_EQUAL(testUniverse.GetMemoryOfComponentByName("a"), 6);

    testUniverse.JoinByName("c", "f");
    BOOST_CHECK_EQUAL(testUniverse.GetWeightOfComponentByName("a"), 6);
    BOOST_CHECK_EQUAL(testUniverse.GetMemoryOfComponentByName("a"), 6);
    BOOST_CHECK_EQUAL(testUniverse.GetNumberOfConnectedComponents(), 4);

    testUniverse.JoinByName("g", "h");
    BOOST_CHECK_EQUAL(testUniverse.GetWeightOfComponentByName("g"), 2);
    BOOST_CHECK_EQUAL(testUniverse.GetMemoryOfComponentByName("g"), 2);

    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("i"), "i");
    BOOST_CHECK_EQUAL(testUniverse.GetWeightOfComponentByName("i"), 1);
    BOOST_CHECK_EQUAL(testUniverse.GetMemoryOfComponentByName("i"), 1);

    testUniverse.JoinByName("h", "i");
    BOOST_CHECK_EQUAL(testUniverse.GetWeightOfComponentByName("i"), 3);
    BOOST_CHECK_EQUAL(testUniverse.GetMemoryOfComponentByName("i"), 3);

    testUniverse.JoinByName("b", "h");
    BOOST_CHECK_EQUAL(testUniverse.GetWeightOfComponentByName("a"), 9);
    BOOST_CHECK_EQUAL(testUniverse.GetMemoryOfComponentByName("a"), 9);

    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("b"));
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("d"), testUniverse.FindOriginByName("i"));
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("e"), testUniverse.FindOriginByName("h"));
    BOOST_CHECK_EQUAL(testUniverse.FindOriginByName("b"), testUniverse.FindOriginByName("i"));

    std::vector<std::pair<std::vector<std::string>, unsigned>> componentsNWeights
        = testUniverse.GetConnectedComponentsAndWeights();
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
