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

#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/trivial.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <iostream>
#include <map>
#include <set>
#include <string>

namespace pt = boost::property_tree;

// main parameters for running simple_schedulers.cpp
struct ConfigParser {
  public:
    pt::ptree global_params;
    pt::ptree scheduler;
    pt::ptree instances;

  private:
    std::string main_config_file = "";
    bool has_config_file = false;

    pt::ptree scheduler_config;

    void usage() {
        std::cout << "Usage: Either read config file: \n"
                  << "     --config *.json          \t\tSpecify config .json file.\n"
                  << "  Or specify command line options:\n"
                  << "     --inputDag, -g <FILE>          \tSpecify the input dag file. Supported formats: *.dot, "
                     "*.mtx,... \n"
                  << "     --inputMachine, -m <FILE>      \tSpecify the input file. Supported format: ... \n"
                  << "     --timeLimit, -t <TIME>         \tSet a time limit in seconds. Applies to each algorithm. \n"
                  << "     --output, -o                   \tOutput schedule file \n"
                  << "     --sankey, -s                   \tOutput sankey schedule file \n"
                  << "     --dot, -d                      \tOutput dot schedule file \n"
                  << "   Available scheduler: \n";

        pt::ptree loadPtreeRoot;
        pt::read_json(main_config_file, loadPtreeRoot);
        pt::ptree scheduler_config_usage = loadPtreeRoot.get_child("algorithms");

        for (auto &algorithm : scheduler_config_usage) {
            std::cout << "     --" << algorithm.second.get_child("name").get_value<std::string>() << "\t\t"
                      << algorithm.second.get_child("description").get_value<std::string>() << "\n";
        }
    }

    void add_algorithm(std::string name) {

        bool algorithm_found = false;
        std::string algorithm_identifier = name;

        while (algorithm_identifier.find("-") == 0) {
            algorithm_identifier = algorithm_identifier.substr(1);
        }

        for (auto &algorithm : scheduler_config) {

            std::string alg_name = algorithm.second.get_child("name").get_value<std::string>();

            std::transform(alg_name.begin(), alg_name.end(), alg_name.begin(),
                           [](unsigned char c) { return c; });

            if (alg_name == algorithm_identifier) {
                scheduler.push_back(algorithm);
                algorithm_found = true;
            }
        }

        if (!algorithm_found) {
            throw std::invalid_argument("Parameter error: wrong input or unknown algorithm \"" + name + "\".\n");
        }
    }

    void parse_config_file(std::string filename) {

        pt::ptree loadPtreeRoot;
        pt::read_json(filename, loadPtreeRoot);

        global_params = loadPtreeRoot.get_child("globalParameters");
        
        try {
            instances = loadPtreeRoot.get_child("inputInstances");
        } catch (const pt::ptree_bad_path &e) {
            
        }      

        pt::ptree scheduler_config_parse = loadPtreeRoot.get_child("algorithms");
        for (auto &algorithm : scheduler_config_parse) {

            if (algorithm.second.get_child("run").get_value<bool>()) {
                scheduler.push_back(algorithm);
            }
        }
    }

  public:
    ConfigParser() = default;
    ConfigParser(std::string main_config_file_) : main_config_file(main_config_file_), has_config_file(true) {}

    void parse_args(const int argc, const char *const argv[]) {

        if (has_config_file) {

            if (argc < 3) {
                usage();
                throw std::invalid_argument("Parameter error: not enough parameters specified.\n");
            } else if (std::string(argv[1]) == "--config") {

                std::string config_file = argv[2];
                if (config_file.empty() || config_file.substr(config_file.size() - 5) != ".json") {
                    throw std::invalid_argument("Parameter error: config file ending is not \".json\".\n");
                }

                parse_config_file(config_file);
                if (scheduler.empty()) {
                    throw std::invalid_argument("Parameter error: config file does not specify scheduler to run!\n");
                }
                if (instances.empty()) {
                    throw std::invalid_argument("Parameter error: config file does not specify input instances!\n");
                }
                if (global_params.empty()) {
                    throw std::invalid_argument("Parameter error: config file does not specify global parameters!\n");
                }
            } else {

                const std::set<std::string> parameters_requiring_value(
                    {"--config", "--inputDag", "--g", "-inputDag", "-g", "--timeLimit", "--t", "-timeLimit", "-t",
                     "--inputMachine", "--m", "-inputMachine", "-m"});

                pt::ptree loadPtreeRoot;
                pt::read_json(main_config_file, loadPtreeRoot);

                global_params = loadPtreeRoot.get_child("globalParameters");
                scheduler_config = loadPtreeRoot.get_child("algorithms");
                pt::ptree instance;

                bool graph_specified = false;
                bool machine_specified = false;

                // PROCESS COMMAND LINE ARGUMENTS
                for (int i = 1; i < argc; ++i) {
                    // Check parameters that require an argument afterwards
                    if (parameters_requiring_value.count(argv[i]) == 1 && i + 1 >= argc) {
                        throw std::invalid_argument("Parameter error: no parameter value after the \"" +
                                                    std::string(argv[i]) + "\" option.\n");
                    }

                    std::string flag = argv[i];
                    std::transform(flag.begin(), flag.end(), flag.begin(),
                                   [](unsigned char c) { return c; });

                    if (std::string(flag) == "--config") {
                        usage();
                        throw std::invalid_argument("Parameter error: usage \"" + std::string(argv[i]) + "\".\n");

                    } else if (std::string(flag) == "--timelimit" || std::string(flag) == "--t" ||
                               std::string(flag) == "-t" || std::string(flag) == "-timelimit") {
                        global_params.put("timeLimit", std::stoi(argv[++i]));

                    } else if (std::string(flag) == "--sankey" || std::string(flag) == "--s" ||
                               std::string(flag) == "-s" || std::string(flag) == "-sankey") {
                        global_params.put("outputSankeySchedule", true);

                    } else if (std::string(flag) == "--dot" || std::string(flag) == "--d" ||
                               std::string(flag) == "-d" || std::string(flag) == "-dot") {
                        global_params.put("outputDotSchedule", true);

                    } else if (std::string(flag) == "--inputDag" || std::string(flag) == "--g" ||
                               std::string(flag) == "-inputDag" || std::string(flag) == "-g") {
                        instance.put("graphFile", argv[++i]);
                        graph_specified = true;

                    } else if (std::string(flag) == "--inputMachine" || std::string(flag) == "--m" ||
                               std::string(flag) == "-inputMachine" || std::string(flag) == "-m") {
                        instance.put("machineParamsFile", argv[++i]);
                        machine_specified = true;

                    } else if (std::string(flag) == "--output" || std::string(flag) == "--o" ||
                               std::string(flag) == "-output" || std::string(flag) == "-o") {
                        global_params.put("outputSchedule", true);
                    } else {
                        add_algorithm(flag);
                    }
                }

                if (!machine_specified || !graph_specified) {
                    usage();
                    throw std::invalid_argument("Parameter error: no graph or machine parameters were specified!\n");
                } else if (scheduler.empty()) {
                    usage();
                    throw std::invalid_argument("Parameter error: no algorithm was specified!\n");
                }

                instances.push_back(std::make_pair("", instance));
            }
        } else {

            if (argc < 3 || std::string(argv[1]) != "--config") {

                std::cout << "Usage: read config file: \n"
                          << "     --config *.json          \t\tSpecify config .json file.\n";

                throw std::invalid_argument("Parameter error: not enough parameters specified.\n");

            } else {

                std::string config_file = argv[2];
                if (config_file.empty() || config_file.substr(config_file.size() - 5) != ".json") {
                    throw std::invalid_argument("Parameter error: config file ending is not \".json\".\n");
                }

                parse_config_file(config_file);
                if (scheduler.empty()) {
                    throw std::invalid_argument("Parameter error: config file does not specify scheduler to run!\n");
                }
                if (global_params.empty()) {
                    throw std::invalid_argument("Parameter error: config file does not specify global parameters!\n");
                }
            }
        }
    }
};
