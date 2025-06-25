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


#include <iostream>
#include <vector>
#include "moe/moe_ilp.hpp"


int main() {
    // 1. Define parameters for the MoE ILP
    unsigned num_gpus = 2;

    std::cout << "Reading token passes from file..." << std::endl;
    std::vector<token_pass> passes =
        read_token_passes_from_file("data/moe/routing_expert_token_0.txt");
    // print_token_passes(passes);

    std::cout << "Creating MoE ILP parameters from token passes..." << std::endl;
    moe_ilp_params params = create_moe_ilp_params_from_token_passes(passes, num_gpus);

    std::cout << "  Number of Experts: " << params.num_experts << std::endl;
    std::cout << "  Number of Layers: " << params.num_layers << std::endl;
    std::cout << "  Number of GPUs: " << params.num_gpus << std::endl;
    std::cout << "  Number of Edges defined: " << params.edges.size() << std::endl;

    //print_layer_details(params, 1);

    // 2. Create the ILP solver instance
    std::cout << "Creating MoE ILP solver..." << std::endl;
    moe_ilp_solver solver(params);

    // 3. Optionally, enable writing intermediate solutions
    // The final solution is written by default to "./moe_final_solution_final_expert_assignment.txt"
    // solver.enableWriteIntermediateSol("./", "test_intermediate_moe");
    // std::cout << "Intermediate solution writing enabled (if any found)." << std::endl;


    solver.solve_ilp();


    return 0;
}