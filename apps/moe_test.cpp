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
    unsigned num_experts = 4;
    unsigned num_layers = 5;
    unsigned num_gpus = 2;

    std::cout << "Setting up MoE ILP parameters..." << std::endl;
    std::cout << "  Number of Experts: " << num_experts << std::endl;
    std::cout << "  Number of Layers: " << num_layers << std::endl;
    std::cout << "  Number of GPUs: " << num_gpus << std::endl;

    // Create parameters. Expert weights will be initialized to 1 by default.
    moe_ilp_params params(num_experts, num_layers, num_gpus);

    for (unsigned i = 0; i < num_experts; i++) {
        for (unsigned j = 0; j < num_layers; j++) {
            params.expert_weights[i][j] = i + j % 6 + 2;
        }
    }



    // Manually add some sample edges (expert_from, expert_to, layer_from, weight)
    // Edge: expert 0 (layer 0) -> expert 0 (layer 1), weight 10
    params.edges.push_back({0, 0, 0, 10});
    // Edge: expert 1 (layer 0) -> expert 1 (layer 1), weight 5
    params.edges.push_back({0, 1, 1, 5});
    // Edge: expert 0 (layer 0) -> expert 2 (layer 1), weight 8. This might create a cut if e0l0 and e2l1 are on different GPUs.
    params.edges.push_back({0, 0, 2, 8});
    params.edges.push_back({0, 1, 2, 12}); // Edge: expert 0 (layer 1) -> expert 1 (layer 2), weight 12
    params.edges.push_back({1, 0, 0, 7}); // Edge: expert 1 (layer 0) -> expert 0 (layer 0), weight 7
    params.edges.push_back({1, 1, 1, 9}); // Edge: expert 1 (layer 1) -> expert 1 (layer 1), weight 9
    params.edges.push_back({2, 3, 0, 15}); // Edge: expert 2 (layer 0) -> expert 3 (layer 0), weight 15
    params.edges.push_back({3, 2, 1, 6}); // Edge: expert 3 (layer 1) -> expert 2 (layer 1), weight 6
    params.edges.push_back({3, 3, 1, 11}); // Edge: expert 3 (layer 1) -> expert 3 (layer 1), weight 11
    params.edges.push_back({0, 3, 0, 18}); // Edge: expert 0 (layer 0) -> expert 3 (layer 0), weight 18
    params.edges.push_back({3, 0, 1, 14}); // Edge: expert 3 (layer 1) -> expert 0 (layer 1), weight 14
    params.edges.push_back({1, 2, 0, 20}); // Edge: expert 1 (layer 0) -> expert 2 (layer 0), weight 20
    params.edges.push_back({2, 1, 1, 17}); // Edge: expert 2 (layer 1) -> expert 1 (layer 1), weight 17
    params.edges.push_back({0, 2, 0, 22}); // Edge: expert 0 (layer 0) -> expert 2 (layer 0), weight 22

    

    std::cout << "  Number of Edges defined: " << params.edges.size() << std::endl;
    // You can also customize params.expert_weights here if needed, e.g.:
    // if (num_experts > 0 && num_layers > 0) {
    //     params.expert_weights[0][0] = 5; // Example: change weight of expert 0 in layer 0
    // }


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