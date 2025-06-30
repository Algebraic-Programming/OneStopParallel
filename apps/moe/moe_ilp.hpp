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

#include <fstream>
#include <string>
#include <vector>

#include <callbackbase.h>
#include <coptcpp_pch.h>

#include "moe_util.hpp"

struct moe_ilp_solution {

    std::vector<std::vector<int>> expert_assignment;
    double objective_value;

    void print_solution() {

        std::cout << "Objective Value: " << objective_value << std::endl;
        std::cout << "Expert Assignment (Expert, Layer, Assigned GPU):" << std::endl;
        for (unsigned expert = 0; expert < expert_assignment.size(); ++expert) {
            for (unsigned layer = 0; layer < expert_assignment[expert].size(); ++layer) {
                if (expert_assignment[expert][layer] != -1) {
                    std::cout << "  Expert " << expert << ", Layer " << layer << ": GPU "
                              << expert_assignment[expert][layer] << std::endl;
                } else {
                    std::cout << "  Expert " << expert << ", Layer " << layer << ": Not assigned (or recomputed)"
                              << std::endl;
                }
            }
        }
    }
};

class moe_ilp_solver {

  private:
    moe_ilp_params &params;

    std::vector<std::vector<VarArray>> expert_layer_proc;
    VarArray edge_cut;

    VarArray max_weight_layer;

    std::string final_solution_path = "./";                        // Default path
    std::string final_solution_file_prefix = "moe_final_solution"; // Default prefix

    void setup_variables(Model &model) {

        // variables indicating if expert i in layer l is assigned to proc p
        expert_layer_proc =
            std::vector<std::vector<VarArray>>(params.num_experts, std::vector<VarArray>(params.num_layers));

        for (unsigned expert = 0; expert < params.num_experts; expert++) {

            for (unsigned layer = 0; layer < params.num_layers; layer++) {
                expert_layer_proc[expert][layer] =
                    model.AddVars(static_cast<int>(params.num_gpus), COPT_BINARY, "expert_layer_proc");
            }
        }

        edge_cut = model.AddVars(static_cast<int>(params.edges.size()), COPT_BINARY, "edge_cut");
    }

    void setup_expert_assign_constr(Model &model, unsigned gpu_expert_capacity = 0, bool allow_recomputation = false) {

        for (unsigned expert = 0; expert < params.num_experts; expert++) {

            for (unsigned layer = 0; layer < params.num_layers; layer++) {

                Expr expr;
                for (unsigned gpu = 0; gpu < params.num_gpus; gpu++) {

                    expr += expert_layer_proc[expert][layer].GetVar(static_cast<int>(gpu));
                }

                model.AddConstr(allow_recomputation ? expr >= .99 : expr == 1);
            }
        }

        if (gpu_expert_capacity > 0) {

            for (unsigned gpu = 0; gpu < params.num_gpus; gpu++) {

                Expr expr;
                for (unsigned expert = 0; expert < params.num_experts; expert++) {

                    for (unsigned layer = 0; layer < params.num_layers; layer++) {
                        expr += expert_layer_proc[expert][layer].GetVar(static_cast<int>(gpu));
                    }
                }

                model.AddConstr(expr <= gpu_expert_capacity);
            }
        }
    }

    void setup_edge_cut_constr(Model &model) {

        for (unsigned edge = 0; edge < params.edges.size(); edge++) {

            for (unsigned gpu_from = 0; gpu_from < params.num_gpus; gpu_from++) {
                for (unsigned gpu_to = 0; gpu_to < params.num_gpus; gpu_to++) {

                    if (gpu_from != gpu_to) {

                        model.AddConstr(
                            edge_cut[static_cast<int>(edge)] >=
                            expert_layer_proc[params.edges[edge].expert_from][params.edges[edge].layer].GetVar(
                                static_cast<int>(gpu_from)) +
                                expert_layer_proc[params.edges[edge].expert_to][params.edges[edge].layer + 1].GetVar(
                                    static_cast<int>(gpu_to)) -
                                1.001);
                    }
                }
            }
        }
    }

    void setup_max_weight_constr(Model &model) {

        max_weight_layer = model.AddVars(static_cast<int>(params.num_layers), COPT_INTEGER, "max_weight_layer");

        for (unsigned layer = 0; layer < params.num_layers; layer++) {

            for (unsigned gpu = 0; gpu < params.num_gpus; gpu++) {

                Expr expr;
                for (unsigned expert = 0; expert < params.num_experts; expert++) {

                    expr += params.expert_weights[expert][layer] *
                            expert_layer_proc[expert][layer].GetVar(static_cast<int>(gpu));
                }

                model.AddConstr(max_weight_layer[static_cast<int>(layer)] >= expr);
            }
        }
    }

    void setup_global_weight_balance_constr(Model &model, double epsilon = 0.1) {

        unsigned total_weight = 0;
        for (unsigned layer = 0; layer < params.num_layers; ++layer) {
            for (unsigned expert = 0; expert < params.num_experts; ++expert) {
                total_weight += params.expert_weights[expert][layer];
            }
        }

        // Constraint to balance the total weight assigned to each GPU across all layers
        for (unsigned gpu = 0; gpu < params.num_gpus; ++gpu) {
            Expr total_weight_gpu;
            for (unsigned layer = 0; layer < params.num_layers; ++layer) {
                for (unsigned expert = 0; expert < params.num_experts; ++expert) {
                    total_weight_gpu += params.expert_weights[expert][layer] *
                                        expert_layer_proc[expert][layer].GetVar(static_cast<int>(gpu));
                }
            }

            model.AddConstr(total_weight_gpu <= (total_weight * (1 + epsilon)) / static_cast<double>(params.num_gpus));
        }
    }

    void setup_layer_weight_balance_constr(Model &model, double epsilon = 0.1) {

        for (unsigned layer = 0; layer < params.num_layers; ++layer) {
            unsigned total_weight_layer = 0;
            for (unsigned expert = 0; expert < params.num_experts; ++expert) {
                total_weight_layer += params.expert_weights[expert][layer];
            }

            for (unsigned gpu = 0; gpu < params.num_gpus; ++gpu) {
                Expr total_weight_gpu_layer;
                for (unsigned expert = 0; expert < params.num_experts; ++expert) {
                    total_weight_gpu_layer += params.expert_weights[expert][layer] *
                                              expert_layer_proc[expert][layer].GetVar(static_cast<int>(gpu));
                }
                model.AddConstr(total_weight_gpu_layer <= (total_weight_layer * (1 + epsilon)) /
                    static_cast<double>(params.num_gpus));
            }
        }
    }

    void setup_objective_max_weight(Model &model) {

        Expr expr;
        for (unsigned edge = 0; edge < params.edges.size(); edge++) {
            expr += params.edges[edge].weight * edge_cut[static_cast<int>(edge)];
        }

        for (unsigned layer = 0; layer < params.num_layers; layer++) {
            expr += max_weight_layer[static_cast<int>(layer)];
        }

        model.SetObjective(expr, COPT_MINIMIZE);
    }

    void setup_objective(Model &model) {

        Expr expr;
        for (unsigned edge = 0; edge < params.edges.size(); edge++) {
            expr += params.edges[edge].weight * edge_cut[static_cast<int>(edge)];
        }

        model.SetObjective(expr, COPT_MINIMIZE);
    }

    class WriteSolutionCallback : public CallbackBase {

      private:
        unsigned counter;
        unsigned max_number_solution;

        double best_obj;

      public:
        WriteSolutionCallback()
            : counter(0), max_number_solution(100), best_obj(COPT_INFINITY), expert_layer_proc_ptr(nullptr),
              num_experts_ptr(nullptr), num_layers_ptr(nullptr), num_gpus_ptr(nullptr), write_solutions_path_cb(""),
              solution_file_prefix_cb("") {}

        const std::vector<std::vector<VarArray>> *expert_layer_proc_ptr;
        const unsigned *num_experts_ptr;
        const unsigned *num_layers_ptr;
        const unsigned *num_gpus_ptr;

        bool write_solutions_enabled_cb = false; // Renamed for clarity
        std::string write_solutions_path_cb = "";
        std::string solution_file_prefix_cb = "";

        void callback() override {
            if (Where() == COPT_CBCONTEXT_MIPSOL && counter < max_number_solution &&
                GetIntInfo(COPT_CBINFO_HASINCUMBENT)) {

                try {

                    if (GetDblInfo(COPT_CBINFO_BESTOBJ) < best_obj && 0.0 < GetDblInfo(COPT_CBINFO_BESTBND)) {

                        best_obj = GetDblInfo(COPT_CBINFO_BESTOBJ);

                        if (write_solutions_enabled_cb && expert_layer_proc_ptr && num_experts_ptr && num_layers_ptr &&
                            num_gpus_ptr) {
                            std::string filepath = write_solutions_path_cb + solution_file_prefix_cb + "_" +
                                                   std::to_string(counter) + "_expert_assignment.txt";
                            std::ofstream outfile(filepath);
                            if (outfile.is_open()) {
                                outfile << "Expert Layer GPU Assignment (Solution " << counter << "):\n";
                                outfile << "Expert Layer GPU Value\n";
                                for (unsigned expert = 0; expert < *num_experts_ptr; ++expert) {
                                    for (unsigned layer = 0; layer < *num_layers_ptr; ++layer) {
                                        for (unsigned gpu = 0; gpu < *num_gpus_ptr; ++gpu) {
                                            double val = GetSolution(
                                                (*expert_layer_proc_ptr)[expert][layer].GetVar(static_cast<int>(gpu)));
                                            if (val >= 0.99) { // Only write assigned ones (value ~1)
                                                outfile << expert << " " << layer << " " << gpu << std::endl;
                                            }
                                        }
                                    }
                                }
                                outfile.close();
                            }
                        }
                        counter++;
                    }

                } catch (const std::exception &e) {
                    // Consider logging the error, e.g., std::cerr << "Exception in callback: " << e.what() <<
                    // std::endl;
                }
            }
        }
    };

    WriteSolutionCallback solution_callback;

  public:
    explicit moe_ilp_solver(moe_ilp_params &p_params) : params(p_params) {}
    virtual ~moe_ilp_solver() = default;

    moe_ilp_solution solve_ilp() {

        Envr env;
        Model model = env.CreateModel("moe_placement");

        setup_variables(model);
        setup_expert_assign_constr(model);
        setup_edge_cut_constr(model);
        setup_max_weight_constr(model);
        setup_objective_max_weight(model);

        // Configure and set callback if enabled
        if (solution_callback.write_solutions_enabled_cb) {
            solution_callback.expert_layer_proc_ptr = &this->expert_layer_proc; // Point to solver's member
            solution_callback.num_experts_ptr = &this->params.num_experts;
            solution_callback.num_layers_ptr = &this->params.num_layers;
            solution_callback.num_gpus_ptr = &this->params.num_gpus;

            model.SetCallback(&solution_callback, COPT_CBCONTEXT_MIPSOL);
        }

        model.SetDblParam(COPT_DBLPARAM_TIMELIMIT, 172800);
        model.SetIntParam(COPT_INTPARAM_THREADS, 128);
        std::cout << "Starting to solve ILP model..." << std::endl;

        model.Solve();

        // Write final solution if enabled and found
        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
            std::string filepath = final_solution_path + final_solution_file_prefix + "_final_expert_assignment.txt";
            _write_model_solution_to_file(filepath);
        }

        return _get_model_solution(model);
    }

    inline void enableWriteIntermediateSol(std::string path, std::string file_prefix) {
        solution_callback.write_solutions_enabled_cb = true;
        solution_callback.write_solutions_path_cb = path;
        solution_callback.solution_file_prefix_cb = file_prefix;
    }

  private:
    moe_ilp_solution _get_model_solution(Model &model) {

        moe_ilp_solution solution;
        solution.objective_value = 0.0; // Initialize with a default value

        // Check if a solution exists
        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
            solution.objective_value = model.GetDblAttr(COPT_DBLINFO_OBJ);

            solution.expert_assignment.resize(params.num_experts);
            for (unsigned expert = 0; expert < params.num_experts; ++expert) {
                solution.expert_assignment[expert].resize(params.num_layers);
                for (unsigned layer = 0; layer < params.num_layers; ++layer) {
                    solution.expert_assignment[expert][layer] = -1; // Default to unassigned

                    for (unsigned gpu = 0; gpu < params.num_gpus; ++gpu) {
                        double val =
                            expert_layer_proc[expert][layer].GetVar(static_cast<int>(gpu)).Get(COPT_DBLINFO_VALUE);
                        if (val >= 0.99) { // If assigned to this GPU
                            solution.expert_assignment[expert][layer] = static_cast<int>(gpu);
                            break; // An expert can only be assigned to one GPU
                        }
                    }
                }
            }
        } else {
            // Handle case where no solution is found (e.g., set all assignments to -1, objective to infinity)
            solution.objective_value = COPT_INFINITY;
            solution.expert_assignment.resize(params.num_experts);
            for (unsigned expert = 0; expert < params.num_experts; ++expert) {
                solution.expert_assignment[expert].resize(params.num_layers, -1);
            }
        }
        return solution;
    }

    void _write_model_solution_to_file(const std::string &filepath) {
        std::ofstream outfile(filepath);
        if (outfile.is_open()) {
            outfile << "Final Expert Layer GPU Assignment:\n";
            outfile << "Expert Layer GPU Value\n";
            for (unsigned expert = 0; expert < params.num_experts; ++expert) {
                for (unsigned layer = 0; layer < params.num_layers; ++layer) {
                    for (unsigned gpu = 0; gpu < params.num_gpus; ++gpu) {
                        // For final solution, get value directly from the variable attached to the solved model
                        double val =
                            expert_layer_proc[expert][layer].GetVar(static_cast<int>(gpu)).Get(COPT_DBLINFO_VALUE);
                        if (val >= 0.99) { // Only write assigned ones
                            outfile << expert << " " << layer << " " << gpu << std::endl;
                        }
                    }
                }
            }
            outfile.close();
        }
    }
};
