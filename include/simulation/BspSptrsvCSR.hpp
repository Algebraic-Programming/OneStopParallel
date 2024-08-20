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

#include <algorithm>
#include <iostream>
#include <list>
#include <map>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <vector>

#include "auxiliary/auxiliary.hpp"
#include "model/BspSchedule.hpp"
#include "model/SetSchedule.hpp"

typedef std::tuple<VertexType, VertexType> edge_csr;

class BspSptrsvCSR {

  private:
    const BspInstance *instance;

    std::unordered_map<VertexType, double> node_value;
    std::unordered_map<EdgeType, double, EdgeType_hash> edge_value;

  public:
    std::vector<double> val;
    std::vector<unsigned> col_idx;

    std::vector<unsigned> row_ptr;
    std::vector<unsigned> num_row_entries;

    std::vector<unsigned> step_ptr;
    std::vector<std::vector<unsigned>> step_proc_ptr;
    std::vector<std::vector<unsigned>> step_proc_num;

    std::vector<double> x;

    unsigned num_supersteps;

    std::vector<std::vector<std::vector<VertexType>>> vector_step_processor_vertices;

    std::vector<int> ready;

    // /**
    //  * @brief Default constructor for the BspSptrsvCSR class.
    //  */
    // BspSptrsvCSR(const BspInstance &inst, bool use_mtx_values = false) : instance(&inst) {

    //     if (use_mtx_values) {

    //         for (const auto &node : inst.getComputationalDag().vertices()) {

    //             node_value[node] = inst.getComputationalDag().get_node_mtx_entry(node);

    //             for (const auto &edge : inst.getComputationalDag().out_edges(node)) {

    //                 edge_value[edge] = inst.getComputationalDag().get_edge_mtx_entry(edge);
    //             }
    //         }
    //     } else {
    //         double lower_bound = -100;
    //         double upper_bound = 100;
    //         std::uniform_real_distribution<double> unif_100(lower_bound, upper_bound);

    //         std::uniform_real_distribution<double> unif_log(-std::log(upper_bound), std::log(upper_bound));
    //         std::default_random_engine re;

    //         for (const auto &node : inst.getComputationalDag().vertices()) {

    //             node_value[node] = (1 - 2 * randInt(2)) * std::exp(unif_log(re));

    //             for (const auto &edge : inst.getComputationalDag().out_edges(node)) {

    //                 edge_value[edge] = unif_100(re);
    //             }
    //         }
    //     }
    // }

    BspSptrsvCSR(BspInstance &inst, bool generate_mtx_values) : instance(&inst) {

        if (generate_mtx_values) {

            double lower_bound = -100;
            double upper_bound = 100;
            std::uniform_real_distribution<double> unif_100(lower_bound, upper_bound);

            std::uniform_real_distribution<double> unif_log(-std::log(upper_bound), std::log(upper_bound));
            std::default_random_engine re;

            for (const auto &node : inst.getComputationalDag().vertices()) {

                // node_value[node] = (1 - 2 * randInt(2)) * std::exp(unif_log(re));

                // if (write_mtx_values) {
                inst.getComputationalDag().set_node_mtx_entry(node, (1 - 2 * randInt(2)) * std::exp(unif_log(re)));
                //}

                for (const auto &edge : inst.getComputationalDag().out_edges(node)) {

                    // edge_value[edge] = unif_100(re);

                    // if (write_mtx_values) {
                    inst.getComputationalDag().set_edge_mtx_entry(edge, unif_100(re));
                    //}
                }
            }
        }
    }

    // /**
    //  * @brief Default constructor for the BspSptrsvCSR class.
    //  */
    // BspSptrsvCSR(const BspInstance &inst,
    //              const std::unordered_map<std::pair<VertexType, VertexType>, double, pair_hash> &matrix_val_ptr)
    //     : instance(&inst) {
    //     for (const auto &node : inst.getComputationalDag().vertices()) {
    //         node_value[node] = matrix_val_ptr.at(std::make_pair(node, node));
    //         for (const auto &edge : inst.getComputationalDag().out_edges(node)) {
    //             edge_value[edge] = matrix_val_ptr.at(std::make_pair(edge.m_source, edge.m_target));
    //         }
    //     }
    // }

    void setup_csr(const BspSchedule &schedule, std::vector<size_t> &perm);

    void setup_csr_snake(const BspSchedule &schedule, std::vector<size_t> &perm);

    void setup_csr_no_permutation(const BspSchedule &schedule);

    void setup_csr_no_barrier(const BspSchedule &schedule, std::vector<size_t> &perm);

    void simulate_sptrsv_serial();

    void simulate_sptrsv();

    void simulate_sptrsv_snake();

    void simulate_sptrsv_no_permutation();

    void simulate_sptrsv_graph_mtx();

    void simulate_sptrsv_no_barrier();

    std::vector<double> compute_sptrsv();

    std::vector<double> get_result() const { return x; }

    void reset_x() {

        for (auto &val : x) {
            val = 1.0;
        }
    }

    void permute_vector(std::vector<double> &vec, const std::vector<size_t> &perm);

    std::unordered_map<VertexType, double> &getNodeValues() { return node_value; }
    std::unordered_map<EdgeType, double, EdgeType_hash> getEdgeValues() { return edge_value; }

    virtual ~BspSptrsvCSR() = default;
};
