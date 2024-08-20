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

#include "simulation/BspSptrsvCSR.hpp"

void BspSptrsvCSR::setup_csr(const BspSchedule &schedule, std::vector<size_t> &perm) {

    std::vector<size_t> perm_inv(perm.size());
    for (size_t i = 0; i < perm.size(); i++) {
        perm_inv[perm[i]] = i;
    }

    num_supersteps = schedule.numberOfSupersteps();

    x = std::vector<double>(instance->numberOfVertices(), 1.0);

    val.clear();
    val.reserve(instance->getComputationalDag().numberOfEdges() + instance->getComputationalDag().numberOfVertices());

    col_idx.clear();
    col_idx.reserve(instance->getComputationalDag().numberOfEdges() +
                    instance->getComputationalDag().numberOfVertices());

    row_ptr.clear();
    row_ptr.reserve(instance->numberOfVertices() + 1);

    step_ptr.clear();
    step_ptr.reserve(instance->numberOfVertices());

    step_proc_ptr =
        std::vector<std::vector<unsigned>>(num_supersteps, std::vector<unsigned>(instance->numberOfProcessors(), 0));

    step_proc_num = schedule.num_assigned_nodes_per_superstep_processor();

    unsigned current_step = 0;
    unsigned current_processor = 0;

    step_proc_ptr[current_step][current_processor] = 0;

    for (const auto &node : perm_inv) {

        if (schedule.assignedProcessor(node) != current_processor || schedule.assignedSuperstep(node) != current_step) {
            
            while (schedule.assignedProcessor(node) != current_processor ||
                   schedule.assignedSuperstep(node) != current_step) {

                if (current_processor < instance->numberOfProcessors() - 1) {
                    current_processor++;
                } else {
                    current_processor = 0;
                    current_step++;
                }
            }

            step_proc_ptr[current_step][current_processor] = row_ptr.size();

        }

        row_ptr.push_back(col_idx.size());

        std::set<VertexType> parents;

        for (const auto &edge : instance->getComputationalDag().in_edges(node)) {
            parents.insert(perm[instance->getComputationalDag().source(edge)]);
        }

        for (const auto &p : parents) {
            col_idx.push_back(p);
            auto pair = boost::edge(perm_inv[p], node, instance->getComputationalDag().getGraph());
            assert(pair.second);
            val.push_back(instance->getComputationalDag().edge_mtx_entry(pair.first));

        }

        col_idx.push_back(perm[node]);
        val.push_back(instance->getComputationalDag().node_mtx_entry(node));

    }

    row_ptr.push_back(col_idx.size());
}

void BspSptrsvCSR::setup_csr_snake(const BspSchedule &schedule, std::vector<size_t> &perm) {

    std::vector<size_t> perm_inv(perm.size());
    for (size_t i = 0; i < perm.size(); i++) {
        perm_inv[perm[i]] = i;
    }

    num_supersteps = schedule.numberOfSupersteps();

    x = std::vector<double>(instance->numberOfVertices(), 1.0);


    val.clear();
    val.reserve(instance->getComputationalDag().numberOfEdges() + instance->getComputationalDag().numberOfVertices());

    col_idx.clear();
    col_idx.reserve(instance->getComputationalDag().numberOfEdges() +
                    instance->getComputationalDag().numberOfVertices());

    row_ptr.clear();
    row_ptr.reserve(instance->numberOfVertices() + 1);

    step_ptr.clear();
    step_ptr.reserve(instance->numberOfVertices());

    step_proc_ptr =
        std::vector<std::vector<unsigned>>(num_supersteps, std::vector<unsigned>(instance->numberOfProcessors(), 0));

    step_proc_num = schedule.num_assigned_nodes_per_superstep_processor();

    unsigned current_step = 0;
    unsigned current_processor = 0;
    bool reverse = false;

    step_proc_ptr[current_step][current_processor] = 0;

    for (const auto &node : perm_inv) {

        if (schedule.assignedProcessor(node) != current_processor || schedule.assignedSuperstep(node) != current_step) {

            while (schedule.assignedProcessor(node) != current_processor ||
                   schedule.assignedSuperstep(node) != current_step) {
                if (reverse) {
                    if (current_processor > 0) {
                        current_processor--;
                    } else {
                        reverse = !reverse;
                        current_step++;
                    }
                } else {
                    if (current_processor < instance->numberOfProcessors() - 1) {
                        current_processor++;
                    } else {
                        reverse = !reverse;
                        current_step++;
                    }
                }
            }

            step_proc_ptr[current_step][current_processor] = row_ptr.size();
        }

        row_ptr.push_back(col_idx.size());

        std::set<VertexType> parents;

        for (const auto &edge : instance->getComputationalDag().in_edges(node)) {
            parents.insert(perm[instance->getComputationalDag().source(edge)]);
        }

        for (const auto &p : parents) {
            col_idx.push_back(p);
            auto pair = boost::edge(perm_inv[p], node, instance->getComputationalDag().getGraph());
            assert(pair.second);
            val.push_back(instance->getComputationalDag().edge_mtx_entry(pair.first));

        }

        col_idx.push_back(perm[node]);
        val.push_back(instance->getComputationalDag().node_mtx_entry(node));

    }

    row_ptr.push_back(col_idx.size());
}

void BspSptrsvCSR::simulate_sptrsv_serial() {

    for (unsigned i = 0; i < instance->numberOfVertices(); i++) {
        for (unsigned j = row_ptr[i]; j < row_ptr[i + 1] - 1; j++) {
            x[i] -= val[j] * x[col_idx[j]];
        }
        x[i] /= val[row_ptr[i + 1] - 1];
    }
}

void BspSptrsvCSR::simulate_sptrsv() {

#pragma omp parallel num_threads(instance->numberOfProcessors())
    {
        for (unsigned step = 0; step < num_supersteps; step++) {

            const unsigned int proc = omp_get_thread_num();
            // #pragma omp for schedule(static, 1)
            //             for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

            // nodes assigned to proc
            const unsigned upper_limit = step_proc_ptr[step][proc] + step_proc_num[step][proc];

            for (unsigned row_idx = step_proc_ptr[step][proc]; row_idx < upper_limit; row_idx++) {

                for (unsigned i = row_ptr[row_idx]; i < row_ptr[row_idx + 1] - 1; i++) {

                    x[row_idx] -= val[i] * x[col_idx[i]];
                }

                x[row_idx] /= val[row_ptr[row_idx + 1] - 1];
            }

#pragma omp barrier
        }
    }
}

void BspSptrsvCSR::permute_vector(std::vector<double> &vec, const std::vector<size_t> &perm) {

    std::vector<double> vec_perm(vec.size());

    for (size_t i = 0; i < perm.size(); i++) {
        vec_perm[i] = vec[perm[i]];
    }

    vec = vec_perm;
}

void BspSptrsvCSR::setup_csr_no_permutation(const BspSchedule &schedule) {

    vector_step_processor_vertices = std::vector<std::vector<std::vector<VertexType>>>(
        schedule.numberOfSupersteps(),
        std::vector<std::vector<VertexType>>(schedule.getInstance().numberOfProcessors()));

    x = std::vector<double>(instance->numberOfVertices(), 1.0);

    num_supersteps = schedule.numberOfSupersteps();

    val.clear();
    val.reserve(instance->getComputationalDag().numberOfEdges() + instance->getComputationalDag().numberOfVertices());

    col_idx.clear();
    col_idx.reserve(instance->getComputationalDag().numberOfEdges() +
                    instance->getComputationalDag().numberOfVertices());

    row_ptr.clear();
    row_ptr.reserve(instance->numberOfVertices() + 1);

    for (const auto& node : instance->getComputationalDag().dfs_topoOrder()) {

        vector_step_processor_vertices[schedule.assignedSuperstep(node)][schedule.assignedProcessor(node)].push_back(
            node);
    }

    for (unsigned node = 0; node < instance->numberOfVertices(); node++) {

        row_ptr.push_back(col_idx.size());

        std::set<VertexType> parents;

        for (const auto &edge : instance->getComputationalDag().in_edges(node)) {
            parents.insert(instance->getComputationalDag().source(edge));
        }

        for (const auto &p : parents) {
            col_idx.push_back(p);
            auto pair = boost::edge(p, node, instance->getComputationalDag().getGraph());
            assert(pair.second);
            val.push_back(instance->getComputationalDag().edge_mtx_entry(pair.first));

        }

        col_idx.push_back(node);
        val.push_back(instance->getComputationalDag().node_mtx_entry(node));

    }

    row_ptr.push_back(col_idx.size());
}

std::vector<double> BspSptrsvCSR::compute_sptrsv() {

    std::vector<double> x(instance->getComputationalDag().numberOfVertices(), 1.0);

    for (const auto &node : instance->getComputationalDag().dfs_topoOrder()) {

        for (const auto &edge : instance->getComputationalDag().in_edges(node)) {

            x[node] -=
                instance->getComputationalDag().edge_mtx_entry(edge) * x[instance->getComputationalDag().source(edge)];
        }

        x[node] /= instance->getComputationalDag().node_mtx_entry(node);
    }

    return x;
}

void BspSptrsvCSR::simulate_sptrsv_no_permutation() {

#pragma omp parallel num_threads(instance->numberOfProcessors())
    {
        for (unsigned step = 0; step < num_supersteps; step++) {

            const unsigned int proc = omp_get_thread_num();

            for (const auto &node : vector_step_processor_vertices[step][proc]) {

                const unsigned row_idx = row_ptr[node];

                for (unsigned i = row_idx; i < row_ptr[node + 1] - 1; i++) {

                    x[node] -= val[i] * x[col_idx[i]];
                }

                x[node] /= val[row_ptr[node + 1] - 1];
            }
#pragma omp barrier
        }
    }
}

void BspSptrsvCSR::simulate_sptrsv_graph_mtx() {

#pragma omp parallel num_threads(instance->numberOfProcessors())
    {
        for (unsigned step = 0; step < num_supersteps; step++) {

            const unsigned int proc = omp_get_thread_num();

            for (const auto &node : vector_step_processor_vertices[step][proc]) {

                for (const auto &edge : instance->getComputationalDag().in_edges(node)) {
                    x[node] -= instance->getComputationalDag().edge_mtx_entry(edge) *
                               x[instance->getComputationalDag().source(edge)];
                }

                x[node] /= instance->getComputationalDag().node_mtx_entry(node);
            }
#pragma omp barrier
        }
    }
}

void BspSptrsvCSR::simulate_sptrsv_no_barrier() {

#pragma omp parallel num_threads(instance->numberOfProcessors())
    {

        for (unsigned step = 0; step < num_supersteps; step++) {

            const unsigned int proc = omp_get_thread_num();

            for (unsigned k = 0; k < vector_step_processor_vertices[step][proc].size();) {

                const unsigned node = vector_step_processor_vertices[step][proc][k];

                bool advance = false;
                if (ready[node] <= 0) {
                    advance = true;
                }

                const unsigned row_idx = row_ptr[node];
                const unsigned row_end = row_idx + instance->getComputationalDag().numberOfParents(node);

                for (unsigned i = row_idx; i < row_end; i++) {

                    x[node] -= val[i] * x[col_idx[i]];
                }

                x[node] /= val[row_end];

                if (advance) {

                    for (const auto &child : instance->getComputationalDag().children(node)) {

#pragma omp atomic update
                        ready[child]--;
                    }
                    k++;
                } else {
                    x[node] = 1.0;
                }
            }
        }
    }
}

void BspSptrsvCSR::setup_csr_no_barrier(const BspSchedule &schedule, std::vector<size_t> &perm) {

    std::vector<size_t> perm_inv(perm.size());
    for (size_t i = 0; i < perm.size(); i++) {
        perm_inv[perm[i]] = i;
    }

    ready = std::vector<int>(instance->numberOfVertices(), 0);

    for (const auto &node : instance->getComputationalDag().vertices()) {

        ready[node] = instance->getComputationalDag().numberOfParents(node);
    }

    vector_step_processor_vertices = std::vector<std::vector<std::vector<VertexType>>>(
        schedule.numberOfSupersteps(),
        std::vector<std::vector<VertexType>>(schedule.getInstance().numberOfProcessors()));

    num_supersteps = schedule.numberOfSupersteps();

    x = std::vector<double>(instance->numberOfVertices(), 1.0);

    for (const auto &node : perm_inv) {

        vector_step_processor_vertices[schedule.assignedSuperstep(node)][schedule.assignedProcessor(node)].push_back(
            node);
    }

    val.clear();
    val.reserve(instance->getComputationalDag().numberOfEdges() + instance->getComputationalDag().numberOfVertices());

    col_idx.clear();
    col_idx.reserve(instance->getComputationalDag().numberOfEdges() +
                    instance->getComputationalDag().numberOfVertices());

    row_ptr = std::vector<unsigned>(instance->numberOfVertices());

    for (const auto &node : perm_inv) {

        row_ptr[node] = col_idx.size();

        std::set<VertexType> parents;

        for (const auto &edge : instance->getComputationalDag().in_edges(node)) {
            parents.insert(perm[instance->getComputationalDag().source(edge)]);
        }

        for (const auto &p : parents) {
            col_idx.push_back(perm_inv[p]);
            auto pair = boost::edge(perm_inv[p], node, instance->getComputationalDag().getGraph());
            assert(pair.second);
            val.push_back(instance->getComputationalDag().edge_mtx_entry(pair.first));
        }

        col_idx.push_back(perm[node]);
        val.push_back(instance->getComputationalDag().node_mtx_entry(node));

    }
}