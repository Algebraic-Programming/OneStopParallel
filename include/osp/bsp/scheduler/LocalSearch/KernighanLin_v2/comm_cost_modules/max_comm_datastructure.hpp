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

#include <vector>
#include <set>
#include <algorithm>
#include "osp/bsp/model/BspInstance.hpp"
#include "lambda_container.hpp"
#include "../kl_active_schedule.hpp"

template<typename comm_weight_t>
struct pre_move_comm_data {

    comm_weight_t from_step_max_comm;
    comm_weight_t from_step_second_max_comm;

    comm_weight_t to_step_max_comm;
    comm_weight_t to_step_second_max_comm;

    pre_move_comm_data() = default;
    pre_move_comm_data(comm_weight_t from_max, comm_weight_t from_second_max,
                       comm_weight_t to_max, comm_weight_t to_second_max)
        : from_step_max_comm(from_max), from_step_second_max_comm(from_second_max),
          to_step_max_comm(to_max), to_step_second_max_comm(to_second_max) {}

};

template<typename Graph_t, typename cost_t, typename kl_active_schedule_t>
struct max_comm_datastructure {

    using comm_weight_t = v_commw_t<Graph_t>;
    using VertexType = vertex_idx_t<Graph_t>;
    using kl_move = kl_move_struct<cost_t, VertexType>;

    const BspInstance<Graph_t> *instance;
    const kl_active_schedule_t *active_schedule;
   
    struct comm_proc {
        comm_weight_t comm;
        unsigned proc;

        comm_proc() : comm(0), proc(0) {}
        comm_proc(comm_weight_t c, unsigned p) : comm(c), proc(p) {}
    
        bool operator<(comm_proc const &rhs) const {
            return (comm > rhs.comm) or (comm == rhs.comm and proc < rhs.proc);
        }
    };

    std::vector<std::vector<comm_proc>> step_proc_send_sorted;
    std::vector<std::vector<comm_proc>> step_proc_receive_sorted;

    std::vector<std::vector<comm_weight_t>> step_proc_send;
    std::vector<std::vector<comm_weight_t>> step_proc_receive;

    std::vector<unsigned> step_max_send_processor_count;
    std::vector<unsigned> step_max_receive_processor_count;

    std::vector<comm_weight_t> step_max_comm_cache;
    std::vector<comm_weight_t> step_second_max_comm_cache;

    comm_weight_t max_comm_weight = 0;

    lambda_vector_container node_lambda_map;

    inline comm_weight_t step_proc_send(unsigned step, unsigned proc) const { return step_proc_send[step][proc]; }
    inline comm_weight_t& step_proc_send(unsigned step, unsigned proc) { return step_proc_send[step][proc]; }
    inline comm_weight_t step_proc_receive(unsigned step, unsigned proc) const { return step_proc_receive[step][proc]; }
    inline comm_weight_t& step_proc_receive(unsigned step, unsigned proc) { return step_proc_receive[step][proc]; }

    inline comm_weight_t step_max_send(unsigned step) const { return step_proc_send_sorted[step][0].comm; }
    inline comm_weight_t step_second_max_send(unsigned step) const {
        return step_proc_send_sorted[step][step_max_send_processor_count[step]].comm;
    }

    inline comm_weight_t step_max_receive(unsigned step) const { return step_proc_receive_sorted[step][0].comm; }
    inline comm_weight_t step_second_max_receive(unsigned step) const {
        return step_proc_receive_sorted[step][step_max_receive_processor_count[step]].comm;
    }

    inline comm_weight_t step_max_comm(unsigned step) const { return step_max_comm_cache[step]; }
    inline comm_weight_t step_second_max_comm(unsigned step) const {
        return step_second_max_comm_cache[step];
    }

    template<typename cost_t, typename vertex_idx_t>
    inline pre_move_comm_data<comm_weight_t> get_pre_move_comm_data(const kl_move_struct<cost_t, vertex_idx_t>& move) {
        return pre_move_comm_data<comm_weight_t>(
            step_max_comm(move.from_step), step_second_max_comm(move.from_step),
            step_max_comm(move.to_step), step_second_max_comm(move.to_step)
        );
    }

    template<typename cost_t>
    inline pre_move_comm_data_step<cost_t> get_pre_move_comm_data_step(unsigned step) const {
        return pre_move_comm_data_step<cost_t>(
            step_max_comm(step), step_second_max_comm(step), 0, 0
        );
    }

    inline void initialize( kl_active_schedule_t &kl_sched) {        
        active_schedule = &kl_sched;
        instance = & active_schedule->getInstance();
        const unsigned num_steps = active_schedule->num_steps();
        const unsigned num_procs = instance->numberOfProcessors();
        max_comm_weight = 0;

        step_proc_send.assign(num_steps, std::vector<comm_weight_t>(num_procs, 0));
        step_proc_receive.assign(num_steps, std::vector<comm_weight_t>(num_procs, 0));

        step_proc_send_sorted.assign(num_steps, std::vector<comm_proc>(num_procs));
        step_proc_receive_sorted.assign(num_steps, std::vector<comm_proc>(num_procs));

        step_max_send_processor_count.assign(num_steps, 0);
        step_max_receive_processor_count.assign(num_steps, 0);
        step_max_comm_cache.assign(num_steps, 0);
        step_second_max_comm_cache.assign(num_steps, 0);

        node_lambda_map.initialize(instance->getComputationalDag().num_vertices(), num_procs);
    }

    inline void clear() {
        step_proc_send.clear();
        step_proc_receive.clear();
        step_proc_send_sorted.clear();
        step_proc_receive_sorted.clear();
        step_max_send_processor_count.clear();
        step_max_receive_processor_count.clear();
        step_max_comm_cache.clear();
        step_second_max_comm_cache.clear();
        node_lambda_map.clear();
    }

    inline void arrange_superstep_comm_data(const unsigned step) {
        for (unsigned p = 0; p < instance->numberOfProcessors(); ++p) {
            step_proc_send_sorted[step][p] = {step_proc_send[step][p], p};
            step_proc_receive_sorted[step][p] = {step_proc_receive[step][p], p};
        }
        std::sort(step_proc_send_sorted[step].begin(), step_proc_send_sorted[step].end());
        std::sort(step_proc_receive_sorted[step].begin(), step_proc_receive_sorted[step].end());

        const comm_weight_t max_send = step_proc_send_sorted[step][0].comm;
        unsigned send_count = 1;
        while (send_count < instance->numberOfProcessors() && step_proc_send_sorted[step][send_count].comm == max_send) {
            send_count++;
        }
        step_max_send_processor_count[step] = send_count;

        const comm_weight_t max_receive = step_proc_receive_sorted[step][0].comm;
        unsigned receive_count = 1;
        while (receive_count < instance->numberOfProcessors() && step_proc_receive_sorted[step][receive_count].comm == max_receive) {
            receive_count++;
        }
        step_max_receive_processor_count[step] = receive_count;

        step_max_comm_cache[step] = std::max(max_send, max_receive);

        const comm_weight_t second_max_send = step_proc_send_sorted[step][send_count].comm;
        const comm_weight_t second_max_receive = step_proc_receive_sorted[step][receive_count].comm;

        step_second_max_comm_cache[step] = std::max(std::max(second_max_send, max_receive), std::max(max_send, second_max_receive));

    }

    void recompute_max_send_receive(unsigned step) {
        arrange_superstep_comm_data(step);
    }
    
    void update_datastructure_after_move(const kl_move& move, unsigned start_step, unsigned end_step) {
 
    }

    void swap_steps(const unsigned step1, const unsigned step2) {
        std::swap(step_proc_send[step1], step_proc_send[step2]);
        std::swap(step_proc_receive[step1], step_proc_receive[step2]);
        std::swap(step_proc_send_sorted[step1], step_proc_send_sorted[step2]);
        std::swap(step_proc_receive_sorted[step1], step_proc_receive_sorted[step2]);
        std::swap(step_max_send_processor_count[step1], step_max_send_processor_count[step2]);
        std::swap(step_max_receive_processor_count[step1], step_max_receive_processor_count[step2]);
        std::swap(step_max_comm_cache[step1], step_max_comm_cache[step2]);
        std::swap(step_second_max_comm_cache[step1], step_second_max_comm_cache[step2]);
    }

    void reset_superstep(unsigned step) {
        std::fill(step_proc_send[step].begin(), step_proc_send[step].end(), 0);
        std::fill(step_proc_receive[step].begin(), step_proc_receive[step].end(), 0);
        arrange_superstep_comm_data(step);
    }

    void compute_comm_datastructures(unsigned start_step, unsigned end_step) {
        for (unsigned step = start_step; step <= end_step; step++) {
            std::fill(step_proc_send[step].begin(), step_proc_send[step].end(), 0);
            std::fill(step_proc_receive[step].begin(), step_proc_receive[step].end(), 0);
        }

        const auto & vec_sched = active_schedule->getVectorSchedule();
        const auto & graph = instance->getComputationalDag();

        for (const auto &u : graph.vertices()) {
            node_lambda_map.reset_node(u);
            const unsigned u_proc = vec_sched.assignedProcessor(u);
            const unsigned u_step = vec_sched.assignedSuperstep(u);
            const comm_weight_t comm_w = graph.vertex_comm_weight(u);
            max_comm_weight = std::max(max_comm_weight, comm_w);

            bool has_child_on_other_proc = false;
            for (const auto &v : graph.children(u)) {
                const unsigned v_proc = vec_sched.assignedProcessor(v);
                if (u_proc != v_proc) {
                    if (node_lambda_map.increase_proc_count(u, v_proc)) {                        
                        has_child_on_other_proc = true;
                        step_proc_receive[u_step][v_proc] += comm_w;
                    }
                }
            }

            if(has_child_on_other_proc)
                step_proc_send[u_step][u_proc] += comm_w;
        }
        
        for (unsigned step = start_step; step <= end_step; step++) {
            arrange_superstep_comm_data(step);
        }
    }
};