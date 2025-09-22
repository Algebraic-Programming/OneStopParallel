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

#include <omp.h>
#include "kl_improver.hpp"

namespace osp {



template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t = no_local_search_memory_constraint,
         unsigned window_size = 1, typename cost_t = double>
class kl_improver_mt : public kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t> {

  protected:

    unsigned max_num_threads = std::numeric_limits<unsigned>::max();

    void set_thread_boundaries(const unsigned num_threads, const unsigned num_steps, bool last_thread_large_range) {

        if (num_threads == 1) {
            this->set_start_step(0, this->thread_data_vec[0]);
            this->thread_data_vec[0].end_step = (num_steps > 0) ? num_steps - 1 : 0;
            this->thread_data_vec[0].original_end_step = this->thread_data_vec[0].end_step;
            return;
        } else {
            const unsigned total_gap_size = (num_threads - 1) * this->parameters.thread_range_gap;
            const unsigned bonus = this->parameters.thread_min_range;
            const unsigned steps_to_distribute = num_steps - total_gap_size - bonus;
            const unsigned base_range = steps_to_distribute / num_threads;
            const unsigned remainder = steps_to_distribute % num_threads;
            const unsigned large_range_thread_idx = last_thread_large_range ? num_threads - 1 : 0;

            unsigned current_start_step = 0;
            for (unsigned i = 0; i < num_threads; ++i) {
                this->thread_finished_vec[i] = false;
                this->set_start_step(current_start_step, this->thread_data_vec[i]);
                unsigned current_range = base_range + (i < remainder ? 1 : 0);
                if (i == large_range_thread_idx) {
                    current_range += bonus;
                }

                const unsigned end_step = current_start_step + current_range - 1;
                this->thread_data_vec[i].end_step = end_step;
                this->thread_data_vec[i].original_end_step = this->thread_data_vec[i].end_step;
                current_start_step = end_step + 1 + this->parameters.thread_range_gap;
#ifdef KL_DEBUG_1
                std::cout << "thread " << i << ": start_step=" << this->thread_data_vec[i].start_step << ", end_step=" << this->thread_data_vec[i].end_step << std::endl;
#endif
            }
        }
    }

    void set_num_threads(unsigned &num_threads, const unsigned num_steps) {
        unsigned max_allowed_threads = 0;
        if (num_steps >= this->parameters.thread_min_range + this->parameters.thread_range_gap) {
            const unsigned divisor = this->parameters.thread_min_range + this->parameters.thread_range_gap;
            if (divisor > 0) {
                // This calculation is based on the constraint that one thread's range is
                // 'min_range' larger than the others, and all ranges are at least 'min_range'.
                max_allowed_threads = (num_steps + this->parameters.thread_range_gap - this->parameters.thread_min_range) / divisor;
            } else {
                max_allowed_threads = num_steps;
            }
        } else if (num_steps >= this->parameters.thread_min_range) {
            max_allowed_threads = 1;
        }

        if (num_threads > max_allowed_threads) {
            num_threads = max_allowed_threads;
        }

        if (num_threads == 0) {
            num_threads = 1;
        }
#ifdef KL_DEBUG_1
        std::cout << "num threads: " << num_threads << " number of supersteps: " << num_steps << ", max allowed threads: " << max_allowed_threads << std::endl;
#endif       
    
    }


  public:
  
  kl_improver_mt() : kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>() { }
    virtual ~kl_improver_mt() = default;

    void set_max_num_threads(const unsigned num_threads) {
        max_num_threads = num_threads;
    }

    virtual RETURN_STATUS improveSchedule(BspSchedule<Graph_t> &schedule) override {
        unsigned num_threads = std::min(max_num_threads, static_cast<unsigned>(omp_get_max_threads()));
        set_num_threads(num_threads, schedule.numberOfSupersteps());

        this->thread_data_vec.resize(num_threads);      
        this->thread_finished_vec.assign(num_threads, true);

        if (num_threads == 1) {
            this->parameters.num_parallel_loops = 1; // no parallelization with one thread. Affects parameters.max_out_iteration calculation in set_parameters()
        }

        this->set_parameters(schedule.getInstance().numberOfVertices());
        this->initialize_datastructures(schedule); 
        const cost_t initial_cost = this->active_schedule.get_cost();

        for (size_t i = 0; i < this->parameters.num_parallel_loops; ++i) {
            set_thread_boundaries(num_threads, schedule.numberOfSupersteps(), i % 2 == 0);                       

            #pragma omp parallel num_threads(num_threads) 
            {
                const size_t thread_id = static_cast<size_t>(omp_get_thread_num());
                auto & thread_data = this->thread_data_vec[thread_id];
                thread_data.active_schedule_data.initialize_cost(this->active_schedule.get_cost());
                thread_data.selection_strategy.setup(thread_data.start_step, thread_data.end_step);
                this->run_local_search(thread_data); 
            }
        
            this->synchronize_active_schedule(num_threads);
            if (num_threads > 1) {
                this->active_schedule.set_cost(this->comm_cost_f.compute_schedule_cost());
                set_num_threads(num_threads, schedule.numberOfSupersteps());
                this->thread_finished_vec.resize(num_threads);
            }
        }               

        if (initial_cost > this->active_schedule.get_cost()) {
            this->active_schedule.write_schedule(schedule);
            this->cleanup_datastructures();
            return RETURN_STATUS::OSP_SUCCESS;
        } else {
            this->cleanup_datastructures();
            return RETURN_STATUS::BEST_FOUND;
        }
    }
};

} // namespace osp