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

// #define KL_DEBUG

#include "model/SetSchedule.hpp"
#include "model/VectorSchedule.hpp"
#include "scheduler/ImprovementScheduler.hpp"

struct kl_move {

    VertexType node;

    double gain;
    double change_in_cost;

    unsigned from_proc;
    unsigned from_step;

    unsigned to_proc;
    unsigned to_step;

    kl_move() : node(0), gain(0), change_in_cost(0), from_proc(0), from_step(0), to_proc(0), to_step(0) {}
    kl_move(VertexType node, double gain, double change_cost, unsigned from_proc, unsigned from_step, unsigned to_proc,
            unsigned to_step)
        : node(node), gain(gain), change_in_cost(change_cost), from_proc(from_proc), from_step(from_step),
          to_proc(to_proc), to_step(to_step) {}

    bool operator<(kl_move const &rhs) const {
        return (gain < rhs.gain) or (gain == rhs.gain and change_in_cost < rhs.change_in_cost) or
               (gain == rhs.gain and change_in_cost == rhs.change_in_cost and node > rhs.node);
    }

    kl_move reverse_move() const {
        return kl_move(node, -gain, -change_in_cost, to_proc, to_step, from_proc, from_step);
    }

};

class Ikl_cost_function {
  public:
    virtual double compute_current_costs() = 0;
};


class kl_current_schedule {

  public:
   
    kl_current_schedule(Ikl_cost_function *cost_f_)
        : cost_f(cost_f_) {}

    virtual ~kl_current_schedule() = default;

    Ikl_cost_function *cost_f;

    bool use_memory_constraint = false;

    const BspInstance *instance;

    VectorSchedule vector_schedule;
    SetSchedule set_schedule;

    std::vector<std::vector<int>> step_processor_memory;
    std::vector<std::vector<std::unordered_set<VertexType>>> step_processor_pred;

    std::vector<int> current_proc_persistent_memory;
    std::vector<int> current_proc_transient_memory;

    std::vector<std::vector<int>> step_processor_work;

    std::vector<int> step_max_work;
    std::vector<int> step_second_max_work;

    double current_cost = 0;
    
    bool current_feasible = true;
    std::unordered_set<EdgeType, EdgeType_hash> current_violations; // edges

    std::unordered_map<VertexType, EdgeType> new_violations; 
    std::unordered_set<EdgeType, EdgeType_hash> resolved_violations; 

    void remove_superstep(unsigned step);
    void reset_superstep(unsigned step);
    void recompute_neighboring_supersteps(unsigned step);

    inline unsigned num_steps() const { return vector_schedule.numberOfSupersteps(); }

    virtual void set_current_schedule(const IBspSchedule &schedule);

    virtual void initialize_superstep_datastructures();
    virtual void cleanup_superstep_datastructures();

    virtual void compute_work_memory_datastructures(unsigned start_step, unsigned end_step);
    virtual void recompute_current_violations();

    virtual void apply_move(kl_move move);

    virtual void initialize_current_schedule(const IBspSchedule &schedule) {

#ifdef KL_DEBUG
      std::cout << "KLCurrentSchedule initialize current schedule" << std::endl;
#endif

      vector_schedule = VectorSchedule(schedule);
      set_schedule = SetSchedule(schedule);

      initialize_superstep_datastructures();

      compute_work_memory_datastructures(0, num_steps() - 1);
      recompute_current_violations();

      cost_f->compute_current_costs();
    }

  private:

    void update_violations(VertexType node);
    void update_max_work_datastructures(kl_move move);
    void recompute_superstep_max_work(unsigned step);
    

};



class kl_current_schedule_max_comm : public kl_current_schedule {

  public:
    std::vector<std::vector<int>> step_processor_send;
    std::vector<int> step_max_send;
    std::vector<int> step_max_receive;

    std::vector<std::vector<int>> step_processor_receive;
    std::vector<int> step_second_max_send;
    std::vector<int> step_second_max_receive;
};
