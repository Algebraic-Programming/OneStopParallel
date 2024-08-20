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
#include <chrono>
#include <deque>
#include <limits>
#include <list>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "boost_extensions/transitive_edge_reduction.hpp"
#include <boost/heap/fibonacci_heap.hpp>

#include "scheduler/ImprovementScheduler.hpp"
#include "auxiliary/auxiliary.hpp"
#include "model/SetSchedule.hpp"
#include "model/VectorSchedule.hpp"

// #define LK_DEBUG

template<typename T>
class LKBase : public ImprovementScheduler {

  protected:
    struct Move {

        VertexType node;

        double gain;
        T change_in_cost;

        unsigned from_proc;
        unsigned from_step;

        unsigned to_proc;
        unsigned to_step;

        Move() : node(0), gain(0), change_in_cost(0), from_proc(0), from_step(0), to_proc(0), to_step(0) {}
        Move(VertexType node, double gain, T change_cost, unsigned from_proc, unsigned from_step, unsigned to_proc,
             unsigned to_step)
            : node(node), gain(gain), change_in_cost(change_cost), from_proc(from_proc), from_step(from_step),
              to_proc(to_proc), to_step(to_step) {}

        bool operator<(Move const &rhs) const { return gain < rhs.gain; }
    };

    boost::heap::fibonacci_heap<Move> max_gain_heap;
    using heap_handle = typename boost::heap::fibonacci_heap<Move>::handle_type;

    std::unordered_map<VertexType, heap_handle> node_heap_handles;

    unsigned counter = 0;
    unsigned step_selection_counter = 1;
    unsigned epoch_counter = 0;

    double current_cost = 0;

    std::mt19937 gen;
    //

    // current schedule
    VectorSchedule vector_schedule;
    SetSchedule set_schedule;

    const BspInstance *instance;

    BspSchedule *best_schedule;
    T best_schedule_costs;

    unsigned num_nodes;
    unsigned num_procs;
    unsigned num_steps;

    // std::vector<unsigned> best_node_step, best_node_proc;

    std::unordered_set<VertexType> node_selection;

    std::vector<std::vector<std::vector<double>>> node_gains;
    std::vector<std::vector<std::vector<T>>> node_change_in_costs;

    std::vector<std::vector<unsigned>> step_processor_memory;

    std::vector<std::vector<T>> step_processor_work;
    std::vector<std::vector<T>> step_processor_send;
    std::vector<std::vector<T>> step_processor_receive;

    std::vector<T> step_max_work;
    std::vector<T> step_max_send;
    std::vector<T> step_max_receive;

    std::vector<T> step_second_max_work;
    std::vector<T> step_second_max_send;
    std::vector<T> step_second_max_receive;

    bool current_feasible;
    std::unordered_set<EdgeType, EdgeType_hash> current_violations; // edges

    std::unordered_set<VertexType> locked_nodes;
    std::vector<unsigned> unlock;

    virtual void commputeCommGain(unsigned node, unsigned current_step, unsigned current_proc, unsigned new_proc) = 0;
    virtual void update_superstep_datastructures(Move move) = 0;
    virtual void compute_superstep_datastructures() = 0;
    virtual T compute_current_costs() = 0;

    virtual void computeNodeGain(unsigned node);
    virtual void computeWorkGain(unsigned node, unsigned current_step, unsigned current_proc, unsigned new_proc);

    virtual void applyMove(Move move, bool update = true);
    virtual void reverseMove(Move move, bool update = true);

    virtual void compute_superstep_work_datastructures(unsigned start_step, unsigned end_step);

    virtual void collectNodesToUpdate(Move move, std::unordered_set<VertexType> &nodes);

    virtual void updateViolations(VertexType node, std::unordered_map<VertexType, EdgeType> &new_violations,
                                  std::unordered_set<EdgeType, EdgeType_hash> *resolved_violations = nullptr);

    virtual double computeMaxGain(VertexType node);
    virtual std::pair<unsigned, unsigned> best_move_change_superstep(VertexType node);
    virtual Move compute_best_move(VertexType node);

    virtual void recompute_superstep_max_work(unsigned step);
    virtual void initalize_datastructures();
    virtual void initalize_superstep_datastructures();

    virtual void initalize_gain_heap(const std::unordered_set<VertexType> &nodes);

    virtual void cleanup_datastructures();
    virtual void cleanup_superstep_datastructures();

    virtual void initializeRewardPenaltyFactors();
    virtual void updateRewardPenaltyFactors();

    virtual void resetLockedNodesAndComputeGains();
    virtual void setup_gain_heap_unlocked_nodes();

    virtual void resetGainHeap();

    virtual Move findMove();

    virtual bool start() = 0;

    virtual void resetLockedNodes();
    virtual bool unlockNode(VertexType node);
    virtual void unlockEdge(EdgeType edge);
    virtual void unlockNeighbours(Move move, std::unordered_set<VertexType> &unlocked);
    virtual void unlockEdgeNeighbors(EdgeType edge, std::unordered_set<VertexType> &unlocked);

    virtual void updateNodesGain(const std::unordered_set<VertexType> &nodes);

    virtual void setCurrentSchedule(const IBspSchedule &schedule);
    virtual void setBestSchedule(const IBspSchedule &schedule);
    virtual void reverseMoveBestSchedule(Move move);

    virtual std::unordered_set<VertexType> selectNodesThreshold();
    virtual std::unordered_set<VertexType> selectNodesConseqSteps(unsigned threshold);
    virtual std::unordered_set<VertexType> selectNodesConseqStepsMaxWork(unsigned threshold);
    virtual std::unordered_set<VertexType> selectNodesThreshold(unsigned threshold);
    virtual std::unordered_set<VertexType> selectNodesPermutationThreshold();
    virtual std::unordered_set<VertexType> selectNodesPermutationThreshold(unsigned threshold);
    virtual std::unordered_set<VertexType> selectNodesCollision(unsigned num = 5);

    virtual std::unordered_set<VertexType> selectNodesConseqStepsReduceNumSteps(unsigned threshold);

    virtual std::unordered_set<VertexType> selectNodesFindRemoveSteps(unsigned threshold);

    virtual void setParameters();

    virtual void select_nodes();

    virtual void checkMergeSupersteps();
    virtual void checkInsertSuperstep();

    virtual void insertSuperstep(unsigned step);

    virtual bool check_remove_superstep(unsigned step);
    virtual void remove_superstep(unsigned step);

    void printHeap();

    unsigned max_epochs;

    unsigned max_inner_iterations;

    unsigned max_num_unlocks;
    unsigned max_iterations;
    unsigned selection_threshold;

    double penalty_factor;
    double base_penalty_factor;
    double reward_factor;
    double base_reward_factor;
    double base_reward;

    bool compute_with_time_limit = false;
    bool use_memory_constraint = false;
    bool contract_transitive_edges = false;

    bool quick_pass = false;
    int initial_depth = 3;

  public:
    LKBase()
        : ImprovementScheduler(), best_schedule_costs(0), current_feasible(true), max_epochs(20), max_num_unlocks(3),
          max_iterations(500), penalty_factor(1.0), base_penalty_factor(20.0), reward_factor(1.0),
          base_reward_factor(5.0), base_reward(1.0) {

        std::random_device rd;
        gen = std::mt19937(rd());
    }

    virtual ~LKBase() = default;

    virtual RETURN_STATUS improveSchedule(BspSchedule &schedule) override;
    virtual RETURN_STATUS improveScheduleWithTimeLimit(BspSchedule &schedule) override;

    virtual void setTimeLimitSeconds(unsigned limit) override;

    virtual void setUseMemoryConstraint(bool use_memory_constraint_) override {
        use_memory_constraint = use_memory_constraint_;
    }

    virtual void setContractTransitiveEdges(bool contract_transitive_edges_) {
        contract_transitive_edges = contract_transitive_edges_;
    }

    virtual std::string getScheduleName() const = 0;

    virtual void set_quick_pass(bool quick_pass_) { quick_pass = quick_pass_; }
};

template<typename T>
void LKBase<T>::setTimeLimitSeconds(unsigned limit) {

    compute_with_time_limit = true;
    timeLimitSeconds = limit;
}

template<typename T>
void LKBase<T>::select_nodes() {

    node_selection = selectNodesFindRemoveSteps(selection_threshold);
}

template<typename T>
void LKBase<T>::setParameters() {

    max_num_unlocks = 3;
    max_inner_iterations = 500;

    if (num_nodes < 250) {

        max_iterations = 100;

        selection_threshold = num_nodes * 0.33;

    } else if (num_nodes < 1000) {

        max_iterations = num_nodes / 2;

        selection_threshold = num_nodes * 0.33;

    } else if (num_nodes < 5000) {

        max_iterations = 4 * std::sqrt(num_nodes);

        selection_threshold = num_nodes * 0.33;

    } else if (num_nodes < 10000) {

        max_iterations = 3 * std::sqrt(num_nodes);

        selection_threshold = num_nodes * 0.33;

    } else if (num_nodes < 50000) {

        max_iterations = std::sqrt(num_nodes);

        selection_threshold = num_nodes * 0.1;

    } else if (num_nodes < 100000) {

        max_iterations = 2 * std::log(num_nodes);

        selection_threshold = num_nodes * 0.1;

    } else {

        max_iterations = std::log(num_nodes);

        selection_threshold = num_nodes * 0.1;
    }
}

template<typename T>
RETURN_STATUS LKBase<T>::improveSchedule(BspSchedule &schedule) {

    best_schedule = &schedule;

    if (contract_transitive_edges) {

        auto g = best_schedule->getInstance().getComputationalDag().getGraph();
        approx_transitive_edge_reduction filter(g);

        boost::filtered_graph<GraphType, approx_transitive_edge_reduction> fg(g, filter);

        ComputationalDag f_dag;
        boost::copy_graph(fg, f_dag.getGraph());

        BspInstance *f_instance = new BspInstance(f_dag, best_schedule->getInstance().getArchitecture());

        instance = f_instance;
    } else {
        instance = &best_schedule->getInstance();
    }

    num_nodes = instance->numberOfVertices();
    num_procs = instance->numberOfProcessors();
    num_steps = best_schedule->numberOfSupersteps();

    bool improvement_found = start();

    assert(best_schedule->satisfiesPrecedenceConstraints());

    if (contract_transitive_edges) {
        delete instance;
    }

    schedule.setImprovedLazyCommunicationSchedule();

    if (improvement_found)
        return SUCCESS;
    else
        return BEST_FOUND;
};

template<typename T>
RETURN_STATUS LKBase<T>::improveScheduleWithTimeLimit(BspSchedule &schedule) {

    best_schedule = &schedule;

    instance = &best_schedule->getInstance();
    num_nodes = instance->numberOfVertices();
    num_procs = instance->numberOfProcessors();
    num_steps = best_schedule->numberOfSupersteps();

    compute_with_time_limit = true;

    bool improvement_found = start();

    assert(best_schedule->satisfiesPrecedenceConstraints());

    schedule.setAutoCommunicationSchedule();

    if (improvement_found)
        return SUCCESS;
    else
        return BEST_FOUND;
};

template<typename T>
void LKBase<T>::remove_superstep(unsigned step) {

    assert(step < num_steps);

    for (unsigned proc = 0; proc < num_procs; proc++) {
        for (const auto &node : set_schedule.step_processor_vertices[step][proc]) {

            computeNodeGain(node);
            auto pair = best_move_change_superstep(node);

            vector_schedule.setAssignedSuperstep(node, pair.first);
            vector_schedule.setAssignedProcessor(node, pair.second);

            set_schedule.step_processor_vertices[pair.first][pair.second].insert(node);

            std::unordered_map<VertexType, EdgeType> q;
            updateViolations(node, q);
        }
        set_schedule.step_processor_vertices[step][proc].clear();
    }

    if (step > 0) {
        vector_schedule.mergeSupersteps(step - 1, step);
        set_schedule.mergeSupersteps(step - 1, step);
    } else {
        vector_schedule.mergeSupersteps(0, 1);
        set_schedule.mergeSupersteps(0, 1);
    }

    num_steps -= 1;
    compute_superstep_work_datastructures(step - 1, step);

    for (unsigned i = step + 1; i < num_steps - 1; i++) {

        step_max_work[i] = step_max_work[i + 1];
        step_second_max_work[i] = step_second_max_work[i + 1];

        for (unsigned proc = 0; proc < num_procs; proc++) {

            step_processor_work[i][proc] = step_processor_work[i + 1][proc];
        }
    }

    step_second_max_work[num_steps - 1] = 0;
    step_max_work[num_steps - 1] = 0;
}

template<typename T>
void LKBase<T>::checkMergeSupersteps() {

    if (num_steps < 2)
        return;

    for (unsigned step = 0; step < num_steps - 1; step++) {

        unsigned min_work = step_max_work[step];
        unsigned avg_work = 0;

        for (unsigned proc = 0; proc < num_procs; proc++) {

            avg_work += step_processor_work[step][proc];

            if (step_processor_work[step][proc] < min_work) {
                min_work = step_processor_work[step][proc];
            }
        }

        avg_work = avg_work / num_procs;

        std::cout << "step " << step << " " << " min work: " << min_work << " avg work: " << avg_work
                  << " max work: " << step_max_work[step] << std::endl;
    }
}

template<typename T>
bool LKBase<T>::check_remove_superstep(unsigned step) {

    unsigned total_work = 0;

    for (unsigned proc = 0; proc < num_procs; proc++) {

        total_work += step_processor_work[step][proc];
    }

    if (total_work < instance->synchronisationCosts()) {

        remove_superstep(step);
        return true;
    }

    return false;
}

template<typename T>
void LKBase<T>::checkInsertSuperstep() {

    if (current_violations.size() < 25)
        return;

    std::vector<unsigned> step_num_viol(num_steps, 0);

    for (const auto &edge : current_violations) {
        const auto &source = instance->getComputationalDag().source(edge);
        const auto &target = instance->getComputationalDag().target(edge);

        step_num_viol[vector_schedule.assignedSuperstep(source)]++;
        step_num_viol[vector_schedule.assignedSuperstep(target)]++;
    }

    unsigned max = 0;
    unsigned max_step = 0;
    for (unsigned step = 0; step < num_steps; step++) {

        if (step_num_viol[step] > max) {
            max = step_num_viol[step];
            max_step = step;
        }
    }

    if (max > current_violations.size() * 0.5) {

        if (max_step == 0) {
            insertSuperstep(0);

        } else if (max_step == num_steps - 1) {

            insertSuperstep(num_steps - 1);
        } else {

            if (step_num_viol[max_step - 1] < step_num_viol[max_step + 1]) {
                insertSuperstep(max_step - 1);

            } else {
                insertSuperstep(max_step);
            }
        }
    }
}

template<typename T>
void LKBase<T>::insertSuperstep(unsigned step_before) {

    set_schedule.insertSupersteps(step_before, 1);
    vector_schedule.insertSupersteps(step_before, 1);

    step_processor_work.push_back(std::move(step_processor_work[num_steps - 1]));
    step_processor_send.push_back(std::move(step_processor_send[num_steps - 1]));
    step_processor_receive.push_back(std::move(step_processor_receive[num_steps - 1]));

    step_max_work.push_back(step_max_work[num_steps - 1]);
    step_max_send.push_back(step_max_send[num_steps - 1]);
    step_max_receive.push_back(step_max_receive[num_steps - 1]);

    step_second_max_work.push_back(step_second_max_work[num_steps - 1]);
    step_second_max_send.push_back(step_second_max_send[num_steps - 1]);
    step_second_max_receive.push_back(step_second_max_receive[num_steps - 1]);

    for (unsigned step = num_steps - 2; step > step_before; step--) {

        step_max_work[step + 1] = step_max_work[step];
        step_max_send[step + 1] = step_max_send[step];
        step_max_receive[step + 1] = step_max_receive[step];

        step_second_max_work[step + 1] = step_second_max_work[step];
        step_second_max_send[step + 1] = step_second_max_send[step];
        step_second_max_receive[step + 1] = step_second_max_receive[step];

        step_processor_work[step + 1] = std::move(step_processor_work[step]);
        step_processor_send[step + 1] = std::move(step_processor_send[step]);
        step_processor_receive[step + 1] = std::move(step_processor_receive[step]);
    }

    num_steps += 1;

    step_max_work[step_before + 1] = 0;
    step_max_send[step_before + 1] = 0;
    step_max_receive[step_before + 1] = 0;

    step_second_max_work[step_before + 1] = 0;
    step_second_max_send[step_before + 1] = 0;
    step_second_max_receive[step_before + 1] = 0;

    step_processor_work[step_before + 1] = std::vector<T>(num_procs, 0);
    step_processor_send[step_before + 1] = std::vector<T>(num_procs, 0);
    step_processor_receive[step_before + 1] = std::vector<T>(num_procs, 0);

    unsigned last_step = std::min(num_steps - 1, step_before + 2);

    for (unsigned i = step_before; i <= last_step; i++) {

        if (i != step_before + 1) {
            for (unsigned proc = 0; proc < num_procs; proc++) {

                for (const auto &node : set_schedule.step_processor_vertices[i][proc]) {

                    if (locked_nodes.find(node) != locked_nodes.end()) {

                        if (unlockNode(node)) {
                            computeNodeGain(node);
                            computeMaxGain(node);
                        }

                    } else {
                        computeNodeGain(node);
                        computeMaxGain(node);
                    }
                }
            }
        }
    }
}

template<typename T>
std::unordered_set<VertexType> LKBase<T>::selectNodesConseqSteps(unsigned threshold) {

    std::unordered_set<VertexType> nodes;

    std::uniform_int_distribution<> dis(0, instance->numberOfProcessors() - 1);

    while (nodes.size() < threshold) {
        auto proc = dis(gen);

        std::sample(set_schedule.step_processor_vertices[step_selection_counter][proc].begin(),
                    set_schedule.step_processor_vertices[step_selection_counter][proc].end(),
                    std::inserter(nodes, nodes.end()), 10, gen);
    }

    step_selection_counter++;
    if (step_selection_counter == num_steps) {
        step_selection_counter = 0;
        epoch_counter++;
    }
    return nodes;
}

template<typename T>
std::unordered_set<VertexType> LKBase<T>::selectNodesFindRemoveSteps(unsigned threshold) {

    for (unsigned step_to_remove = step_selection_counter; step_to_remove < num_steps; step_to_remove++) {

        if (check_remove_superstep(step_to_remove)) {

#ifdef LK_DEBUG
            std::cout << "trying to remove superstep " << step_to_remove << std::endl;
#endif
            std::unordered_set<VertexType> nodes;

            for (unsigned proc = 0; proc < num_procs; proc++) {

                nodes.insert(set_schedule.step_processor_vertices[step_selection_counter][proc].begin(),
                             set_schedule.step_processor_vertices[step_selection_counter][proc].end());
                nodes.insert(set_schedule.step_processor_vertices[step_selection_counter - 1][proc].begin(),
                             set_schedule.step_processor_vertices[step_selection_counter - 1][proc].end());
            }

            step_selection_counter = step_to_remove + 1;

            if (step_selection_counter >= num_steps) {
                epoch_counter++;
            }

            return nodes;
        }
    }

    return selectNodesThreshold(threshold);
}

template<typename T>
std::unordered_set<VertexType> LKBase<T>::selectNodesConseqStepsReduceNumSteps(unsigned threshold) {

    if (step_selection_counter > 0 && check_remove_superstep(step_selection_counter)) {

        std::cout << "trying to reduce nr of supersteps" << std::endl;

        std::unordered_set<VertexType> nodes;

        for (unsigned proc = 0; proc < num_procs; proc++) {

            nodes.insert(set_schedule.step_processor_vertices[step_selection_counter][proc].begin(),
                         set_schedule.step_processor_vertices[step_selection_counter][proc].end());
            nodes.insert(set_schedule.step_processor_vertices[step_selection_counter - 1][proc].begin(),
                         set_schedule.step_processor_vertices[step_selection_counter - 1][proc].end());
        }

        step_selection_counter++;
        if (step_selection_counter >= num_steps) {
            step_selection_counter = 0;
            epoch_counter++;
        }

        return nodes;

    } else {
        return selectNodesConseqStepsMaxWork(threshold);
    }
}

template<typename T>
std::unordered_set<VertexType> LKBase<T>::selectNodesConseqStepsMaxWork(unsigned threshold) {

    std::unordered_set<VertexType> nodes;

    unsigned max_work_step = 0;
    unsigned max_step = 0;
    unsigned second_max_work_step = 0;
    unsigned second_max_step = 0;

    for (unsigned proc = 0; proc < num_procs; proc++) {

        if (step_processor_work[step_selection_counter][proc] > max_work_step) {
            second_max_work_step = max_work_step;
            second_max_step = max_step;
            max_work_step = step_processor_work[step_selection_counter][proc];
            max_step = proc;

        } else if (step_processor_work[step_selection_counter][proc] > second_max_work_step) {
            second_max_work_step = step_processor_work[step_selection_counter][proc];
            second_max_step = proc;
        }
    }

    if (set_schedule.step_processor_vertices[step_selection_counter][max_step].size() < threshold * .66) {

        nodes.insert(set_schedule.step_processor_vertices[step_selection_counter][max_step].begin(),
                     set_schedule.step_processor_vertices[step_selection_counter][max_step].end());

    } else {

        std::sample(set_schedule.step_processor_vertices[step_selection_counter][max_step].begin(),
                    set_schedule.step_processor_vertices[step_selection_counter][max_step].end(),
                    std::inserter(nodes, nodes.end()), (unsigned)std::round(threshold * .66), gen);
    }

    if (set_schedule.step_processor_vertices[step_selection_counter][second_max_step].size() < threshold * .33) {

        nodes.insert(set_schedule.step_processor_vertices[step_selection_counter][second_max_step].begin(),
                     set_schedule.step_processor_vertices[step_selection_counter][second_max_step].end());

    } else {

        std::sample(set_schedule.step_processor_vertices[step_selection_counter][second_max_step].begin(),
                    set_schedule.step_processor_vertices[step_selection_counter][second_max_step].end(),
                    std::inserter(nodes, nodes.end()), (unsigned)std::round(threshold * .33), gen);
    }

    step_selection_counter++;
    if (step_selection_counter >= num_steps) {
        step_selection_counter = 0;
        epoch_counter++;
    }

    return nodes;
}

template<typename T>
std::unordered_set<VertexType> LKBase<T>::selectNodesThreshold() {

    std::unordered_set<VertexType> nodes;

    std::uniform_int_distribution<> dis(0, num_nodes - 1);

    unsigned threshold = num_nodes * 0.33;

    while (nodes.size() < threshold) {
        auto node = dis(gen);

        nodes.insert(node);
    }

    return nodes;
}

template<typename T>
std::unordered_set<VertexType> LKBase<T>::selectNodesThreshold(unsigned threshold) {

    std::unordered_set<VertexType> nodes;

    std::uniform_int_distribution<> dis(0, num_nodes - 1);

    while (nodes.size() < threshold) {

        auto node = dis(gen);
        nodes.insert(node);
    }

    return nodes;
}

template<typename T>
std::unordered_set<VertexType> LKBase<T>::selectNodesPermutationThreshold() {

    std::unordered_set<VertexType> nodes;
    std::vector<VertexType> permutation(num_nodes);
    std::iota(std::begin(permutation), std::end(permutation), 0);

    std::shuffle(permutation.begin(), permutation.end(), gen);

    unsigned threshold = num_nodes * 0.33;
    for (unsigned i = 0; i < threshold; i++) {
        nodes.insert(permutation[i]);
    }
    return nodes;
}

template<typename T>
std::unordered_set<VertexType> LKBase<T>::selectNodesPermutationThreshold(unsigned threshold) {

    std::unordered_set<VertexType> nodes;
    std::vector<VertexType> permutation(num_nodes);
    std::iota(std::begin(permutation), std::end(permutation), 0);

    std::shuffle(permutation.begin(), permutation.end(), gen);

    for (unsigned i = 0; i < threshold; i++) {
        nodes.insert(permutation[i]);
    }

    return nodes;
}

template<typename T>
std::unordered_set<VertexType> LKBase<T>::selectNodesCollision(unsigned num_coll) {

    std::unordered_set<VertexType> nodes;
    std::uniform_int_distribution<> dis(0, num_nodes - 1);
    unsigned coll_counter = 0;

    while (coll_counter <= num_coll) {

        auto pair = nodes.insert(dis(gen));

        if (!pair.second)
            coll_counter++;
    }

    return nodes;
}

template<typename T>
void LKBase<T>::initializeRewardPenaltyFactors() {

    penalty_factor = base_penalty_factor * instance->communicationCosts();
    base_reward = base_reward_factor * penalty_factor;
}

template<typename T>
void LKBase<T>::updateRewardPenaltyFactors() {

    penalty_factor = base_penalty_factor * instance->communicationCosts() + std::sqrt(current_violations.size());
    reward_factor = base_reward + current_violations.size();
}

template<typename T>
typename LKBase<T>::Move LKBase<T>::findMove() {

    const unsigned local_max = 50;
    std::vector<VertexType> max_nodes(local_max);
    unsigned count = 0;
    for (auto iter = max_gain_heap.ordered_begin(); iter != max_gain_heap.ordered_end(); ++iter) {

        if (iter->gain == max_gain_heap.top().gain && count < local_max) {
            max_nodes[count] = (iter->node);
            count++;

        } else {
            break;
        }
    }

    unsigned i = randInt(count);
    Move best_move = Move((*node_heap_handles[max_nodes[i]]));

    max_gain_heap.erase(node_heap_handles[max_nodes[i]]);
    node_heap_handles.erase(max_nodes[i]);

    return best_move;
};

template<typename T>
void LKBase<T>::compute_superstep_work_datastructures(unsigned start_step, unsigned end_step) {

    for (unsigned step = start_step; step <= end_step; step++) {

        step_max_work[step] = 0;
        step_second_max_work[step] = 0;

        for (unsigned proc = 0; proc < num_procs; proc++) {

            step_processor_work[step][proc] = 0;

            for (const auto &node : set_schedule.step_processor_vertices[step][proc]) {
                step_processor_work[step][proc] += instance->getComputationalDag().nodeWorkWeight(node);
            }

            if (step_processor_work[step][proc] > step_max_work[step]) {

                step_second_max_work[step] = step_max_work[step];
                step_max_work[step] = step_processor_work[step][proc];

            } else if (step_processor_work[step][proc] > step_second_max_work[step]) {

                step_second_max_work[step] = step_processor_work[step][proc];
            }
        }
    }
}

template<typename T>
void LKBase<T>::applyMove(Move move, bool update) {

    vector_schedule.setAssignedProcessor(move.node, move.to_proc);
    vector_schedule.setAssignedSuperstep(move.node, move.to_step);

    set_schedule.step_processor_vertices[move.from_step][move.from_proc].erase(move.node);
    set_schedule.step_processor_vertices[move.to_step][move.to_proc].insert(move.node);

    if (update) {
        update_superstep_datastructures(move);
    }

    locked_nodes.insert(move.node);

    current_cost -= move.change_in_cost;
}

template<typename T>
void LKBase<T>::reverseMove(Move move, bool update) {

    vector_schedule.setAssignedProcessor(move.node, move.from_proc);
    vector_schedule.setAssignedSuperstep(move.node, move.from_step);

    set_schedule.step_processor_vertices[move.to_step][move.to_proc].erase(move.node);
    set_schedule.step_processor_vertices[move.from_step][move.from_proc].insert(move.node);

    if (update) {
        update_superstep_datastructures(move);
    }

    current_cost += move.change_in_cost;
}

template<typename T>
void LKBase<T>::updateViolations(VertexType node, std::unordered_map<VertexType, EdgeType> &new_violations,
                                 std::unordered_set<EdgeType, EdgeType_hash> *resolved_violations) {

    for (const auto &edge : instance->getComputationalDag().out_edges(node)) {

        const auto &child = instance->getComputationalDag().target(edge);

        if (current_violations.find(edge) == current_violations.end()) {

            if (vector_schedule.assignedSuperstep(node) >= vector_schedule.assignedSuperstep(child)) {

                if (vector_schedule.assignedProcessor(node) != vector_schedule.assignedProcessor(child) ||
                    vector_schedule.assignedSuperstep(node) > vector_schedule.assignedSuperstep(child)) {

                    current_violations.insert(edge);
                    new_violations[child] = edge;
                }
            }
        } else {

            if (vector_schedule.assignedSuperstep(node) <= vector_schedule.assignedSuperstep(child)) {

                if (vector_schedule.assignedProcessor(node) == vector_schedule.assignedProcessor(child) ||
                    vector_schedule.assignedSuperstep(node) < vector_schedule.assignedSuperstep(child)) {

                    current_violations.erase(edge);

                    if (resolved_violations != nullptr) {
                        resolved_violations->insert(edge);
                    }
                }
            }
        }
    }

    for (const auto &edge : instance->getComputationalDag().in_edges(node)) {

        const auto &parent = instance->getComputationalDag().source(edge);

        if (current_violations.find(edge) == current_violations.end()) {

            if (vector_schedule.assignedSuperstep(node) <= vector_schedule.assignedSuperstep(parent)) {

                if (vector_schedule.assignedProcessor(node) != vector_schedule.assignedProcessor(parent) ||
                    vector_schedule.assignedSuperstep(node) < vector_schedule.assignedSuperstep(parent)) {

                    current_violations.insert(edge);
                    new_violations[parent] = edge;
                }
            }
        } else {

            if (vector_schedule.assignedSuperstep(node) >= vector_schedule.assignedSuperstep(parent)) {

                if (vector_schedule.assignedProcessor(node) == vector_schedule.assignedProcessor(parent) ||
                    vector_schedule.assignedSuperstep(node) > vector_schedule.assignedSuperstep(parent)) {

                    current_violations.erase(edge);

                    if (resolved_violations != nullptr) {
                        resolved_violations->insert(edge);
                    }
                }
            }
        }
    }
}

template<typename T>
void LKBase<T>::setBestSchedule(const IBspSchedule &schedule) {

    for (unsigned node = 0; node < num_nodes; node++) {

        best_schedule->setAssignedProcessor(node, schedule.assignedProcessor(node));
        best_schedule->setAssignedSuperstep(node, schedule.assignedSuperstep(node));
    }

    best_schedule->updateNumberOfSupersteps();
    // best_schedule->setNumberOfSupersteps(schedule.numberOfSupersteps());
}

template<typename T>
void LKBase<T>::reverseMoveBestSchedule(Move move) {
    best_schedule->setAssignedProcessor(move.node, move.from_proc);
    best_schedule->setAssignedSuperstep(move.node, move.from_step);
}

template<typename T>
void LKBase<T>::computeNodeGain(unsigned node) {

    const unsigned &current_proc = vector_schedule.assignedProcessor(node);
    const unsigned &current_step = vector_schedule.assignedSuperstep(node);

    for (unsigned new_proc = 0; new_proc < num_procs; new_proc++) {

        node_gains[node][new_proc][0] = 0.0;
        node_gains[node][new_proc][1] = 0.0;
        node_gains[node][new_proc][2] = 0.0;

        node_change_in_costs[node][new_proc][0] = 0;
        node_change_in_costs[node][new_proc][1] = 0;
        node_change_in_costs[node][new_proc][2] = 0;

        commputeCommGain(node, current_step, current_proc, new_proc);
        computeWorkGain(node, current_step, current_proc, new_proc);

        if (use_memory_constraint) {

            if (step_processor_memory[vector_schedule.assignedSuperstep(node)][new_proc] +
                    instance->getComputationalDag().nodeMemoryWeight(node) >
                instance->memoryBound()) {

                node_gains[node][new_proc][1] = std::numeric_limits<T>::lowest();
            }
            if (vector_schedule.assignedSuperstep(node) > 0) {
                if (step_processor_memory[vector_schedule.assignedSuperstep(node) - 1][new_proc] +
                        instance->getComputationalDag().nodeMemoryWeight(node) >
                    instance->memoryBound()) {

                    node_gains[node][new_proc][0] = std::numeric_limits<T>::lowest();
                }
            }

            if (vector_schedule.assignedSuperstep(node) < num_steps - 1) {
                if (step_processor_memory[vector_schedule.assignedSuperstep(node) + 1][new_proc] +
                        instance->getComputationalDag().nodeMemoryWeight(node) >
                    instance->memoryBound()) {

                    node_gains[node][new_proc][2] = std::numeric_limits<T>::lowest();
                }
            }
        }
    }
}

template<typename T>
void LKBase<T>::computeWorkGain(unsigned node, unsigned current_step, unsigned current_proc, unsigned new_proc) {

    if (current_proc == new_proc) {

        node_gains[node][current_proc][1] = std::numeric_limits<double>::lowest();

    } else {

        if (step_max_work[current_step] == step_processor_work[current_step][current_proc] &&
            step_processor_work[current_step][current_proc] > step_second_max_work[current_step]) {

            // new max
            const T new_max_work = std::max(step_processor_work[current_step][current_proc] -
                                                instance->getComputationalDag().nodeWorkWeight(node),
                                            step_second_max_work[current_step]);

            if (step_processor_work[current_step][new_proc] + instance->getComputationalDag().nodeWorkWeight(node) >
                new_max_work) {

                T gain = 0;
                if (step_max_work[current_step] > step_processor_work[current_step][new_proc] +
                                                      instance->getComputationalDag().nodeWorkWeight(node)) {

                    gain = step_max_work[current_step] - (step_processor_work[current_step][new_proc] +
                                                          instance->getComputationalDag().nodeWorkWeight(node));
                } else {
                    gain = (step_processor_work[current_step][new_proc] +
                            instance->getComputationalDag().nodeWorkWeight(node) - step_max_work[current_step]) *
                           -1.0;
                }

                node_gains[node][new_proc][1] += gain;
                node_change_in_costs[node][new_proc][1] += gain;

            } else {
                node_gains[node][new_proc][1] += (step_max_work[current_step] - new_max_work);
                node_change_in_costs[node][new_proc][1] += (step_max_work[current_step] - new_max_work);
            }

        } else {

            if (step_max_work[current_step] <
                instance->getComputationalDag().nodeWorkWeight(node) + step_processor_work[current_step][new_proc]) {

                node_gains[node][new_proc][1] -=
                    (instance->getComputationalDag().nodeWorkWeight(node) +
                     step_processor_work[current_step][new_proc] - step_max_work[current_step]);

                node_change_in_costs[node][new_proc][1] -=
                    (instance->getComputationalDag().nodeWorkWeight(node) +
                     step_processor_work[current_step][new_proc] - step_max_work[current_step]);
            }
        }
    }

    if (current_step > 0) {

        if (step_max_work[current_step - 1] <
            step_processor_work[current_step - 1][new_proc] + instance->getComputationalDag().nodeWorkWeight(node)) {

            node_gains[node][new_proc][0] -=
                (step_processor_work[current_step - 1][new_proc] +
                 instance->getComputationalDag().nodeWorkWeight(node) - step_max_work[current_step - 1]);

            node_change_in_costs[node][new_proc][0] -=
                (step_processor_work[current_step - 1][new_proc] +
                 instance->getComputationalDag().nodeWorkWeight(node) - step_max_work[current_step - 1]);
        }

        if (step_max_work[current_step] == step_processor_work[current_step][current_proc] &&
            step_processor_work[current_step][current_proc] > step_second_max_work[current_step]) {

            if (step_max_work[current_step] - instance->getComputationalDag().nodeWorkWeight(node) >
                step_second_max_work[current_step]) {

                node_gains[node][new_proc][0] += instance->getComputationalDag().nodeWorkWeight(node);
                node_change_in_costs[node][new_proc][0] += instance->getComputationalDag().nodeWorkWeight(node);
            } else {

                node_gains[node][new_proc][0] += (step_max_work[current_step] - step_second_max_work[current_step]);
                node_change_in_costs[node][new_proc][0] +=
                    (step_max_work[current_step] - step_second_max_work[current_step]);
            }
        }

    } else {

        node_gains[node][new_proc][0] = std::numeric_limits<T>::lowest();
    }

    if (current_step < num_steps - 1) {

        if (step_max_work[current_step + 1] <
            step_processor_work[current_step + 1][new_proc] + instance->getComputationalDag().nodeWorkWeight(node)) {

            node_gains[node][new_proc][2] -=
                (step_processor_work[current_step + 1][new_proc] +
                 instance->getComputationalDag().nodeWorkWeight(node) - step_max_work[current_step + 1]);

            node_change_in_costs[node][new_proc][2] -=
                (step_processor_work[current_step + 1][new_proc] +
                 instance->getComputationalDag().nodeWorkWeight(node) - step_max_work[current_step + 1]);
        }

        if (step_max_work[current_step] == step_processor_work[current_step][current_proc] &&
            step_processor_work[current_step][current_proc] > step_second_max_work[current_step]) {

            if (step_max_work[current_step] - instance->getComputationalDag().nodeWorkWeight(node) >
                step_second_max_work[current_step]) {

                node_gains[node][new_proc][2] += instance->getComputationalDag().nodeWorkWeight(node);
                node_change_in_costs[node][new_proc][2] += instance->getComputationalDag().nodeWorkWeight(node);

            } else {

                node_gains[node][new_proc][2] += (step_max_work[current_step] - step_second_max_work[current_step]);
                node_change_in_costs[node][new_proc][2] +=
                    (step_max_work[current_step] - step_second_max_work[current_step]);
            }
        }
    } else {

        node_gains[node][new_proc][2] = std::numeric_limits<double>::lowest();
    }
}

template<typename T>
void LKBase<T>::collectNodesToUpdate(Move move, std::unordered_set<VertexType> &nodes_to_update) {

    for (const auto &target : instance->getComputationalDag().children(move.node)) {

        if (node_selection.find(target) != node_selection.end() && locked_nodes.find(target) == locked_nodes.end()) {
            nodes_to_update.insert(target);
        }
    }

    for (const auto &source : instance->getComputationalDag().parents(move.node)) {

        if (node_selection.find(source) != node_selection.end() && locked_nodes.find(source) == locked_nodes.end()) {
            nodes_to_update.insert(source);
        }
    }

    const unsigned start_step =
        std::min(move.from_step, move.to_step) == 0 ? 0 : std::min(move.from_step, move.to_step) - 1;
    const unsigned end_step = std::min(num_steps, std::max(move.from_step, move.to_step) + 2);

    for (unsigned step = start_step; step < end_step; step++) {

        for (unsigned proc = 0; proc < num_procs; proc++) {

            for (const auto &node : set_schedule.step_processor_vertices[step][proc]) {

                if (node_selection.find(node) != node_selection.end() &&
                    locked_nodes.find(node) == locked_nodes.end()) {
                    nodes_to_update.insert(node);
                }
            }
        }
    }
}

template<typename T>
void LKBase<T>::recompute_superstep_max_work(unsigned step) {

    step_max_work[step] = 0;
    step_second_max_work[step] = 0;

    for (unsigned proc = 0; proc < num_procs; proc++) {

        if (step_processor_work[step][proc] > step_max_work[step]) {

            step_second_max_work[step] = step_max_work[step];
            step_max_work[step] = step_processor_work[step][proc];

        } else if (step_processor_work[step][proc] > step_second_max_work[step]) {

            step_second_max_work[step] = step_processor_work[step][proc];
        }
    }
}

template<typename T>
typename LKBase<T>::Move LKBase<T>::compute_best_move(VertexType node) {

    // max_node_gains[node] = std::numeric_limits<double>::lowest();
    double node_max_gain = std::numeric_limits<double>::lowest();
    T node_change_in_cost = 0;
    unsigned node_best_step = 0;
    unsigned node_best_proc = 0;

    T proc_change_in_cost = 0;
    double proc_max = 0;
    unsigned best_step = 0;
    for (unsigned proc = 0; proc < num_procs; proc++) {

        unsigned rand_count = 0;

        if (vector_schedule.assignedSuperstep(node) > 0 && vector_schedule.assignedSuperstep(node) < num_steps - 1) {

            if (node_gains[node][proc][0] > node_gains[node][proc][1]) {

                if (node_gains[node][proc][0] > node_gains[node][proc][2]) {
                    proc_max = node_gains[node][proc][0];
                    proc_change_in_cost = node_change_in_costs[node][proc][0];
                    best_step = 0;

                } else {
                    proc_max = node_gains[node][proc][2];
                    proc_change_in_cost = node_change_in_costs[node][proc][2];
                    best_step = 2;
                }

            } else {

                if (node_gains[node][proc][1] > node_gains[node][proc][2]) {

                    proc_max = node_gains[node][proc][1];
                    proc_change_in_cost = node_change_in_costs[node][proc][1];
                    best_step = 1;
                } else {

                    proc_max = node_gains[node][proc][2];
                    proc_change_in_cost = node_change_in_costs[node][proc][2];
                    best_step = 2;
                }
            }

        } else if (vector_schedule.assignedSuperstep(node) == 0 &&
                   vector_schedule.assignedSuperstep(node) < num_steps - 1) {

            if (node_gains[node][proc][2] > node_gains[node][proc][1]) {

                proc_max = node_gains[node][proc][2];
                proc_change_in_cost = node_change_in_costs[node][proc][2];
                best_step = 2;
            } else {

                proc_max = node_gains[node][proc][1];
                proc_change_in_cost = node_change_in_costs[node][proc][1];
                best_step = 1;
            }

        } else if (vector_schedule.assignedSuperstep(node) > 0 &&
                   vector_schedule.assignedSuperstep(node) == num_steps - 1) {

            if (node_gains[node][proc][1] > node_gains[node][proc][0]) {

                proc_max = node_gains[node][proc][1];
                proc_change_in_cost = node_change_in_costs[node][proc][1];
                best_step = 1;
            } else {

                proc_max = node_gains[node][proc][0];
                proc_change_in_cost = node_change_in_costs[node][proc][0];
                best_step = 0;
            }
        } else {
            proc_max = node_gains[node][proc][1];
            proc_change_in_cost = node_change_in_costs[node][proc][1];
            best_step = 1;
        }

        if (node_max_gain < proc_max) {

            node_max_gain = proc_max;
            node_change_in_cost = proc_change_in_cost;
            node_best_step = vector_schedule.assignedSuperstep(node) + best_step - 1;
            node_best_proc = proc;
            rand_count = 0;

        } else if (node_max_gain == proc_max) {

            if (rand() % (2 + rand_count) == 0) {
                node_max_gain = proc_max;
                node_change_in_cost = proc_change_in_cost;
                node_best_step = vector_schedule.assignedSuperstep(node) + best_step - 1;
                node_best_proc = proc;
                rand_count++;
            }
        }
    }

    return Move(node, node_max_gain, node_change_in_cost, vector_schedule.assignedProcessor(node),
                vector_schedule.assignedSuperstep(node), node_best_proc, node_best_step);
}

template<typename T>
double LKBase<T>::computeMaxGain(VertexType node) {

    double node_max_gain = std::numeric_limits<double>::lowest();
    T node_change_in_cost = 0;
    unsigned node_best_step = 0;
    unsigned node_best_proc = 0;

    T proc_change_in_cost = 0;
    double proc_max = 0;
    unsigned best_step = 0;
    for (unsigned proc = 0; proc < num_procs; proc++) {

        unsigned rand_count = 0;

        if (vector_schedule.assignedSuperstep(node) > 0 && vector_schedule.assignedSuperstep(node) < num_steps - 1) {

            if (node_gains[node][proc][0] > node_gains[node][proc][1]) {

                if (node_gains[node][proc][0] > node_gains[node][proc][2]) {
                    proc_max = node_gains[node][proc][0];
                    proc_change_in_cost = node_change_in_costs[node][proc][0];
                    best_step = 0;

                } else {
                    proc_max = node_gains[node][proc][2];
                    proc_change_in_cost = node_change_in_costs[node][proc][2];
                    best_step = 2;
                }

            } else {

                if (node_gains[node][proc][1] > node_gains[node][proc][2]) {

                    proc_max = node_gains[node][proc][1];
                    proc_change_in_cost = node_change_in_costs[node][proc][1];
                    best_step = 1;
                } else {

                    proc_max = node_gains[node][proc][2];
                    proc_change_in_cost = node_change_in_costs[node][proc][2];
                    best_step = 2;
                }
            }

        } else if (vector_schedule.assignedSuperstep(node) == 0 &&
                   vector_schedule.assignedSuperstep(node) < num_steps - 1) {

            if (node_gains[node][proc][2] > node_gains[node][proc][1]) {

                proc_max = node_gains[node][proc][2];
                proc_change_in_cost = node_change_in_costs[node][proc][2];
                best_step = 2;
            } else {

                proc_max = node_gains[node][proc][1];
                proc_change_in_cost = node_change_in_costs[node][proc][1];
                best_step = 1;
            }

        } else if (vector_schedule.assignedSuperstep(node) > 0 &&
                   vector_schedule.assignedSuperstep(node) == num_steps - 1) {

            if (node_gains[node][proc][1] > node_gains[node][proc][0]) {

                proc_max = node_gains[node][proc][1];
                proc_change_in_cost = node_change_in_costs[node][proc][1];
                best_step = 1;
            } else {

                proc_max = node_gains[node][proc][0];
                proc_change_in_cost = node_change_in_costs[node][proc][0];
                best_step = 0;
            }
        } else {
            proc_max = node_gains[node][proc][1];
            proc_change_in_cost = node_change_in_costs[node][proc][1];
            best_step = 1;
        }

        if (node_max_gain < proc_max) {

            node_max_gain = proc_max;
            node_change_in_cost = proc_change_in_cost;
            node_best_step = vector_schedule.assignedSuperstep(node) + best_step - 1;
            node_best_proc = proc;
            rand_count = 0;

        } else if (node_max_gain == proc_max) {

            if (rand() % (2 + rand_count) == 0) {
                node_max_gain = proc_max;
                node_change_in_cost = proc_change_in_cost;
                node_best_step = vector_schedule.assignedSuperstep(node) + best_step - 1;
                node_best_proc = proc;
                rand_count++;
            }
        }
    }

    if (node_heap_handles.find(node) != node_heap_handles.end()) {

        (*node_heap_handles[node]).to_proc = node_best_proc;
        (*node_heap_handles[node]).to_step = node_best_step;
        (*node_heap_handles[node]).change_in_cost = node_change_in_cost;

        if ((*node_heap_handles[node]).gain != node_max_gain) {

            (*node_heap_handles[node]).gain = node_max_gain;
            max_gain_heap.update(node_heap_handles[node]);
        }

    } else {

        if (node_max_gain < -1.0 && node_change_in_cost < 0)
            return node_max_gain;

        Move move(node, node_max_gain, node_change_in_cost, vector_schedule.assignedProcessor(node),
                  vector_schedule.assignedSuperstep(node), node_best_proc, node_best_step);
        node_heap_handles[node] = max_gain_heap.push(move);
    }

    return node_max_gain;
}

template<typename T>
std::pair<unsigned, unsigned> LKBase<T>::best_move_change_superstep(VertexType node) {

    // max_node_gains[node] = std::numeric_limits<double>::lowest();
    double node_max_gain = std::numeric_limits<double>::lowest();
    // T node_change_in_cost = 0;
    unsigned node_best_step = 0;
    unsigned node_best_proc = 0;

    // T proc_change_in_cost = 0;
    double proc_max = 0;
    unsigned best_step = 0;
    for (unsigned proc = 0; proc < num_procs; proc++) {

        if (vector_schedule.assignedSuperstep(node) > 0 && vector_schedule.assignedSuperstep(node) < num_steps - 1) {

            if (node_gains[node][proc][0] > node_gains[node][proc][2]) {
                proc_max = node_gains[node][proc][0];
                best_step = 0;

            } else {
                proc_max = node_gains[node][proc][2];
                best_step = 2;
            }

        } else if (vector_schedule.assignedSuperstep(node) == 0 &&
                   vector_schedule.assignedSuperstep(node) < num_steps - 1) {

            proc_max = node_gains[node][proc][2];
            best_step = 2;

        } else if (vector_schedule.assignedSuperstep(node) > 0 &&
                   vector_schedule.assignedSuperstep(node) == num_steps - 1) {

            proc_max = node_gains[node][proc][0];
            best_step = 0;

        } else {
            throw std::invalid_argument("error lk base best_move_change_superstep");
        }

        if (node_max_gain < proc_max) {

            node_max_gain = proc_max;

            node_best_step = vector_schedule.assignedSuperstep(node) + best_step - 1;
            node_best_proc = proc;
        }
    }

    return {node_best_step, node_best_proc};
}

template<typename T>
void LKBase<T>::printHeap() {

    std::cout << "heap current size: " << max_gain_heap.size() << std::endl;
    std::cout << "heap top node " << max_gain_heap.top().node << " gain " << max_gain_heap.top().gain << std::endl;

    unsigned count = 0;
    for (auto it = max_gain_heap.ordered_begin(); it != max_gain_heap.ordered_end(); ++it) {
        std::cout << "node " << it->node << " gain " << it->gain << " to proc " << it->to_proc << " to step "
                  << it->to_step << std::endl;

        if (count++ > 25) {
            break;
        }
    }
}

template<typename T>
void LKBase<T>::cleanup_datastructures() {

    node_change_in_costs.clear();
    node_gains.clear();
    node_heap_handles.clear();

    unlock.clear();

    max_gain_heap.clear();

    cleanup_superstep_datastructures();
}

/* needed */
template<typename T>
void LKBase<T>::initalize_datastructures() {

    node_gains = std::vector<std::vector<std::vector<double>>>(
        num_nodes, std::vector<std::vector<double>>(num_procs, std::vector<double>(3, 0)));

    node_change_in_costs = std::vector<std::vector<std::vector<T>>>(
        num_nodes, std::vector<std::vector<T>>(num_procs, std::vector<T>(3, 0)));

    unlock = std::vector<unsigned>(num_nodes, max_num_unlocks);

    initalize_superstep_datastructures();
}

template<typename T>
void LKBase<T>::resetLockedNodesAndComputeGains() {

    for (const auto &i : locked_nodes) {

        unlock[i] = max_num_unlocks;

        computeNodeGain(i);
        computeMaxGain(i);
    }

    locked_nodes.clear();
}

template<typename T>
void LKBase<T>::resetLockedNodes() {

    for (const auto &i : locked_nodes) {

        unlock[i] = max_num_unlocks;
    }

    locked_nodes.clear();
}

template<typename T>
void LKBase<T>::cleanup_superstep_datastructures() {

    step_processor_work.clear();
    step_processor_send.clear();
    step_processor_receive.clear();

    step_max_work.clear();
    step_max_send.clear();
    step_max_receive.clear();

    step_second_max_work.clear();
    step_second_max_send.clear();
    step_second_max_receive.clear();

    step_processor_memory.clear();
}

template<typename T>
void LKBase<T>::initalize_superstep_datastructures() {

    if (use_memory_constraint) {
        step_processor_memory = std::vector<std::vector<unsigned>>(num_steps, std::vector<unsigned>(num_procs, 0));
    }

    step_processor_work = std::vector<std::vector<T>>(num_steps, std::vector<T>(num_procs, 0));
    step_processor_send = std::vector<std::vector<T>>(num_steps, std::vector<T>(num_procs, 0));
    step_processor_receive = std::vector<std::vector<T>>(num_steps, std::vector<T>(num_procs, 0));

    step_max_work = std::vector<T>(num_steps, 0);
    step_max_send = std::vector<T>(num_steps, 0);
    step_max_receive = std::vector<T>(num_steps, 0);

    step_second_max_work = std::vector<T>(num_steps, 0);
    step_second_max_send = std::vector<T>(num_steps, 0);
    step_second_max_receive = std::vector<T>(num_steps, 0);
}

template<typename T>
void LKBase<T>::setCurrentSchedule(const IBspSchedule &schedule) {

    num_steps = schedule.numberOfSupersteps();

    for (unsigned step = 0; step < num_steps; step++) {

        for (unsigned proc = 0; proc < num_procs; proc++) {

            set_schedule.step_processor_vertices[step][proc].clear();
        }
    }

    for (unsigned node = 0; node < num_nodes; node++) {

        vector_schedule.setAssignedProcessor(node, schedule.assignedProcessor(node));
        vector_schedule.setAssignedSuperstep(node, schedule.assignedSuperstep(node));

        set_schedule.step_processor_vertices[schedule.assignedSuperstep(node)][schedule.assignedProcessor(node)].insert(
            node);
    }
}

template<typename T>
void LKBase<T>::resetGainHeap() {

    max_gain_heap.clear();
    node_heap_handles.clear();
}

template<typename T>
void LKBase<T>::initalize_gain_heap(const std::unordered_set<VertexType> &nodes) {

    for (const auto &node : nodes) {

        computeNodeGain(node);
        computeMaxGain(node);
    }
}

template<typename T>
void LKBase<T>::setup_gain_heap_unlocked_nodes() {

    for (const auto &node : node_selection) {

        if (locked_nodes.find(node) == locked_nodes.end()) {

            computeNodeGain(node);
            computeMaxGain(node);
        }
    }
}

template<typename T>
bool LKBase<T>::unlockNode(VertexType node) {

    //    if (!super_locked[node]) {
    if (locked_nodes.find(node) != locked_nodes.end() && unlock[node] > 0) {
        unlock[node]--;

        locked_nodes.erase(node);

        return true;
    } else if (locked_nodes.find(node) == locked_nodes.end()) {
        return true;
    }
    //    }
    return false;
};

template<typename T>
void LKBase<T>::updateNodesGain(const std::unordered_set<VertexType> &nodes) {

    for (const auto &node : nodes) {

        if (locked_nodes.find(node) == locked_nodes.end()) {
            computeNodeGain(node);
            computeMaxGain(node);
        }
    }
};

template<typename T>
void LKBase<T>::unlockNeighbours(Move move, std::unordered_set<VertexType> &unlocked) {

    for (const auto &edge : instance->getComputationalDag().out_edges(move.node)) {

        const auto &target = instance->getComputationalDag().target(edge);
        if (vector_schedule.assignedProcessor(target) != move.to_proc ||
            vector_schedule.assignedSuperstep(target) < move.to_step) {
            if (unlockNode(target))
                unlocked.insert(target);
        }
    }

    for (const auto &edge : instance->getComputationalDag().in_edges(move.node)) {

        const auto &source = instance->getComputationalDag().source(edge);
        if (vector_schedule.assignedProcessor(source) != move.to_proc ||
            vector_schedule.assignedSuperstep(source) > move.to_step) {
            if (unlockNode(source))
                unlocked.insert(source);
        }
    }
}

template<typename T>
void LKBase<T>::unlockEdge(EdgeType edge) {

    const auto &source = instance->getComputationalDag().source(edge);
    const auto &target = instance->getComputationalDag().target(edge);

    unlockNode(source);
    unlockNode(target);
}

template<typename T>
void LKBase<T>::unlockEdgeNeighbors(EdgeType edge, std::unordered_set<VertexType> &unlocked) {

    const auto &source = instance->getComputationalDag().source(edge);
    const auto &target = instance->getComputationalDag().target(edge);

    // unlockNode(source);
    // unlockNode(target);

    for (const auto &child : instance->getComputationalDag().children(target)) {
        if (unlockNode(child))
            unlocked.insert(child);
    }

    for (const auto &child : instance->getComputationalDag().children(source)) {
        if (child != target)
            if (unlockNode(child))
                unlocked.insert(child);
    }

    for (const auto &parent : instance->getComputationalDag().parents(target)) {
        if (parent != source)
            if (unlockNode(parent))
                unlocked.insert(parent);
    }

    for (const auto &parent : instance->getComputationalDag().parents(source)) {

        if (unlockNode(parent))
            unlocked.insert(parent);
    }
}
