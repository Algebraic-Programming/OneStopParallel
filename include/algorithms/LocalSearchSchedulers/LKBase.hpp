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

#include <chrono>
#include <limits>
#include <list>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include <boost/heap/fibonacci_heap.hpp>

#include "algorithms/ImprovementScheduler.hpp"
#include "auxiliary/auxiliary.hpp"
#include "model/SetSchedule.hpp"
#include "model/VectorSchedule.hpp"

template<typename T>
class LKBase : public ImprovementScheduler {

  protected:
    struct Move {

        VertexType node;

        double gain;
        T change_in_cost;

        unsigned to_proc;
        unsigned to_step;

        Move() : node(0), gain(0), change_in_cost(0), to_proc(0), to_step(0) {}
        Move(VertexType node, double gain, T change_cost, unsigned to_proc, unsigned to_step)
            : node(node), gain(gain), change_in_cost(change_cost), to_proc(to_proc), to_step(to_step) {}

        bool operator<(Move const &rhs) const { return gain < rhs.gain; }
    };

    boost::heap::fibonacci_heap<Move> max_gain_heap;
    using heap_handle = typename boost::heap::fibonacci_heap<Move>::handle_type;

    std::vector<heap_handle> node_heap_handles;

    unsigned counter = 0;

    std::mt19937 gen;

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

    std::vector<std::vector<std::vector<double>>> node_gains;
    std::vector<std::vector<std::vector<T>>> node_change_in_costs;

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

    std::vector<bool> in_heap;
    std::vector<bool> locked;
    std::vector<unsigned> unlock;
    std::unordered_set<VertexType> locked_nodes;

    virtual void commputeCommGain(unsigned node, unsigned current_step, unsigned current_proc, unsigned new_proc) = 0;
    virtual void update_superstep_datastructures(Move move, unsigned from_proc, unsigned from_step) = 0;
    virtual void compute_superstep_datastructures() = 0;
    virtual T current_costs() = 0;

    virtual void computeNodeGain(unsigned node);
    virtual void computeWorkGain(unsigned node, unsigned current_step, unsigned current_proc, unsigned new_proc);

    virtual void applyMove(Move move, unsigned from_proc, unsigned from_step);
    virtual void updateNodesGainAfterMove(Move move, unsigned from_proc, unsigned from_step);
    virtual std::unordered_set<VertexType> collectNodesToUpdate(Move move, unsigned from_proc, unsigned from_step);
    virtual void updateMaxGainUnlockedNeighbors(VertexType node);

    virtual void updateViolations(VertexType node);

    virtual double computeMaxGain(VertexType node);

    virtual void recompute_superstep_max_work(unsigned step);
    virtual void initalize_datastructures();
    virtual void initalize_superstep_datastructures();
    virtual void initalize_gain_heap();
    virtual void initalize_gain_heap(const std::unordered_set<VertexType> &nodes);

    virtual void cleanup_datastructures();
    virtual void cleanup_superstep_datastructures();

    virtual void initializeRewardPenaltyFactors();
    virtual void updateRewardPenaltyFactors();

    virtual void resetLockedNodesAndComputeGains();
    virtual void setup_gain_heap_unlocked_nodes();

    virtual void resetGainHeap();
    virtual void computeUnlockedNodesGain();

    virtual Move findMove();

    virtual bool start();
    virtual void resetLockedNodes();
    virtual bool unlockNode(VertexType node);
    virtual void unlockEdge(EdgeType edge);
    virtual void unlockNeighbours(VertexType node, std::unordered_set<VertexType> &unlocked);
    virtual void unlockEdgeNeighbors(EdgeType edge);
    virtual void lockAll();
    virtual void unlockViolationEdges();

    virtual void updateNodesGain(const std::unordered_set<VertexType> &nodes);

    virtual void setCurrentSchedule(const IBspSchedule &schedule);
    virtual void setBestSchedule(const IBspSchedule &schedule);
    virtual void reverseMoveBestSchedule(Move move, unsigned from_proc, unsigned from_step);

    virtual std::unordered_set<VertexType> selectNodesThreshold();
    virtual std::unordered_set<VertexType> selectNodesThreshold(unsigned threshold);
    virtual std::unordered_set<VertexType> selectNodesPermutationThreshold();
    virtual std::unordered_set<VertexType> selectNodesPermutationThreshold(unsigned threshold);
    virtual std::unordered_set<VertexType> selectNodesCollision(unsigned num = 5);

    virtual bool checkAbortCondition(Move &move);
    std::vector<double> circularBuffer;

    virtual void setParameters();

    virtual void checkMergeSupersteps();
    virtual void checkInsertSuperstep();

    virtual void insertSuperstep(unsigned step);

    void printHeap();

    unsigned max_num_unlocks;
    unsigned max_iterations;
    unsigned max_iterations_inner;
    unsigned selection_threshold;
    unsigned abort_threshold;

    double penalty_factor;
    double base_penalty_factor;
    double reward_factor;
    double base_reward_factor;
    double base_reward;

  public:
    LKBase()
        : ImprovementScheduler(), best_schedule_costs(0), current_feasible(true), max_num_unlocks(3),
          max_iterations(500), penalty_factor(1.0), base_penalty_factor(7.0), reward_factor(1.0),
          base_reward_factor(10.0), base_reward(1.0) {

        std::random_device rd;
        gen = std::mt19937(rd());
    }

    virtual ~LKBase() = default;

    virtual RETURN_STATUS improveSchedule(BspSchedule &schedule) override;

    virtual std::string getScheduleName() const = 0;
};

template<typename T>
void LKBase<T>::setParameters() {

    max_num_unlocks = 3;
    abort_threshold = 20;

    if (num_nodes < 250) {

        max_iterations = 50;

        selection_threshold = num_nodes * 0.33;

        max_iterations_inner = num_nodes;

    } else if (num_nodes < 1000) {

        max_iterations = std::sqrt(num_nodes) * 2;

        selection_threshold = num_nodes * 0.15;

        max_iterations_inner = 8 * std::sqrt(num_nodes);

    } else if (num_nodes < 5000) {

        max_iterations = std::log(num_nodes) * 14;

        selection_threshold = num_nodes * 0.10;

        max_iterations_inner = 4 * std::sqrt(num_nodes);

    } else {

        max_iterations = 4 * std::sqrt(num_nodes) ;

        selection_threshold = num_nodes * 0.20;

        max_iterations_inner = 6 * std::sqrt(num_nodes);

    }
}

template<typename T>
RETURN_STATUS LKBase<T>::improveSchedule(BspSchedule &schedule) {

    best_schedule = &schedule;

    instance = &best_schedule->getInstance();
    num_nodes = instance->numberOfVertices();
    num_procs = instance->numberOfProcessors();
    num_steps = best_schedule->numberOfSupersteps();

    bool improvement_found = start();

    assert(best_schedule->satisfiesPrecedenceConstraints());

    schedule.setAutoCommunicationSchedule();

    if (improvement_found)
        return SUCCESS;
    else
        return BEST_FOUND;
};

template<typename T>
bool LKBase<T>::start() {

    vector_schedule = VectorSchedule(*best_schedule);
    set_schedule = SetSchedule(*best_schedule);

    initializeRewardPenaltyFactors();
    initalize_datastructures();
    compute_superstep_datastructures();

    best_schedule_costs = current_costs();

    T initial_costs = best_schedule_costs;
    T current_costs = initial_costs;

    std::cout << getScheduleName() << " start() best schedule costs: " << best_schedule_costs << std::endl;

    setParameters();

    initalize_gain_heap();

    //  std::cout << "Initial costs " << current_costs() << std::endl;
    unsigned improvement_counter = 0;

    for (unsigned i = 0; i < max_iterations; i++) {

        // std::cout << "begin iter costs: " << current_costs() << std::endl;

        unsigned failed_branches = 0;
        T best_iter_costs = current_costs;
        counter = 0;

        while (failed_branches < 3 && locked_nodes.size() < max_iterations_inner && max_gain_heap.size() > 0) {

            // std::cout << "failed branches: " << failed_branches << std::endl;

            Move best_move = findMove(); // O(log n)

            unsigned current_proc = vector_schedule.assignedProcessor(best_move.node);
            unsigned current_step = vector_schedule.assignedSuperstep(best_move.node);

            applyMove(best_move, current_proc, current_step); // O(p + log n)

            current_costs -= best_move.change_in_cost;

            // std::cout << "current costs: " << current_costs << " best move gain: " << best_move.gain
            //           << " best move costs: " << best_move.change_in_cost << std::endl;

            locked_nodes.insert(best_move.node);
            locked[best_move.node] = true;

            updateViolations(best_move.node); // O(Delta_max * log(current_violations.size())

            updateRewardPenaltyFactors();

            std::unordered_set<VertexType> nodes_to_update =
                collectNodesToUpdate(best_move, current_proc, current_step);
            unlockNeighbours(best_move.node, nodes_to_update);
            updateNodesGain(nodes_to_update);

            if (best_move.change_in_cost < 0 && current_violations.empty() && current_feasible) {

                if (best_schedule_costs > current_costs + best_move.change_in_cost) {

                    //                    std::cout << "costs increased .. save best schedule with costs "
                    //                              << current_costs + best_move.change_in_cost << std::endl;

                    best_schedule_costs = current_costs + best_move.change_in_cost;
                    setBestSchedule(vector_schedule); // O(n)
                    reverseMoveBestSchedule(best_move, current_proc, current_step);
                }
            }

            if (current_violations.empty() && not current_feasible) {

                //                std::cout << "<=============== moved from infeasible to feasible" << std::endl;
                current_feasible = true;

                if (current_costs <= best_schedule_costs) {
                    //                    std::cout << "new schdule better than previous best schedule" << std::endl;

                    //                setBestSchedule(vector_schedule);
                    //                best_schedule_costs = current_costs;
                } else {

                    //                    std::cout << "... but costs did not improve: " << current_costs
                    //                              << " vs best schedule: " << best_schedule_costs << std::endl;

                    if (current_costs > (1.1 - counter * 0.001) * best_schedule_costs) {
                        //                        std::cout << "rollback to best schedule" << std::endl;
                        setCurrentSchedule(*best_schedule); // O(n + p*s)
                        compute_superstep_datastructures(); // O(n)
                        current_costs = best_schedule_costs;
                        current_violations.clear();
                        current_feasible = true;
                        resetGainHeap();
                        setup_gain_heap_unlocked_nodes();
                        counter = 0;
                        failed_branches++;
                    }
                }

            } else if (not current_violations.empty() && current_feasible) {
                //                std::cout << "================> moved from feasible to infeasible" << std::endl;
                current_feasible = false;

                // unlockNeighbours(best_move.node);

                if (current_costs + best_move.change_in_cost <= best_schedule_costs) {
                    //                    std::cout << "save best schedule with costs " << current_costs +
                    //                    best_move.change_in_cost
                    //                              << std::endl;
                    best_schedule_costs = current_costs + best_move.change_in_cost;
                    setBestSchedule(vector_schedule); // O(n)
                    reverseMoveBestSchedule(best_move, current_proc, current_step);
                }
            }

            if (not current_feasible) {

                if (current_costs > (1.2 - counter * 0.001) * best_schedule_costs) {

                    // std::cout << "current cost " << current_costs
                    //           << " too far away from best schedule costs: " << best_schedule_costs << "rollback to
                    //           best schedule" << std::endl;

                    setCurrentSchedule(*best_schedule); // O(n + p*s)
                    compute_superstep_datastructures(); // O(n)
                    current_costs = best_schedule_costs;
                    current_violations.clear();
                    current_feasible = true;
                    resetGainHeap();
                    setup_gain_heap_unlocked_nodes();
                    counter = 0;
                    failed_branches++;
                }
            }

            counter++;

        } // while

        // std::cout << "current costs end while: " << current_costs_double() << std::endl;
        // std::cout << "number of violations " << current_violations.size() << std::endl;
        if (current_violations.empty()) {

            if (current_costs <= best_schedule_costs) {
                setBestSchedule(vector_schedule);
                best_schedule_costs = current_costs;
            }

            resetLockedNodesAndComputeGains();

        } else {

            //            std::cout << "current solution not feasible .. rolling back to best solution with costs "
            //                      << best_schedule_costs << std::endl;

            resetLockedNodes();

            setCurrentSchedule(*best_schedule); // O (n + p*s)

            compute_superstep_datastructures(); // O(n)
            current_costs = best_schedule_costs;
            current_violations.clear();
            current_feasible = true;

            resetGainHeap();
            initalize_gain_heap();
        }

        if (best_iter_costs <= current_costs) {

            if (improvement_counter++ == std::log(max_iterations)) {
                // std::cout << "no improvement ... end local search " << std::endl;
                break;
            }
        } else {
            improvement_counter = 0;
        }

    } // for

    std::cout << getScheduleName() << " end best schedule costs: " << best_schedule_costs << std::endl;
    cleanup_datastructures();

    if (initial_costs > current_costs)
        return true;
    else
        return false;
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

            if (step_max_work[step] < min_work) {
                min_work = step_processor_work[step][proc];
            }
        }
    }

    unsigned step = 0;
    while (step < num_steps - 1) {

        if (set_schedule.step_processor_vertices[step].empty()) {
            set_schedule.mergeSupersteps(step, step + 1);
            num_steps--;
        } else {
            step++;
        }
    }
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

                    if (locked[node]) {

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
bool LKBase<T>::checkAbortCondition(Move &move) {

    if (circularBuffer.size() < abort_threshold) {
        circularBuffer.push_back(move.gain);

    } else {
        circularBuffer.erase(circularBuffer.begin());
        circularBuffer.push_back(move.gain);
    }

    if (std::accumulate(circularBuffer.begin(), circularBuffer.end(), 0) <= 0) {
        return true;
    }

    return false;
}

template<typename T>
std::unordered_set<VertexType> LKBase<T>::selectNodesThreshold() {

    std::unordered_set<VertexType> nodes;

    std::uniform_int_distribution<> dis(0, num_nodes - 1);

    unsigned threshold = num_nodes * 0.33;

    while (nodes.size() < threshold) {
        nodes.insert(dis(gen));
    }

    return nodes;
}

template<typename T>
std::unordered_set<VertexType> LKBase<T>::selectNodesThreshold(unsigned threshold) {

    std::unordered_set<VertexType> nodes;

    std::uniform_int_distribution<> dis(0, num_nodes - 1);

    while (nodes.size() < threshold) {
        nodes.insert(dis(gen));
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
    in_heap[best_move.node] = false;

    return best_move;
};

template<typename T>
void LKBase<T>::applyMove(Move move, unsigned from_proc, unsigned from_step) {

    vector_schedule.setAssignedProcessor(move.node, move.to_proc);
    vector_schedule.setAssignedSuperstep(move.node, move.to_step);

    set_schedule.step_processor_vertices[from_step][from_proc].erase(move.node);
    set_schedule.step_processor_vertices[move.to_step][move.to_proc].insert(move.node);

    update_superstep_datastructures(move, from_proc, from_step);
}

template<typename T>
void LKBase<T>::updateViolations(VertexType node) {

    for (const auto &edge : instance->getComputationalDag().out_edges(node)) {

        const auto &child = instance->getComputationalDag().target(edge);

        if (current_violations.find(edge) == current_violations.end()) {

            if (vector_schedule.assignedSuperstep(node) >= vector_schedule.assignedSuperstep(child)) {

                if (vector_schedule.assignedProcessor(node) != vector_schedule.assignedProcessor(child) ||
                    vector_schedule.assignedSuperstep(node) > vector_schedule.assignedSuperstep(child)) {

                    current_violations.insert(edge);
                }
            }
        } else {

            if (vector_schedule.assignedSuperstep(node) <= vector_schedule.assignedSuperstep(child)) {

                if (vector_schedule.assignedProcessor(node) == vector_schedule.assignedProcessor(child) ||
                    vector_schedule.assignedSuperstep(node) < vector_schedule.assignedSuperstep(child)) {

                    current_violations.erase(edge);
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
                }
            }
        } else {

            if (vector_schedule.assignedSuperstep(node) >= vector_schedule.assignedSuperstep(parent)) {

                if (vector_schedule.assignedProcessor(node) == vector_schedule.assignedProcessor(parent) ||
                    vector_schedule.assignedSuperstep(node) > vector_schedule.assignedSuperstep(parent)) {

                    current_violations.erase(edge);
                }
            }
        }
    }
}

template<typename T>
void LKBase<T>::lockAll() {
    for (unsigned i = 0; i < num_nodes; i++) {
        locked[i] = true;
    }
}

template<typename T>
void LKBase<T>::unlockViolationEdges() {

    for (const auto &edge : current_violations) {
        const auto &source = instance->getComputationalDag().source(edge);
        const auto &target = instance->getComputationalDag().target(edge);

        locked[source] = false;
        locked[target] = false;
    }
}

template<typename T>
void LKBase<T>::setBestSchedule(const IBspSchedule &schedule) {

    for (unsigned node = 0; node < num_nodes; node++) {

        best_schedule->setAssignedProcessor(node, schedule.assignedProcessor(node));
        best_schedule->setAssignedSuperstep(node, schedule.assignedSuperstep(node));
    }
}

template<typename T>
void LKBase<T>::reverseMoveBestSchedule(Move move, unsigned from_proc, unsigned from_step) {
    best_schedule->setAssignedProcessor(move.node, from_proc);
    best_schedule->setAssignedSuperstep(move.node, from_step);
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
void LKBase<T>::updateNodesGainAfterMove(Move move, unsigned from_proc, unsigned from_step) {

    for (const auto &ep : instance->getComputationalDag().out_edges(move.node)) {
        const auto &target = instance->getComputationalDag().target(ep);

        if (!locked[target]) {
            computeNodeGain(target);
            computeMaxGain(target);
        }
    }

    for (const auto &ep : instance->getComputationalDag().in_edges(move.node)) {
        const auto &source = instance->getComputationalDag().source(ep);

        if (!locked[source]) {
            computeNodeGain(source);
            computeMaxGain(source);
        }
    }

    const unsigned start_step = std::min(from_step, move.to_step) == 0 ? 0 : std::min(from_step, move.to_step) - 1;
    const unsigned end_step = std::min(num_steps, std::max(from_step, move.to_step) + 2);

    for (unsigned step = start_step; step < end_step; step++) {

        for (unsigned proc = 0; proc < num_procs; proc++) {

            for (const auto &node : set_schedule.step_processor_vertices[step][proc]) {

                if (locked[node] == false) {
                    computeNodeGain(node);
                    computeMaxGain(node);
                }
            }
        }
    }
}

template<typename T>
std::unordered_set<VertexType> LKBase<T>::collectNodesToUpdate(Move move, unsigned from_proc, unsigned from_step) {

    std::unordered_set<VertexType> nodes_to_update;

    for (const auto &target : instance->getComputationalDag().children(move.node)) {

        if (!locked[target]) {
            nodes_to_update.insert(target);
        }
    }

    for (const auto &source : instance->getComputationalDag().parents(move.node)) {

        if (!locked[source]) {
            nodes_to_update.insert(source);
        }
    }

    const unsigned start_step = std::min(from_step, move.to_step) == 0 ? 0 : std::min(from_step, move.to_step) - 1;
    const unsigned end_step = std::min(num_steps, std::max(from_step, move.to_step) + 2);

    for (unsigned step = start_step; step < end_step; step++) {

        for (unsigned proc = 0; proc < num_procs; proc++) {

            for (const auto &node : set_schedule.step_processor_vertices[step][proc]) {

                if (locked[node] == false) {
                    nodes_to_update.insert(node);
                }
            }
        }
    }

    return nodes_to_update;
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
void LKBase<T>::updateMaxGainUnlockedNeighbors(VertexType node) {

    for (const auto &edge : instance->getComputationalDag().out_edges(node)) {
        const auto &target = instance->getComputationalDag().target(edge);
        if (!locked[target]) {
            computeMaxGain(target);
        }
    }

    for (const auto &edge : instance->getComputationalDag().in_edges(node)) {

        const auto &source = instance->getComputationalDag().source(edge);
        if (!locked[source]) {
            computeMaxGain(source);
        }
    }
}

template<typename T>
double LKBase<T>::computeMaxGain(VertexType node) {

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

    if (in_heap[node]) {

        (*node_heap_handles[node]).to_proc = node_best_proc;
        (*node_heap_handles[node]).to_step = node_best_step;
        (*node_heap_handles[node]).change_in_cost = node_change_in_cost;

        if ((*node_heap_handles[node]).gain != node_max_gain) {

            (*node_heap_handles[node]).gain = node_max_gain;
            max_gain_heap.update(node_heap_handles[node]);
        }

    } else {

        Move move(node, node_max_gain, node_change_in_cost, node_best_proc, node_best_step);
        node_heap_handles[node] = max_gain_heap.push(move);
        in_heap[node] = true;
    }

    return node_max_gain;
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

    locked.clear();
    in_heap.clear();
    unlock.clear();

    max_gain_heap.clear();

    cleanup_superstep_datastructures();
}

template<typename T>
void LKBase<T>::initalize_datastructures() {

    node_gains = std::vector<std::vector<std::vector<double>>>(
        num_nodes, std::vector<std::vector<double>>(num_procs, std::vector<double>(3, 0)));

    node_change_in_costs = std::vector<std::vector<std::vector<T>>>(
        num_nodes, std::vector<std::vector<T>>(num_procs, std::vector<T>(3, 0)));

    node_heap_handles = std::vector<heap_handle>(num_nodes);

    locked = std::vector<bool>(num_nodes, true);
    in_heap = std::vector<bool>(num_nodes, false);
    unlock = std::vector<unsigned>(num_nodes, max_num_unlocks);

    initalize_superstep_datastructures();
}

template<typename T>
void LKBase<T>::resetLockedNodesAndComputeGains() {

    for (const auto &i : locked_nodes) {
        locked[i] = false;
        unlock[i] = max_num_unlocks;

        computeNodeGain(i);
        computeMaxGain(i);
    }

    locked_nodes.clear();
}

template<typename T>
void LKBase<T>::resetLockedNodes() {

    for (const auto &i : locked_nodes) {
        locked[i] = false;
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
}

template<typename T>
void LKBase<T>::initalize_superstep_datastructures() {

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
    for (unsigned node = 0; node < num_nodes; node++) {
        in_heap[node] = false;
    }
}

template<typename T>
void LKBase<T>::initalize_gain_heap() {

    for (unsigned i = 0; i < num_nodes; i++) {

        computeNodeGain(i);
        computeMaxGain(i);
    }
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

    for (unsigned i = 0; i < num_nodes; i++) {

        if (!locked[i]) {

            computeNodeGain(i);
            computeMaxGain(i);
        }
    }
}

template<typename T>
void LKBase<T>::computeUnlockedNodesGain() {

    for (unsigned i = 0; i < num_nodes; i++) {

        if (!locked[i]) {
            computeNodeGain(i);
        }
    }
}

template<typename T>
bool LKBase<T>::unlockNode(VertexType node) {

    if (locked[node] && unlock[node] > 0) {
        unlock[node]--;
        locked[node] = false;

        locked_nodes.erase(node);

        return true;
    }

    return false;
}

template<typename T>
void LKBase<T>::updateNodesGain(const std::unordered_set<VertexType> &nodes) {

    for (const auto &node : nodes) {
        computeNodeGain(node);
        computeMaxGain(node);
    }
}

template<typename T>
void LKBase<T>::unlockNeighbours(VertexType node, std::unordered_set<VertexType> &unlocked) {

    for (const auto &edge : instance->getComputationalDag().out_edges(node)) {

        if (unlockNode(instance->getComputationalDag().target(edge)))
            unlocked.insert(instance->getComputationalDag().target(edge));
    }

    for (const auto &edge : instance->getComputationalDag().in_edges(node)) {

        if (unlockNode(instance->getComputationalDag().source(edge)))
            unlocked.insert(instance->getComputationalDag().source(edge));
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
void LKBase<T>::unlockEdgeNeighbors(EdgeType edge) {

    const auto &source = instance->getComputationalDag().source(edge);
    const auto &target = instance->getComputationalDag().target(edge);

    for (const auto &child : instance->getComputationalDag().children(target)) {
        unlockNode(child);
    }

    for (const auto &child : instance->getComputationalDag().children(source)) {
        if (child != target)
            unlockNode(child);
    }

    for (const auto &parent : instance->getComputationalDag().parents(target)) {
        if (parent != source)
            unlockNode(parent);
    }

    for (const auto &parent : instance->getComputationalDag().parents(source)) {

        unlockNode(parent);
    }
}
