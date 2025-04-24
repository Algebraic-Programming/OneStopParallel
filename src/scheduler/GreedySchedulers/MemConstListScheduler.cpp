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

#include "scheduler/GreedySchedulers/MemConstListScheduler.hpp"

void MemConstListScheduler::init_mem_const_data_structures(const BspArchitecture &arch) {

    memory_const_type = arch.getMemoryConstraintType();

    if (use_memory_constraint) {

        switch (memory_const_type) {

        case LOCAL:
            current_proc_persistent_memory = std::vector<int>(arch.numberOfProcessors(), 0);
            break;

        case LOCAL_IN_OUT:
            current_proc_persistent_memory = std::vector<int>(arch.numberOfProcessors(), 0);
            break;

        case LOCAL_INC_EDGES:
            current_proc_persistent_memory = std::vector<int>(arch.numberOfProcessors(), 0);
            current_proc_predec = std::vector<std::unordered_set<VertexType>>(arch.numberOfProcessors());
            break;

        case LOCAL_INC_EDGES_2:
            current_proc_persistent_memory = std::vector<int>(arch.numberOfProcessors(), 0);
            current_proc_predec = std::vector<std::unordered_set<VertexType>>(arch.numberOfProcessors());
            break;

        case PERSISTENT_AND_TRANSIENT:
            current_proc_persistent_memory = std::vector<int>(arch.numberOfProcessors(), 0);
            current_proc_transient_memory = std::vector<int>(arch.numberOfProcessors(), 0);
            break;

        case GLOBAL:
            throw std::invalid_argument("Global memory constraint not supported");

        case NONE:
            use_memory_constraint = false;
            std::cerr << "Warning: Memory constraint type set to NONE, ignoring memory constraint" << std::endl;
            break;

        default:
            throw std::invalid_argument("Unknown memory constraint not supported");
        }
    }
}

void MemConstListScheduler::reset_mem_const_datastructures_new_superstep(unsigned proc) {

    if (use_memory_constraint) {

        if (memory_const_type == LOCAL) {
            current_proc_persistent_memory[proc] = 0;

        } else if (memory_const_type == LOCAL_IN_OUT) {
            current_proc_persistent_memory[proc] = 0;

        } else if (memory_const_type == LOCAL_INC_EDGES) {
            current_proc_persistent_memory[proc] = 0;
            current_proc_predec[proc].clear();
        } else if (memory_const_type == LOCAL_INC_EDGES_2) {
            current_proc_persistent_memory[proc] = 0;
            current_proc_predec[proc].clear();
        }
    }
}

bool MemConstListScheduler::check_can_add(const BspSchedule &schedule, const BspInstance &instance, unsigned node,
                                          unsigned succ, unsigned supstepIdx) {

    if (use_memory_constraint) {

        if (memory_const_type == LOCAL) {

            if (current_proc_persistent_memory[schedule.assignedProcessor(node)] +
                    instance.getComputationalDag().nodeMemoryWeight(succ) >
                instance.getArchitecture().memoryBound(schedule.assignedProcessor(node))) {
                return false;
            }

        } else if (memory_const_type == PERSISTENT_AND_TRANSIENT) {

            if (current_proc_persistent_memory[schedule.assignedProcessor(node)] +
                    instance.getComputationalDag().nodeMemoryWeight(succ) +
                    std::max(current_proc_transient_memory[schedule.assignedProcessor(node)],
                             instance.getComputationalDag().nodeCommunicationWeight(succ)) >
                instance.getArchitecture().memoryBound(schedule.assignedProcessor(node))) {
                return false;
            }
        } else if (memory_const_type == LOCAL_IN_OUT) {

            int inc_memory = instance.getComputationalDag().nodeMemoryWeight(succ) +
                             instance.getComputationalDag().nodeCommunicationWeight(succ);

            for (const auto &pred : instance.getComputationalDag().parents(succ)) {

                if (schedule.assignedProcessor(pred) == schedule.assignedProcessor(node) &&
                    schedule.assignedSuperstep(pred) == supstepIdx) {
                    inc_memory -= instance.getComputationalDag().nodeCommunicationWeight(pred);
                }
            }

            if (current_proc_persistent_memory[schedule.assignedProcessor(node)] + inc_memory >
                static_cast<int>(instance.getArchitecture().memoryBound(schedule.assignedProcessor(node)))) {
                return false;
            }
        } else if (memory_const_type == LOCAL_INC_EDGES) {

            int inc_memory = instance.getComputationalDag().nodeCommunicationWeight(succ);
            for (const auto &pred : instance.getComputationalDag().parents(succ)) {

                if (schedule.assignedSuperstep(pred) != supstepIdx &&
                    current_proc_predec[schedule.assignedProcessor(node)].find(pred) ==
                        current_proc_predec[schedule.assignedProcessor(node)].end()) {
                    inc_memory += instance.getComputationalDag().nodeCommunicationWeight(pred);
                }
            }

            if (current_proc_persistent_memory[schedule.assignedProcessor(node)] + inc_memory >
                instance.getArchitecture().memoryBound(schedule.assignedProcessor(node))) {
                return false;
            }
        } else if (memory_const_type == LOCAL_INC_EDGES_2) {

            int inc_memory = 0;

            if (instance.getComputationalDag().isSource(succ)) {
                inc_memory += instance.getComputationalDag().nodeMemoryWeight(succ);
            }

            for (const auto &pred : instance.getComputationalDag().parents(succ)) {

                if (schedule.assignedSuperstep(pred) != supstepIdx &&
                    current_proc_predec[schedule.assignedProcessor(node)].find(pred) ==
                        current_proc_predec[schedule.assignedProcessor(node)].end()) {
                    inc_memory += instance.getComputationalDag().nodeCommunicationWeight(pred);
                }
            }

            if (current_proc_persistent_memory[schedule.assignedProcessor(node)] + inc_memory >
                instance.getArchitecture().memoryBound(schedule.assignedProcessor(node))) {
                return false;
            }
        }
    }
    return true;
}

std::vector<VertexType> MemConstListScheduler::update_mem_const_datastructure_after_assign(
    const BspSchedule &schedule, const BspInstance &instance, unsigned nextNode, unsigned nextProc, unsigned supstepIdx,
    std::vector<std::set<VertexType>> &procReady) {

    std::vector<VertexType> toErase;

    if (use_memory_constraint) {

        if (instance.getArchitecture().getMemoryConstraintType() == LOCAL) {

            current_proc_persistent_memory[nextProc] += instance.getComputationalDag().nodeMemoryWeight(nextNode);

            for (const auto &node : procReady[nextProc]) {
                if (current_proc_persistent_memory[nextProc] + instance.getComputationalDag().nodeMemoryWeight(node) >
                    instance.getArchitecture().memoryBound(nextProc)) {
                    toErase.push_back(node);
                }
            }

        } else if (instance.getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {

            current_proc_persistent_memory[nextProc] += instance.getComputationalDag().nodeMemoryWeight(nextNode);
            current_proc_transient_memory[nextProc] =
                std::max(current_proc_transient_memory[nextProc],
                         instance.getComputationalDag().nodeCommunicationWeight(nextNode));

            for (const auto &node : procReady[nextProc]) {
                if (current_proc_persistent_memory[nextProc] + instance.getComputationalDag().nodeMemoryWeight(node) +
                        std::max(current_proc_transient_memory[nextProc],
                                 instance.getComputationalDag().nodeCommunicationWeight(node)) >
                    instance.getArchitecture().memoryBound(nextProc)) {
                    toErase.push_back(node);
                }
            }
        } else if (instance.getArchitecture().getMemoryConstraintType() == LOCAL_IN_OUT) {

            current_proc_persistent_memory[nextProc] +=
                instance.getComputationalDag().nodeMemoryWeight(nextNode) +
                instance.getComputationalDag().nodeCommunicationWeight(nextNode);

            for (const auto &pred : instance.getComputationalDag().parents(nextNode)) {

                if (schedule.assignedProcessor(pred) == schedule.assignedProcessor(nextNode) &&
                    schedule.assignedSuperstep(pred) == supstepIdx) {
                    current_proc_persistent_memory[nextProc] -=
                        instance.getComputationalDag().nodeCommunicationWeight(pred);
                }
            }

            for (const auto &node : procReady[nextProc]) {

                int inc_memory = instance.getComputationalDag().nodeMemoryWeight(node) +
                                 instance.getComputationalDag().nodeCommunicationWeight(node);

                for (const auto &pred : instance.getComputationalDag().parents(node)) {
                    if (schedule.assignedProcessor(pred) == schedule.assignedProcessor(nextNode) &&
                        schedule.assignedSuperstep(pred) == supstepIdx) {
                        inc_memory -= instance.getComputationalDag().nodeCommunicationWeight(pred);
                    }
                }

                if (current_proc_persistent_memory[schedule.assignedProcessor(nextNode)] + inc_memory >
                    instance.getArchitecture().memoryBound(schedule.assignedProcessor(nextNode))) {
                    toErase.push_back(node);
                }
            }
        } else if (instance.getArchitecture().getMemoryConstraintType() == LOCAL_INC_EDGES) {

            current_proc_persistent_memory[nextProc] +=
                instance.getComputationalDag().nodeCommunicationWeight(nextNode);

            for (const auto &pred : instance.getComputationalDag().parents(nextNode)) {

                if (schedule.assignedSuperstep(pred) != supstepIdx) {
                    const auto pair = current_proc_predec[nextProc].insert(pred);
                    if (pair.second) {
                        current_proc_persistent_memory[nextProc] +=
                            instance.getComputationalDag().nodeCommunicationWeight(pred);
                    }
                }
            }

            for (const auto &node : procReady[nextProc]) {
                int inc_memory = instance.getComputationalDag().nodeCommunicationWeight(node);
                for (const auto &pred : instance.getComputationalDag().parents(node)) {

                    if (schedule.assignedSuperstep(pred) != supstepIdx &&
                        current_proc_predec[schedule.assignedProcessor(nextNode)].find(pred) ==
                            current_proc_predec[schedule.assignedProcessor(nextNode)].end()) {
                        inc_memory += instance.getComputationalDag().nodeCommunicationWeight(pred);
                    }
                }

                if (current_proc_persistent_memory[schedule.assignedProcessor(nextNode)] + inc_memory >
                    instance.getArchitecture().memoryBound(schedule.assignedProcessor(nextNode))) {

                    toErase.push_back(node);
                }
            }
        } else if (instance.getArchitecture().getMemoryConstraintType() == LOCAL_INC_EDGES_2) {

            if (instance.getComputationalDag().isSource(nextNode)) {
                current_proc_persistent_memory[nextProc] += instance.getComputationalDag().nodeMemoryWeight(nextNode);
            }

            for (const auto &pred : instance.getComputationalDag().parents(nextNode)) {

                if (schedule.assignedSuperstep(pred) != supstepIdx) {
                    const auto pair = current_proc_predec[nextProc].insert(pred);
                    if (pair.second) {
                        current_proc_persistent_memory[nextProc] +=
                            instance.getComputationalDag().nodeCommunicationWeight(pred);
                    }
                }
            }

            for (const auto &node : procReady[nextProc]) {

                int inc_memory = 0;

                if (instance.getComputationalDag().isSource(node)) {
                    inc_memory += instance.getComputationalDag().nodeMemoryWeight(node);
                }

                for (const auto &pred : instance.getComputationalDag().parents(node)) {

                    if (schedule.assignedSuperstep(pred) != supstepIdx &&
                        current_proc_predec[schedule.assignedProcessor(nextNode)].find(pred) ==
                            current_proc_predec[schedule.assignedProcessor(nextNode)].end()) {
                        inc_memory += instance.getComputationalDag().nodeCommunicationWeight(pred);
                    }
                }

                if (current_proc_persistent_memory[schedule.assignedProcessor(nextNode)] + inc_memory >
                    instance.getArchitecture().memoryBound(schedule.assignedProcessor(nextNode))) {

                    toErase.push_back(node);
                }
            }
        }
    }
    return toErase;
}

bool MemConstListScheduler::check_choose_node(const BspSchedule &schedule, const BspInstance &instance, VertexType node,
                                              unsigned proc, unsigned current_superstep) {

    if (instance.getArchitecture().getMemoryConstraintType() == LOCAL) {

        if (current_proc_persistent_memory[proc] + instance.getComputationalDag().nodeMemoryWeight(node) <=
            instance.getArchitecture().memoryBound(proc)) {

            // best_node = top_node;
            // node = top_node.node;
            // p = proc;
            return true;
        } else {
            return false;
        }

    } else if (instance.getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {
        if (current_proc_persistent_memory[proc] + instance.getComputationalDag().nodeMemoryWeight(node) +
                std::max(current_proc_transient_memory[proc],
                         instance.getComputationalDag().nodeCommunicationWeight(node)) <=
            instance.getArchitecture().memoryBound(proc)) {

            // best_node = top_node;
            // node = top_node.node;
            // p = proc;
            return true;
        } else {
            return false;
        }
    } else if (instance.getArchitecture().getMemoryConstraintType() == LOCAL_IN_OUT) {

        int inc_memory = instance.getComputationalDag().nodeMemoryWeight(node) +
                         instance.getComputationalDag().nodeCommunicationWeight(node);

        for (const auto &pred : instance.getComputationalDag().parents(node)) {

            if (schedule.assignedProcessor(pred) == proc && schedule.assignedSuperstep(pred) == current_superstep) {
                inc_memory -= instance.getComputationalDag().nodeCommunicationWeight(pred);
            }
        }

        if (current_proc_persistent_memory[proc] + inc_memory <= instance.getArchitecture().memoryBound(proc)) {

            // best_node = top_node;
            // node = top_node.node;
            // p = proc;
            return true;
        } else {
            return false;
        }

    } else if (instance.getArchitecture().getMemoryConstraintType() == LOCAL_INC_EDGES) {

        int inc_memory = instance.getComputationalDag().nodeCommunicationWeight(node);
        for (const auto &pred : instance.getComputationalDag().parents(node)) {

            if (schedule.assignedSuperstep(pred) != current_superstep &&
                current_proc_predec[proc].find(pred) == current_proc_predec[proc].end()) {
                inc_memory += instance.getComputationalDag().nodeCommunicationWeight(pred);
            }
        }

        if (current_proc_persistent_memory[proc] + inc_memory <= instance.getArchitecture().memoryBound(proc)) {
            // best_node = top_node;
            // node = top_node.node;
            // p = proc;
            return true;
        } else {
            return false;
        }
    } else if (instance.getArchitecture().getMemoryConstraintType() == LOCAL_INC_EDGES_2) {

        int inc_memory = 0;
        
        if (instance.getComputationalDag().isSource(node)) {
            inc_memory += instance.getComputationalDag().nodeMemoryWeight(node); 
        }
        
        for (const auto &pred : instance.getComputationalDag().parents(node)) {

            if (schedule.assignedSuperstep(pred) != current_superstep &&
                current_proc_predec[proc].find(pred) == current_proc_predec[proc].end()) {
                inc_memory += instance.getComputationalDag().nodeCommunicationWeight(pred);
            }
        }

        if (current_proc_persistent_memory[proc] + inc_memory <= instance.getArchitecture().memoryBound(proc)) {
            // best_node = top_node;
            // node = top_node.node;
            // p = proc;
            return true;
        } else {
            return false;
        }
    }

    return true;
}