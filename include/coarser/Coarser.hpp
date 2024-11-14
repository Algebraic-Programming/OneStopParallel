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
#include <future>
#include <iostream>
#include <thread>

#include "model/ComputationalDag.hpp"
#include "scheduler/Scheduler.hpp"
#include "model/DAGPartition.hpp"

/**
 * @class Coarser
 * @brief Abstract base class for coarsening ComputationalDags.
 * 
 */
class Coarser {

    public:

        /**
         * @brief Destructor for the Coarser class.
         */
        virtual ~Coarser() = default;

        /**
         * @brief Get the name of the coarsening algorithm.
         * @return The name of the coarsening algorithm.
         */
        virtual std::string getCoarserName() const = 0;
        
        virtual RETURN_STATUS coarseDag(const ComputationalDag &dag_in, ComputationalDag &dag_out, std::vector<std::vector<VertexType>>& vertex_map) = 0;
        
};

std::pair<RETURN_STATUS, BspSchedule> pull_back_schedule(const BspInstance &instance_large, const BspSchedule &schedule_in, const std::vector<std::vector<VertexType>>& vertex_map);

std::pair<RETURN_STATUS, DAGPartition> pull_back_partition(const BspInstance &instance_large, const DAGPartition &partition_in, const std::vector<std::vector<VertexType>>& vertex_map);

class CoarseAndSchedule : public Scheduler {

    private:

        Coarser* coarser;
        Scheduler* scheduler;

    public:

        CoarseAndSchedule(Coarser& coarser_, Scheduler& scheduler_) : coarser(&coarser_), scheduler(&scheduler_) {}    


        std::string getScheduleName() const override {
            return "CoarseAndSchedule";
        }

        std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override {

            ComputationalDag dag_coarse;
            std::vector<std::vector<VertexType>> vertex_map;
            RETURN_STATUS status = coarser->coarseDag(instance.getComputationalDag(), dag_coarse, vertex_map);
            if (status != RETURN_STATUS::SUCCESS) {
                return {status, BspSchedule()};
            }

            BspInstance instance_coarse(dag_coarse, instance.getArchitecture());

            std::pair<RETURN_STATUS, BspSchedule> schedule_coarse = scheduler->computeSchedule(instance_coarse);
            if (schedule_coarse.first != RETURN_STATUS::SUCCESS) {
                return {schedule_coarse.first, BspSchedule()};
            }

            std::pair<RETURN_STATUS, BspSchedule> schedule_large = pull_back_schedule(instance, schedule_coarse.second, vertex_map);
            return schedule_large;
        }



};