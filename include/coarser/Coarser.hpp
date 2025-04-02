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
#include "concepts/computational_dag_concept.hpp"
#include "concepts/constructable_computational_dag_concept.hpp"

/**
 * @class Coarser
 * @brief Abstract base class for coarsening ComputationalDags.
 * 
 */
template<typename Graph_t>
class Coarser {

    public:

        /**
         * @brief Destructor for the Coarser class.
         */
        virtual ~Coarser() = default;

        /**
         * @brief Get the name of the coarsening algorithm.
         * @return A human-readable name of the coarsening algorithm, typically used for identification or logging purposes.
         */
        virtual std::string getCoarserName() const = 0;
        
        /**
        * @brief Coarsens the input computational DAG into a simplified version.
        * 
        * @param dag_in The input computational DAG to be coarsened. It is expected to be a valid graph structure.
        * @param coarsened_dag The output computational DAG after coarsening. It will be populated by this method.
        * @param vertex_map A mapping from vertices in the coarse DAG to the corresponding vertices in the original DAG.
        *                   Each entry in the outer vector corresponds to a vertex in the coarse DAG, and the inner vector
        *                   contains the indices of the original vertices that were merged.
        * @return A status code indicating the success or failure of the coarsening operation.
        */
        virtual bool coarseDag(const Graph_t &dag_in, Graph_t &coarsened_dag, std::vector<std::vector<vertex_idx_t<Graph_t>>>& vertex_map) = 0;
        
};


// std::pair<RETURN_STATUS, BspSchedule> pull_back_schedule(const BspInstance &instance_large, const BspSchedule &schedule_in, const std::vector<std::vector<VertexType>>& vertex_map);

// std::pair<RETURN_STATUS, DAGPartition> pull_back_partition(const BspInstance &instance_large, const DAGPartition &partition_in, const std::vector<std::vector<VertexType>>& vertex_map);

// class CoarseAndSchedule : public Scheduler {

//     private:

//         Coarser* coarser;
//         Scheduler* scheduler;

//     public:

//         CoarseAndSchedule(Coarser& coarser_, Scheduler& scheduler_) : coarser(&coarser_), scheduler(&scheduler_) {}    


//         std::string getScheduleName() const override {
//             return "CoarseAndSchedule";
//         }

//         std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override {

//             ComputationalDag dag_coarse;
//             std::vector<std::vector<VertexType>> vertex_map;
//             RETURN_STATUS status = coarser->coarseDag(instance.getComputationalDag(), dag_coarse, vertex_map);
//             if (status != RETURN_STATUS::SUCCESS) {
//                 return {status, BspSchedule()};
//             }

//             BspInstance instance_coarse(dag_coarse, instance.getArchitecture());
//             instance_coarse.setNodeProcessorCompatibility(instance.getProcessorCompatibilityMatrix());

//             std::pair<RETURN_STATUS, BspSchedule> schedule_coarse = scheduler->computeSchedule(instance_coarse);
//             if (schedule_coarse.first != RETURN_STATUS::SUCCESS and schedule_coarse.first != RETURN_STATUS::BEST_FOUND) {
//                 return {schedule_coarse.first, BspSchedule()};
//             }
             
//             return pull_back_schedule(instance, schedule_coarse.second, vertex_map);
//         }



// };