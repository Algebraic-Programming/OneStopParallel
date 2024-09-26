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
