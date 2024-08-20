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


#include "model/BspSchedule.hpp"
#include "MultiBspInstance.hpp"

class MultiBspSchedule {

  private:
    const MultiBspInstance *instance;

    unsigned number_of_supersteps;

    std::vector<unsigned> node_to_processor_assignment;
    std::vector<unsigned> node_to_superstep_assignment;


    public:

    MultiBspSchedule() = default;

    MultiBspSchedule(const MultiBspInstance &instance) : instance(&instance), number_of_supersteps(0) {

    }

};