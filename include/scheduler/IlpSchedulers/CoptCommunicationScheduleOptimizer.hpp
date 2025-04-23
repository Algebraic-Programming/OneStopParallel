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

#include "COPTEnv.hpp"
#include "scheduler/CommunicationScheduler.hpp"
#include "scheduler/ImprovementScheduler.hpp"
#include "model/VectorSchedule.hpp"

typedef std::tuple<unsigned int, unsigned int> KeyTuple;

class CoptCommunicationScheduleOptimizer : public ImprovementScheduler, public ICommunicationScheduler {

  protected:
    Model coptModel;

    VarArray superstep_used_var;
    VarArray max_comm_superstep_var;
    std::vector<std::vector<std::vector<VarArray>>> comm_processor_to_processor_superstep_node_var;

    void setupVariablesConstraintsObjective(const IBspSchedule &initial_scheduler,
                                            bool num_supersteps_can_change = true);

    bool numberOfSuperstepsChanged(const BspSchedule &initial_schedule);

    std::map<KeyTriple, unsigned int> reduceNumberOfSuperstepsAndAddCommScheduleConFromSolution(VectorSchedule &schedule,
                                                                   const BspSchedule &initial_schedule);

    std::map<KeyTriple, unsigned int> constructCommScheduleFromSolution(const IBspSchedule &initial_schedule);

  public:
    CoptCommunicationScheduleOptimizer()
        : ImprovementScheduler(), coptModel(COPTEnv::getInstance().CreateModel("BspCommOptimizer")) {}

    virtual ~CoptCommunicationScheduleOptimizer() = default;

    virtual RETURN_STATUS improveSchedule(BspSchedule &schedule) override;
    virtual std::pair<RETURN_STATUS, BspSchedule> constructImprovedSchedule(const BspSchedule &schedule) override;

    virtual std::map<KeyTriple, unsigned> computeCommunicationSchedule(const IBspSchedule &sched) override;

    virtual std::string getScheduleName() const override { return "ILPCommunication"; }
};
