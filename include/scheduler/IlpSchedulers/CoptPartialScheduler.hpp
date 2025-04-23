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
#include "scheduler/ImprovementScheduler.hpp"
#include <unordered_map>

class CoptPartialScheduler : public ImprovementScheduler {

  protected:
    unsigned max_number_supersteps;
    unsigned start_superstep;
    unsigned end_superstep;

    Model coptModel;

    VarArray superstep_used_var;
    std::vector<std::vector<VarArray>> node_to_processor_superstep_var;
    std::vector<std::vector<std::vector<VarArray>>> comm_processor_to_processor_superstep_node_var;
    std::vector<std::vector<std::vector<VarArray>>> comm_processor_to_processor_superstep_source_var;

    VarArray max_comm_superstep_var;
    VarArray max_work_superstep_var;

    std::vector<unsigned> vertex_map;
    std::unordered_map<unsigned, unsigned> backward_vertex_map;

    std::vector<unsigned> source_map;
    std::unordered_map<unsigned, unsigned> backward_source_map;

    std::vector<std::vector<unsigned>> source_predecessors;
    std::vector<std::vector<unsigned>> vertex_predecessors;

    std::unordered_set<unsigned> target_vertices_set;

    unsigned num_vertices;
    unsigned num_source_vertices;

    void setupVertexMaps(const BspSchedule& initial_schedule);
    void setupPartialVariablesConstraintsObjective(const BspSchedule& initial_schedule);

    unsigned constructBspScheduleFromSolution(const BspSchedule& initial_schedule, std::vector<unsigned>& processor_assignment, std::vector<unsigned>& superstep_assignment, std::map<KeyTriple, unsigned>& commSchedule);


  public:
    CoptPartialScheduler() = delete;
    CoptPartialScheduler(unsigned start, unsigned end)
        : max_number_supersteps(end - start + 1), start_superstep(start),
          end_superstep(end), coptModel(COPTEnv::getInstance().CreateModel("BspSchedule")), num_vertices(0),
          num_source_vertices(0) {}

    CoptPartialScheduler(unsigned start, unsigned end, unsigned max_num_step_)
        : max_number_supersteps(max_num_step_), start_superstep(start),
          end_superstep(end), coptModel(COPTEnv::getInstance().CreateModel("BspSchedule")), num_vertices(0),
          num_source_vertices(0) {}

    virtual ~CoptPartialScheduler() = default;

    virtual RETURN_STATUS improveSchedule(BspSchedule& schedule) override;
    virtual std::pair<RETURN_STATUS, BspSchedule> constructImprovedSchedule(const BspSchedule& schedule) override;

    inline unsigned maxNumberSupersteps() const { return max_number_supersteps; }
    inline unsigned endSuperstep() const { return end_superstep; }
    inline unsigned startSuperstep() const { return start_superstep; }

    inline void setEndSuperstep(const unsigned &step) { end_superstep = step; }
    inline void setStartSuperstep(const unsigned &step) { start_superstep = step; }
    inline void setMaxNumberSupersteps(const unsigned &step) { max_number_supersteps = step; }

    virtual std::string getScheduleName() const override { return "PartialIlp"; }
};
