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

#include "coarser/Coarser.hpp"

class top_order : public Coarser {

  private:
    
    // input
    const std::vector<VertexType> &top_ordering;
    
    // parameters
    int work_threshold = 100;
    int memory_threshold = 100;
    int communication_threshold = 100;
    unsigned degree_threshold = 10;

    // internal data strauctures
    int current_memory = 0;
    int current_work = 0;
    int current_communication = 0;
    VertexType current_super_node_idx = 0;
    std::vector<VertexType> reverse_vertex_map;

    void finish_super_node_add_edges(const ComputationalDag &dag_in, ComputationalDag &dag_out,
                                     const std::vector<VertexType> &nodes);


    void add_new_super_node(const ComputationalDag &dag_in, ComputationalDag &dag_out, VertexType node);
  

  public:
    top_order(const std::vector<VertexType> &top_, int work_threshold_ = 100, int memory_threshold_ = 100,
              int communication_threshold_ = 100)
        : top_ordering(top_), work_threshold(work_threshold_), memory_threshold(memory_threshold_),
          communication_threshold(communication_threshold_) {};

    virtual ~top_order() = default;

    void set_degree_threshold(unsigned degree_threshold_) { degree_threshold = degree_threshold_; }
    void set_work_threshold(int work_threshold_) { work_threshold = work_threshold_; }
    void set_memory_threshold(int memory_threshold_) { memory_threshold = memory_threshold_; }
    void set_communication_threshold(int communication_threshold_) { communication_threshold = communication_threshold_; }

    virtual std::string getCoarserName() const override { return "top_order_coarser"; };
    virtual RETURN_STATUS coarseDag(const ComputationalDag &dag_in, ComputationalDag &dag_out, std::vector<std::vector<VertexType>> &vertex_map) override;
};