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

#include "coarser/Coarser.hpp"

class hdagg_coarser : public Coarser {

  protected:
    int work_threshold = 100;
    int memory_threshold = 100;
    int communication_threshold = 100;

    unsigned super_node_size_threshold = 10;
    
    MEMORY_CONSTRAINT_TYPE memory_constraint_type = NONE;  

    // internal data strauctures
    int current_memory = 0;
    int current_work = 0;
    int current_communication = 0;
    VertexType current_super_node_idx = 0;

    void finish_super_node(ComputationalDag &dag_out);
    void add_edges_between_super_nodes(const ComputationalDag &dag_in, ComputationalDag &dag_out,
                                       std::vector<std::vector<VertexType>> &vertex_map,
                                       std::vector<VertexType> &reverse_vertex_map);
    void add_new_super_node(const ComputationalDag &dag_in, ComputationalDag &dag_out, VertexType node);

  public:
    hdagg_coarser() {};

    virtual ~hdagg_coarser() = default;

    virtual std::string getCoarserName() const override { return "hdagg_order_coarser"; };

    virtual RETURN_STATUS coarseDag(const ComputationalDag &dag_in, ComputationalDag &dag_out,
                                    std::vector<std::vector<VertexType>> &vertex_map) override;

    inline void set_work_threshold(int work_threshold_) { work_threshold = work_threshold_; }
    inline void set_memory_threshold(int memory_threshold_) { memory_threshold = memory_threshold_; }
    inline void set_communication_threshold(int communication_threshold_) { communication_threshold = communication_threshold_;  }
    inline void set_super_node_size_threshold(unsigned super_node_size_threshold_) { super_node_size_threshold = super_node_size_threshold_; }
    inline void set_memory_constraint_type(MEMORY_CONSTRAINT_TYPE memory_constraint_type_) { memory_constraint_type = memory_constraint_type_; }
};