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

#include <limits>
#include <string>

#include "scheduler/Scheduler.hpp"
#include "structures/union_find.hpp"
#include "scheduler/Partitioners/partitioners.hpp"

struct weighted_memory_component {
    unsigned id;
    unsigned weight;
    unsigned memory;
    unsigned vertex_type;

    weighted_memory_component(unsigned id_, unsigned weight_, unsigned memory_, unsigned vertex_type_) : id(id_), weight(weight_), memory(memory_), vertex_type(vertex_type_) {};

    bool operator<(const weighted_memory_component& other) const { return (weight < other.weight) || ((weight == other.weight) && (memory < other.memory)) || ((weight == other.weight) && (memory == other.memory) && (id < other.id));}
    bool operator>(const weighted_memory_component& other) const { return (weight > other.weight) || ((weight == other.weight) && (memory > other.memory)) || ((weight == other.weight) && (memory == other.memory) && (id > other.id));}
};

struct weighted_memory_bin {
    const unsigned id;
    const unsigned bin_type;
    unsigned weight;
    unsigned memory;

    weighted_memory_bin(const unsigned id_, const unsigned bin_type_, unsigned weight_ = 0, unsigned memory_ = 0) : id(id_), bin_type(bin_type_), weight(weight_), memory(memory_) {};

    weighted_memory_bin operator+=(const weighted_memory_component &comp) { weight+=comp.weight; memory+=comp.memory; return *this; }
    weighted_memory_bin operator-=(const weighted_memory_component &comp) { weight-=comp.weight; memory-=comp.memory; return *this; }

    bool operator<(const weighted_memory_bin& other) const { return (weight < other.weight) || ((weight == other.weight) && (memory < other.memory)) || ((weight == other.weight) && (memory == other.memory) && (id < other.id)) ;}
};

/**
 * @brief Parameters used for HCoreHDagg
 * 
 */
struct HCoreHDagg_parameters {
    enum FRONT_TYPE { WAVEFRONT, WAVEFRONT_VERTEXTYPE };
    enum SCORE_FUNC { WEIGHT_BALANCE, SCALED_SUPERSTEP_COST };

    unsigned min_total_work_weight_check;
    unsigned min_max_work_weight_check;
    unsigned max_repeated_failures_to_improve; // must be at least one

    FRONT_TYPE front_type;
    SCORE_FUNC score_func;

    bool consider_future_score;
    float future_score_devalue;


    HCoreHDagg_parameters(unsigned min_total_work_weight_check_ = 0, unsigned min_max_work_weight_check_ = 0, unsigned max_repeated_failures_to_improve_ = 2, FRONT_TYPE front_type_ = FRONT_TYPE::WAVEFRONT_VERTEXTYPE, SCORE_FUNC score_func_ = SCORE_FUNC::SCALED_SUPERSTEP_COST, bool consider_future_score_ = true, float future_score_devalue_ = 0.75)
        : min_total_work_weight_check(min_total_work_weight_check_), min_max_work_weight_check(min_max_work_weight_check_), max_repeated_failures_to_improve(max_repeated_failures_to_improve_), front_type(front_type_), score_func(score_func_), consider_future_score(consider_future_score_), future_score_devalue(future_score_devalue_) {}
};

/**
 * @brief Scheduler based on HDagg without the coarsening step and different balancing heuristic implemented.
 * @brief Zarebavani, Behrooz, et al. "HDagg: hybrid aggregation of loop-carried dependence iterations in sparse matrix computations." 2022 IEEE International Parallel and Distributed Processing Symposium (IPDPS). IEEE, 2022.
 * 
 */
class HCoreHDagg : public Scheduler {
  private:
    const HCoreHDagg_parameters params;
    bool use_memory_constraint = false;

    // scoring functions
    float score_weight_balance(unsigned max_work_weight, unsigned total_work_weight);
    float score_scaled_superstep_cost(unsigned max_work_weight, unsigned total_work_weight, const BspInstance &instance);

    // future scoring functions
    float future_score_weight_balance(const std::set<VertexType> &ready_vertices_forward_check, const BspInstance &instance);
    float future_score_scaled_superstep_cost(const std::set<VertexType> &ready_vertices_forward_check, const BspInstance &instance);

    // For wavefront vertex additions
    std::vector<VertexType> wavefront_initial_vertices(const std::vector<unsigned> &number_of_unprocessed_parents, const BspInstance &instance);
    std::vector<VertexType> wavefront_get_next_vertices_forward_check(const std::vector<VertexType> &task_to_schedule_forward_check, std::vector<unsigned> &number_of_unprocessed_parents_forward_check, Union_Find_Universe<VertexType> &uf_structure, const BspInstance &instance);
    std::vector<VertexType> wavefront_get_next_vertices(const std::vector<std::vector<VertexType>> &best_allocation, std::vector<unsigned> &number_of_unprocessed_parents, std::vector<bool> &vertex_scheduled, std::set<VertexType> &ready_vertices, const BspInstance &instance);

    // For wavefront vertextype additions
    std::vector<VertexType> wavefront_vertextype_initial_vertices(const std::vector<unsigned> &number_of_unprocessed_parents, const BspInstance &instance);
    std::vector<VertexType> wavefront_vertextype_get_next_vertices_forward_check(const std::vector<VertexType> &task_to_schedule_forward_check, std::vector<unsigned> &number_of_unprocessed_parents_forward_check, std::set<VertexType> &ready_vertices_forward_check, const std::vector<unsigned> &preferred_order_of_adding_vertex_type_wavefront, Union_Find_Universe<VertexType> &uf_structure, const BspInstance &instance);
    std::vector<VertexType> wavefront_vertextype_get_next_vertices(const std::vector<std::vector<VertexType>> &best_allocation, std::vector<unsigned> &number_of_unprocessed_parents, std::vector<bool> &vertex_scheduled, std::set<VertexType> &ready_vertices, const BspInstance &instance);    
    
    // For allocation of components to processors
    std::vector<unsigned> component_allocation(const std::vector<std::tuple<std::vector<VertexType>, unsigned, unsigned>> &components_weights_and_memory, unsigned &max_work_weight, std::vector<unsigned> &preferred_order_of_adding_vertex_type_wavefront, const BspInstance &instanc, RETURN_STATUS &status);

  public:
    HCoreHDagg() : HCoreHDagg(HCoreHDagg_parameters()){};
    HCoreHDagg(HCoreHDagg_parameters params_) : Scheduler(), params(params_){};
    HCoreHDagg(unsigned timelimit) : HCoreHDagg(timelimit, HCoreHDagg_parameters()){};
    HCoreHDagg(unsigned timelimit, HCoreHDagg_parameters params_) : Scheduler(timelimit), params(params_){};
    virtual ~HCoreHDagg() = default;

    virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override;

    virtual std::string getScheduleName() const override {
        return "HCoreHDagg";
    }

    virtual void setUseMemoryConstraint(bool use_memory_constraint_) override { use_memory_constraint = use_memory_constraint_; }
};