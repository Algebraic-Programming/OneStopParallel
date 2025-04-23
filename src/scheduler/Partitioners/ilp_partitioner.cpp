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

#include "coptcpp_pch.h"
#include "scheduler/Partitioners/partitioners.hpp"

#include <vector>

std::vector<unsigned> ilp_partitioner(const unsigned num_parts, const std::multiset<int, std::greater<int>> &weights, unsigned time) {

    const unsigned num_elements = weights.size();

    const std::vector<unsigned> weights_vec(weights.cbegin(), weights.cend());

    Envr env;
    Model coptModel = env.CreateModel("Partitioner");

    std::vector<VarArray> element_partition_assignment(num_elements);

    for (unsigned element = 0; element < num_elements; element++) {
        element_partition_assignment[element] = coptModel.AddVars(num_parts, COPT_BINARY, "element_partition");
    }

    VarArray partition_weight = coptModel.AddVars(num_parts, COPT_INTEGER, "partition_weight");

    VarArray max_weight = coptModel.AddVars(1, COPT_CONTINUOUS, "partition_weight");

    for (unsigned element = 0; element < num_elements; element++) {

        Expr expr;
        for (unsigned part = 0; part < num_parts; part++) {
            expr += element_partition_assignment[element][part];
        }
        coptModel.AddConstr(expr == 1);
    }

    for (unsigned part = 0; part < num_parts; part++) {

        Expr expr;
        for (unsigned element = 0; element < num_elements; element++) {
            expr += weights_vec[element] * element_partition_assignment[element][part];
        }

        coptModel.AddConstr(expr == partition_weight[part]);

        coptModel.AddConstr(max_weight[0] >= partition_weight[part]);
    }

    for (unsigned part = 1; part < num_parts; part++) {
        coptModel.AddConstr(partition_weight[0] >= partition_weight[part]);
    }

    // if (num_parts == 2) {
    //   coptModel.AddConstr(partition_weight[0] >= partition_weight[1]);
    //   coptModel.SetObjective(partition_weight[0] - partition_weight[1], COPT_MINIMIZE);

    // } else {
    coptModel.SetObjective(max_weight[0], COPT_MINIMIZE);
    //}

    coptModel.SetDblParam(COPT_DBLPARAM_TIMELIMIT, time);
    coptModel.Solve();

    if (coptModel.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {

        std::vector<unsigned> partitioning(num_elements);

        for (unsigned element = 0; element < num_elements; element++) {

            for (unsigned part = 0; part < num_parts; part++) {
                if (element_partition_assignment[element][part].Get(COPT_DBLINFO_VALUE) == 1) {
                    partitioning[element] = part;
                }
            }
        }

        return partitioning;

    } else {
        throw std::runtime_error("ILP partitioner did not find a solution :(");
    }
}