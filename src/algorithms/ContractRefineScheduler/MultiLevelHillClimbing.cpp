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

#include "algorithms/ContractRefineScheduler/MultiLevelHillClimbing.hpp"

std::pair<RETURN_STATUS, BspSchedule> MultiLevelHillClimbingScheduler::computeSchedule(const BspInstance &instance) {

    DAG dag(instance.getComputationalDag());
    BSPproblem params;

    params.ConvertFromNewBspParam(instance.getArchitecture());

    Multilevel Multi(dag);
    const DAG coarseG = Multi.Coarsify(contraction_factor * instance.numberOfVertices(), "", fast_coarsification);

    int bestCost = INT_MAX;
    Schedule bestCoarseSchedule;

    ComputationalDag coarse_com_dag = coarseG.ConvertToNewDAG();
    BspInstance coarse_instance(coarse_com_dag, instance.getArchitecture());

    {
        GreedyLayers greedy_layers;

        auto [layers_status, layers_schedule] = greedy_layers.computeSchedule(coarse_instance);

        Schedule currentSchedule;
        currentSchedule.ConvertFromNewSchedule(layers_schedule);

        HillClimbing hillClimb(currentSchedule);
        hillClimb.HillClimb(timeLimitSeconds * 0.9);
        currentSchedule = hillClimb.getSchedule();
        if (currentSchedule.GetCost() < bestCost) {
            bestCost = currentSchedule.cost;
            bestCoarseSchedule = currentSchedule;
        }
    }

    {
        GreedyBspScheduler greedy_bsp;
        auto [bsp_status, bsp_schedule] = greedy_bsp.computeSchedule(coarse_instance);

        Schedule currentSchedule;
        currentSchedule.ConvertFromNewSchedule(bsp_schedule);

        HillClimbing hillClimb(currentSchedule);
        hillClimb.HillClimb(timeLimitSeconds * 0.9);
        currentSchedule = hillClimb.getSchedule();
        if (currentSchedule.GetCost() < bestCost) {
            bestCost = currentSchedule.cost;
            bestCoarseSchedule = currentSchedule;
        }
    }

    /*
        const std::vector<std::string> modes = {"BSP", "Layers"};
        for (std::string mode : modes) {
            Schedule currentSchedule = RunGreedyMode(coarseG, params, mode);
            HillClimbing hillClimb(currentSchedule);
            hillClimb.HillClimb(timeLimitSeconds * 0.9);
            currentSchedule = hillClimb.getSchedule();
            if (currentSchedule.GetCost() < bestCost) {
                bestCost = currentSchedule.cost;
                bestCoarseSchedule = currentSchedule;
            }
        }
    */

    // set refinement points - linear or exponential
    // Multi.setLinearRefinementPoints(bestCoarseSchedule.G.n, 5);
    Multi.setExponentialRefinementPoints(bestCoarseSchedule.G.n, 1.02);

    Schedule refinedSchedule = Multi.Refine(bestCoarseSchedule, hc_steps);

    HillClimbingCS hillClimbCS(refinedSchedule);
    hillClimbCS.HillClimb(timeLimitSeconds * 0.1);
    refinedSchedule = hillClimbCS.getSchedule();

    BspSchedule bsp_schedule = refinedSchedule.ConvertToNewSchedule(instance);

    return std::make_pair(SUCCESS, bsp_schedule);
}
