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

#include "scheduler/GreedySchedulers/MetaGreedyScheduler.hpp"
#include <stdexcept>

std::pair<RETURN_STATUS, BspSchedule> MetaGreedyScheduler::runGreedyMode(const BspInstance &instance,
                                                                         const std::string &mode) {

    if (mode.substr(0, 3) == "BSP") {

        GreedyBspScheduler scheduler;
        return scheduler.computeSchedule(instance);

    } else if (mode == "cilk") {
        GreedyCilkScheduler scheduler(CILK);

        return scheduler.computeSchedule(instance);

    } else if (mode == "SJF") {

        GreedyCilkScheduler scheduler(SJF);

        return scheduler.computeSchedule(instance);
    } else if (mode == "RNG") {

        RandomGreedy scheduler;

        return scheduler.computeSchedule(instance);

    } else if (mode == "CHLDRN") {

        GreedyChildren scheduler;

        return scheduler.computeSchedule(instance);

    } else if (mode == "Layers") {

        GreedyLayers scheduler;

        return scheduler.computeSchedule(instance);

    } else if (mode == "random") {

        GreedyCilkScheduler scheduler(RANDOM);

        return scheduler.computeSchedule(instance);

    } else if (mode == "Variance") {

        GreedyVarianceScheduler scheduler;
        return scheduler.computeSchedule(instance);

    } else if (mode.substr(0, 6) == "BL-EST") {

        GreedyEtfScheduler scheduler(BL_EST);

        if (mode == "BL-EST-NUMA")
            scheduler.setUseNuma(true);

        return scheduler.computeSchedule(instance);

    } else if (mode.substr(0, 3) == "ETF") {

        GreedyEtfScheduler scheduler(ETF);

        if (mode == "ETF-NUMA")
            scheduler.setUseNuma(true);

        return scheduler.computeSchedule(instance);
    } else {
        throw std::invalid_argument("MetaGreedyScheduler: unsupportet mode" + mode);
    }
};

std::pair<RETURN_STATUS, BspSchedule> MetaGreedyScheduler::computeSchedule(const BspInstance &instance) {

    std::vector<Scheduler *> greedyScheduler;

    GreedyBspScheduler bsp_greedy_scheduler;
    greedyScheduler.push_back(&bsp_greedy_scheduler);

    GreedyBspFillupScheduler fillup_greedy_scheduler;
    greedyScheduler.push_back(&fillup_greedy_scheduler);

    GreedyVarianceFillupScheduler variance_fillup_greedy_scheduler;
    greedyScheduler.push_back(&variance_fillup_greedy_scheduler);

    GreedyVarianceScheduler variance_greedy_scheduler;
    greedyScheduler.push_back(&variance_greedy_scheduler);

    GreedyBspLocking locking_greedy_scheduler;
    greedyScheduler.push_back(&locking_greedy_scheduler);

    


    // GreedyCilkScheduler cilk_scheduler(CILK);
    // greedyScheduler.push_back(&cilk_scheduler);

    // GreedyCilkScheduler sfj_scheduler(SJF);
    // greedyScheduler.push_back(&sfj_scheduler);

    // GreedyCilkScheduler rand_scheduler(RANDOM);
    // greedyScheduler.push_back(&rand_scheduler);

    // GreedyLayers layers_scheduler;
    // greedyScheduler.push_back(&layers_scheduler);

    // GreedyEtfScheduler etf_scheduler(ETF);
    // greedyScheduler.push_back(&etf_scheduler);

    // GreedyEtfScheduler est_scheduler(BL_EST);
    // greedyScheduler.push_back(&est_scheduler);

    // RandomGreedy rng_greedy_scheduler_s(true);
    // greedyScheduler.push_back(&rng_greedy_scheduler_s);

    // RandomGreedy rng_greedy_scheduler(false);
    // greedyScheduler.push_back(&rng_greedy_scheduler);

    // GreedyChildren children_scheduler_s(true);
    // greedyScheduler.push_back(&children_scheduler_s);

    // GreedyChildren children_scheduler(false);
    // greedyScheduler.push_back(&children_scheduler);

    bool schedule_found = false;
    unsigned min_costs = UINT_MAX;
    BspSchedule best_schedule;

    for (Scheduler *scheduler : greedyScheduler) {

        scheduler->setTimeLimitSeconds(timeLimitSeconds);
        auto [return_status, return_schedule] = scheduler->computeSchedule(instance);

        if (return_status == SUCCESS || return_status == BEST_FOUND) {

            unsigned costs = 0;

            switch (cost_function) {
            case BSP:
                costs = return_schedule.computeCosts();
                break;

            case SUPERSTEPS:
                costs = return_schedule.numberOfSupersteps();
                break;

            default:
                costs = return_schedule.computeCosts();
                break;
            }

            if (costs < min_costs) {
                min_costs = costs;
                best_schedule = return_schedule;
                schedule_found = true;
            }
        }
    }

    if (schedule_found) {
        return {SUCCESS, best_schedule};

    } else {

        return {ERROR, BspSchedule()};
    }
};
