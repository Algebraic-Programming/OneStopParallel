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

#include <algorithm>
#include <set>
#include <tuple>
#include <vector>

#include "auxiliary/auxiliary.hpp"
#include "scheduler/InstanceContractor.hpp"

struct SarkarParameters {
    double geomDecay = 0.875;
    std::vector<int> commCostSeq = std::vector<int>({});
    double leniency = 0.0;
};

class Sarkar : public InstanceContractor {
    private:
        SarkarParameters params;

        bool useTopPoset = true;

        void init();

        std::vector<unsigned> getBotPosetMap();
        std::vector<int> getTopDistance(int commCost);
        std::vector<int> getBotDistance(int commCost);


    protected:
        std::pair<RETURN_STATUS, unsigned> singleContraction(int commCost);
        std::pair<RETURN_STATUS, unsigned> allChildrenContraction(int commCost);
        std::pair<RETURN_STATUS, unsigned> allParentsContraction(int commCost);
        
        RETURN_STATUS run_contractions() override;

    public:
        Sarkar(SarkarParameters params_ = SarkarParameters()) : InstanceContractor(), params(params_) {}
        Sarkar(Scheduler *sched_, ImprovementScheduler *improver_ = nullptr, SarkarParameters params_ = SarkarParameters()) : InstanceContractor(sched_, improver_), params(params_) {}
        Sarkar(unsigned timelimit, Scheduler *sched_, ImprovementScheduler *improver_ = nullptr, SarkarParameters params_ = SarkarParameters()) : InstanceContractor(timelimit, sched_, improver_), params(params_) {}
        virtual ~Sarkar() = default;

        std::string getCoarserName() const override { return "Sarkar"; }
};