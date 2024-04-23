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

#include "algorithms/InstanceContractor.hpp"

class CacheLineGluer : public InstanceContractor {
    private:
        unsigned cacheline_shift;
        unsigned cacheline_size;

    protected:
        RETURN_STATUS run_contractions() override;

    public:
        CacheLineGluer(unsigned cacheline_size_, unsigned cacheline_shift_ = 0) : InstanceContractor(), cacheline_shift(cacheline_shift_), cacheline_size(cacheline_size_){ assert(cacheline_size != 0); }
        CacheLineGluer(Scheduler* sched_, unsigned cacheline_size_, unsigned cacheline_shift_ = 0) : CacheLineGluer(sched_, nullptr, cacheline_size_, cacheline_shift_) { assert(cacheline_size != 0); }
        CacheLineGluer(Scheduler* sched_, ImprovementScheduler* improver_, unsigned cacheline_size_, unsigned cacheline_shift_ = 0) : InstanceContractor(sched_, improver_), cacheline_shift(cacheline_shift_), cacheline_size(cacheline_size_) { assert(cacheline_size != 0); }
        CacheLineGluer(unsigned timelimit, Scheduler* sched_, unsigned cacheline_size_, unsigned cacheline_shift_ = 0) : CacheLineGluer(timelimit, sched_, nullptr, cacheline_size_, cacheline_shift_) { assert(cacheline_size != 0); }
        CacheLineGluer(unsigned timelimit, Scheduler* sched_, ImprovementScheduler* improver_, unsigned cacheline_size_, unsigned cacheline_shift_ = 0) : InstanceContractor(timelimit, sched_, improver_), cacheline_shift(cacheline_shift_), cacheline_size(cacheline_size_) { assert(cacheline_size != 0); }
        virtual ~CacheLineGluer() = default;

        std::string getCoarserName() const override { return "Cacheline"+cacheline_size; }
};