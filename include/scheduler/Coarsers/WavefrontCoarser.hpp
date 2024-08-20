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

#include "structures/union_find.hpp"
#include "scheduler/InstanceContractor.hpp"

class WavefrontCoarser : public InstanceContractor {
    private:
        double min_wavefront_size;
        void auto_wavefront_size();
        bool check_wavefront_size() const;

    protected:
        RETURN_STATUS run_contractions() override;

    public:
        WavefrontCoarser(double min_wavefront_size_ = 0) : InstanceContractor(), min_wavefront_size(min_wavefront_size_) { }
        WavefrontCoarser(Scheduler* sched_, double min_wavefront_size_ = 0) : WavefrontCoarser(sched_, nullptr, min_wavefront_size_) { }
        WavefrontCoarser(Scheduler* sched_, ImprovementScheduler* improver_, double min_wavefront_size_  = 0) : InstanceContractor(sched_, improver_), min_wavefront_size(min_wavefront_size_) { }
        WavefrontCoarser(unsigned timelimit, Scheduler* sched_, double min_wavefront_size_ = 0) : WavefrontCoarser(timelimit, sched_, nullptr, min_wavefront_size_) { }
        WavefrontCoarser(unsigned timelimit, Scheduler* sched_, ImprovementScheduler* improver_, double min_wavefront_size_ = 0) : InstanceContractor(timelimit, sched_, improver_), min_wavefront_size(min_wavefront_size_) { }
        virtual ~WavefrontCoarser() = default;

        std::string getCoarserName() const override { return "WavefrontCoarser"; }
};