#pragma once

#include <chrono>
#include <climits>
#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "scheduler/Scheduler.hpp"
#include "auxiliary/auxiliary.hpp"
#include "model/BspSchedule.hpp"

enum Mode { BSP, ETF, CILK, SJF, RANDOM, BL_EST };

class DummyScheduler : public Scheduler {
  public:
    DummyScheduler() = delete;
    explicit DummyScheduler(unsigned int time_limit) : Scheduler(time_limit) {}

    ~DummyScheduler() override = default;

    virtual std::string getScheduleName() const override { return "Dummy"; };

    virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override {
        // Sleep for 2*getTimeLimitSeconds() to simulate a long computation
        std::this_thread::sleep_for(std::chrono::seconds(2 * getTimeLimitSeconds()));
        return std::make_pair(SUCCESS, BspSchedule(instance));
    }

    BspSchedule computeBspGreedy();
    BspSchedule computeClassicGreedy();
    BspSchedule computeETFGreedy();
};
