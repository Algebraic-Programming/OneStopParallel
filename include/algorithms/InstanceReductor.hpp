#pragma once

#include "algorithms/Scheduler.hpp"
#include "model/BspInstance.hpp"

class InstanceReductor : public Scheduler {

  protected:
  
    Scheduler *scheduler;

  public:
    InstanceReductor(Scheduler &s) : scheduler(&s) {}

    InstanceReductor() : scheduler(nullptr) {}

    virtual ~InstanceReductor() = default;

    virtual BspInstance reduce(const BspInstance &instance) = 0;

    virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override {

        if (scheduler == nullptr) {
            throw std::runtime_error("InstanceReductor: No scheduler set.");
        }

        BspInstance reduced_instance = reduce(instance);
        auto [status, schedule] = scheduler->computeSchedule(reduced_instance);

        BspSchedule full_schedule(instance, schedule.assignedProcessors(), schedule.assignedSupersteps());
        full_schedule.setAutoCommunicationSchedule();

        return {status, full_schedule};
    }
};