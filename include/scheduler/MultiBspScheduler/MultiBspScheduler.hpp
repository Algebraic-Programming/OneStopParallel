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

#include <boost/graph/filtered_graph.hpp>
#include <unordered_set>

#include "scheduler/Scheduler.hpp"
#include "model/MultiBsp/MultiBspInstance.hpp"
#include "model/MultiBsp/MultiBspSchedule.hpp"

class MultiBspScheduler {

    virtual std::pair<RETURN_STATUS, MultiBspSchedule> computeSchedule(const MultiBspInstance &instance) = 0;

    /**
     * @brief Get the name of the scheduling algorithm.
     * @return The name of the scheduling algorithm.
     */
    virtual std::string getScheduleName() const = 0;

    virtual void set_scheduler(Scheduler &scheduler) = 0;
};

struct dag_projection_predicate {

    std::unordered_set<VertexType> vertices;

    dag_projection_predicate(std::unordered_set<VertexType> vertices_) : vertices(vertices_) {}

    template<typename Edge>
    bool operator()(const Edge &e) const {

        return vertices.find(e.source()) != vertices.end() && vertices.find(e.target()) != vertices.end();
    }

    bool operator()(const VertexType &v) const { return vertices.find(v) == vertices.end(); }
};

using dag_projection = boost::filtered_graph<GraphType, dag_projection_predicate, dag_projection_predicate>;
class HierarchichalMultiBspScheduler : public MultiBspScheduler {

  private:
    Scheduler *scheduler;

  public:
    HierarchichalMultiBspScheduler(Scheduler &scheduler) : scheduler(&scheduler) {}
    virtual ~HierarchichalMultiBspScheduler() = default;

    virtual std::pair<RETURN_STATUS, MultiBspSchedule> computeSchedule(const MultiBspInstance &instance) override {

        const std::vector<BspArchitecture> &arch = instance.getArchitecture().getArchitectures();

        if (arch.empty()) {
            return {ERROR, MultiBspSchedule()};
        }

        BspInstance bsp_instance = BspInstance(instance.getComputationalDag(), arch[0]);

        std::cout << "Computing schedule for first architecture: " << arch[0].numberOfProcessors() << 
        " g: " << arch[0].communicationCosts() << 
        " l: " << arch[0].synchronisationCosts() <<  std::endl;

        std::pair<RETURN_STATUS, BspSchedule> result = scheduler->computeSchedule(bsp_instance);

        if (result.first != SUCCESS) {
            std::cout << "Error in scheduler" << std::endl;
            return {result.first, MultiBspSchedule()};
        }

        std::vector<std::vector<std::unordered_set<VertexType>>> processor_step_node_sets =
            split_dag(instance.getComputationalDag(), result.second);

        for (unsigned i = 0; i < processor_step_node_sets.size(); i++) {
            for (unsigned j = 0; j < processor_step_node_sets[i].size(); j++) {

                std::cout << "Processor " << i << " Step " << j << " cout: " << processor_step_node_sets[i][j].size()
                          << std::endl;
            }
        }

        dag_projection_predicate p_dag_pred(processor_step_node_sets[0][0]);

        dag_projection p_dag(instance.getComputationalDag().getGraph(), p_dag_pred, p_dag_pred);

        return {SUCCESS, MultiBspSchedule()};
    }

    virtual void set_scheduler(Scheduler &scheduler) override { this->scheduler = &scheduler; }

    virtual std::string getScheduleName() const override { return "HMultiBspScheduler"; }

    std::vector<std::vector<std::unordered_set<VertexType>>> split_dag(const ComputationalDag &dag,
                                                                       const BspSchedule &schedule);
};