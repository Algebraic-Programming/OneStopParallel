#pragma once

#include "scheduler/InstanceReductor.hpp"
#include "boost_extensions/transitive_edge_reduction.hpp"

class AppTransEdgeReductor : public InstanceReductor {

  public:
    AppTransEdgeReductor(Scheduler &s) : InstanceReductor(s) {}
    AppTransEdgeReductor() : InstanceReductor() {}

    BspInstance reduce(const BspInstance &instance) override {

        approx_transitive_edge_reduction filter(instance.getComputationalDag().getGraph());

        boost::filtered_graph<GraphType, approx_transitive_edge_reduction> fg(instance.getComputationalDag().getGraph(),
                                                                              filter);

        ComputationalDag f_dag;
        boost::copy_graph(fg, f_dag.getGraph());

        BspInstance f_instance(f_dag, instance.getArchitecture());

        return f_instance;
    }

    std::string getScheduleName() const override {

        if (scheduler == nullptr) {

            return "TransitiveEdgeReduction";
        } else {

            return "TransitiveEdgeReduction-" + scheduler->getScheduleName();
        }
    }
};