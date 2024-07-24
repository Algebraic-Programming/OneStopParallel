#pragma once

#include "algorithms/Coarsers/HDaggCoarser.hpp"
#include "algorithms/GreedySchedulers/GreedyBspScheduler.hpp"
#include "algorithms/Scheduler.hpp"
#include "boost_extensions/transitive_edge_reduction.hpp"

class BspHDagg : public Scheduler {
  private:
    Scheduler *sched;

  public:
    BspHDagg(Scheduler &s) : sched(&s){};
    virtual ~BspHDagg() = default;

    virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override {

        HDaggCoarser coarser;

        auto pair = coarser.get_contracted_graph_and_mapping(instance.getComputationalDag());



        approx_transitive_edge_reduction filter(pair.first.getGraph());

        // boost::filtered_graph<GraphType , approx_transitive_edge_reduction>  fg(pair.first.getGraph(), filter);
        for (auto e : filter.deleted_edges) {
            boost::remove_edge(e, pair.first.getGraph());
        }

        std::cout << "contracted graph has " << boost::num_vertices(pair.first.getGraph()) << " vertices and "
                  << boost::num_edges(pair.first.getGraph()) << " edges\n";

        BspInstance contr_instance(pair.first, instance.getArchitecture());

        auto [status, schedule] = sched->computeSchedule(contr_instance);

        if (status != RETURN_STATUS::SUCCESS) {
            return {status, BspSchedule()};
        }

        return {status, coarser.expand_schedule(schedule, pair, instance)};
    }

    virtual std::string getScheduleName() const override { return "BspHDagg"; }
};