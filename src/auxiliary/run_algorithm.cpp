#include "auxiliary/run_algorithm.hpp"

std::pair<RETURN_STATUS, BspSchedule> run_algorithm(const CommandLineParser &parser, const pt::ptree &algorithm,
                                                    const BspInstance &bsp_instance, unsigned timeLimit) {

    std::cout << "Running algorithm: " << algorithm.get_child("name").get_value<std::string>() << std::endl;

    if (algorithm.get_child("name").get_value<std::string>() == "Serial") {

        Serial scheduler;
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyBsp") {

        float max_percent_idle_processors =
            algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();

        GreedyBspScheduler scheduler(max_percent_idle_processors);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    }
    else if (algorithm.get_child("name").get_value<std::string>() == "GreedyBspLocking") {

        float max_percent_idle_processors =
            algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();

        GreedyBspLocking scheduler(max_percent_idle_processors);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    }
    else if (algorithm.get_child("name").get_value<std::string>() == "GreedyVariance") {

        float max_percent_idle_processors =
            algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();

        GreedyVarianceScheduler scheduler(max_percent_idle_processors);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "Bsp") {

        float max_percent_idle_processors =
            algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();

        GreedyBspFillupScheduler scheduler(max_percent_idle_processors);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "Variance") {

        float max_percent_idle_processors =
            algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();

        GreedyVarianceFillupScheduler scheduler(max_percent_idle_processors);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);

    } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyCilk") {

        GreedyCilkScheduler scheduler;
        scheduler.setTimeLimitSeconds(timeLimit);

        algorithm.get_child("parameters").get_child("mode").get_value<std::string>() == "RANDOM"
            ? scheduler.setMode(CilkMode::RANDOM)
        : algorithm.get_child("parameters").get_child("mode").get_value<std::string>() == "SJF"
            ? scheduler.setMode(CilkMode::SJF)
            : scheduler.setMode(CilkMode::CILK);

        return scheduler.computeSchedule(bsp_instance);

    } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyEtf") {

        GreedyEtfScheduler scheduler;
        scheduler.setTimeLimitSeconds(timeLimit);

        algorithm.get_child("parameters").get_child("mode").get_value<std::string>() == "BL_EST"
            ? scheduler.setMode(EtfMode::BL_EST)
            : scheduler.setMode(EtfMode::ETF);

        return scheduler.computeSchedule(bsp_instance);

    } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyLayers") {

        GreedyLayers scheduler;
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);

    } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyRandom") {

        RandomGreedy scheduler;
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);

    } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyBadRandom") {

        RandomBadGreedy scheduler;
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);

    } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyChildren") {

        GreedyChildren scheduler;
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);

    } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyMeta") {

        MetaGreedyScheduler scheduler;
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);

    } else if (algorithm.get_child("name").get_value<std::string>() == "MultiHC") {

        MultiLevelHillClimbingScheduler scheduler;
        scheduler.setTimeLimitSeconds(timeLimit);

        //        bool comp_contr_rate =
        // algorithm.get_child("parameters").get_child("compute_best_contraction_rate").get_value<bool>();

        double contraction_rate = algorithm.get_child("parameters").get_child("contraction_rate").get_value<double>();
        unsigned step = algorithm.get_child("parameters").get_child("hill_climbing_steps").get_value<unsigned>();
        bool fast_coarsification = algorithm.get_child("parameters").get_child("fast_coarsification").get_value<bool>();

        scheduler.setContractionFactor(contraction_rate);
        scheduler.setHcSteps(step);
        scheduler.setFastCoarsification(fast_coarsification);
        // scheduler.

        return scheduler.computeSchedule(bsp_instance);

    } else if (algorithm.get_child("name").get_value<std::string>() == "Wavefront") {

        unsigned hillclimb_balancer_iterations =
            algorithm.get_child("parameters").get_child("hillclimb_balancer_iterations").get_value<unsigned>();
        bool hungarian_alg = algorithm.get_child("parameters").get_child("hungarian_alg").get_value<bool>();

        Wavefront_parameters params(hillclimb_balancer_iterations, hungarian_alg);
        Wavefront scheduler(params);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "CoarseWavefront") {

        unsigned hillclimb_balancer_iterations =
            algorithm.get_child("parameters").get_child("hillclimb_balancer_iterations").get_value<unsigned>();
        bool hungarian_alg = algorithm.get_child("parameters").get_child("hungarian_alg").get_value<bool>();

        Wavefront_parameters params(hillclimb_balancer_iterations, hungarian_alg);
        Wavefront wave_front_scheduler(params);
        HillClimbingScheduler wavefront_hillclimb;
        WavefrontCoarser wavefront_coarse_scheduler(&wave_front_scheduler);
        ComboScheduler scheduler(&wavefront_coarse_scheduler, &wavefront_hillclimb);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "HDagg") {

        float balance_threshhold = algorithm.get_child("parameters").get_child("balance_threshhold").get_value<float>();
        unsigned hillclimb_balancer_iterations =
            algorithm.get_child("parameters").get_child("hillclimb_balancer_iterations").get_value<unsigned>();
        bool hungarian_alg = algorithm.get_child("parameters").get_child("hungarian_alg").get_value<bool>();
        HDagg_parameters::BALANCE_FUNC balance_function = algorithm.get_child("parameters").get_child("balance_func").get_value<std::string>() == "xlogx" ? HDagg_parameters::XLOGX : HDagg_parameters::MAXIMUM;

        HDagg_parameters params(balance_threshhold, hillclimb_balancer_iterations, hungarian_alg, balance_function);
        HDagg_simple scheduler(params);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "HDagg_original") {

        float balance_threshhold = algorithm.get_child("parameters").get_child("balance_threshhold").get_value<float>();
        unsigned hillclimb_balancer_iterations =
            algorithm.get_child("parameters").get_child("hillclimb_balancer_iterations").get_value<unsigned>();
        bool hungarian_alg = algorithm.get_child("parameters").get_child("hungarian_alg").get_value<bool>();
        HDagg_parameters::BALANCE_FUNC balance_function = algorithm.get_child("parameters").get_child("balance_func").get_value<std::string>() == "xlogx" ? HDagg_parameters::XLOGX : HDagg_parameters::MAXIMUM;

        HDagg_parameters params(balance_threshhold, hillclimb_balancer_iterations, hungarian_alg, balance_function);
        HDagg_simple scheduler_inner(params);
        HDaggCoarser scheduler(&scheduler_inner);

        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "HDagg_original_xlogx") {

        float balance_threshhold = algorithm.get_child("parameters").get_child("balance_threshhold").get_value<float>();
        unsigned hillclimb_balancer_iterations =
            algorithm.get_child("parameters").get_child("hillclimb_balancer_iterations").get_value<unsigned>();
        bool hungarian_alg = algorithm.get_child("parameters").get_child("hungarian_alg").get_value<bool>();
        HDagg_parameters::BALANCE_FUNC balance_function = algorithm.get_child("parameters").get_child("balance_func").get_value<std::string>() == "xlogx" ? HDagg_parameters::XLOGX : HDagg_parameters::MAXIMUM;

        HDagg_parameters params(balance_threshhold, hillclimb_balancer_iterations, hungarian_alg, balance_function);
        HDagg_simple scheduler_inner(params);
        HDaggCoarser scheduler(&scheduler_inner);

        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "BalDMixR") {

        unsigned number_of_partitions = bsp_instance.numberOfProcessors();
        float balance_threshhold = algorithm.get_child("parameters").get_child("balance_threshhold").get_value<float>();

        float nodes_per_clump = algorithm.get_child("parameters").get_child("nodes_per_clump").get_value<float>();
        float nodes_per_partition =
            algorithm.get_child("parameters").get_child("nodes_per_partition").get_value<float>();
        float clumps_per_partition =
            algorithm.get_child("parameters").get_child("clumps_per_partition").get_value<float>();
        float max_weight_for_flag =
            algorithm.get_child("parameters").get_child("max_weight_for_flag").get_value<float>();
        float balanced_cut_ratio = algorithm.get_child("parameters").get_child("balanced_cut_ratio").get_value<float>();
        float min_weight_for_split =
            algorithm.get_child("parameters").get_child("min_weight_for_split").get_value<float>();
        unsigned hill_climb_simple_improvement_attemps =
            algorithm.get_child("parameters").get_child("hill_climb_simple_improvement_attemps").get_value<unsigned>();
        int min_comp_generation_when_shaving =
            algorithm.get_child("parameters").get_child("min_comp_generation_when_shaving").get_value<int>();

        PartitionAlgorithm part_algo;
        if (algorithm.get_child("parameters").get_child("part_algo").get_value<std::string>() == "KarmarkarKarp") {
            part_algo = KarmarkarKarp;
        } else if (algorithm.get_child("parameters").get_child("part_algo").get_value<std::string>() == "ILP") {
            part_algo = ILP;
        } else if (algorithm.get_child("parameters").get_child("part_algo").get_value<std::string>() == "Greedy") {
            part_algo = Greedy;
        } else {
            part_algo = Greedy;
        }

        CoinType coin_type;
        if (algorithm.get_child("parameters").get_child("coin_type").get_value<std::string>() == "Thue_Morse") {
            coin_type = Thue_Morse;
        } else if (algorithm.get_child("parameters").get_child("coin_type").get_value<std::string>() ==
                   "Biased_Randomly") {
            coin_type = Biased_Randomly;
        } else {
            coin_type = Thue_Morse;
        }

        Coarse_Scheduler_Params params(number_of_partitions, balance_threshhold, part_algo, coin_type,
                                       clumps_per_partition, nodes_per_clump, nodes_per_partition, max_weight_for_flag,
                                       balanced_cut_ratio, min_weight_for_split, hill_climb_simple_improvement_attemps,
                                       min_comp_generation_when_shaving);

        BalDMixR scheduler(params);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "CoBalDMixR") {

        unsigned number_of_partitions = bsp_instance.numberOfProcessors();

        float geom_decay_num_nodes =
            algorithm.get_child("parameters").get_child("coarsen").get_child("geom_decay_num_nodes").get_value<float>();
        double poisson_par =
            algorithm.get_child("parameters").get_child("coarsen").get_child("poisson_par").get_value<double>();
        unsigned noise =
            algorithm.get_child("parameters").get_child("coarsen").get_child("noise").get_value<unsigned>();
        std::pair<unsigned, unsigned> edge_sort_ratio = std::make_pair(algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_triangle")
                                                                           .get_value<unsigned>(),
                                                                       algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_weight")
                                                                           .get_value<unsigned>());
        int num_rep_without_node_decrease = algorithm.get_child("parameters")
                                                .get_child("coarsen")
                                                .get_child("num_rep_without_node_decrease")
                                                .get_value<int>();
        float temperature_multiplier = algorithm.get_child("parameters")
                                           .get_child("coarsen")
                                           .get_child("temperature_multiplier")
                                           .get_value<float>();
        float number_of_temperature_increases = algorithm.get_child("parameters")
                                                    .get_child("coarsen")
                                                    .get_child("number_of_temperature_increases")
                                                    .get_value<float>();

        CoarsenParams coarse_params(geom_decay_num_nodes, poisson_par, noise, edge_sort_ratio,
                                    num_rep_without_node_decrease, temperature_multiplier,
                                    number_of_temperature_increases);

        int min_nodes_after_coarsen_per_partition = algorithm.get_child("parameters")
                                                        .get_child("coarsen")
                                                        .get_child("min_nodes_after_coarsen_per_partition")
                                                        .get_value<int>();
        int number_of_final_no_change_reps = algorithm.get_child("parameters")
                                                 .get_child("coarsen")
                                                 .get_child("number_of_final_no_change_reps")
                                                 .get_value<int>();

        float initial_balance_threshhold =
            algorithm.get_child("parameters").get_child("initial").get_child("balance_threshhold").get_value<float>();

        float initial_nodes_per_clump =
            algorithm.get_child("parameters").get_child("initial").get_child("nodes_per_clump").get_value<float>();
        float initial_nodes_per_partition =
            algorithm.get_child("parameters").get_child("initial").get_child("nodes_per_partition").get_value<float>();
        float initial_clumps_per_partition =
            algorithm.get_child("parameters").get_child("initial").get_child("clumps_per_partition").get_value<float>();
        float initial_max_weight_for_flag =
            algorithm.get_child("parameters").get_child("initial").get_child("max_weight_for_flag").get_value<float>();
        float initial_balanced_cut_ratio =
            algorithm.get_child("parameters").get_child("initial").get_child("balanced_cut_ratio").get_value<float>();
        float initial_min_weight_for_split =
            algorithm.get_child("parameters").get_child("initial").get_child("min_weight_for_split").get_value<float>();
        unsigned initial_hill_climb_simple_improvement_attemps = algorithm.get_child("parameters")
                                                                     .get_child("initial")
                                                                     .get_child("hill_climb_simple_improvement_attemps")
                                                                     .get_value<unsigned>();
        int initial_min_comp_generation_when_shaving = algorithm.get_child("parameters")
                                                           .get_child("initial")
                                                           .get_child("min_comp_generation_when_shaving")
                                                           .get_value<int>();

        PartitionAlgorithm initial_part_algo;
        if (algorithm.get_child("parameters").get_child("initial").get_child("part_algo").get_value<std::string>() ==
            "KarmarkarKarp") {
            initial_part_algo = KarmarkarKarp;
        } else if (algorithm.get_child("parameters")
                       .get_child("initial")
                       .get_child("part_algo")
                       .get_value<std::string>() == "ILP") {
            initial_part_algo = ILP;
        } else if (algorithm.get_child("parameters")
                       .get_child("initial")
                       .get_child("part_algo")
                       .get_value<std::string>() == "Greedy") {
            initial_part_algo = Greedy;
        } else {
            initial_part_algo = Greedy;
        }

        CoinType initial_coin_type;
        if (algorithm.get_child("parameters").get_child("initial").get_child("coin_type").get_value<std::string>() ==
            "Thue_Morse") {
            initial_coin_type = Thue_Morse;
        } else if (algorithm.get_child("parameters")
                       .get_child("initial")
                       .get_child("coin_type")
                       .get_value<std::string>() == "Biased_Randomly") {
            initial_coin_type = Biased_Randomly;
        } else {
            initial_coin_type = Thue_Morse;
        }

        Coarse_Scheduler_Params initial_params(
            number_of_partitions, initial_balance_threshhold, initial_part_algo, initial_coin_type,
            initial_clumps_per_partition, initial_nodes_per_clump, initial_nodes_per_partition,
            initial_max_weight_for_flag, initial_balanced_cut_ratio, initial_min_weight_for_split,
            initial_hill_climb_simple_improvement_attemps, initial_min_comp_generation_when_shaving);

        float final_balance_threshhold =
            algorithm.get_child("parameters").get_child("final").get_child("balance_threshhold").get_value<float>();

        float final_nodes_per_clump =
            algorithm.get_child("parameters").get_child("final").get_child("nodes_per_clump").get_value<float>();
        float final_nodes_per_partition =
            algorithm.get_child("parameters").get_child("final").get_child("nodes_per_partition").get_value<float>();
        float final_clumps_per_partition =
            algorithm.get_child("parameters").get_child("final").get_child("clumps_per_partition").get_value<float>();
        float final_max_weight_for_flag =
            algorithm.get_child("parameters").get_child("final").get_child("max_weight_for_flag").get_value<float>();
        float final_balanced_cut_ratio =
            algorithm.get_child("parameters").get_child("final").get_child("balanced_cut_ratio").get_value<float>();
        float final_min_weight_for_split =
            algorithm.get_child("parameters").get_child("final").get_child("min_weight_for_split").get_value<float>();
        unsigned final_hill_climb_simple_improvement_attemps = algorithm.get_child("parameters")
                                                                   .get_child("final")
                                                                   .get_child("hill_climb_simple_improvement_attemps")
                                                                   .get_value<unsigned>();
        int final_min_comp_generation_when_shaving = algorithm.get_child("parameters")
                                                         .get_child("final")
                                                         .get_child("min_comp_generation_when_shaving")
                                                         .get_value<int>();

        PartitionAlgorithm final_part_algo;
        if (algorithm.get_child("parameters").get_child("final").get_child("part_algo").get_value<std::string>() ==
            "KarmarkarKarp") {
            final_part_algo = KarmarkarKarp;
        } else if (algorithm.get_child("parameters")
                       .get_child("final")
                       .get_child("part_algo")
                       .get_value<std::string>() == "ILP") {
            final_part_algo = ILP;
        } else if (algorithm.get_child("parameters")
                       .get_child("final")
                       .get_child("part_algo")
                       .get_value<std::string>() == "Greedy") {
            final_part_algo = Greedy;
        } else {
            final_part_algo = Greedy;
        }

        CoinType final_coin_type;
        if (algorithm.get_child("parameters").get_child("final").get_child("coin_type").get_value<std::string>() ==
            "Thue_Morse") {
            final_coin_type = Thue_Morse;
        } else if (algorithm.get_child("parameters")
                       .get_child("final")
                       .get_child("coin_type")
                       .get_value<std::string>() == "Biased_Randomly") {
            final_coin_type = Biased_Randomly;
        } else {
            final_coin_type = Thue_Morse;
        }

        Coarse_Scheduler_Params final_params(
            number_of_partitions, final_balance_threshhold, final_part_algo, final_coin_type,
            final_clumps_per_partition, final_nodes_per_clump, final_nodes_per_partition, final_max_weight_for_flag,
            final_balanced_cut_ratio, final_min_weight_for_split, final_hill_climb_simple_improvement_attemps,
            final_min_comp_generation_when_shaving);

        CoarseRefineScheduler_parameters params(initial_params, final_params, coarse_params,
                                                min_nodes_after_coarsen_per_partition, number_of_final_no_change_reps);

        CoBalDMixR scheduler(params);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "CoBalDMixRLK") {

        unsigned number_of_partitions = bsp_instance.numberOfProcessors();

        float geom_decay_num_nodes =
            algorithm.get_child("parameters").get_child("coarsen").get_child("geom_decay_num_nodes").get_value<float>();
        double poisson_par =
            algorithm.get_child("parameters").get_child("coarsen").get_child("poisson_par").get_value<double>();
        unsigned noise =
            algorithm.get_child("parameters").get_child("coarsen").get_child("noise").get_value<unsigned>();
        std::pair<unsigned, unsigned> edge_sort_ratio = std::make_pair(algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_triangle")
                                                                           .get_value<unsigned>(),
                                                                       algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_weight")
                                                                           .get_value<unsigned>());
        int num_rep_without_node_decrease = algorithm.get_child("parameters")
                                                .get_child("coarsen")
                                                .get_child("num_rep_without_node_decrease")
                                                .get_value<int>();
        float temperature_multiplier = algorithm.get_child("parameters")
                                           .get_child("coarsen")
                                           .get_child("temperature_multiplier")
                                           .get_value<float>();
        float number_of_temperature_increases = algorithm.get_child("parameters")
                                                    .get_child("coarsen")
                                                    .get_child("number_of_temperature_increases")
                                                    .get_value<float>();

        CoarsenParams coarse_params(geom_decay_num_nodes, poisson_par, noise, edge_sort_ratio,
                                    num_rep_without_node_decrease, temperature_multiplier,
                                    number_of_temperature_increases);

        int min_nodes_after_coarsen_per_partition = algorithm.get_child("parameters")
                                                        .get_child("coarsen")
                                                        .get_child("min_nodes_after_coarsen_per_partition")
                                                        .get_value<int>();
        int number_of_final_no_change_reps = algorithm.get_child("parameters")
                                                 .get_child("coarsen")
                                                 .get_child("number_of_final_no_change_reps")
                                                 .get_value<int>();

        float initial_balance_threshhold =
            algorithm.get_child("parameters").get_child("initial").get_child("balance_threshhold").get_value<float>();

        float initial_nodes_per_clump =
            algorithm.get_child("parameters").get_child("initial").get_child("nodes_per_clump").get_value<float>();
        float initial_nodes_per_partition =
            algorithm.get_child("parameters").get_child("initial").get_child("nodes_per_partition").get_value<float>();
        float initial_clumps_per_partition =
            algorithm.get_child("parameters").get_child("initial").get_child("clumps_per_partition").get_value<float>();
        float initial_max_weight_for_flag =
            algorithm.get_child("parameters").get_child("initial").get_child("max_weight_for_flag").get_value<float>();
        float initial_balanced_cut_ratio =
            algorithm.get_child("parameters").get_child("initial").get_child("balanced_cut_ratio").get_value<float>();
        float initial_min_weight_for_split =
            algorithm.get_child("parameters").get_child("initial").get_child("min_weight_for_split").get_value<float>();
        unsigned initial_hill_climb_simple_improvement_attemps = algorithm.get_child("parameters")
                                                                     .get_child("initial")
                                                                     .get_child("hill_climb_simple_improvement_attemps")
                                                                     .get_value<unsigned>();
        int initial_min_comp_generation_when_shaving = algorithm.get_child("parameters")
                                                           .get_child("initial")
                                                           .get_child("min_comp_generation_when_shaving")
                                                           .get_value<int>();

        PartitionAlgorithm initial_part_algo;
        if (algorithm.get_child("parameters").get_child("initial").get_child("part_algo").get_value<std::string>() ==
            "KarmarkarKarp") {
            initial_part_algo = KarmarkarKarp;
        } else if (algorithm.get_child("parameters")
                       .get_child("initial")
                       .get_child("part_algo")
                       .get_value<std::string>() == "ILP") {
            initial_part_algo = ILP;
        } else if (algorithm.get_child("parameters")
                       .get_child("initial")
                       .get_child("part_algo")
                       .get_value<std::string>() == "Greedy") {
            initial_part_algo = Greedy;
        } else {
            initial_part_algo = Greedy;
        }

        CoinType initial_coin_type;
        if (algorithm.get_child("parameters").get_child("initial").get_child("coin_type").get_value<std::string>() ==
            "Thue_Morse") {
            initial_coin_type = Thue_Morse;
        } else if (algorithm.get_child("parameters")
                       .get_child("initial")
                       .get_child("coin_type")
                       .get_value<std::string>() == "Biased_Randomly") {
            initial_coin_type = Biased_Randomly;
        } else {
            initial_coin_type = Thue_Morse;
        }

        Coarse_Scheduler_Params initial_params(
            number_of_partitions, initial_balance_threshhold, initial_part_algo, initial_coin_type,
            initial_clumps_per_partition, initial_nodes_per_clump, initial_nodes_per_partition,
            initial_max_weight_for_flag, initial_balanced_cut_ratio, initial_min_weight_for_split,
            initial_hill_climb_simple_improvement_attemps, initial_min_comp_generation_when_shaving);

        float final_balance_threshhold =
            algorithm.get_child("parameters").get_child("final").get_child("balance_threshhold").get_value<float>();

        float final_nodes_per_clump =
            algorithm.get_child("parameters").get_child("final").get_child("nodes_per_clump").get_value<float>();
        float final_nodes_per_partition =
            algorithm.get_child("parameters").get_child("final").get_child("nodes_per_partition").get_value<float>();
        float final_clumps_per_partition =
            algorithm.get_child("parameters").get_child("final").get_child("clumps_per_partition").get_value<float>();
        float final_max_weight_for_flag =
            algorithm.get_child("parameters").get_child("final").get_child("max_weight_for_flag").get_value<float>();
        float final_balanced_cut_ratio =
            algorithm.get_child("parameters").get_child("final").get_child("balanced_cut_ratio").get_value<float>();
        float final_min_weight_for_split =
            algorithm.get_child("parameters").get_child("final").get_child("min_weight_for_split").get_value<float>();
        unsigned final_hill_climb_simple_improvement_attemps = algorithm.get_child("parameters")
                                                                   .get_child("final")
                                                                   .get_child("hill_climb_simple_improvement_attemps")
                                                                   .get_value<unsigned>();
        int final_min_comp_generation_when_shaving = algorithm.get_child("parameters")
                                                         .get_child("final")
                                                         .get_child("min_comp_generation_when_shaving")
                                                         .get_value<int>();

        PartitionAlgorithm final_part_algo;
        if (algorithm.get_child("parameters").get_child("final").get_child("part_algo").get_value<std::string>() ==
            "KarmarkarKarp") {
            final_part_algo = KarmarkarKarp;
        } else if (algorithm.get_child("parameters")
                       .get_child("final")
                       .get_child("part_algo")
                       .get_value<std::string>() == "ILP") {
            final_part_algo = ILP;
        } else if (algorithm.get_child("parameters")
                       .get_child("final")
                       .get_child("part_algo")
                       .get_value<std::string>() == "Greedy") {
            final_part_algo = Greedy;
        } else {
            final_part_algo = Greedy;
        }

        CoinType final_coin_type;
        if (algorithm.get_child("parameters").get_child("final").get_child("coin_type").get_value<std::string>() ==
            "Thue_Morse") {
            final_coin_type = Thue_Morse;
        } else if (algorithm.get_child("parameters")
                       .get_child("final")
                       .get_child("coin_type")
                       .get_value<std::string>() == "Biased_Randomly") {
            final_coin_type = Biased_Randomly;
        } else {
            final_coin_type = Thue_Morse;
        }

        Coarse_Scheduler_Params final_params(
            number_of_partitions, final_balance_threshhold, final_part_algo, final_coin_type,
            final_clumps_per_partition, final_nodes_per_clump, final_nodes_per_partition, final_max_weight_for_flag,
            final_balanced_cut_ratio, final_min_weight_for_split, final_hill_climb_simple_improvement_attemps,
            final_min_comp_generation_when_shaving);

        CoarseRefineScheduler_parameters params(initial_params, final_params, coarse_params,
                                                min_nodes_after_coarsen_per_partition, number_of_final_no_change_reps);

        CoBalDMixR cob_scheduler(params);
        LKTotalCommScheduler improver;
        improver.set_quick_pass(true);
        ComboScheduler scheduler(&cob_scheduler, &improver);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "BestGreedyLK") {

        MetaGreedyScheduler best_greedy;
        LKTotalCommScheduler improver;
        improver.set_quick_pass(true);
        ComboScheduler scheduler(&best_greedy, &improver);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "BestGreedyHC") {

        MetaGreedyScheduler best_greedy;
        HillClimbingScheduler hill_climbing;
        ComboScheduler scheduler(&best_greedy, &hill_climbing);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "CoarseBestGreedyHC") {

        unsigned number_of_partitions = bsp_instance.numberOfProcessors();

        float geom_decay_num_nodes =
            algorithm.get_child("parameters").get_child("coarsen").get_child("geom_decay_num_nodes").get_value<float>();
        double poisson_par =
            algorithm.get_child("parameters").get_child("coarsen").get_child("poisson_par").get_value<double>();
        unsigned noise =
            algorithm.get_child("parameters").get_child("coarsen").get_child("noise").get_value<unsigned>();
        std::pair<unsigned, unsigned> edge_sort_ratio = std::make_pair(algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_triangle")
                                                                           .get_value<unsigned>(),
                                                                       algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_weight")
                                                                           .get_value<unsigned>());
        int num_rep_without_node_decrease = algorithm.get_child("parameters")
                                                .get_child("coarsen")
                                                .get_child("num_rep_without_node_decrease")
                                                .get_value<int>();
        float temperature_multiplier = algorithm.get_child("parameters")
                                           .get_child("coarsen")
                                           .get_child("temperature_multiplier")
                                           .get_value<float>();
        float number_of_temperature_increases = algorithm.get_child("parameters")
                                                    .get_child("coarsen")
                                                    .get_child("number_of_temperature_increases")
                                                    .get_value<float>();

        CoarsenParams coarse_params(geom_decay_num_nodes, poisson_par, noise, edge_sort_ratio,
                                    num_rep_without_node_decrease, temperature_multiplier,
                                    number_of_temperature_increases);

        int min_nodes_after_coarsen_per_partition = algorithm.get_child("parameters")
                                                        .get_child("coarsen")
                                                        .get_child("min_nodes_after_coarsen_per_partition")
                                                        .get_value<int>();

        MetaGreedyScheduler best_greedy;
        HillClimbingScheduler hill_climbing;
        SquashA scheduler(&best_greedy, &hill_climbing, coarse_params,
                          min_nodes_after_coarsen_per_partition * number_of_partitions);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "CoarseBestGreedy") {

        unsigned number_of_partitions = bsp_instance.numberOfProcessors();

        float geom_decay_num_nodes =
            algorithm.get_child("parameters").get_child("coarsen").get_child("geom_decay_num_nodes").get_value<float>();
        double poisson_par =
            algorithm.get_child("parameters").get_child("coarsen").get_child("poisson_par").get_value<double>();
        unsigned noise =
            algorithm.get_child("parameters").get_child("coarsen").get_child("noise").get_value<unsigned>();
        std::pair<unsigned, unsigned> edge_sort_ratio = std::make_pair(algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_triangle")
                                                                           .get_value<unsigned>(),
                                                                       algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_weight")
                                                                           .get_value<unsigned>());
        int num_rep_without_node_decrease = algorithm.get_child("parameters")
                                                .get_child("coarsen")
                                                .get_child("num_rep_without_node_decrease")
                                                .get_value<int>();
        float temperature_multiplier = algorithm.get_child("parameters")
                                           .get_child("coarsen")
                                           .get_child("temperature_multiplier")
                                           .get_value<float>();
        float number_of_temperature_increases = algorithm.get_child("parameters")
                                                    .get_child("coarsen")
                                                    .get_child("number_of_temperature_increases")
                                                    .get_value<float>();

        CoarsenParams coarse_params(geom_decay_num_nodes, poisson_par, noise, edge_sort_ratio,
                                    num_rep_without_node_decrease, temperature_multiplier,
                                    number_of_temperature_increases);

        int min_nodes_after_coarsen_per_partition = algorithm.get_child("parameters")
                                                        .get_child("coarsen")
                                                        .get_child("min_nodes_after_coarsen_per_partition")
                                                        .get_value<int>();

        MetaGreedyScheduler best_greedy;
        SquashA scheduler(&best_greedy, coarse_params, min_nodes_after_coarsen_per_partition * number_of_partitions);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "SquashHDagg") {

        unsigned number_of_partitions = bsp_instance.numberOfProcessors();

        float geom_decay_num_nodes =
            algorithm.get_child("parameters").get_child("coarsen").get_child("geom_decay_num_nodes").get_value<float>();
        double poisson_par =
            algorithm.get_child("parameters").get_child("coarsen").get_child("poisson_par").get_value<double>();
        unsigned noise =
            algorithm.get_child("parameters").get_child("coarsen").get_child("noise").get_value<unsigned>();
        std::pair<unsigned, unsigned> edge_sort_ratio = std::make_pair(algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_triangle")
                                                                           .get_value<unsigned>(),
                                                                       algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_weight")
                                                                           .get_value<unsigned>());
        int num_rep_without_node_decrease = algorithm.get_child("parameters")
                                                .get_child("coarsen")
                                                .get_child("num_rep_without_node_decrease")
                                                .get_value<int>();
        float temperature_multiplier = algorithm.get_child("parameters")
                                           .get_child("coarsen")
                                           .get_child("temperature_multiplier")
                                           .get_value<float>();
        float number_of_temperature_increases = algorithm.get_child("parameters")
                                                    .get_child("coarsen")
                                                    .get_child("number_of_temperature_increases")
                                                    .get_value<float>();

        CoarsenParams coarse_params(geom_decay_num_nodes, poisson_par, noise, edge_sort_ratio,
                                    num_rep_without_node_decrease, temperature_multiplier,
                                    number_of_temperature_increases);

        int min_nodes_after_coarsen_per_partition = algorithm.get_child("parameters")
                                                        .get_child("coarsen")
                                                        .get_child("min_nodes_after_coarsen_per_partition")
                                                        .get_value<int>();

        float balance_threshhold =
            algorithm.get_child("parameters").get_child("HDagg").get_child("balance_threshhold").get_value<float>();
        unsigned hillclimb_balancer_iterations = algorithm.get_child("parameters")
                                                     .get_child("HDagg")
                                                     .get_child("hillclimb_balancer_iterations")
                                                     .get_value<unsigned>();
        bool hungarian_alg =
            algorithm.get_child("parameters").get_child("HDagg").get_child("hungarian_alg").get_value<bool>();
        HDagg_parameters::BALANCE_FUNC balance_function = algorithm.get_child("parameters").get_child("HDagg").get_child("balance_func").get_value<std::string>() == "xlogx" ? HDagg_parameters::XLOGX : HDagg_parameters::MAXIMUM;

        HDagg_parameters params(balance_threshhold, hillclimb_balancer_iterations, hungarian_alg, balance_function);

        HDagg_simple hdagg_scheduler(params);
        SquashA scheduler(&hdagg_scheduler, coarse_params,
                          min_nodes_after_coarsen_per_partition * number_of_partitions);

        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "SquashComboBestGreedyLK") {

        unsigned number_of_partitions = bsp_instance.numberOfProcessors();

        float geom_decay_num_nodes =
            algorithm.get_child("parameters").get_child("coarsen").get_child("geom_decay_num_nodes").get_value<float>();
        double poisson_par =
            algorithm.get_child("parameters").get_child("coarsen").get_child("poisson_par").get_value<double>();
        unsigned noise =
            algorithm.get_child("parameters").get_child("coarsen").get_child("noise").get_value<unsigned>();
        std::pair<unsigned, unsigned> edge_sort_ratio = std::make_pair(algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_triangle")
                                                                           .get_value<unsigned>(),
                                                                       algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_weight")
                                                                           .get_value<unsigned>());
        int num_rep_without_node_decrease = algorithm.get_child("parameters")
                                                .get_child("coarsen")
                                                .get_child("num_rep_without_node_decrease")
                                                .get_value<int>();
        float temperature_multiplier = algorithm.get_child("parameters")
                                           .get_child("coarsen")
                                           .get_child("temperature_multiplier")
                                           .get_value<float>();
        float number_of_temperature_increases = algorithm.get_child("parameters")
                                                    .get_child("coarsen")
                                                    .get_child("number_of_temperature_increases")
                                                    .get_value<float>();

        CoarsenParams coarse_params(geom_decay_num_nodes, poisson_par, noise, edge_sort_ratio,
                                    num_rep_without_node_decrease, temperature_multiplier,
                                    number_of_temperature_increases);

        int min_nodes_after_coarsen_per_partition = algorithm.get_child("parameters")
                                                        .get_child("coarsen")
                                                        .get_child("min_nodes_after_coarsen_per_partition")
                                                        .get_value<int>();

        MetaGreedyScheduler best_greedy_sched;
        LKTotalCommScheduler improver;
        improver.set_quick_pass(true);
        ComboScheduler combo_sched(&best_greedy_sched, &improver);
        SquashA scheduler(&combo_sched, coarse_params, min_nodes_after_coarsen_per_partition * number_of_partitions);

        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "CoarseWaveBestGreedy") {

        MetaGreedyScheduler best_greedy;
        WavefrontCoarser scheduler(&best_greedy);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "CoarseWaveBestGreedyHC") {

        MetaGreedyScheduler best_greedy;
        HillClimbingScheduler hill_climbing;
        WavefrontCoarser wave_coarse(&best_greedy);
        ComboScheduler scheduler(&wave_coarse, &hill_climbing);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "CoarseHDaggGreedyBsp") {

        float max_percent_idle_processor =
            algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();

        GreedyBspScheduler bsp_greedy(max_percent_idle_processor);
        HDaggCoarser scheduler(&bsp_greedy);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);

    } else if (algorithm.get_child("name").get_value<std::string>() == "CoarseHDaggGreedyBspFillup") {

        float max_percent_idle_processor =
            algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();

        GreedyBspFillupScheduler bsp_greedy(max_percent_idle_processor);
        HDaggCoarser scheduler(&bsp_greedy);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);

    }
    else if (algorithm.get_child("name").get_value<std::string>() == "CoarseHDaggGreedyLocking") {

        float max_percent_idle_processor =
            algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();

        GreedyBspLocking bsp_greedy(max_percent_idle_processor);
        HDaggCoarser scheduler(&bsp_greedy);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);

    } else if (algorithm.get_child("name").get_value<std::string>() == "CoarseHDaggGreedyVariance") {

        float max_percent_idle_processor =
            algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();

        GreedyVarianceScheduler variance_greedy(max_percent_idle_processor);
        HDaggCoarser scheduler(&variance_greedy);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "CoarseHDaggGreedyVarianceFillup") {

        float max_percent_idle_processor =
            algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();

        GreedyVarianceFillupScheduler variance_greedy(max_percent_idle_processor);
        HDaggCoarser scheduler(&variance_greedy);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    
    
    } else if (algorithm.get_child("name").get_value<std::string>() == "CoarseHDaggGreedyLocking") {

        float max_percent_idle_processor =
            algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();

        GreedyBspLocking variance_greedy(max_percent_idle_processor);
        HDaggCoarser scheduler(&variance_greedy);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "CoarseHDaggBestGreedy") {

        MetaGreedyScheduler meta_greedy;
        HDaggCoarser scheduler(&meta_greedy);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "CoarseHDaggBestGreedyHC") {

        MetaGreedyScheduler meta_greedy;
        HillClimbingScheduler improver;
        HDaggCoarser scheduler(&meta_greedy, &improver);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "CoarseHDaggBestGreedyLK") {

        MetaGreedyScheduler meta_greedy;
        LKTotalCommScheduler improver;
        improver.set_quick_pass(true);
        HDaggCoarser scheduler(&meta_greedy, &improver);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "CoarseHDaggCoBalDMixR") {

        unsigned number_of_partitions = bsp_instance.numberOfProcessors();

        float geom_decay_num_nodes =
            algorithm.get_child("parameters").get_child("coarsen").get_child("geom_decay_num_nodes").get_value<float>();
        double poisson_par =
            algorithm.get_child("parameters").get_child("coarsen").get_child("poisson_par").get_value<double>();
        unsigned noise =
            algorithm.get_child("parameters").get_child("coarsen").get_child("noise").get_value<unsigned>();
        std::pair<unsigned, unsigned> edge_sort_ratio = std::make_pair(algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_triangle")
                                                                           .get_value<unsigned>(),
                                                                       algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_weight")
                                                                           .get_value<unsigned>());
        int num_rep_without_node_decrease = algorithm.get_child("parameters")
                                                .get_child("coarsen")
                                                .get_child("num_rep_without_node_decrease")
                                                .get_value<int>();
        float temperature_multiplier = algorithm.get_child("parameters")
                                           .get_child("coarsen")
                                           .get_child("temperature_multiplier")
                                           .get_value<float>();
        float number_of_temperature_increases = algorithm.get_child("parameters")
                                                    .get_child("coarsen")
                                                    .get_child("number_of_temperature_increases")
                                                    .get_value<float>();

        CoarsenParams coarse_params(geom_decay_num_nodes, poisson_par, noise, edge_sort_ratio,
                                    num_rep_without_node_decrease, temperature_multiplier,
                                    number_of_temperature_increases);

        int min_nodes_after_coarsen_per_partition = algorithm.get_child("parameters")
                                                        .get_child("coarsen")
                                                        .get_child("min_nodes_after_coarsen_per_partition")
                                                        .get_value<int>();
        int number_of_final_no_change_reps = algorithm.get_child("parameters")
                                                 .get_child("coarsen")
                                                 .get_child("number_of_final_no_change_reps")
                                                 .get_value<int>();

        float initial_balance_threshhold =
            algorithm.get_child("parameters").get_child("initial").get_child("balance_threshhold").get_value<float>();

        float initial_nodes_per_clump =
            algorithm.get_child("parameters").get_child("initial").get_child("nodes_per_clump").get_value<float>();
        float initial_nodes_per_partition =
            algorithm.get_child("parameters").get_child("initial").get_child("nodes_per_partition").get_value<float>();
        float initial_clumps_per_partition =
            algorithm.get_child("parameters").get_child("initial").get_child("clumps_per_partition").get_value<float>();
        float initial_max_weight_for_flag =
            algorithm.get_child("parameters").get_child("initial").get_child("max_weight_for_flag").get_value<float>();
        float initial_balanced_cut_ratio =
            algorithm.get_child("parameters").get_child("initial").get_child("balanced_cut_ratio").get_value<float>();
        float initial_min_weight_for_split =
            algorithm.get_child("parameters").get_child("initial").get_child("min_weight_for_split").get_value<float>();
        unsigned initial_hill_climb_simple_improvement_attemps = algorithm.get_child("parameters")
                                                                     .get_child("initial")
                                                                     .get_child("hill_climb_simple_improvement_attemps")
                                                                     .get_value<unsigned>();
        int initial_min_comp_generation_when_shaving = algorithm.get_child("parameters")
                                                           .get_child("initial")
                                                           .get_child("min_comp_generation_when_shaving")
                                                           .get_value<int>();

        PartitionAlgorithm initial_part_algo;
        if (algorithm.get_child("parameters").get_child("initial").get_child("part_algo").get_value<std::string>() ==
            "KarmarkarKarp") {
            initial_part_algo = KarmarkarKarp;
        } else if (algorithm.get_child("parameters")
                       .get_child("initial")
                       .get_child("part_algo")
                       .get_value<std::string>() == "ILP") {
            initial_part_algo = ILP;
        } else if (algorithm.get_child("parameters")
                       .get_child("initial")
                       .get_child("part_algo")
                       .get_value<std::string>() == "Greedy") {
            initial_part_algo = Greedy;
        } else {
            initial_part_algo = Greedy;
        }

        CoinType initial_coin_type;
        if (algorithm.get_child("parameters").get_child("initial").get_child("coin_type").get_value<std::string>() ==
            "Thue_Morse") {
            initial_coin_type = Thue_Morse;
        } else if (algorithm.get_child("parameters")
                       .get_child("initial")
                       .get_child("coin_type")
                       .get_value<std::string>() == "Biased_Randomly") {
            initial_coin_type = Biased_Randomly;
        } else {
            initial_coin_type = Thue_Morse;
        }

        Coarse_Scheduler_Params initial_params(
            number_of_partitions, initial_balance_threshhold, initial_part_algo, initial_coin_type,
            initial_clumps_per_partition, initial_nodes_per_clump, initial_nodes_per_partition,
            initial_max_weight_for_flag, initial_balanced_cut_ratio, initial_min_weight_for_split,
            initial_hill_climb_simple_improvement_attemps, initial_min_comp_generation_when_shaving);

        float final_balance_threshhold =
            algorithm.get_child("parameters").get_child("final").get_child("balance_threshhold").get_value<float>();

        float final_nodes_per_clump =
            algorithm.get_child("parameters").get_child("final").get_child("nodes_per_clump").get_value<float>();
        float final_nodes_per_partition =
            algorithm.get_child("parameters").get_child("final").get_child("nodes_per_partition").get_value<float>();
        float final_clumps_per_partition =
            algorithm.get_child("parameters").get_child("final").get_child("clumps_per_partition").get_value<float>();
        float final_max_weight_for_flag =
            algorithm.get_child("parameters").get_child("final").get_child("max_weight_for_flag").get_value<float>();
        float final_balanced_cut_ratio =
            algorithm.get_child("parameters").get_child("final").get_child("balanced_cut_ratio").get_value<float>();
        float final_min_weight_for_split =
            algorithm.get_child("parameters").get_child("final").get_child("min_weight_for_split").get_value<float>();
        unsigned final_hill_climb_simple_improvement_attemps = algorithm.get_child("parameters")
                                                                   .get_child("final")
                                                                   .get_child("hill_climb_simple_improvement_attemps")
                                                                   .get_value<unsigned>();
        int final_min_comp_generation_when_shaving = algorithm.get_child("parameters")
                                                         .get_child("final")
                                                         .get_child("min_comp_generation_when_shaving")
                                                         .get_value<int>();

        PartitionAlgorithm final_part_algo;
        if (algorithm.get_child("parameters").get_child("final").get_child("part_algo").get_value<std::string>() ==
            "KarmarkarKarp") {
            final_part_algo = KarmarkarKarp;
        } else if (algorithm.get_child("parameters")
                       .get_child("final")
                       .get_child("part_algo")
                       .get_value<std::string>() == "ILP") {
            final_part_algo = ILP;
        } else if (algorithm.get_child("parameters")
                       .get_child("final")
                       .get_child("part_algo")
                       .get_value<std::string>() == "Greedy") {
            final_part_algo = Greedy;
        } else {
            final_part_algo = Greedy;
        }

        CoinType final_coin_type;
        if (algorithm.get_child("parameters").get_child("final").get_child("coin_type").get_value<std::string>() ==
            "Thue_Morse") {
            final_coin_type = Thue_Morse;
        } else if (algorithm.get_child("parameters")
                       .get_child("final")
                       .get_child("coin_type")
                       .get_value<std::string>() == "Biased_Randomly") {
            final_coin_type = Biased_Randomly;
        } else {
            final_coin_type = Thue_Morse;
        }

        Coarse_Scheduler_Params final_params(
            number_of_partitions, final_balance_threshhold, final_part_algo, final_coin_type,
            final_clumps_per_partition, final_nodes_per_clump, final_nodes_per_partition, final_max_weight_for_flag,
            final_balanced_cut_ratio, final_min_weight_for_split, final_hill_climb_simple_improvement_attemps,
            final_min_comp_generation_when_shaving);

        CoarseRefineScheduler_parameters params(initial_params, final_params, coarse_params,
                                                min_nodes_after_coarsen_per_partition, number_of_final_no_change_reps);

        CoBalDMixR sched(params);
        HDaggCoarser scheduler(&sched);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);

    } else if (algorithm.get_child("name").get_value<std::string>() == "CoarseHDaggCoBalDMixRHC") {

        unsigned number_of_partitions = bsp_instance.numberOfProcessors();

        float geom_decay_num_nodes =
            algorithm.get_child("parameters").get_child("coarsen").get_child("geom_decay_num_nodes").get_value<float>();
        double poisson_par =
            algorithm.get_child("parameters").get_child("coarsen").get_child("poisson_par").get_value<double>();
        unsigned noise =
            algorithm.get_child("parameters").get_child("coarsen").get_child("noise").get_value<unsigned>();
        std::pair<unsigned, unsigned> edge_sort_ratio = std::make_pair(algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_triangle")
                                                                           .get_value<unsigned>(),
                                                                       algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_weight")
                                                                           .get_value<unsigned>());
        int num_rep_without_node_decrease = algorithm.get_child("parameters")
                                                .get_child("coarsen")
                                                .get_child("num_rep_without_node_decrease")
                                                .get_value<int>();
        float temperature_multiplier = algorithm.get_child("parameters")
                                           .get_child("coarsen")
                                           .get_child("temperature_multiplier")
                                           .get_value<float>();
        float number_of_temperature_increases = algorithm.get_child("parameters")
                                                    .get_child("coarsen")
                                                    .get_child("number_of_temperature_increases")
                                                    .get_value<float>();

        CoarsenParams coarse_params(geom_decay_num_nodes, poisson_par, noise, edge_sort_ratio,
                                    num_rep_without_node_decrease, temperature_multiplier,
                                    number_of_temperature_increases);

        int min_nodes_after_coarsen_per_partition = algorithm.get_child("parameters")
                                                        .get_child("coarsen")
                                                        .get_child("min_nodes_after_coarsen_per_partition")
                                                        .get_value<int>();
        int number_of_final_no_change_reps = algorithm.get_child("parameters")
                                                 .get_child("coarsen")
                                                 .get_child("number_of_final_no_change_reps")
                                                 .get_value<int>();

        float initial_balance_threshhold =
            algorithm.get_child("parameters").get_child("initial").get_child("balance_threshhold").get_value<float>();

        float initial_nodes_per_clump =
            algorithm.get_child("parameters").get_child("initial").get_child("nodes_per_clump").get_value<float>();
        float initial_nodes_per_partition =
            algorithm.get_child("parameters").get_child("initial").get_child("nodes_per_partition").get_value<float>();
        float initial_clumps_per_partition =
            algorithm.get_child("parameters").get_child("initial").get_child("clumps_per_partition").get_value<float>();
        float initial_max_weight_for_flag =
            algorithm.get_child("parameters").get_child("initial").get_child("max_weight_for_flag").get_value<float>();
        float initial_balanced_cut_ratio =
            algorithm.get_child("parameters").get_child("initial").get_child("balanced_cut_ratio").get_value<float>();
        float initial_min_weight_for_split =
            algorithm.get_child("parameters").get_child("initial").get_child("min_weight_for_split").get_value<float>();
        unsigned initial_hill_climb_simple_improvement_attemps = algorithm.get_child("parameters")
                                                                     .get_child("initial")
                                                                     .get_child("hill_climb_simple_improvement_attemps")
                                                                     .get_value<unsigned>();
        int initial_min_comp_generation_when_shaving = algorithm.get_child("parameters")
                                                           .get_child("initial")
                                                           .get_child("min_comp_generation_when_shaving")
                                                           .get_value<int>();

        PartitionAlgorithm initial_part_algo;
        if (algorithm.get_child("parameters").get_child("initial").get_child("part_algo").get_value<std::string>() ==
            "KarmarkarKarp") {
            initial_part_algo = KarmarkarKarp;
        } else if (algorithm.get_child("parameters")
                       .get_child("initial")
                       .get_child("part_algo")
                       .get_value<std::string>() == "ILP") {
            initial_part_algo = ILP;
        } else if (algorithm.get_child("parameters")
                       .get_child("initial")
                       .get_child("part_algo")
                       .get_value<std::string>() == "Greedy") {
            initial_part_algo = Greedy;
        } else {
            initial_part_algo = Greedy;
        }

        CoinType initial_coin_type;
        if (algorithm.get_child("parameters").get_child("initial").get_child("coin_type").get_value<std::string>() ==
            "Thue_Morse") {
            initial_coin_type = Thue_Morse;
        } else if (algorithm.get_child("parameters")
                       .get_child("initial")
                       .get_child("coin_type")
                       .get_value<std::string>() == "Biased_Randomly") {
            initial_coin_type = Biased_Randomly;
        } else {
            initial_coin_type = Thue_Morse;
        }

        Coarse_Scheduler_Params initial_params(
            number_of_partitions, initial_balance_threshhold, initial_part_algo, initial_coin_type,
            initial_clumps_per_partition, initial_nodes_per_clump, initial_nodes_per_partition,
            initial_max_weight_for_flag, initial_balanced_cut_ratio, initial_min_weight_for_split,
            initial_hill_climb_simple_improvement_attemps, initial_min_comp_generation_when_shaving);

        float final_balance_threshhold =
            algorithm.get_child("parameters").get_child("final").get_child("balance_threshhold").get_value<float>();

        float final_nodes_per_clump =
            algorithm.get_child("parameters").get_child("final").get_child("nodes_per_clump").get_value<float>();
        float final_nodes_per_partition =
            algorithm.get_child("parameters").get_child("final").get_child("nodes_per_partition").get_value<float>();
        float final_clumps_per_partition =
            algorithm.get_child("parameters").get_child("final").get_child("clumps_per_partition").get_value<float>();
        float final_max_weight_for_flag =
            algorithm.get_child("parameters").get_child("final").get_child("max_weight_for_flag").get_value<float>();
        float final_balanced_cut_ratio =
            algorithm.get_child("parameters").get_child("final").get_child("balanced_cut_ratio").get_value<float>();
        float final_min_weight_for_split =
            algorithm.get_child("parameters").get_child("final").get_child("min_weight_for_split").get_value<float>();
        unsigned final_hill_climb_simple_improvement_attemps = algorithm.get_child("parameters")
                                                                   .get_child("final")
                                                                   .get_child("hill_climb_simple_improvement_attemps")
                                                                   .get_value<unsigned>();
        int final_min_comp_generation_when_shaving = algorithm.get_child("parameters")
                                                         .get_child("final")
                                                         .get_child("min_comp_generation_when_shaving")
                                                         .get_value<int>();

        PartitionAlgorithm final_part_algo;
        if (algorithm.get_child("parameters").get_child("final").get_child("part_algo").get_value<std::string>() ==
            "KarmarkarKarp") {
            final_part_algo = KarmarkarKarp;
        } else if (algorithm.get_child("parameters")
                       .get_child("final")
                       .get_child("part_algo")
                       .get_value<std::string>() == "ILP") {
            final_part_algo = ILP;
        } else if (algorithm.get_child("parameters")
                       .get_child("final")
                       .get_child("part_algo")
                       .get_value<std::string>() == "Greedy") {
            final_part_algo = Greedy;
        } else {
            final_part_algo = Greedy;
        }

        CoinType final_coin_type;
        if (algorithm.get_child("parameters").get_child("final").get_child("coin_type").get_value<std::string>() ==
            "Thue_Morse") {
            final_coin_type = Thue_Morse;
        } else if (algorithm.get_child("parameters")
                       .get_child("final")
                       .get_child("coin_type")
                       .get_value<std::string>() == "Biased_Randomly") {
            final_coin_type = Biased_Randomly;
        } else {
            final_coin_type = Thue_Morse;
        }

        Coarse_Scheduler_Params final_params(
            number_of_partitions, final_balance_threshhold, final_part_algo, final_coin_type,
            final_clumps_per_partition, final_nodes_per_clump, final_nodes_per_partition, final_max_weight_for_flag,
            final_balanced_cut_ratio, final_min_weight_for_split, final_hill_climb_simple_improvement_attemps,
            final_min_comp_generation_when_shaving);

        CoarseRefineScheduler_parameters params(initial_params, final_params, coarse_params,
                                                min_nodes_after_coarsen_per_partition, number_of_final_no_change_reps);

        CoBalDMixR sched(params);
        HillClimbingScheduler improver;
        HDaggCoarser scheduler(&sched, &improver);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "CoarseHDaggCoBalDMixRLK") {

        unsigned number_of_partitions = bsp_instance.numberOfProcessors();

        float geom_decay_num_nodes =
            algorithm.get_child("parameters").get_child("coarsen").get_child("geom_decay_num_nodes").get_value<float>();
        double poisson_par =
            algorithm.get_child("parameters").get_child("coarsen").get_child("poisson_par").get_value<double>();
        unsigned noise =
            algorithm.get_child("parameters").get_child("coarsen").get_child("noise").get_value<unsigned>();
        std::pair<unsigned, unsigned> edge_sort_ratio = std::make_pair(algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_triangle")
                                                                           .get_value<unsigned>(),
                                                                       algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_weight")
                                                                           .get_value<unsigned>());
        int num_rep_without_node_decrease = algorithm.get_child("parameters")
                                                .get_child("coarsen")
                                                .get_child("num_rep_without_node_decrease")
                                                .get_value<int>();
        float temperature_multiplier = algorithm.get_child("parameters")
                                           .get_child("coarsen")
                                           .get_child("temperature_multiplier")
                                           .get_value<float>();
        float number_of_temperature_increases = algorithm.get_child("parameters")
                                                    .get_child("coarsen")
                                                    .get_child("number_of_temperature_increases")
                                                    .get_value<float>();

        CoarsenParams coarse_params(geom_decay_num_nodes, poisson_par, noise, edge_sort_ratio,
                                    num_rep_without_node_decrease, temperature_multiplier,
                                    number_of_temperature_increases);

        int min_nodes_after_coarsen_per_partition = algorithm.get_child("parameters")
                                                        .get_child("coarsen")
                                                        .get_child("min_nodes_after_coarsen_per_partition")
                                                        .get_value<int>();
        int number_of_final_no_change_reps = algorithm.get_child("parameters")
                                                 .get_child("coarsen")
                                                 .get_child("number_of_final_no_change_reps")
                                                 .get_value<int>();

        float initial_balance_threshhold =
            algorithm.get_child("parameters").get_child("initial").get_child("balance_threshhold").get_value<float>();

        float initial_nodes_per_clump =
            algorithm.get_child("parameters").get_child("initial").get_child("nodes_per_clump").get_value<float>();
        float initial_nodes_per_partition =
            algorithm.get_child("parameters").get_child("initial").get_child("nodes_per_partition").get_value<float>();
        float initial_clumps_per_partition =
            algorithm.get_child("parameters").get_child("initial").get_child("clumps_per_partition").get_value<float>();
        float initial_max_weight_for_flag =
            algorithm.get_child("parameters").get_child("initial").get_child("max_weight_for_flag").get_value<float>();
        float initial_balanced_cut_ratio =
            algorithm.get_child("parameters").get_child("initial").get_child("balanced_cut_ratio").get_value<float>();
        float initial_min_weight_for_split =
            algorithm.get_child("parameters").get_child("initial").get_child("min_weight_for_split").get_value<float>();
        unsigned initial_hill_climb_simple_improvement_attemps = algorithm.get_child("parameters")
                                                                     .get_child("initial")
                                                                     .get_child("hill_climb_simple_improvement_attemps")
                                                                     .get_value<unsigned>();
        int initial_min_comp_generation_when_shaving = algorithm.get_child("parameters")
                                                           .get_child("initial")
                                                           .get_child("min_comp_generation_when_shaving")
                                                           .get_value<int>();

        PartitionAlgorithm initial_part_algo;
        if (algorithm.get_child("parameters").get_child("initial").get_child("part_algo").get_value<std::string>() ==
            "KarmarkarKarp") {
            initial_part_algo = KarmarkarKarp;
        } else if (algorithm.get_child("parameters")
                       .get_child("initial")
                       .get_child("part_algo")
                       .get_value<std::string>() == "ILP") {
            initial_part_algo = ILP;
        } else if (algorithm.get_child("parameters")
                       .get_child("initial")
                       .get_child("part_algo")
                       .get_value<std::string>() == "Greedy") {
            initial_part_algo = Greedy;
        } else {
            initial_part_algo = Greedy;
        }

        CoinType initial_coin_type;
        if (algorithm.get_child("parameters").get_child("initial").get_child("coin_type").get_value<std::string>() ==
            "Thue_Morse") {
            initial_coin_type = Thue_Morse;
        } else if (algorithm.get_child("parameters")
                       .get_child("initial")
                       .get_child("coin_type")
                       .get_value<std::string>() == "Biased_Randomly") {
            initial_coin_type = Biased_Randomly;
        } else {
            initial_coin_type = Thue_Morse;
        }

        Coarse_Scheduler_Params initial_params(
            number_of_partitions, initial_balance_threshhold, initial_part_algo, initial_coin_type,
            initial_clumps_per_partition, initial_nodes_per_clump, initial_nodes_per_partition,
            initial_max_weight_for_flag, initial_balanced_cut_ratio, initial_min_weight_for_split,
            initial_hill_climb_simple_improvement_attemps, initial_min_comp_generation_when_shaving);

        float final_balance_threshhold =
            algorithm.get_child("parameters").get_child("final").get_child("balance_threshhold").get_value<float>();

        float final_nodes_per_clump =
            algorithm.get_child("parameters").get_child("final").get_child("nodes_per_clump").get_value<float>();
        float final_nodes_per_partition =
            algorithm.get_child("parameters").get_child("final").get_child("nodes_per_partition").get_value<float>();
        float final_clumps_per_partition =
            algorithm.get_child("parameters").get_child("final").get_child("clumps_per_partition").get_value<float>();
        float final_max_weight_for_flag =
            algorithm.get_child("parameters").get_child("final").get_child("max_weight_for_flag").get_value<float>();
        float final_balanced_cut_ratio =
            algorithm.get_child("parameters").get_child("final").get_child("balanced_cut_ratio").get_value<float>();
        float final_min_weight_for_split =
            algorithm.get_child("parameters").get_child("final").get_child("min_weight_for_split").get_value<float>();
        unsigned final_hill_climb_simple_improvement_attemps = algorithm.get_child("parameters")
                                                                   .get_child("final")
                                                                   .get_child("hill_climb_simple_improvement_attemps")
                                                                   .get_value<unsigned>();
        int final_min_comp_generation_when_shaving = algorithm.get_child("parameters")
                                                         .get_child("final")
                                                         .get_child("min_comp_generation_when_shaving")
                                                         .get_value<int>();

        PartitionAlgorithm final_part_algo;
        if (algorithm.get_child("parameters").get_child("final").get_child("part_algo").get_value<std::string>() ==
            "KarmarkarKarp") {
            final_part_algo = KarmarkarKarp;
        } else if (algorithm.get_child("parameters")
                       .get_child("final")
                       .get_child("part_algo")
                       .get_value<std::string>() == "ILP") {
            final_part_algo = ILP;
        } else if (algorithm.get_child("parameters")
                       .get_child("final")
                       .get_child("part_algo")
                       .get_value<std::string>() == "Greedy") {
            final_part_algo = Greedy;
        } else {
            final_part_algo = Greedy;
        }

        CoinType final_coin_type;
        if (algorithm.get_child("parameters").get_child("final").get_child("coin_type").get_value<std::string>() ==
            "Thue_Morse") {
            final_coin_type = Thue_Morse;
        } else if (algorithm.get_child("parameters")
                       .get_child("final")
                       .get_child("coin_type")
                       .get_value<std::string>() == "Biased_Randomly") {
            final_coin_type = Biased_Randomly;
        } else {
            final_coin_type = Thue_Morse;
        }

        Coarse_Scheduler_Params final_params(
            number_of_partitions, final_balance_threshhold, final_part_algo, final_coin_type,
            final_clumps_per_partition, final_nodes_per_clump, final_nodes_per_partition, final_max_weight_for_flag,
            final_balanced_cut_ratio, final_min_weight_for_split, final_hill_climb_simple_improvement_attemps,
            final_min_comp_generation_when_shaving);

        CoarseRefineScheduler_parameters params(initial_params, final_params, coarse_params,
                                                min_nodes_after_coarsen_per_partition, number_of_final_no_change_reps);

        CoBalDMixR sched(params);
        LKTotalCommScheduler improver;
        improver.set_quick_pass(true);
        HDaggCoarser scheduler(&sched, &improver);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "FunnelBspGreedy") {

        float max_relative_weight =
            algorithm.get_child("parameters").get_child("coarsen").get_child("max_relative_weight").get_value<float>();
        bool funnel_incoming =
            algorithm.get_child("parameters").get_child("coarsen").get_child("funnel_incoming").get_value<bool>();
        bool funnel_outgoing =
            algorithm.get_child("parameters").get_child("coarsen").get_child("funnel_outgoing").get_value<bool>();
        bool first_funnel_incoming =
            algorithm.get_child("parameters").get_child("coarsen").get_child("first_funnel_incoming").get_value<bool>();
        bool use_approx_transitive_reduction = algorithm.get_child("parameters")
                                                   .get_child("coarsen")
                                                   .get_child("use_approx_transitive_reduction")
                                                   .get_value<bool>();

        Funnel_parameters params(max_relative_weight, funnel_incoming, funnel_outgoing, first_funnel_incoming,
                                 use_approx_transitive_reduction);

        float max_percent_idle_processors = algorithm.get_child("parameters")
                                                .get_child("bsp")
                                                .get_child("max_percent_idle_processors")
                                                .get_value<float>();

        GreedyBspScheduler scheduler_inner(max_percent_idle_processors);
        Funnel scheduler(&scheduler_inner, params);

        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "FunnelBspLocking") {

        float max_relative_weight =
            algorithm.get_child("parameters").get_child("coarsen").get_child("max_relative_weight").get_value<float>();
        bool funnel_incoming =
            algorithm.get_child("parameters").get_child("coarsen").get_child("funnel_incoming").get_value<bool>();
        bool funnel_outgoing =
            algorithm.get_child("parameters").get_child("coarsen").get_child("funnel_outgoing").get_value<bool>();
        bool first_funnel_incoming =
            algorithm.get_child("parameters").get_child("coarsen").get_child("first_funnel_incoming").get_value<bool>();
        bool use_approx_transitive_reduction = algorithm.get_child("parameters")
                                                   .get_child("coarsen")
                                                   .get_child("use_approx_transitive_reduction")
                                                   .get_value<bool>();

        Funnel_parameters params(max_relative_weight, funnel_incoming, funnel_outgoing, first_funnel_incoming,
                                 use_approx_transitive_reduction);

        float max_percent_idle_processors = algorithm.get_child("parameters")
                                                .get_child("bsp")
                                                .get_child("max_percent_idle_processors")
                                                .get_value<float>();

        GreedyBspLocking scheduler_inner(max_percent_idle_processors);
        Funnel scheduler(&scheduler_inner, params);

        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    
    
    } else if (algorithm.get_child("name").get_value<std::string>() == "FunnelVarianceGreedy") {

        float max_relative_weight =
            algorithm.get_child("parameters").get_child("coarsen").get_child("max_relative_weight").get_value<float>();
        bool funnel_incoming =
            algorithm.get_child("parameters").get_child("coarsen").get_child("funnel_incoming").get_value<bool>();
        bool funnel_outgoing =
            algorithm.get_child("parameters").get_child("coarsen").get_child("funnel_outgoing").get_value<bool>();
        bool first_funnel_incoming =
            algorithm.get_child("parameters").get_child("coarsen").get_child("first_funnel_incoming").get_value<bool>();
        bool use_approx_transitive_reduction = algorithm.get_child("parameters")
                                                   .get_child("coarsen")
                                                   .get_child("use_approx_transitive_reduction")
                                                   .get_value<bool>();

        Funnel_parameters params(max_relative_weight, funnel_incoming, funnel_outgoing, first_funnel_incoming,
                                 use_approx_transitive_reduction);

        float max_percent_idle_processors = algorithm.get_child("parameters")
                                                .get_child("variance")
                                                .get_child("max_percent_idle_processors")
                                                .get_value<float>();

        GreedyVarianceScheduler scheduler_inner(max_percent_idle_processors);
        Funnel scheduler(&scheduler_inner, params);

        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "FunnelBsp") {

        float max_relative_weight =
            algorithm.get_child("parameters").get_child("coarsen").get_child("max_relative_weight").get_value<float>();
        bool funnel_incoming =
            algorithm.get_child("parameters").get_child("coarsen").get_child("funnel_incoming").get_value<bool>();
        bool funnel_outgoing =
            algorithm.get_child("parameters").get_child("coarsen").get_child("funnel_outgoing").get_value<bool>();
        bool first_funnel_incoming =
            algorithm.get_child("parameters").get_child("coarsen").get_child("first_funnel_incoming").get_value<bool>();
        bool use_approx_transitive_reduction = algorithm.get_child("parameters")
                                                   .get_child("coarsen")
                                                   .get_child("use_approx_transitive_reduction")
                                                   .get_value<bool>();

        Funnel_parameters params(max_relative_weight, funnel_incoming, funnel_outgoing, first_funnel_incoming,
                                 use_approx_transitive_reduction);

        float max_percent_idle_processors = algorithm.get_child("parameters")
                                                .get_child("bsp")
                                                .get_child("max_percent_idle_processors")
                                                .get_value<float>();

        GreedyBspFillupScheduler scheduler_inner(max_percent_idle_processors);
        Funnel scheduler(&scheduler_inner, params);

        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "FunnelVariance") {

        float max_relative_weight =
            algorithm.get_child("parameters").get_child("coarsen").get_child("max_relative_weight").get_value<float>();
        bool funnel_incoming =
            algorithm.get_child("parameters").get_child("coarsen").get_child("funnel_incoming").get_value<bool>();
        bool funnel_outgoing =
            algorithm.get_child("parameters").get_child("coarsen").get_child("funnel_outgoing").get_value<bool>();
        bool first_funnel_incoming =
            algorithm.get_child("parameters").get_child("coarsen").get_child("first_funnel_incoming").get_value<bool>();
        bool use_approx_transitive_reduction = algorithm.get_child("parameters")
                                                   .get_child("coarsen")
                                                   .get_child("use_approx_transitive_reduction")
                                                   .get_value<bool>();

        Funnel_parameters params(max_relative_weight, funnel_incoming, funnel_outgoing, first_funnel_incoming,
                                 use_approx_transitive_reduction);

        float max_percent_idle_processors = algorithm.get_child("parameters")
                                                .get_child("variance")
                                                .get_child("max_percent_idle_processors")
                                                .get_value<float>();

        GreedyVarianceFillupScheduler scheduler_inner(max_percent_idle_processors);
        Funnel scheduler(&scheduler_inner, params);

        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);

   } else if (algorithm.get_child("name").get_value<std::string>() == "FunnelBestGreedy") {

        float max_relative_weight =
            algorithm.get_child("parameters").get_child("coarsen").get_child("max_relative_weight").get_value<float>();
        bool funnel_incoming =
            algorithm.get_child("parameters").get_child("coarsen").get_child("funnel_incoming").get_value<bool>();
        bool funnel_outgoing =
            algorithm.get_child("parameters").get_child("coarsen").get_child("funnel_outgoing").get_value<bool>();
        bool first_funnel_incoming =
            algorithm.get_child("parameters").get_child("coarsen").get_child("first_funnel_incoming").get_value<bool>();
        bool use_approx_transitive_reduction = algorithm.get_child("parameters")
                                                   .get_child("coarsen")
                                                   .get_child("use_approx_transitive_reduction")
                                                   .get_value<bool>();

        Funnel_parameters params(max_relative_weight, funnel_incoming, funnel_outgoing, first_funnel_incoming,
                                 use_approx_transitive_reduction);

     

        MetaGreedyScheduler scheduler_inner;
        Funnel scheduler(&scheduler_inner, params);

        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);

    } else if (algorithm.get_child("name").get_value<std::string>() == "HDaggCoarseBspGLK+HC") {

        bool apply_trans_edge_contraction =
            algorithm.get_child("parameters").get_child("trans_edge_contraction").get_value<bool>();

        float max_percent_idle_processor =
            algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();

        GreedyBspFillupScheduler bsp_greedy(max_percent_idle_processor);
        LKTotalCommScheduler lk;
        ComboScheduler bsp_greedy_lk(&bsp_greedy, &lk);
        HillClimbingScheduler hc;
        HDaggCoarser scheduler(&bsp_greedy_lk, &hc);

        if (apply_trans_edge_contraction) {

            AppTransEdgeReductor red(scheduler);
            return red.computeSchedule(bsp_instance);

        } else {

            return scheduler.computeSchedule(bsp_instance);
        }

    } else if (algorithm.get_child("name").get_value<std::string>() == "HDaggCoarseVarGLK+HC") {

        bool apply_trans_edge_contraction =
            algorithm.get_child("parameters").get_child("trans_edge_contraction").get_value<bool>();

        float max_percent_idle_processors = algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();

        GreedyVarianceFillupScheduler greedy(max_percent_idle_processors);
        LKTotalCommScheduler lk;
        ComboScheduler greedy_lk(&greedy, &lk);
        HillClimbingScheduler hc;
        HDaggCoarser scheduler(&greedy_lk, &hc);

        if (apply_trans_edge_contraction) {

            AppTransEdgeReductor red(scheduler);
            return red.computeSchedule(bsp_instance);

        } else {

            return scheduler.computeSchedule(bsp_instance);
        }

    } else if (algorithm.get_child("name").get_value<std::string>() == "HDaggCoarseLockGLK+HC") {

        bool apply_trans_edge_contraction =
            algorithm.get_child("parameters").get_child("trans_edge_contraction").get_value<bool>();

        float max_percent_idle_processors = algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();

        
        GreedyBspLocking greedy(max_percent_idle_processors);
        LKTotalCommScheduler lk;
        ComboScheduler greedy_lk(&greedy, &lk);
        HillClimbingScheduler hc;
        HDaggCoarser scheduler(&greedy_lk, &hc);

        if (apply_trans_edge_contraction) {

            AppTransEdgeReductor red(scheduler);
            return red.computeSchedule(bsp_instance);

        } else {

            return scheduler.computeSchedule(bsp_instance);
        }

    } else if (algorithm.get_child("name").get_value<std::string>() == "HDaggCoarseCobaldLK+HC") {

        unsigned number_of_partitions = bsp_instance.numberOfProcessors();

        bool apply_trans_edge_contraction =
            algorithm.get_child("parameters").get_child("trans_edge_contraction").get_value<bool>();

        float geom_decay_num_nodes =
            algorithm.get_child("parameters").get_child("coarsen").get_child("geom_decay_num_nodes").get_value<float>();
        double poisson_par =
            algorithm.get_child("parameters").get_child("coarsen").get_child("poisson_par").get_value<double>();
        unsigned noise =
            algorithm.get_child("parameters").get_child("coarsen").get_child("noise").get_value<unsigned>();
        std::pair<unsigned, unsigned> edge_sort_ratio = std::make_pair(algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_triangle")
                                                                           .get_value<unsigned>(),
                                                                       algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_weight")
                                                                           .get_value<unsigned>());
        int num_rep_without_node_decrease = algorithm.get_child("parameters")
                                                .get_child("coarsen")
                                                .get_child("num_rep_without_node_decrease")
                                                .get_value<int>();
        float temperature_multiplier = algorithm.get_child("parameters")
                                           .get_child("coarsen")
                                           .get_child("temperature_multiplier")
                                           .get_value<float>();
        float number_of_temperature_increases = algorithm.get_child("parameters")
                                                    .get_child("coarsen")
                                                    .get_child("number_of_temperature_increases")
                                                    .get_value<float>();

        CoarsenParams coarse_params(geom_decay_num_nodes, poisson_par, noise, edge_sort_ratio,
                                    num_rep_without_node_decrease, temperature_multiplier,
                                    number_of_temperature_increases);

        int min_nodes_after_coarsen_per_partition = algorithm.get_child("parameters")
                                                        .get_child("coarsen")
                                                        .get_child("min_nodes_after_coarsen_per_partition")
                                                        .get_value<int>();
        int number_of_final_no_change_reps = algorithm.get_child("parameters")
                                                 .get_child("coarsen")
                                                 .get_child("number_of_final_no_change_reps")
                                                 .get_value<int>();

        float initial_balance_threshhold =
            algorithm.get_child("parameters").get_child("initial").get_child("balance_threshhold").get_value<float>();

        float initial_nodes_per_clump =
            algorithm.get_child("parameters").get_child("initial").get_child("nodes_per_clump").get_value<float>();
        float initial_nodes_per_partition =
            algorithm.get_child("parameters").get_child("initial").get_child("nodes_per_partition").get_value<float>();
        float initial_clumps_per_partition =
            algorithm.get_child("parameters").get_child("initial").get_child("clumps_per_partition").get_value<float>();
        float initial_max_weight_for_flag =
            algorithm.get_child("parameters").get_child("initial").get_child("max_weight_for_flag").get_value<float>();
        float initial_balanced_cut_ratio =
            algorithm.get_child("parameters").get_child("initial").get_child("balanced_cut_ratio").get_value<float>();
        float initial_min_weight_for_split =
            algorithm.get_child("parameters").get_child("initial").get_child("min_weight_for_split").get_value<float>();
        unsigned initial_hill_climb_simple_improvement_attemps = algorithm.get_child("parameters")
                                                                     .get_child("initial")
                                                                     .get_child("hill_climb_simple_improvement_attemps")
                                                                     .get_value<unsigned>();
        int initial_min_comp_generation_when_shaving = algorithm.get_child("parameters")
                                                           .get_child("initial")
                                                           .get_child("min_comp_generation_when_shaving")
                                                           .get_value<int>();

        PartitionAlgorithm initial_part_algo;
        if (algorithm.get_child("parameters").get_child("initial").get_child("part_algo").get_value<std::string>() ==
            "KarmarkarKarp") {
            initial_part_algo = KarmarkarKarp;
        } else if (algorithm.get_child("parameters")
                       .get_child("initial")
                       .get_child("part_algo")
                       .get_value<std::string>() == "ILP") {
            initial_part_algo = ILP;
        } else if (algorithm.get_child("parameters")
                       .get_child("initial")
                       .get_child("part_algo")
                       .get_value<std::string>() == "Greedy") {
            initial_part_algo = Greedy;
        } else {
            initial_part_algo = Greedy;
        }

        CoinType initial_coin_type;
        if (algorithm.get_child("parameters").get_child("initial").get_child("coin_type").get_value<std::string>() ==
            "Thue_Morse") {
            initial_coin_type = Thue_Morse;
        } else if (algorithm.get_child("parameters")
                       .get_child("initial")
                       .get_child("coin_type")
                       .get_value<std::string>() == "Biased_Randomly") {
            initial_coin_type = Biased_Randomly;
        } else {
            initial_coin_type = Thue_Morse;
        }

        Coarse_Scheduler_Params initial_params(
            number_of_partitions, initial_balance_threshhold, initial_part_algo, initial_coin_type,
            initial_clumps_per_partition, initial_nodes_per_clump, initial_nodes_per_partition,
            initial_max_weight_for_flag, initial_balanced_cut_ratio, initial_min_weight_for_split,
            initial_hill_climb_simple_improvement_attemps, initial_min_comp_generation_when_shaving);

        float final_balance_threshhold =
            algorithm.get_child("parameters").get_child("final").get_child("balance_threshhold").get_value<float>();

        float final_nodes_per_clump =
            algorithm.get_child("parameters").get_child("final").get_child("nodes_per_clump").get_value<float>();
        float final_nodes_per_partition =
            algorithm.get_child("parameters").get_child("final").get_child("nodes_per_partition").get_value<float>();
        float final_clumps_per_partition =
            algorithm.get_child("parameters").get_child("final").get_child("clumps_per_partition").get_value<float>();
        float final_max_weight_for_flag =
            algorithm.get_child("parameters").get_child("final").get_child("max_weight_for_flag").get_value<float>();
        float final_balanced_cut_ratio =
            algorithm.get_child("parameters").get_child("final").get_child("balanced_cut_ratio").get_value<float>();
        float final_min_weight_for_split =
            algorithm.get_child("parameters").get_child("final").get_child("min_weight_for_split").get_value<float>();
        unsigned final_hill_climb_simple_improvement_attemps = algorithm.get_child("parameters")
                                                                   .get_child("final")
                                                                   .get_child("hill_climb_simple_improvement_attemps")
                                                                   .get_value<unsigned>();
        int final_min_comp_generation_when_shaving = algorithm.get_child("parameters")
                                                         .get_child("final")
                                                         .get_child("min_comp_generation_when_shaving")
                                                         .get_value<int>();

        PartitionAlgorithm final_part_algo;
        if (algorithm.get_child("parameters").get_child("final").get_child("part_algo").get_value<std::string>() ==
            "KarmarkarKarp") {
            final_part_algo = KarmarkarKarp;
        } else if (algorithm.get_child("parameters")
                       .get_child("final")
                       .get_child("part_algo")
                       .get_value<std::string>() == "ILP") {
            final_part_algo = ILP;
        } else if (algorithm.get_child("parameters")
                       .get_child("final")
                       .get_child("part_algo")
                       .get_value<std::string>() == "Greedy") {
            final_part_algo = Greedy;
        } else {
            final_part_algo = Greedy;
        }

        CoinType final_coin_type;
        if (algorithm.get_child("parameters").get_child("final").get_child("coin_type").get_value<std::string>() ==
            "Thue_Morse") {
            final_coin_type = Thue_Morse;
        } else if (algorithm.get_child("parameters")
                       .get_child("final")
                       .get_child("coin_type")
                       .get_value<std::string>() == "Biased_Randomly") {
            final_coin_type = Biased_Randomly;
        } else {
            final_coin_type = Thue_Morse;
        }

        Coarse_Scheduler_Params final_params(
            number_of_partitions, final_balance_threshhold, final_part_algo, final_coin_type,
            final_clumps_per_partition, final_nodes_per_clump, final_nodes_per_partition, final_max_weight_for_flag,
            final_balanced_cut_ratio, final_min_weight_for_split, final_hill_climb_simple_improvement_attemps,
            final_min_comp_generation_when_shaving);

        CoarseRefineScheduler_parameters params(initial_params, final_params, coarse_params,
                                                min_nodes_after_coarsen_per_partition, number_of_final_no_change_reps);

        CoBalDMixR cobald(params);

        LKTotalCommScheduler lk;
        ComboScheduler cobald_lk(&cobald, &lk);
        HillClimbingScheduler hc;
        HDaggCoarser scheduler(&cobald_lk, &hc);

        if (apply_trans_edge_contraction) {

            AppTransEdgeReductor red(scheduler);
            return red.computeSchedule(bsp_instance);

        } else {

            return scheduler.computeSchedule(bsp_instance);
        }

    } else if (algorithm.get_child("name").get_value<std::string>() == "FunnelCoarseBspGLK+HC") {

        bool apply_trans_edge_contraction =
            algorithm.get_child("parameters").get_child("trans_edge_contraction").get_value<bool>();



        float max_relative_weight =
            algorithm.get_child("parameters").get_child("coarsen").get_child("max_relative_weight").get_value<float>();
        bool funnel_incoming =
            algorithm.get_child("parameters").get_child("coarsen").get_child("funnel_incoming").get_value<bool>();
        bool funnel_outgoing =
            algorithm.get_child("parameters").get_child("coarsen").get_child("funnel_outgoing").get_value<bool>();
        bool first_funnel_incoming =
            algorithm.get_child("parameters").get_child("coarsen").get_child("first_funnel_incoming").get_value<bool>();
        bool use_approx_transitive_reduction = algorithm.get_child("parameters")
                                                   .get_child("coarsen")
                                                   .get_child("use_approx_transitive_reduction")
                                                   .get_value<bool>();

        Funnel_parameters params(max_relative_weight, funnel_incoming, funnel_outgoing, first_funnel_incoming,
                                 use_approx_transitive_reduction);


        float max_percent_idle_processor =
            algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();

        GreedyBspFillupScheduler bsp_greedy(max_percent_idle_processor);
        LKTotalCommScheduler lk;
        ComboScheduler bsp_greedy_lk(&bsp_greedy, &lk);
        HillClimbingScheduler hc;
        Funnel scheduler(&bsp_greedy_lk, &hc, params);

        if (apply_trans_edge_contraction) {

            AppTransEdgeReductor red(scheduler);
            return red.computeSchedule(bsp_instance);

        } else {

            return scheduler.computeSchedule(bsp_instance);
        }
    
    
    } else if (algorithm.get_child("name").get_value<std::string>() == "FunnelCoarseVarGLK+HC") {

        bool apply_trans_edge_contraction =
            algorithm.get_child("parameters").get_child("trans_edge_contraction").get_value<bool>();

        float max_percent_idle_processors = algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();

        float max_relative_weight =
            algorithm.get_child("parameters").get_child("coarsen").get_child("max_relative_weight").get_value<float>();
        bool funnel_incoming =
            algorithm.get_child("parameters").get_child("coarsen").get_child("funnel_incoming").get_value<bool>();
        bool funnel_outgoing =
            algorithm.get_child("parameters").get_child("coarsen").get_child("funnel_outgoing").get_value<bool>();
        bool first_funnel_incoming =
            algorithm.get_child("parameters").get_child("coarsen").get_child("first_funnel_incoming").get_value<bool>();
        bool use_approx_transitive_reduction = algorithm.get_child("parameters")
                                                   .get_child("coarsen")
                                                   .get_child("use_approx_transitive_reduction")
                                                   .get_value<bool>();

        Funnel_parameters params(max_relative_weight, funnel_incoming, funnel_outgoing, first_funnel_incoming,
                                 use_approx_transitive_reduction);

        GreedyVarianceFillupScheduler greedy(max_percent_idle_processors);
        LKTotalCommScheduler lk;
        ComboScheduler greedy_lk(&greedy, &lk);
        HillClimbingScheduler hc;
        Funnel scheduler(&greedy_lk, &hc);
  
        if (apply_trans_edge_contraction) {

            AppTransEdgeReductor red(scheduler);
            return red.computeSchedule(bsp_instance);

        } else {

            return scheduler.computeSchedule(bsp_instance);
        }

    } else if (algorithm.get_child("name").get_value<std::string>() == "FunnelCoarseCobaldLK+HC") {

        unsigned number_of_partitions = bsp_instance.numberOfProcessors();

        bool apply_trans_edge_contraction =
            algorithm.get_child("parameters").get_child("trans_edge_contraction").get_value<bool>();

        float geom_decay_num_nodes =
            algorithm.get_child("parameters").get_child("coarsen").get_child("geom_decay_num_nodes").get_value<float>();
        double poisson_par =
            algorithm.get_child("parameters").get_child("coarsen").get_child("poisson_par").get_value<double>();
        unsigned noise =
            algorithm.get_child("parameters").get_child("coarsen").get_child("noise").get_value<unsigned>();
        std::pair<unsigned, unsigned> edge_sort_ratio = std::make_pair(algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_triangle")
                                                                           .get_value<unsigned>(),
                                                                       algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_weight")
                                                                           .get_value<unsigned>());
        int num_rep_without_node_decrease = algorithm.get_child("parameters")
                                                .get_child("coarsen")
                                                .get_child("num_rep_without_node_decrease")
                                                .get_value<int>();
        float temperature_multiplier = algorithm.get_child("parameters")
                                           .get_child("coarsen")
                                           .get_child("temperature_multiplier")
                                           .get_value<float>();
        float number_of_temperature_increases = algorithm.get_child("parameters")
                                                    .get_child("coarsen")
                                                    .get_child("number_of_temperature_increases")
                                                    .get_value<float>();

        CoarsenParams coarse_params(geom_decay_num_nodes, poisson_par, noise, edge_sort_ratio,
                                    num_rep_without_node_decrease, temperature_multiplier,
                                    number_of_temperature_increases);

        int min_nodes_after_coarsen_per_partition = algorithm.get_child("parameters")
                                                        .get_child("coarsen")
                                                        .get_child("min_nodes_after_coarsen_per_partition")
                                                        .get_value<int>();
        int number_of_final_no_change_reps = algorithm.get_child("parameters")
                                                 .get_child("coarsen")
                                                 .get_child("number_of_final_no_change_reps")
                                                 .get_value<int>();

        float initial_balance_threshhold =
            algorithm.get_child("parameters").get_child("initial").get_child("balance_threshhold").get_value<float>();

        float initial_nodes_per_clump =
            algorithm.get_child("parameters").get_child("initial").get_child("nodes_per_clump").get_value<float>();
        float initial_nodes_per_partition =
            algorithm.get_child("parameters").get_child("initial").get_child("nodes_per_partition").get_value<float>();
        float initial_clumps_per_partition =
            algorithm.get_child("parameters").get_child("initial").get_child("clumps_per_partition").get_value<float>();
        float initial_max_weight_for_flag =
            algorithm.get_child("parameters").get_child("initial").get_child("max_weight_for_flag").get_value<float>();
        float initial_balanced_cut_ratio =
            algorithm.get_child("parameters").get_child("initial").get_child("balanced_cut_ratio").get_value<float>();
        float initial_min_weight_for_split =
            algorithm.get_child("parameters").get_child("initial").get_child("min_weight_for_split").get_value<float>();
        unsigned initial_hill_climb_simple_improvement_attemps = algorithm.get_child("parameters")
                                                                     .get_child("initial")
                                                                     .get_child("hill_climb_simple_improvement_attemps")
                                                                     .get_value<unsigned>();
        int initial_min_comp_generation_when_shaving = algorithm.get_child("parameters")
                                                           .get_child("initial")
                                                           .get_child("min_comp_generation_when_shaving")
                                                           .get_value<int>();

        PartitionAlgorithm initial_part_algo;
        if (algorithm.get_child("parameters").get_child("initial").get_child("part_algo").get_value<std::string>() ==
            "KarmarkarKarp") {
            initial_part_algo = KarmarkarKarp;
        } else if (algorithm.get_child("parameters")
                       .get_child("initial")
                       .get_child("part_algo")
                       .get_value<std::string>() == "ILP") {
            initial_part_algo = ILP;
        } else if (algorithm.get_child("parameters")
                       .get_child("initial")
                       .get_child("part_algo")
                       .get_value<std::string>() == "Greedy") {
            initial_part_algo = Greedy;
        } else {
            initial_part_algo = Greedy;
        }

        CoinType initial_coin_type;
        if (algorithm.get_child("parameters").get_child("initial").get_child("coin_type").get_value<std::string>() ==
            "Thue_Morse") {
            initial_coin_type = Thue_Morse;
        } else if (algorithm.get_child("parameters")
                       .get_child("initial")
                       .get_child("coin_type")
                       .get_value<std::string>() == "Biased_Randomly") {
            initial_coin_type = Biased_Randomly;
        } else {
            initial_coin_type = Thue_Morse;
        }

        Coarse_Scheduler_Params initial_params(
            number_of_partitions, initial_balance_threshhold, initial_part_algo, initial_coin_type,
            initial_clumps_per_partition, initial_nodes_per_clump, initial_nodes_per_partition,
            initial_max_weight_for_flag, initial_balanced_cut_ratio, initial_min_weight_for_split,
            initial_hill_climb_simple_improvement_attemps, initial_min_comp_generation_when_shaving);

        float final_balance_threshhold =
            algorithm.get_child("parameters").get_child("final").get_child("balance_threshhold").get_value<float>();

        float final_nodes_per_clump =
            algorithm.get_child("parameters").get_child("final").get_child("nodes_per_clump").get_value<float>();
        float final_nodes_per_partition =
            algorithm.get_child("parameters").get_child("final").get_child("nodes_per_partition").get_value<float>();
        float final_clumps_per_partition =
            algorithm.get_child("parameters").get_child("final").get_child("clumps_per_partition").get_value<float>();
        float final_max_weight_for_flag =
            algorithm.get_child("parameters").get_child("final").get_child("max_weight_for_flag").get_value<float>();
        float final_balanced_cut_ratio =
            algorithm.get_child("parameters").get_child("final").get_child("balanced_cut_ratio").get_value<float>();
        float final_min_weight_for_split =
            algorithm.get_child("parameters").get_child("final").get_child("min_weight_for_split").get_value<float>();
        unsigned final_hill_climb_simple_improvement_attemps = algorithm.get_child("parameters")
                                                                   .get_child("final")
                                                                   .get_child("hill_climb_simple_improvement_attemps")
                                                                   .get_value<unsigned>();
        int final_min_comp_generation_when_shaving = algorithm.get_child("parameters")
                                                         .get_child("final")
                                                         .get_child("min_comp_generation_when_shaving")
                                                         .get_value<int>();

        PartitionAlgorithm final_part_algo;
        if (algorithm.get_child("parameters").get_child("final").get_child("part_algo").get_value<std::string>() ==
            "KarmarkarKarp") {
            final_part_algo = KarmarkarKarp;
        } else if (algorithm.get_child("parameters")
                       .get_child("final")
                       .get_child("part_algo")
                       .get_value<std::string>() == "ILP") {
            final_part_algo = ILP;
        } else if (algorithm.get_child("parameters")
                       .get_child("final")
                       .get_child("part_algo")
                       .get_value<std::string>() == "Greedy") {
            final_part_algo = Greedy;
        } else {
            final_part_algo = Greedy;
        }

        CoinType final_coin_type;
        if (algorithm.get_child("parameters").get_child("final").get_child("coin_type").get_value<std::string>() ==
            "Thue_Morse") {
            final_coin_type = Thue_Morse;
        } else if (algorithm.get_child("parameters")
                       .get_child("final")
                       .get_child("coin_type")
                       .get_value<std::string>() == "Biased_Randomly") {
            final_coin_type = Biased_Randomly;
        } else {
            final_coin_type = Thue_Morse;
        }

        Coarse_Scheduler_Params final_params(
            number_of_partitions, final_balance_threshhold, final_part_algo, final_coin_type,
            final_clumps_per_partition, final_nodes_per_clump, final_nodes_per_partition, final_max_weight_for_flag,
            final_balanced_cut_ratio, final_min_weight_for_split, final_hill_climb_simple_improvement_attemps,
            final_min_comp_generation_when_shaving);

        CoarseRefineScheduler_parameters params(initial_params, final_params, coarse_params,
                                                min_nodes_after_coarsen_per_partition, number_of_final_no_change_reps);


        float max_relative_weight =
            algorithm.get_child("parameters").get_child("funnel").get_child("max_relative_weight").get_value<float>();
        bool funnel_incoming =
            algorithm.get_child("parameters").get_child("funnel").get_child("funnel_incoming").get_value<bool>();
        bool funnel_outgoing =
            algorithm.get_child("parameters").get_child("funnel").get_child("funnel_outgoing").get_value<bool>();
        bool first_funnel_incoming =
            algorithm.get_child("parameters").get_child("funnel").get_child("first_funnel_incoming").get_value<bool>();
        bool use_approx_transitive_reduction = algorithm.get_child("parameters")
                                                   .get_child("funnel")
                                                   .get_child("use_approx_transitive_reduction")
                                                   .get_value<bool>();

        Funnel_parameters f_params(max_relative_weight, funnel_incoming, funnel_outgoing, first_funnel_incoming,
                                 use_approx_transitive_reduction);

        CoBalDMixR cobald(params);

        LKTotalCommScheduler lk;
        ComboScheduler cobald_lk(&cobald, &lk);
        HillClimbingScheduler hc;
        Funnel scheduler(&cobald_lk, &hc, f_params);

        if (apply_trans_edge_contraction) {

            AppTransEdgeReductor red(scheduler);
            return red.computeSchedule(bsp_instance);

        } else {

            return scheduler.computeSchedule(bsp_instance);
        }

    } else if (algorithm.get_child("name").get_value<std::string>() == "SquashBspGQLK") {

        unsigned number_of_partitions = bsp_instance.numberOfProcessors();

        float geom_decay_num_nodes =
            algorithm.get_child("parameters").get_child("coarsen").get_child("geom_decay_num_nodes").get_value<float>();
        double poisson_par =
            algorithm.get_child("parameters").get_child("coarsen").get_child("poisson_par").get_value<double>();
        unsigned noise =
            algorithm.get_child("parameters").get_child("coarsen").get_child("noise").get_value<unsigned>();
        std::pair<unsigned, unsigned> edge_sort_ratio = std::make_pair(algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_triangle")
                                                                           .get_value<unsigned>(),
                                                                       algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_weight")
                                                                           .get_value<unsigned>());
        int num_rep_without_node_decrease = algorithm.get_child("parameters")
                                                .get_child("coarsen")
                                                .get_child("num_rep_without_node_decrease")
                                                .get_value<int>();
        float temperature_multiplier = algorithm.get_child("parameters")
                                           .get_child("coarsen")
                                           .get_child("temperature_multiplier")
                                           .get_value<float>();
        float number_of_temperature_increases = algorithm.get_child("parameters")
                                                    .get_child("coarsen")
                                                    .get_child("number_of_temperature_increases")
                                                    .get_value<float>();

        CoarsenParams coarse_params(geom_decay_num_nodes, poisson_par, noise, edge_sort_ratio,
                                    num_rep_without_node_decrease, temperature_multiplier,
                                    number_of_temperature_increases);

        int min_nodes_after_coarsen_per_partition = algorithm.get_child("parameters")
                                                        .get_child("coarsen")
                                                        .get_child("min_nodes_after_coarsen_per_partition")
                                                        .get_value<int>();

        float max_percent_idle_processor =
            algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();

        GreedyBspScheduler bsp_greedy(max_percent_idle_processor);
        LKTotalCommScheduler lk;
        lk.set_quick_pass(true);
        SquashA scheduler(&bsp_greedy, &lk, coarse_params,
                          min_nodes_after_coarsen_per_partition * number_of_partitions);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    }  else {

        throw std::invalid_argument("Parameter error: Unknown algorithm.\n");
    }
}