#include "dag_divider/IsomorphismGroups.hpp"

void IsomorphismGroups::compute_isomorphism_groups(std::vector<std::vector<std::vector<unsigned>>> &vertex_maps,
                                                   const ComputationalDag &dag) {

    isomorphism_groups = std::vector<std::vector<std::vector<unsigned>>>(vertex_maps.size());

    isomorphism_groups_subgraphs = std::vector<std::vector<ComputationalDag>>(vertex_maps.size());

    for (size_t i = 0; i < vertex_maps.size(); i++) {

        for (unsigned j = 0; j < vertex_maps[i].size(); j++) {

            ComputationalDag current_subgraph = dag_algorithms::create_induced_subgraph_sorted(dag, vertex_maps[i][j]);

            bool isomorphism_group_found = false;
            for (size_t k = 0; k < isomorphism_groups[i].size(); k++) {

                if (isomorphism_groups_subgraphs[i][k].checkOrderedIsomorphism(current_subgraph)) {

                    isomorphism_groups[i][k].emplace_back(j);
                    isomorphism_group_found = true;
                    break;
                }
            }

            if (!isomorphism_group_found) {

                isomorphism_groups[i].emplace_back(std::vector<unsigned>{j});
                isomorphism_groups_subgraphs[i].emplace_back(current_subgraph);
            }
        }
    }

    // for (size_t i = 0; i < vertex_maps.size(); i++) {

    //     if (isomorphism_groups[i].size() > 1) 
    //         continue;

    //     for (size_t j = 0; j < isomorphism_groups[i].size(); j++) {

    //         const size_t size = static_cast<int>(isomorphism_groups[i][j].size());

    //         if (size > 8u) {

    //             std::cout << "iso group more than 8 components " << size << std::endl;

    //             if ((size & (size - 1)) == 0) {

    //                 size_t mult = size / 8;
    //                 std::cout << "mult: " << mult << std::endl;
                    
    //                 std::vector<std::vector<unsigned>> new_groups(8);

    //                 unsigned idx = 0;
    //                 for (auto& group : new_groups) {

    //                     for (size_t k = 0; k < mult; k++) {
    //                         group.insert(group.end(), vertex_maps[i][isomorphism_groups[i][j][idx]].begin(), vertex_maps[i][isomorphism_groups[i][j][idx]].end());
    //                         idx++;
    //                     }
    //                     std::sort(group.begin(), group.end());
    //                 }

    //                 vertex_maps[i] = new_groups;
    //                 isomorphism_groups[i] = std::vector<std::vector<unsigned>>(1, std::vector<unsigned>({0,1,2,3,4,5,6,7}));
    //                 isomorphism_groups_subgraphs[i] = std::vector<ComputationalDag>(1, dag_algorithms::create_induced_subgraph_sorted(dag, new_groups[0]));

    //             }
    //         }
    //     }
    // }

    print_isomorphism_groups();
}

void IsomorphismGroups::print_isomorphism_groups() const {

    std::cout << "Isomorphism groups: " << std::endl;
    for (size_t i = 0; i < isomorphism_groups.size(); i++) {
        std::cout << "Level " << i << std::endl;
        for (size_t j = 0; j < isomorphism_groups[i].size(); j++) {
            std::cout << "Group " << j << " of size " << isomorphism_groups_subgraphs[i][j].numberOfVertices() << " : ";

            // ComputationalDagWriter writer(isomorphism_groups_subgraphs[i][j]);
            // writer.write_dot("isomorphism_group_" + std::to_string(i) + "_" + std::to_string(j) + ".dot");

            for (const auto &vertex : isomorphism_groups[i][j]) {
                std::cout << vertex << " ";
            }
            std::cout << std::endl;
        }
    }

    std::cout << "Isomorphism groups end" << std::endl;
}
