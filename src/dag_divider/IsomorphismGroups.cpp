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

    print_isomorphism_groups();
}

void IsomorphismGroups::print_isomorphism_groups() const {

    std::cout << "Isomorphism groups: " << std::endl;
    for (size_t i = 0; i < isomorphism_groups.size(); i++) {
        std::cout << "Level " << i << std::endl;
        for (size_t j = 0; j < isomorphism_groups[i].size(); j++) {
            std::cout << "Group " << j << " of size " << isomorphism_groups_subgraphs[i][j].numberOfVertices() << " : ";

            for (const auto &vertex : isomorphism_groups[i][j]) {
                std::cout << vertex << " ";
            }
            std::cout << std::endl;
        }
    }

    std::cout << "Isomorphism groups end" << std::endl;
}
