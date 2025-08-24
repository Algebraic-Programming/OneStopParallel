#pragma once

#include "common_mocks.hpp" // In your project, include your actual files

namespace osp {

/**
 * @class IsomorphicComponentDivider
 * @brief Divides a DAG by reacting to structural changes at each wavefront.
 */
template<typename Graph_t>
class IsomorphicComponentDivider : public IDagDivider<Graph_t> {
    static_assert(is_computational_dag_v<Graph_t>,
                  "IsomorphicComponentDivider can only be used with computational DAGs.");

public:
    explicit IsomorphicComponentDivider(double major_merge_threshold = 0.7,
                                        double diversity_loss_threshold = 0.5,
                                        size_t absolute_merge_threshold = 1000000)
        : major_merge_threshold_(major_merge_threshold),
          diversity_loss_threshold_(diversity_loss_threshold),
          absolute_merge_threshold_(absolute_merge_threshold) {}

    std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> divide(const Graph_t &dag) override {
        dag_ptr_ = &dag;
        std::vector<std::vector<VertexType>> level_sets = compute_wavefronts(dag);
        if (level_sets.empty()) return {};

        std::vector<size_t> cut_levels;
        union_find_universe_t<Graph_t> uf;
        MerkleHashComputer<Graph_t> hash_computer(dag);
        std::map<MerkleHash, size_t> previous_census;

        for (size_t i = 0; i < level_sets.size(); ++i) {
            update_components(uf, level_sets[i]);
            auto current_components = uf.get_connected_components();
            std::map<MerkleHash, size_t> current_census;
            for (const auto& component_vertices : current_components) {
                current_census[hash_computer.compute_hash(component_vertices)]++;
            }

            if (i > 0) {
                bool cut_inserted = check_for_major_merge(previous_census, current_census) ||
                                    check_for_diversity_loss(previous_census, current_census);
                if (cut_inserted && (cut_levels.empty() || cut_levels.back() != i)) {
                    cut_levels.push_back(i);
                }
            }
            previous_census = std::move(current_census);
        }
        return create_vertex_maps_from_cuts(cut_levels, level_sets);
    }

private:
    using VertexType = vertex_idx_t<Graph_t>;

    const Graph_t* dag_ptr_ = nullptr;
    double major_merge_threshold_;
    double diversity_loss_threshold_;
    size_t absolute_merge_threshold_;

    bool check_for_major_merge(const std::map<MerkleHash, size_t>& prev, const std::map<MerkleHash, size_t>& curr) {
        for (const auto& [hash, prev_count] : prev) {
            if (prev_count <= 1) continue;
            size_t current_count = curr.count(hash) ? curr.at(hash) : 0;
            if (current_count < prev_count) {
                if (static_cast<double>(prev_count - current_count) / prev_count >= major_merge_threshold_) return true;
                if (prev_count >= absolute_merge_threshold_) return true;
            }
        }
        return false;
    }

    bool check_for_diversity_loss(const std::map<MerkleHash, size_t>& prev, const std::map<MerkleHash, size_t>& curr) {
        if (prev.empty() || prev.size() <= 1) return false;
        if (static_cast<double>(prev.size() - curr.size()) / prev.size() >= diversity_loss_threshold_) return true;
        return false;
    }

    void update_components(union_find_universe_t<Graph_t>& uf, const std::vector<VertexType>& wavefront) {
        for (const auto vertex : wavefront) {
            uf.add_object(vertex, 0, 0);
            for (const auto& parent : dag_ptr_->parents(vertex)) {
                if (uf.is_in_universe(parent)) uf.join_by_name(parent, vertex);
            }
        }
    }
    
    std::vector<std::vector<VertexType>> get_components_for_range(
        size_t start_level, size_t end_level, const std::vector<std::vector<VertexType>>& levels) {
        if (start_level >= end_level) return {};
        union_find_universe_t<Graph_t> uf;
        for (size_t i = start_level; i < end_level; ++i) {
            update_components(uf, levels[i]);
        }
        return uf.get_connected_components();
    }

    std::vector<std::vector<std::vector<VertexType>>> create_vertex_maps_from_cuts(
        const std::vector<size_t>& cut_levels, const std::vector<std::vector<VertexType>>& levels) {
        if (levels.empty()) return {};
        std::vector<std::vector<std::vector<VertexType>>> partitions;
        size_t last_cut = 0;
        for (const auto& cut_level : cut_levels) {
            partitions.push_back(get_components_for_range(last_cut, cut_level, levels));
            last_cut = cut_level;
        }
        partitions.push_back(get_components_for_range(last_cut, levels.size(), levels));
        partitions.erase(std::remove_if(partitions.begin(), partitions.end(), [](const auto& p){ return p.empty(); }), partitions.end());
        return partitions;
    }

    std::vector<std::vector<VertexType>> compute_wavefronts(const Graph_t& dag) const {
        std::vector<std::vector<VertexType>> level_sets;
        std::vector<int> in_degree(dag.num_vertices(), 0);
        std::queue<VertexType> q;
        for (VertexType u = 0; u < dag.num_vertices(); ++u) {
            in_degree[u] = dag.parents(u).size();
            if (in_degree[u] == 0) q.push(u);
        }
        while (!q.empty()) {
            size_t level_size = q.size();
            std::vector<VertexType> current_level;
            for (size_t i = 0; i < level_size; ++i) {
                VertexType u = q.front(); q.pop();
                current_level.push_back(u);
                for (const auto& v : dag.children(u)) {
                    if (--in_degree[v] == 0) q.push(v);
                }
            }
            level_sets.push_back(current_level);
        }
        return level_sets;
    }
};

} // namespace osp