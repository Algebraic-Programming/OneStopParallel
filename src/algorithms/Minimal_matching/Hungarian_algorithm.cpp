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

# include "algorithms/Minimal_matching/Hungarian_algorithm.hpp"

std::vector<long unsigned> dfs_for_Hungarian_algorithm(  std::set<long unsigned>& Z_S,
                                                    std::set<long unsigned>& Z_T,
                                                    const std::set<long unsigned>& not_assigned_S,
                                                    const std::set<long unsigned>& not_assigned_T,
                                                    const std::vector<std::set<long unsigned>>& tight_forward,
                                                    const std::vector<long int>& tight_backward,
                                                    std::vector<long unsigned>& path,
                                                    bool last_in_S )
{
    if (path.empty()) return path;

    if (last_in_S) {
        for ( auto& t : tight_forward[ path.back() ] ) {
            if ( Z_T.find(t) != Z_T.cend() ) continue;
            Z_T.emplace(t);
            path.emplace_back(t);
            if ( not_assigned_T.find(t) != not_assigned_T.cend() ) return path;
            std::vector<long unsigned> returned_path = dfs_for_Hungarian_algorithm(Z_S, Z_T, not_assigned_S, not_assigned_T, tight_forward, tight_backward, path, false);
            if ( returned_path.size() != 0 ) return returned_path;

            auto it = path.end();
            it--;
            path.erase( it );
        }
        return std::vector<long unsigned>({});
    }
    else {
        unsigned s = tight_backward[ path.back() ];
        if ( Z_S.find(s) != Z_S.cend() ) return std::vector<long unsigned>({});
        Z_S.emplace(s);
        path.emplace_back(s);
        std::vector<long unsigned> returned_path = dfs_for_Hungarian_algorithm(Z_S, Z_T, not_assigned_S, not_assigned_T, tight_forward, tight_backward, path, true);
        if ( returned_path.size() != 0 ) return returned_path;

        auto it = path.end();
        it--;
        path.erase( it );

        return std::vector<long unsigned>({});
    }
}






std::vector<unsigned> min_perfect_matching_for_complete_bipartite(const std::vector<std::vector<long long unsigned>>& costs ) {
    long unsigned dim = costs.size();
    for (long unsigned i = 0; i<dim; i++) {
        assert( costs[i].size()==dim );
        for (long unsigned j = 0; j < costs[i].size(); j++) {
            assert( costs[i][j] <= LLONG_MAX );
        }
    }

    std::vector<long long int> potential_S(dim,0);
    std::vector<long long int> potential_T(dim,0);

    std::set<long unsigned> not_assigned_S;
    std::set<long unsigned> not_assigned_T;
    for (long unsigned i = 0; i<dim; i++) {
        not_assigned_S.emplace(i);
        not_assigned_T.emplace(i);
    }

    std::vector<std::set<long unsigned>> tight_forward(dim);
    std::vector<long int> tight_backward(dim, -1); // -1 means unassigned

    // initialising tightness
    for (long unsigned i = 0; i<dim; i++ ) {
        for (long unsigned j = 0; j<dim; j++) {
            if (potential_S[i]+potential_T[j] == (long long) costs[i][j]) {
                tight_forward[i].emplace(j);
            }
        }
    }

    while (not_assigned_S.size() != 0) {
        // std::cout << "Not assigned S: ";
        // for (auto s : not_assigned_S) {
        //     std::cout << s << " ";
        // }
        // std::cout << std::endl;
        // std::cout << "Not assigned T: ";
        // for (auto t : not_assigned_T) {
        //     std::cout << t << " ";
        // }
        // std::cout << std::endl;

        std::set<long unsigned> Z_S = not_assigned_S;
        std::set<long unsigned> Z_T;

        std::vector<long unsigned> dfs_returned_path;
        for ( long unsigned s : not_assigned_S ) {
            std::vector<long unsigned> path({s});
            dfs_returned_path = dfs_for_Hungarian_algorithm(Z_S, Z_T, not_assigned_S, not_assigned_T, tight_forward, tight_backward, path, true);
            if (dfs_returned_path.size() != 0) break;
        }

        // std::cout << "Z_S: ";
        // for (auto s : Z_S) {
        //     std::cout << s << " ";
        // }
        // std::cout << std::endl;
        // std::cout << "Z_T: ";
        // for (auto t : Z_T) {
        //     std::cout << t << " ";
        // }
        // std::cout << std::endl;

        if (dfs_returned_path.size() != 0) {
            // std::cout << "Path: ";
            // for (auto t : dfs_returned_path) {
            //     std::cout << t << " ";
            // }
            // std::cout << std::endl;

            for (long unsigned i = 1; i<dfs_returned_path.size(); i+=2) {
                tight_forward[ dfs_returned_path[i-1] ].erase( dfs_returned_path[i] );
                tight_backward[ dfs_returned_path[i] ] = dfs_returned_path[i-1];

                if (i >= dfs_returned_path.size()-1) continue;

                tight_forward[ dfs_returned_path[i+1] ].emplace( dfs_returned_path[i] );
            }
            not_assigned_S.erase(dfs_returned_path[0]);
            not_assigned_T.erase(dfs_returned_path.back());

            // std::cout << "backwards t->s: ";
            // for (unsigned t = 0; t < dim; t++) {
            //     std::cout << t << "->" << tight_backward[t] << " ";
            // }
            // std::cout << std::endl;
        }
        else {
            long long unsigned delta = ULLONG_MAX;
            for (auto& s : Z_S) {
                for (long unsigned t = 0; t<dim; t++) {
                    if (Z_T.find(t) != Z_T.cend() ) continue;

                    // std::cout << "s, t: " << s << ", " << t << std::endl;
                    // std::cout << "P(s), P(t): " << potential_S[s] << ", " << potential_T[t] << std::endl;
                    // std::cout << "costs: " << costs[s][t] << ", sum: " << potential_S[s] + potential_T[t] << std::endl;

                    long long int potential_sum = potential_S[s]+potential_T[t];

                    assert( (potential_sum < 0) || ( (long long) costs[s][t] >= potential_sum) );
                    delta = std::min(delta, costs[s][t]-potential_S[s]-potential_T[t]);
                }
            }

            // std::cout << "delta: " << delta << std::endl;

            // potential change
            for (auto& s : Z_S) {
                potential_S[s] += delta;
            }
            for (auto& t : Z_T) {
                potential_T[t] -= delta;
            }
            
            // std::cout << "potential_S: ";
            // for (auto s : potential_S) {
            //     std::cout << s << " ";
            // }
            // std::cout << std::endl;
            // std::cout << "potential_T: ";
            // for (auto t : potential_T) {
            //     std::cout << t << " ";
            // }
            // std::cout << std::endl;


            // tight graph update
            for (long unsigned s = 0; s < dim; s++) {
                for (long unsigned t = 0; t < dim; t++) {
                    if ( (long long) costs[s][t] == potential_S[s]+potential_T[t] ) {
                        tight_forward[s].emplace(t);
                    }
                    else {
                        tight_forward[s].erase(t);
                    }
                }
            }

        }

        // std::cout<< "end iteration" << std::endl << std::endl;
    }

    // std::cout << "backwards t->s: ";
    // for (unsigned t = 0; t < dim; t++) {
    //     std::cout << t << "->" << tight_backward[t] << " ";
    // }
    // std::cout << std::endl;

    std::vector<unsigned> output(dim);
    for (long unsigned t = 0; t<dim; t++) {
        assert(tight_backward[t] != -1);
        output[ tight_backward[t] ]  = t;
    }

    // std::cout << "Pairing output s->t: ";
    // for (unsigned s = 0; s < dim; s++) {
    //     std::cout << s << "->" << output[s] << " ";
    // }
    // std::cout << std::endl;

    return output;
}


std::vector<unsigned> max_perfect_matching_for_complete_bipartite( const std::vector<std::vector<long long unsigned>> &savings) {
    long long unsigned maximum = 0;
    std::vector<std::vector<long long unsigned>> costs( savings.size(), std::vector<long long unsigned>( savings[0].size() ) );

    for (long unsigned i = 0; i<savings.size(); i++) {
        for (long unsigned j = 0; j<savings[0].size(); j++) {
            maximum = std::max(maximum, savings[i][j]);
        }
    }

    for (long unsigned i = 0; i<savings.size(); i++) {
        for (long unsigned j = 0; j<savings[0].size(); j++) {
            assert( maximum >= savings[i][j] );
            costs[i][j] = maximum - savings[i][j];
        }
    }

    return min_perfect_matching_for_complete_bipartite(costs);
}