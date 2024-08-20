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

#include "model/BspScheduleRecomp.hpp"
#include "model/ComputationalDag.hpp"
#include <boost/graph/graphviz.hpp>

#include <fstream>
#include <string>

class BspScheduleRecompDuplicateWriter {
  private:
    const BspScheduleRecomp &schedule;

  public:
    struct EdgeWriterSchedule_DOT {
        const GraphType &graph;

        EdgeWriterSchedule_DOT(const GraphType &graph_) : graph(graph_) {}

        template<class VertexOrEdge>
        void operator()(std::ostream &out, const VertexOrEdge &i) const {
            out << "[" << "comm_weight=\"" << graph[i].communicationWeight << "\";" << "]";
        }
    };
   
    struct VertexWriterDuplicateSchedule_DOT {
        const BspScheduleRecomp &schedule;
        const std::vector<std::string> name;
        const std::vector<unsigned> node_to_proc;
        const std::vector<unsigned> node_to_superstep;

        VertexWriterDuplicateSchedule_DOT(const BspScheduleRecomp &schedule_, const std::vector<std::string> &name_, std::vector<unsigned> &node_to_proc_,
                                          std::vector<unsigned> &node_to_superstep_)
            : schedule(schedule_), name(name_), node_to_proc(node_to_proc_), node_to_superstep(node_to_superstep_) {}

        template<class VertexOrEdge>
        void operator()(std::ostream &out, const VertexOrEdge &i) const {
            out << "[" << "label=\"" << name[i] << "\";" << "work_weight=\""
                << schedule.getInstance().getComputationalDag().nodeWorkWeight(i) << "\";" << "comm_weight=\""
                << schedule.getInstance().getComputationalDag().nodeCommunicationWeight(i) << "\";" << "mem_weight=\""
                << schedule.getInstance().getComputationalDag().nodeMemoryWeight(i) << "\";" << "proc=\""
                << node_to_proc[i] << "\";" << "superstep=\"" << node_to_superstep[i] << "\";";

            out << "]";
        }
    };

    /**
     * Constructs a BspScheduleWriter object with the given BspSchdule.
     *
     * @param schdule_
     */
    BspScheduleRecompDuplicateWriter(const BspScheduleRecomp &schedule_) : schedule(schedule_) {}

    /**
     * Writes the BspSchedule to the specified output stream in DOT format.
     *
     * @tparam VertexWriterType The type of the vertex writer to be used.
     *         Default is VertexWriter_DOT.
     * @tparam EdgeWriterType The type of the edge writer to be used.
     *         Default is EdgeWriter_DOT.
     *
     * @param os The output stream to write the DOT representation of the computational DAG.
     */
    template<class VertexWriterType = VertexWriterDuplicateSchedule_DOT, class EdgeWriterType = EdgeWriterSchedule_DOT>
    void write_dot(std::ostream &os) const;

    /**
     * Writes the BspSchedule to the specified file in DOT format.
     *
     * @note The file will be overwritten if it already exists.
     * @note The file will be created if it does not exist.
     * @note This function is an alias for `write_dot(std::ostream &os)`
     *
     * @tparam VertexWriterType The type of the vertex writer to be used.
     *         Default is VertexWriter_DOT.
     * @tparam EdgeWriterType The type of the edge writer to be used.
     *         Default is EdgeWriter_DOT.
     *
     * @param filename The name of the file to write the DOT representation of the computational DAG.
     */
    template<class VertexWriterType = VertexWriterDuplicateSchedule_DOT, class EdgeWriterType = EdgeWriterSchedule_DOT>
    void write_dot(const std::string &filename) const {
        std::ofstream os(filename);
        write_dot(os);
    }

};



template<class VertexWriterType, class EdgeWriterType>
void BspScheduleRecompDuplicateWriter::write_dot(std::ostream &os) const {

    const auto &g = schedule.getInstance().getComputationalDag().getGraph();
   
    std::vector<std::string> names(schedule.total_node_assignments());
    std::vector<unsigned> node_to_proc(schedule.total_node_assignments());
    std::vector<unsigned> node_to_superstep(schedule.total_node_assignments());

    std::unordered_map<VertexType, std::vector<unsigned>> vertex_to_idx;

    GraphType g2;

    unsigned idx_new = 0;

    for (const auto &node : boost::make_iterator_range(boost::vertices(g))) {

        if (schedule.assignedProcessors(node).size() == 1) {

            boost::add_vertex(g2);

            names[idx_new] = std::to_string(node);
            node_to_proc[idx_new] = schedule.assignedProcessors(node)[0];
            node_to_superstep[idx_new] = schedule.assignedSupersteps(node)[0];

            vertex_to_idx.insert({node, {idx_new}});
            idx_new++;

        } else {

            std::vector<unsigned> idxs;
            for (unsigned i = 0; i < schedule.assignedProcessors(node).size(); ++i) {

                boost::add_vertex(g2);

                names[idx_new] = std::to_string(node).append("_").append(std::to_string(i));
                node_to_proc[idx_new] = schedule.assignedProcessors(node)[i];
                node_to_superstep[idx_new] = schedule.assignedSupersteps(node)[i];

                idxs.push_back(idx_new++);
            }
            vertex_to_idx.insert({node, idxs});
        }
    }

    for (const auto &[key, val] : vertex_to_idx) {

        if (val.size() == 1) {

            for (const auto &edge : boost::make_iterator_range(boost::out_edges(key, g))) {

                const auto target = boost::target(edge, g);
                
                for (const auto &new_node_target : vertex_to_idx[target]) {
                    boost::add_edge(val[0], new_node_target, g2);
                }
            }

        } else {

            std::unordered_set<unsigned> assigned_processors; 
            
            for(auto val : schedule.assignedProcessors(key)) {
                
                assigned_processors.insert(val);
            }

            for (unsigned i = 0; i < val.size(); i++) {

                for (const auto &edge : boost::make_iterator_range(boost::out_edges(key, g))) {

                    const auto target = boost::target(edge, g);

                    for (unsigned j = 0; j < vertex_to_idx[target].size(); j++) {

                        if (assigned_processors.find(node_to_proc[vertex_to_idx[target][j]]) ==
                                assigned_processors.end() ||
                            node_to_proc[val[i]] == node_to_proc[vertex_to_idx[target][j]]) {

                            boost::add_edge(val[i], vertex_to_idx[target][j], g2);
                        }
                    }
                }
            }
        }
    }

    boost::write_graphviz(os, g2, VertexWriterDuplicateSchedule_DOT(schedule, names, node_to_proc, node_to_superstep), EdgeWriterType(g2));
}