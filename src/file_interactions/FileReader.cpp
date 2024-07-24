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

#include "file_interactions/FileReader.hpp"
#include <boost/algorithm/string.hpp>
#include "model/ComputationalDag.hpp"

std::pair<bool, BspInstance> FileReader::readBspInstance(const std::string &filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cout << "Unable to find/open input schedule file.\n";

        return {false, BspInstance()};
    }

    auto [ret_status_dag, dag] = FileReader::readComputationalDagHyperdagFormat(infile);

    auto [ret_status_architecture, architecture] = FileReader::readBspArchitecture(infile);

    infile.close();
    if (ret_status_dag && ret_status_architecture) {

        return {true, BspInstance(dag, architecture)};

    } else {

        return {false, BspInstance()};
    }
}

std::pair<bool, BspSchedule> FileReader::readBspSchdeuleTxtFormat(const BspInstance &instance,
                                                                  const std::string &filename) {

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cout << "Unable to find/open input machine parameter file.\n";

        return {false, BspSchedule()};
    }

    return FileReader::readBspSchdeuleTxtFormat(instance, infile);
}

std::pair<bool, BspSchedule> FileReader::readBspSchdeuleTxtFormat(const BspInstance &instance, std::ifstream &infile) {

    std::string line;
    getline(infile, line);
    while (!infile.eof() && line.at(0) == '%')
        getline(infile, line);

    unsigned num_nodes;
    unsigned num_proc;
    unsigned num_supersteps;
    unsigned has_comm_schedule;
    sscanf(line.c_str(), "%d %d %d %d", &num_nodes, &num_proc, &num_supersteps, &has_comm_schedule);

    if (num_nodes != instance.numberOfVertices() || num_proc != instance.numberOfProcessors() || num_supersteps < 0) {
        std::cout << "Input file schedule is not compatible with given instance.\n";
        return {false, BspSchedule()};
    }

    std::vector<unsigned> processor_assignment(num_nodes, 0);
    std::vector<unsigned> superstep_assignment(num_nodes, 0);
    std::vector<bool> node_found(num_nodes, false);

    for (unsigned i = 0; i < num_nodes; ++i) {
        if (infile.eof()) {
            std::cout << "Incorrect input file format (file terminated too early).\n";
            return {false, BspSchedule()};
        }
        getline(infile, line);
        while (!infile.eof() && line.at(0) == '%')
            getline(infile, line);

        unsigned node, proc, superstep;
        sscanf(line.c_str(), "%d %d %d", &node, &proc, &superstep);

        if (node < 0 || proc < 0 || superstep < 0 || node > num_nodes || proc > num_proc ||
            superstep > num_supersteps) {
            std::cout << "Incorrect input file format (index out of range).\n";
            return {false, BspSchedule()};
        }

        node_found[node] = true;
        processor_assignment[node] = proc;
        superstep_assignment[node] = superstep;
    }

    for (unsigned i = 0; i < num_nodes; ++i) {
        if (!node_found[i]) {
            std::cout << "Incorrect input file format (node not found).\n";
            return {false, BspSchedule()};
        }
    }

    if (has_comm_schedule == 0) {
        return {true, BspSchedule(instance, processor_assignment, superstep_assignment)};
    }

    std::map<KeyTriple, unsigned> comm_schedule;

    while (!infile.eof()) {

        getline(infile, line);
        while (!infile.eof() && line.at(0) == '%')
            getline(infile, line);

        unsigned node, from, to, comm;
        sscanf(line.c_str(), "%d %d %d %d", &node, &from, &to, &comm);

        if (node < 0 || from < 0 || to < 0 || comm < 0 || node >= num_nodes || from >= num_proc || to >= num_proc ||
            comm >= num_supersteps) {
            std::cout << "Incorrect input file format (index out of range).\n";
            return {false, BspSchedule()};
        }

        comm_schedule[{node, from, to}] = comm;
    }

    return {true, BspSchedule(instance, processor_assignment, superstep_assignment, comm_schedule)};
}

std::pair<bool, BspArchitecture> FileReader::readBspArchitecture(const std::string &filename) {

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cout << "Unable to find/open input machine parameter file.\n";

        return {false, BspArchitecture()};
    }

    return FileReader::readBspArchitecture(infile);
}

std::pair<bool, ComputationalDag> FileReader::readComputationalDagHyperdagFormat(const std::string &filename) {

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cout << "Unable to find/open input dag file.\n";

        return {false, ComputationalDag()};
    }

    return FileReader::readComputationalDagHyperdagFormat(infile);
}

std::pair<bool, ComputationalDag> FileReader::readComputationalDagHyperdagFormat(std::ifstream &infile) {

    std::string line;
    getline(infile, line);
    while (!infile.eof() && line.at(0) == '%')
        getline(infile, line);

    int hEdges, pins, N;
    sscanf(line.c_str(), "%d %d %d", &hEdges, &N, &pins);

    if (N <= 0 || hEdges <= 0 || pins <= 0) {
        std::cout << "Incorrect input file format (number of nodes/hyperedges/pins is not positive).\n";
        return {false, ComputationalDag()};
    }

    // for (size_t i = 0; i < N; i++) {
    //     dag.addVertex(1,1);
    // }

    ComputationalDag dag(N);

    // Resize(N);
    std::vector<int> edgeSource(hEdges, -1);
    // read edges
    for (int i = 0; i < pins; ++i) {
        if (infile.eof()) {
            std::cout << "Incorrect input file format (file terminated too early).\n";
            return {false, dag};
        }
        getline(infile, line);
        while (!infile.eof() && line.at(0) == '%')
            getline(infile, line);

        int hEdge, node;
        sscanf(line.c_str(), "%d %d", &hEdge, &node);

        if (hEdge < 0 || node < 0 || hEdge >= hEdges || node >= N) {
            std::cout << "Incorrect input file format (index out of range).\n";
            return {false, dag};
        }

        if (edgeSource[hEdge] == -1)
            edgeSource[hEdge] = node;
        else
            dag.addEdge(edgeSource[hEdge], node);
    }

    for (int i = 0; i < N; ++i) {
        if (infile.eof()) {
            std::cout << "Incorrect input file format (file terminated too early).\n";
            return {false, dag};
        }

        getline(infile, line);
        while (!infile.eof() && line.at(0) == '%')
            getline(infile, line);

        int node, work, comm; //, mem;
        sscanf(line.c_str(), "%d %d %d", &node, &work, &comm);

        if (node < 0 || work < 0 || comm < 0 || node >= N) {
            std::cout << "Incorrect input file format (index out of range, our weight below 0).\n";
            return {false, dag};
        }

        dag.setNodeCommunicationWeight(node, comm);
        dag.setNodeWorkWeight(node, work);
        dag.setNodeMemoryWeight(node, comm);
    }

    return {true, dag};
};

std::pair<bool, ComputationalDag> FileReader::readComputationalDagMartixMarketFormat(const std::string &filename) {

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cout << "Unable to find/open input dag file.\n";

        return {false, ComputationalDag()};
    }

    return FileReader::readComputationalDagMartixMarketFormat(infile);
}

std::pair<bool, ComputationalDag> FileReader::readComputationalDagMartixMarketFormat(std::ifstream &infile) {

    std::string line;
    getline(infile, line);
    while (!infile.eof() && line.at(0) == '%')
        getline(infile, line);

    int nEntries, M_row, M_col;
    sscanf(line.c_str(), "%d %d %d", &M_row, &M_col, &nEntries);

    if (M_row <= 0 || M_col <= 0 || M_col != M_row) {
        std::cout << "Incorrect input file format (No rows/columns or not a square matrix).\n";
        return {false, ComputationalDag()};
    }

    ComputationalDag dag(M_row);

    // Initialise data;
    std::vector<int> node_work_wts(M_row, 0);
    std::vector<int> node_comm_wts(M_row, 1);
    // read edges
    for (int i = 0; i < nEntries; ++i) {
        getline(infile, line);
        while (!infile.eof() && line.at(0) == '%')
            getline(infile, line);
        
        if (infile.eof()) {
            std::cout << "Incorrect input file format (file terminated too early).\n";
            return {false, dag};
        }

        int row, col;
        double val;
        sscanf(line.c_str(), "%d %d %lf", &row, &col, &val);
        // Indexing starting at 0
        row -= 1;
        col -= 1;

        if (row < 0 || col < 0 || row >= M_row || col >= M_col) {
            std::cout << "Incorrect input file format (index out of range).\n";
            return {false, dag};
        }
        if (row < col) {
            std::cout << "Incorrect input file format (matrix is not lower triangular).\n";
            return {false, dag};
        } else if (col != row) {
            dag.addEdge(col, row , val, 1);
//            mtx.emplace(std::make_pair(col, row), val);
        } else {
            dag.set_node_mtx_entry(row, val);
             //mtx.emplace(std::make_pair(col, row), val);
         }
        node_work_wts[row] += 1;
    }

    for (int i = 0; i < M_row; i++) {
        if (node_work_wts[i] == 0) {
            node_work_wts[i]++;
        }
        dag.setNodeCommunicationWeight(i, node_comm_wts[i]);
        dag.setNodeWorkWeight(i, node_work_wts[i]);
        dag.setNodeMemoryWeight(i, node_work_wts[i]);
    }

    getline(infile, line);
    if (! infile.eof()) {
        std::cout << "Incorrect input file format (file has remaining lines).\n";
        return {false, dag};
    }

    return {true, dag};
}


std::pair<bool, ComputationalDag> FileReader::readComputationalDagMartixMarketFormat(const std::string &filename, std::unordered_map<std::pair<VertexType, VertexType>, double, pair_hash> &mtx) {

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cout << "Unable to find/open input dag file.\n";

        return {false, ComputationalDag()};
    }

    return FileReader::readComputationalDagMartixMarketFormat(infile, mtx);
}

std::pair<bool, ComputationalDag> FileReader::readComputationalDagMartixMarketFormat(std::ifstream &infile, std::unordered_map<std::pair<VertexType, VertexType>, double, pair_hash> &mtx) {

    std::string line;
    getline(infile, line);
    while (!infile.eof() && line.at(0) == '%')
        getline(infile, line);

    int nEntries, M_row, M_col;
    sscanf(line.c_str(), "%d %d %d", &M_row, &M_col, &nEntries);

    if (M_row <= 0 || M_col <= 0 || M_col != M_row) {
        std::cout << "Incorrect input file format (No rows/columns or not a square matrix).\n";
        return {false, ComputationalDag()};
    }

    ComputationalDag dag(M_row);

    // Initialise data;
    std::vector<int> node_work_wts(M_row, 0);
    std::vector<int> node_comm_wts(M_row, 1);
    //std::unordered_map<std::pair<VertexType, VertexType>, double, pair_hash> matrix_entries;
    // read edges
    for (int i = 0; i < nEntries; ++i) {
        getline(infile, line);
        while (!infile.eof() && line.at(0) == '%')
            getline(infile, line);
        
        if (infile.eof()) {
            std::cout << "Incorrect input file format (file terminated too early).\n";
            return {false, dag};
        }

        int row, col;
        double val;
        sscanf(line.c_str(), "%d %d %lf", &row, &col, &val);
        // Indexing starting at 0
        row -= 1;
        col -= 1;

        if (row < 0 || col < 0 || row >= M_row || col >= M_col) {
            std::cout << "Incorrect input file format (index out of range).\n";
            return {false, dag};
        }
        if (row < col) {
            std::cout << "Incorrect input file format (matrix is not lower triangular).\n";
            return {false, dag};
        } else if (col != row) {
             dag.addEdge(col, row , val, 1);
            mtx.emplace(std::make_pair(col, row), val);
        } else {
            dag.set_node_mtx_entry(row, val);
            mtx.emplace(std::make_pair(col, row), val);
        }
        node_work_wts[row] += 1;
    }

    for (int i = 0; i < M_row; i++) {
        if (node_work_wts[i] == 0) {
            node_work_wts[i]++;
        }
        dag.setNodeCommunicationWeight(i, node_comm_wts[i]);
        dag.setNodeWorkWeight(i, node_work_wts[i]);
        dag.setNodeMemoryWeight(i, node_work_wts[i]);
    }

    getline(infile, line);
    if (! infile.eof()) {
        std::cout << "Incorrect input file format (file has remaining lines).\n";
        return {false, dag};
    }

    return {true, dag};
}

// read problem parameters from file
std::pair<bool, BspArchitecture> FileReader::readBspArchitecture(std::ifstream &infile) {

    std::string line;
    getline(infile, line);
    while (!infile.eof() && line.at(0) == '%')
        getline(infile, line);

    unsigned p, g, L;
    sscanf(line.c_str(), "%d %d %d", &p, &g, &L);

    BspArchitecture architecture(p, g, L);

    for (unsigned i = 0; i < p * p; ++i) {
        if (infile.eof()) {
            std::cout << "Incorrect input file format (file terminated too early).\n";

            architecture.SetUniformSendCost();
            return {false, architecture};
        }
        getline(infile, line);
        while (!infile.eof() && line.at(0) == '%')
            getline(infile, line);

        unsigned fromProc, toProc, value;
        sscanf(line.c_str(), "%d %d %d", &fromProc, &toProc, &value);

        if (fromProc < 0 || toProc < 0 || fromProc >= p || toProc >= p || value < 0) {
            std::cout << "Incorrect input file format (index out of range or "
                         "negative NUMA value).\n";

            architecture.SetUniformSendCost();
            return {false, architecture};
        }
        if (fromProc == toProc && value != 0) {
            std::cout << "Incorrect input file format (main diagonal of NUMA cost "
                         "matrix must be 0).\n";

            architecture.SetUniformSendCost();
            return {false, architecture};
        }
        architecture.setSendCosts(fromProc, toProc, value);
    }

    architecture.computeCommAverage();

    return {true, architecture};
};

void parseNode(std::string line, ComputationalDag& G) {

    // Extract node id and properties
    std::size_t pos = line.find('[');
    int nodeId = std::stoi(line.substr(0, pos));
    std::string properties = line.substr(pos + 1, line.find(']') - pos - 1);

    // Split properties into key-value pairs
    std::vector<std::string> keyValuePairs;
    boost::split(keyValuePairs, properties, boost::is_any_of(" "));


    // Create node with properties
    int work_weight = 0;
    int mem_weight = 0;
    int comm_weight = 0;
    for (const std::string& keyValuePair : keyValuePairs) {
        std::vector<std::string> keyValue;
        boost::split(keyValue, keyValuePair, boost::is_any_of("="));

        std::string key = keyValue[0];
        std::string value = keyValue[1];

        if (key == "work_weight") {
            work_weight = std::stoi(value);
        } else if (key == "mem_weight") {
            mem_weight = std::stoi(value);
        } else if (key == "comm_weight") {
            comm_weight = std::stoi(value);
        }
    }

    G.addVertex(work_weight, comm_weight, mem_weight);

}

void parseEdge(std::string line, ComputationalDag& G) {

   

    // Extract source, target and properties
    std::size_t pos = line.find('[');
    std::string nodes = line.substr(0, pos);
    std::string properties = line.substr(pos + 1, line.find(']') - pos - 1);

    // Split nodes into source and target
    std::vector<std::string> sourceTarget;
    boost::split(sourceTarget, nodes, boost::is_any_of("-"));

    int source = std::stoi(sourceTarget[0]);
    int target = std::stoi(sourceTarget[1].substr(1));

    // Split properties into key-value pairs
    std::vector<std::string> keyValuePairs;
    boost::split(keyValuePairs, properties, boost::is_any_of(" "));

    // Create edge with properties
    int comm_weight = 0;
    for (const std::string& keyValuePair : keyValuePairs) {
        std::vector<std::string> keyValue;
        boost::split(keyValue, keyValuePair, boost::is_any_of("="));

        std::string key = keyValue[0];
        std::string value = keyValue[1];

        if (key == "comm_weight") {
            comm_weight = std::stoi(value);
        }
    }


    G.addEdge(source, target, comm_weight);
    // Add edge to graph
    
}


std::pair<bool, ComputationalDag> FileReader::readComputationalDagDotFormat(std::ifstream &infile) {


    ComputationalDag G;
   
    std::string line;
    while (std::getline(infile, line)) {
        // Skip lines that do not contain opening or closing brackets
        if (line.find('{') != std::string::npos || line.find('}') != std::string::npos) {
            continue;
        }

        // Check if the line represents a node or an edge
        if (line.find("->") != std::string::npos) {
            // This is an edge
            parseEdge(line, G);
            // Add the edge to the graph
        } else {
            // This is a node
            parseNode(line, G);
            // Add the node to the graph
        }
    }

    return std::make_pair(true, G);

}


std::pair<bool, ComputationalDag> FileReader::readComputationalDagDotFormat(const std::string &filename) {

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cout << "Unable to find/open input dag file.\n";

        return {false, ComputationalDag()};
    }

    return FileReader::readComputationalDagDotFormat(infile);
}

bool FileReader::readProblem(const std::string &filename, DAG &G, BSPproblem &params, bool NoNUMA) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cout << "Unable to find/open input problem file.\n";
        return false;
    }

    G.read(infile);
    params.read(infile, NoNUMA);

    infile.close();
    return true;
};

std::tuple<bool, BspInstance, BspSchedule> FileReader::readBspScheduleDotFormat(const std::string &filename) {

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cout << "Unable to find/open input dag file.\n";

        return {false, BspInstance(), BspSchedule()};
    }

    return FileReader::readBspScheduleDotFormat(infile);
}

std::tuple<bool, BspInstance, BspSchedule> FileReader::readBspScheduleDotFormat(std::ifstream &infile) {

    struct Node {
        unsigned proc = 0;
        unsigned superstep = 0;
        std::string cs;
        int workWeight = 0;
        int communicationWeight = 0;
        int memoryWeight = 0;
    };

    struct Line {
        int communicationWeight = 0;
    };

    using graph_t = boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, Node, Line>;

    graph_t graph(0);

    boost::dynamic_properties dp(boost::ignore_other_properties);

    dp.property("work_weight", boost::get(&Node::workWeight, graph));
    dp.property("comm_weight", boost::get(&Node::communicationWeight, graph));
    dp.property("mem_weight", boost::get(&Node::memoryWeight, graph));

    dp.property("proc", boost::get(&Node::proc, graph));
    dp.property("superstep", boost::get(&Node::superstep, graph));
    dp.property("cs", boost::get(&Node::cs, graph));
    dp.property("comm_weight", boost::get(&Line::communicationWeight, graph));

    bool status = boost::read_graphviz(infile, graph, dp);

    ComputationalDag G;
    std::vector<unsigned> processor_assignment;
    std::vector<unsigned> superstep_assignment;
    std::map<KeyTriple, unsigned> comm_schedule;

    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        auto &node = graph[v];
        G.addVertex(node.workWeight, node.communicationWeight, node.memoryWeight);
        processor_assignment[v] = node.proc;
        superstep_assignment[v] = node.superstep;

        if (not node.cs.empty()) {

            std::string cs_strip = node.cs.substr(1, node.cs.size() - 2);
            std::vector<std::string> sub_strs;
            boost::split(sub_strs, cs_strip, boost::is_any_of(";"));

            for (const auto &entry : sub_strs) {

                std::string entry_strip = node.cs.substr(1, entry.size() - 2);
                std::vector<std::string> parts;
                boost::split(parts, entry_strip, boost::is_any_of(","));
                comm_schedule[{v, std::stoi(parts[0]), std::stoi(parts[1])}] = std::stoi(parts[2]);
            }
        }
    }

    for (auto e : boost::make_iterator_range(boost::edges(graph))) {
        auto &edge = graph[e];
        G.addEdge(boost::source(e, graph), boost::target(e, graph), edge.communicationWeight);
    }

    BspInstance instance(G, BspArchitecture());
    BspSchedule schedule(instance, processor_assignment, superstep_assignment, comm_schedule);

    return std::make_tuple(status, instance, schedule);
};

std::pair<bool, ComputationalDag> FileReader::readComputationalDagMetisFormat(std::string &filename) {

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cout << "Unable to find/open input dag file.\n";

        return {false, ComputationalDag()};
    }

    return FileReader::readComputationalDagMetisFormat(infile);

}

std::pair<bool, ComputationalDag> FileReader::readComputationalDagMetisFormat(std::ifstream &infile) {
    // graph_t *ReadGraph(params_t *params)

    // idx_t i, j, k, l, fmt, ncon, nfields, readew, readvw, readvs, edge, ewgt;
    // idx_t *xadj, *adjncy, *vwgt, *adjwgt, *vsize;
    // char *line = NULL, fmtstr[256], *curstr, *newstr;
    // size_t lnlen = 0;
    // FILE *fpin;

    
    std::string line;
    getline(infile, line);
    while (!infile.eof() && line.at(0) == '%')
        getline(infile, line);

    unsigned num_nodes = 0;
    unsigned num_edges = 0;
    std::string fmt = "000";
    unsigned ncon = 0;

    std::stringstream line2stream(line);

    line2stream >> num_nodes >> num_edges >> fmt >> ncon;

    std::cout << "num_nodes: " << num_nodes << " num_edges: " << num_edges << " fmt: " << fmt << " ncon: " << ncon
              << std::endl;

    if (num_nodes <= 0 || num_edges < 0) {
        std::cout << "The supplied number of nodes: " << num_nodes
                  << " must be positive and number of edges: " << num_edges << " must be non-negative." << std::endl;
        return {false, ComputationalDag()};
    }

    if (!(fmt.size() == 3 && (fmt[0] == '0' || fmt[0] == '1') && (fmt[1] == '0' || fmt[1] == '1') &&
          (fmt[2] == '0' || fmt[2] == '1'))) {
        std::cout << "Cannot read this type of file format fmt= " << fmt << std::endl;
    }

    bool readvs = (fmt[0] == '1');
    bool readvw = (fmt[1] == '1');
    bool readew = (fmt[2] == '1');

    ComputationalDag dag(num_nodes);

    if (ncon > 0 && !readvw) {
        std::cout << "------------------------------------------------------------------------------\n"
                     "***  I detected an error in your input file  ***\n\n"
                     "You specified ncon="
                  << ncon
                  << ", but the fmt parameter does not specify vertex weights\n"
                     "Make sure that the fmt parameter is set to either 10 or 11.\n"
                     "------------------------------------------------------------------------------\n";
    }

    if (ncon > 3) {
        std::cout << "Only up to 3 vertex weights are supported" << std::endl;
    }

    for (unsigned i = 0; i < num_nodes; i++) {

        do {
            if (infile.eof()) {
                std::cout << "Premature end of input file while reading vertex " << i + 1 << std::endl;

                std::getline(infile, line);
            }
        } while (line[0] == '%');

        std::stringstream line2stream(line);

        if (readvs) {

            unsigned vertex_size;
            line2stream >> vertex_size;

            dag.setNodeCommunicationWeight(i, vertex_size);
        }

        if (readvw) {
            for (unsigned l = 0; l < ncon; l++) {

                unsigned vertex_weight;
                line2stream >> vertex_weight;

                if (l == 0) {
                    dag.setNodeWorkWeight(i, vertex_weight);
                } else if (l == 1) {
                    dag.setNodeMemoryWeight(i, vertex_weight);
                }
            }
        }

        while (line2stream) {

            unsigned target;
            line2stream >> target;
            
            target--;

            if (target < 0 || target >= num_nodes)
                std::cout << "Edge " << target  << " for vertex " << i  << " is out of bounds" << std::endl;

            if (readew) {
                unsigned edge_weight;
                line2stream >> edge_weight;

                dag.addEdge(i, target, 1.0 , edge_weight);
            } else {

                dag.addEdge(i, target);
            }
        }
    }


    return {true, dag};
}

// gk_fclose(fpin);

// if (k != graph->nedges) {
//     printf("------------------------------------------------------------------------------\n");
//     printf("***  I detected an error in your input file  ***\n\n");
//     printf("In the first line of the file, you specified that the graph contained\n"
//            "%" PRIDX " edges. However, I only found %" PRIDX " edges in the file.\n",
//            graph->nedges / 2, k / 2);
//     if (2 * k == graph->nedges) {
//         printf("\n *> I detected that you specified twice the number of edges that you have in\n");
//         printf("    the file. Remember that the number of edges specified in the first line\n");
//         printf("    counts each edge between vertices v and u only once.\n\n");
//     }
//     printf("Please specify the correct number of edges in the first line of the file.\n");
//     printf("------------------------------------------------------------------------------\n");
//     exit(0);
// }

// gk_free((void *)&line, LTERM);

// return {true, dag};
// }