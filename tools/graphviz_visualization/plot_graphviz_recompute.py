# Copyright 2024 Huawei Technologies Co., Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# @author Toni Boehnlein, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner   



import os
import sys
import seaborn as sns
import networkx as nx


def read_dot_file(file_path: str) -> nx.DiGraph:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not os.path.isfile(file_path):
        raise ValueError(f"File is not a file: {file_path}")

    return nx.drawing.nx_agraph.read_dot(file_path)

def count_num_proc(graph: nx.DiGraph) -> int:

    num_proc = 0
    for node in graph.nodes(data=True):
        if int(node[1].get("proc")) > num_proc:
            num_proc = int(node[1].get("proc"))

    return num_proc + 1

def generate_graphviz_content(graph: nx.DiGraph) -> str:

    colors = sns.color_palette("hls", count_num_proc(graph)).as_hex()

    gv_content = """
digraph {
    fontname="Helvetica,Arial,sans-serif"
    node [
        fontname="Helvetica,Arial,sans-serif",
        fontsize=12,
        penwidth=1.0,
        margin=0.05,
        width=0.5,
        height=0.3,
        fixedsize=true
    ]
    edge [
        fontname="Helvetica,Arial,sans-serif",
        arrowsize=.3,
        style=dashed,
        color="#00000040"
    ]
    compound=true;
    rank=same;
    rankdir=BT;
    splines=false;
    overlap=false;

"""
    # Iterate over nodes with the attributes "superstep" being equal to i
    superstep_idx = 0
    while True:
        procs_in_superstep = 0
        found_in_superstep = False
        nodes_in_superstep = []
        node_labels = {}
        for node in graph.nodes(data=True):
            if int(node[1].get("superstep")) == superstep_idx:
                found_in_superstep = True
                nodes_in_superstep.append(node)
                node_labels[node[0]] = node[1].get("label")
                procs_in_superstep = max(procs_in_superstep, int(node[1].get("proc")) + 1)

        # If no node was found in the superstep, break the loop
        if not found_in_superstep:
            break

        # Add the nodes to the graphviz content
        gv_content += f"    subgraph cluster_ss{superstep_idx} {{\n"
        for proc in range(procs_in_superstep):
            gv_content += f"        subgraph cluster_ss{superstep_idx}_p{proc} {{\n"
            gv_content += f'            node [ color="{colors[proc]}", style= filled ];\n'
            gv_content += f"            rankdir=LR;\n"
            gv_content += f'            label="Processor #{proc}";\n'
            gv_content += f"            rank=same;\n"
            gv_content += f"            "
            for node in nodes_in_superstep:
                if int(node[1].get("proc")) == proc:
                    gv_content += f"{node[0]} [label=\"{node_labels[node[0]]}\"]; "
            gv_content += f"\n"
            gv_content += f"        }};\n"
        gv_content += f'        label="Super-step #{superstep_idx}";\n'
        gv_content += f"    }};\n\n"

        # Increment the superstep index
        superstep_idx += 1

    # Iterate over the edges
    for edge in graph.edges(data=True):

        if graph.nodes[edge[0]].get('proc') == graph.nodes[edge[1]].get('proc'):
            gv_content += f"    {edge[0]} -> {edge[1]};\n"
        else:
            gv_content += f"    {edge[0]} -> {edge[1]} [style=solid, color=black, penwidth=1];\n"

    gv_content += "}\n"
    return gv_content


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <dot-file>")
        sys.exit(1)

    dot_file = sys.argv[1]
    graph = read_dot_file(dot_file)
    gv_content = generate_graphviz_content(graph)
    #print(gv_content)
    file = open(dot_file[:-3] + "gv","w")
    file.write(gv_content)
    file.close()


if __name__ == "__main__":
    main()