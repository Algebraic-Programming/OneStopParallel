This directory contains a collection of computational DAGs. They represent a wide variety of different computational tasks, with some of them in a coarse-grained, others in a fine-grained representation.


# florida spd sparse, mtx format

The folder "mtx_tests" contains sparse matrices stored in the matrix market file format (for a description see https://networkrepository.com/mtx-matrix-market-format.html). We assume the matrices to be symmetric and that the files only store the lower triangular part. The sparse symmetric matrices are then naturally interpreted as DAGs such that each row corresponds to a vertex and there is a directed edge between (i,j) if the corresponding matrix entry is non-zero. 


# spaa and ai, hyperDag format

The folder 'spaa' contains hyperDAGs stored in the format described here. The format derives from a standard ‘MatrixMarket coordinate pattern general’ format where the rows of the matrix are understood as hyperedges, while the columns of the matrix are understood as nodes. The format, however, is extended to capture vertex and edge weights corresponding to costs such as operation counts or byte sizes.

This file describes version one (v1) of the hyperDAG file format. Further format revisions will strive to be backwards compatible, in that later file versions still may be read successfully by older version parsers-- even though additional information that future formats may standardise may be ignored by such older parsers.

1. Optionally, the file starts with a MatrixMarket(-like) header. If present, this header line must be the first line and start with `%%`.

2. Optionally, the file then follows with a version indicator. If present, this header line must read `% HyperDAG file format vV`, where `V` is any integer greater than or equal to one and indicates the file format version. This line must appear either as the first line of the file, or immediately following after the header line described above. If this line is missing, then `V` equals one-- i.e., the first hyperDAG file format version, v1, is assumed.

3. A hyperDAG file then may continue (or begin) with an arbitrary number of single-line comments, all starting with a `%` symbol. These lines can be used to describe the properties or origin of the hyperDAG in question. 

4. The first non-comment line of the file contains three integers `M N P`, separated by spaces. Any characters between `P` and the end-of-line character can be ignored.

- The first integer (`M`) describes the number of hyperedges,
- the second integer (`N`) describes the number of nodes, while
- the third integer (`P`) describes the number of pins (i.e. sum of the total size of all hyperedges).

5. This is then followed by `M` distinct lines, each describing the properties of a specific hyperedge. Each of these lines begins with the index of a hyperedge, i.e., an integer between `0` and `M-1` (inclusive). Each of the `M` lines must begin with a unique integer in this range. Each line can contain further integers, separated by spaces, describing properties of this hyperedge (e.g. communication weight).

6. This is then followed by `N` distinct lines, each describing the properties of a specific node. Each of these lines begins with the index of a node, i.e., an integer between `0` and `N-1` (inclusive). Each of the `N` lines must begin with a unique integer in this range. Each line can contain further integers, separated by spaces, describing properties of this node (e.g. computational weight).

7. This is then followed by `P` distinct lines, each describing a specific pin of the hyperDAG. Each of these lines contain two integers `E V`, separated by a space. Integer `E` lies between `0` and `M-1` (inclusive) while `V` lies between `0` and `N-1`-- the line indicates a pin connecting hyperedge `E` to node `V`. Each line can contain further integers, separated by spaces, describing properties of this pin.

- The first pin listed for each hyperedge describes the *source node* of the given hyperedge in the hyperDAG: in the original DAG representation, there is a directed edge from this source node to all the other nodes contained in this hyperedge.

The file shall not have any trailing empty or comment lines. It also shall not inject comment lines between any of the aforementioned `M+N+P` lines. Text comments may appear at the end of each line described by items 4-7, provided they are prepended by the percent (`%`) character. Text comments that end a data line (items 4-7) may be prepended by a space before the `%`.

All lines should be ended by the line feed character (`\n`).