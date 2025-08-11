# Input Machine File Format

This document describes the **BSP architecture machine parameter file format**.

---

## **1. Header Line**

The first **non-comment** line (comments start with `%`) specifies:

```
p g L [mem_type M]
```

Where:

- `p` – Number of processors
- `g` – Communication cost weight (per unit of data sent)
- `L` – Synchronisation cost weight (per superstep)
-  `mem_type` *(optional)* – Memory constraint type:
    - `0` = **NONE** – No memory constraint.
    - `1` = **LOCAL** – Each processor has its own local memory bound per superstep.
    - `2` = **GLOBAL** – Each processor has a total memory bound for all its assigned tasks.
    - `3` = **PERSISTENT\_AND\_TRANSIENT** – Bound applies to persistent memory + the largest transient buffer.
- `M` *(optional, required if **`mem_type > 0`**)* – Memory bound (integer).

If `mem_type` is not given, it defaults to `NONE`.

---

## **2. NUMA Matrix**

After the header, list **p² lines**, each with:

```
fromProc toProc value
```

Where:

- `fromProc`, `toProc` – Processor indices `[0 … p-1]`
- `value` – Communication cost between processors (integer)
- **Diagonal entries** (`fromProc == toProc`) **must be 0**

This matrix describes the relative cost of communication between any two processors. If all off-diagonal values are the same, the architecture is **uniform**; otherwise, it's **NUMA** (Non-Uniform Memory Access).

---

## **Memory Constraint Types Explained**

| mem\_type | Name                           | Description                                                                                                                                    |
| --------- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `0`       | **NONE**                       | No memory constraint is applied; scheduling ignores memory limits.                                                                             |
| `1`       | **LOCAL**                      | Each processor has a local memory bound; only data for tasks in the current superstep must fit.                                            |
| `2`       | **GLOBAL**                     | Each processor has a total memory bound for all assigned tasks across the whole schedule.                                                  |
| `3`       | **PERSISTENT\_AND\_TRANSIENT** | Memory bound applies to persistent storage (sum of all task memory) plus the largest single communication buffer needed at any moment. |

---

## **Example File** 

```
% BSP Data
3 3 5
% NUMA Data
0 0 0
0 1 1
0 2 1
1 0 1
1 1 0
1 2 1
2 0 1
2 1 1
2 2 0
```

**Explanation:**

- **3 processors** (`p = 3`)
- **Communication cost** `g = 3`
- **Synchronisation cost** `L = 5`
- No memory constraint specified → defaults to `NONE`