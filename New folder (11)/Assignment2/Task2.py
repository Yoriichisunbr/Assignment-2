"""
Task2.py - Nearest Neighbour Search using Best-First (BF)
COMP3210/6210 Assignment 2, Individual Task 2

Three methods are compared on the same 200 query points:
  1. Sequential scan  - check the distance to every single point
  2. BF               - Best-First search on one R-tree
  3. BF + D&C         - sort points by y, cut into 4 equal parts,
                        one R-tree per part, BF on each, keep the closest

How to run (from the Assignment2 folder):
    python Task2.py                      -> uses the dataset set in DATASET below
    python Task2.py library              -> pick the dataset on the command line
    python Task2.py library 16           -> dataset + node capacity B
"""

import sys
import time
import math
import heapq

from rtree import RTree, Point, load_points, load_sample_file


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# which dataset to use: (sum of last digits of the 3 student IDs) mod 3
#   0 -> hospital, 1 -> library, 2 -> ev_charging_station
DATASET = "library"
B = 16  # max node capacity used for the real dataset

DATA_DIR = "datasets/Task2_Datasets/"
QUERY_FILE = DATA_DIR + "query_points_BF.txt"
SAMPLE_FILE = "datasets/Sample_Data/NN_BF_search.txt"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def distance(p, qx, qy):
    """Euclidean distance between point p and the query location (qx, qy)."""
    dx = p.x - qx  # difference in x
    dy = p.y - qy  # difference in y
    return math.sqrt(dx * dx + dy * dy)


# ---------------------------------------------------------------------------
# Method 1: Sequential scan
# ---------------------------------------------------------------------------

def sequential_scan(points, q):
    """Check every point and return (nearest point, distance)."""
    best_point = None
    best_dist = float("inf")  # nothing found yet, so start at infinity
    # go through every single point in the dataset
    for p in points:
        d = distance(p, q.x, q.y)
        # keep this point if it is closer than the best one so far
        if d < best_dist:
            best_dist = d
            best_point = p
    return best_point, best_dist



# ---------------------------------------------------------------------------
# Method 2: Best-First search (Week 9 pseudocode)
# ---------------------------------------------------------------------------

def best_first(tree, q, trace=False):
    """Best-First NN search on one R-tree. Returns (nearest point, distance).

    H is a min-heap of (mindist, counter, node). The counter is only there so
    two entries with the same mindist never get compared as Node objects.
    trace=True prints every step (used for the sample data in the report)."""
    # an empty tree has nothing to search
    if tree.root.mbr is None:
        return None, float("inf")

    p_best = None
    best_dist = float("inf")  # ||q, p_best|| = infinity while p_best is empty
    counter = 0               # tie breaker for the heap

    # line 1: insert the MBR of the root into H
    H = [(tree.root.mbr.min_dist(q.x, q.y), counter, tree.root)]
    if trace:
        print(f"Start: H = {heap_text(H)}")

    # line 2: keep going while p_best is further than the smallest mindist in H
    while H and best_dist > H[0][0]:
        # line 3: take out the entry with the smallest mindist
        mind, _, node = heapq.heappop(H)
        if trace:
            print(f"\nPop {node.name} (mindist {mind:.2f})")

        # line 5: intermediate node -> put all its child MBRs into H
        if not node.is_leaf:
            # push every child with its mindist as the sorting key
            for child in node.entries:
                counter += 1
                heapq.heappush(H, (child.mbr.min_dist(q.x, q.y), counter, child))
            if trace:
                print(f"  internal node -> add children, H = {heap_text(H)}")
        # line 7: leaf node -> find the NN of q inside this leaf
        else:
            # compare every point in the leaf with the current best
            for p in node.entries:
                d = distance(p, q.x, q.y)
                if trace:
                    print(f"  dist(q, p{p.id}) = {d:.2f}")
                # line 8-9: closer than p_best -> it becomes the new p_best
                if d < best_dist:
                    best_dist = d
                    p_best = p
            if trace:
                print(f"  current best = p{p_best.id}, distance {best_dist:.2f}")
                print(f"  H = {heap_text(H)}")

    if trace:
        # explain why the loop stopped
        if H:
            print(f"\nStop: {best_dist:.2f} <= smallest mindist in H ({H[0][0]:.2f})")
        else:
            print("\nStop: H is empty")
        print(f"Nearest neighbour = p{p_best.id} ({p_best.x:g},{p_best.y:g}), distance {best_dist:.2f}")
    return p_best, best_dist


def heap_text(H):
    """Turn the heap into readable text sorted by mindist, e.g. {(N1, 0.00), (N2, 2.00)}."""
    items = sorted(H)  # sorted copy, the real heap is not changed
    return "{" + ", ".join(f"({n.name}, {d:.2f})" for d, _, n in items) + "}"



# ---------------------------------------------------------------------------
# Method 3: BF with Divide-and-Conquer
# ---------------------------------------------------------------------------

def split_by_y(points, parts=4):
    """Sort points by y (ascending) and cut them into `parts` groups of
    almost equal size. Returns a list of point lists."""
    ordered = sorted(points, key=lambda p: (p.y, p.x))  # sort on y, x breaks ties
    n = len(ordered)
    groups = []
    start = 0
    # work out where each group starts and ends
    for i in range(parts):
        # spread the leftover points over the first few groups
        size = n // parts + (1 if i < n % parts else 0)
        groups.append(ordered[start:start + size])
        start += size
    return groups


def build_dc_trees(points, B, parts=4, verbose=False):
    """Build one R-tree per subspace with the shared R-tree module."""
    trees = []
    # one tree for every group of points
    for i, group in enumerate(split_by_y(points, parts)):
        if verbose:
            ys = [p.y for p in group]
            print(f"\n===== Subspace S{i + 1}: y from {min(ys):g} to {max(ys):g}, "
                  f"points {[f'p{p.id}' for p in group]} =====")
        trees.append(RTree(B=B, verbose=verbose).build(group))
    return trees


def best_first_dc(trees, q, trace=False):
    """Run BF on every subspace tree and return the closest of the candidates."""
    best_point = None
    best_dist = float("inf")
    # search each of the four R-trees on its own
    for i, tree in enumerate(trees):
        if trace:
            print(f"\n===== BF on subspace S{i + 1} =====")
        p, d = best_first(tree, q, trace)
        if trace:
            print(f"Candidate from S{i + 1}: p{p.id}, distance {d:.2f}")
        # keep the candidate if it beats the best one from the other subspaces
        if d < best_dist:
            best_dist = d
            best_point = p
    if trace:
        print(f"\nFinal answer (closest candidate): p{best_point.id}, distance {best_dist:.2f}")
    return best_point, best_dist


# ---------------------------------------------------------------------------
# Running everything
# ---------------------------------------------------------------------------

def run_method(name, search, queries):
    """Run one search function on all queries. Only the search loop is timed.
    Returns (list of result lines, total time in seconds)."""
    results = []  # (query, point, distance) for every query
    start = time.perf_counter()
    # search for the NN of each query point
    for q in queries:
        p, d = search(q)
        results.append((q, p, d))
    total = time.perf_counter() - start

    # turn results into text AFTER the timer stops (formatting is not timed)
    lines = [f"===== {name} ====="]
    for q, p, d in results:
        lines.append(f"Query {q.id}: id={p.id}, x={p.x:g}, y={p.y:g}, distance={d:.2f}")
    lines.append(f"Total running time for {len(queries)} queries: {total:.4f} seconds")
    lines.append(f"Average time per query: {total / len(queries) * 1000:.4f} ms")
    lines.append("")
    return lines, total


def run_sample():
    """Step-by-step run on the 10-point sample (B=4) for the report."""
    data, queries = load_sample_file(SAMPLE_FILE)
    q = queries[0]
    print("########## R-tree construction (B=4) ##########")
    tree = RTree(B=4, verbose=True).build(data)
    print("\n########## BF search ##########")
    best_first(tree, q, trace=True)
    print("\n########## Divide-and-Conquer: split by y, B=4 ##########")
    trees = build_dc_trees(data, B=4, verbose=True)
    best_first_dc(trees, q, trace=True)


def main():
    # optional command-line arguments: dataset name and B
    dataset = sys.argv[1] if len(sys.argv) > 1 else DATASET
    node_cap = int(sys.argv[2]) if len(sys.argv) > 2 else B
    # "sample" -> just print the worked example for the report
    if dataset == "sample":
        run_sample()
        return

    # reading the files is NOT timed
    points = load_points(DATA_DIR + dataset + ".txt")
    queries = load_points(QUERY_FILE)
    print(f"Dataset: {dataset} ({len(points)} points), {len(queries)} queries, B={node_cap}")

    # building the trees is NOT timed
    t0 = time.perf_counter()
    tree = RTree(B=node_cap).build(points)
    print(f"R-tree built in {time.perf_counter() - t0:.1f}s (height {tree.height()})")
    t0 = time.perf_counter()
    dc_trees = build_dc_trees(points, node_cap)
    print(f"4 D&C R-trees built in {time.perf_counter() - t0:.1f}s")

    # run the three methods (only the searching is timed inside run_method)
    seq_lines, seq_t = run_method("Sequential Scan", lambda q: sequential_scan(points, q), queries)
    bf_lines, bf_t = run_method("Best-First (BF)", lambda q: best_first(tree, q), queries)
    dc_lines, dc_t = run_method("BF with Divide-and-Conquer", lambda q: best_first_dc(dc_trees, q), queries)

    # quick check that all three methods agree on every distance
    mismatches = 0
    for a, b, c in zip(seq_lines[1:201], bf_lines[1:201], dc_lines[1:201]):
        # compare only the distance part at the end of each line
        if not (a.split("distance=")[1] == b.split("distance=")[1] == c.split("distance=")[1]):
            mismatches += 1

    summary = [
        "===== Summary =====",
        f"Dataset: {dataset}, B={node_cap}",
        f"Sequential Scan : {seq_t:.4f} s",
        f"BF              : {bf_t:.4f} s  ({seq_t / bf_t:.1f}x faster than sequential)",
        f"BF + D&C        : {dc_t:.4f} s  ({seq_t / dc_t:.1f}x faster than sequential)",
        f"Queries where the methods disagree on distance: {mismatches}",
    ]

    # print everything and save it to one txt file
    out_lines = seq_lines + bf_lines + dc_lines + summary
    print("\n".join(out_lines))
    out_file = f"Task2_output_{dataset}.txt"
    with open(out_file, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()