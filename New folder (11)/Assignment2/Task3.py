import sys
import time
import math
import heapq

from rtree import RTree, Node, Point, load_points, load_sample_file


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# which dataset to use: (sum of last digits of the 3 student IDs) mod 3
DATASET = "city1"
B = 16  # max node capacity used for the real dataset
DATA_DIR = "datasets/Task3_Datasets/"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Method 1: Sequential scan
# ---------------------------------------------------------------------------
def sequential_skyline_scan(points):
    """Check every point and return the skyline points."""
    skylines = []

    for p in points:
        dominated = False

        for s in skylines:
            if s.x <= p.x and s.y <= p.y:
                dominated = True
                break

        if dominated:
            continue

        skylines = [
            s for s in skylines
            if not (p.x <= s.x and p.y <= s.y)
        ]

        skylines.append(p)

    return skylines
# ---------------------------------------------------------------------------
# Method 2: Best-First search (Week 9 pseudocode)
# ---------------------------------------------------------------------------
def bbs_skyline(tree):
    """Build an R-tree and find skyline points using BBS."""

    skylines = []
    heap = []
    counter = 0

    # 1. Insert root MBR into H, sorted by mindist to origin
    heapq.heappush(
        heap,
        (tree.root.mbr.x1 + tree.root.mbr.y1, counter, tree.root)
    )
    counter += 1

    # 2. SKY = empty

    while heap:
        # 4. Remove first entry e from H
        _, _, entry = heapq.heappop(heap)

        # 5. If e is the MBR of a node u
        if isinstance(entry, Node):
            # 6. If no skyline point dominates the lower-left corner
            mbr = entry.mbr

            if not any(s.x <= mbr.x1 and s.y <= mbr.y1
                       for s in skylines):

                # 7. Visit u and insert its children into H
                for child in entry.entries:
                    child_mbr = child.mbr

                    heapq.heappush(
                        heap,
                        (child_mbr.x1 + child_mbr.y1, counter, child)
                    )
                    counter += 1

        else:
            # 8–9. e is a point; check whether any skyline point dominates it
            if not any(s.x <= entry.x and s.y <= entry.y
                       for s in skylines):

                # 10. Add e to SKY
                skylines.append(entry)

    return skylines

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



# ---------------------------------------------------------------------------
# Running everything
# ---------------------------------------------------------------------------


def time_algorithm(name, algorithm, *args, **kwargs):
    start = time.perf_counter()
    result = algorithm(*args, **kwargs)
    elapsed = time.perf_counter() - start

    print(f"{name:<30} {elapsed:.6f} seconds")
    return result, elapsed

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

    points = load_points(DATA_DIR + dataset + ".txt")
    print(f"Dataset: {dataset} ({len(points)} points), B={node_cap}")

    t0 = time.perf_counter()
    tree = RTree(B=node_cap).build(points)
    print(f"R-tree built in {time.perf_counter() - t0:.1f}s (height {tree.height()})")

    sequential, seq_time = time_algorithm(
        "Sequential Scan",
        sequential_skyline_scan,
        points
    )

    bbs, bbs_time = time_algorithm(
        "BBS",
        bbs_skyline,
        tree
    )
    print(sequential)
    print(bbs)
    summary = [
        "===== Summary =====",
        f"Dataset: {dataset}, B={node_cap}",
        f"Sequential Scan : {seq_time:.4f} s",
        f"BBS              : {bbs_time:.4f} s  ({seq_time / bbs_time:.1f}x faster than sequential)",
    ]
    print(summary)
'''
    # print everything and save it to one txt file
    out_file = f"Task3_output_{dataset}.txt"
    with open(out_file, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\nSaved to {out_file}")
'''

if __name__ == "__main__":
    main()
