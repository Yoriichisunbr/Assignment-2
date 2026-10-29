import sys
import time
import heapq

from rtree import RTree, Node, Point, load_points, load_sample_file

DATASET = "city1"
B = 5  # max node capacity used for the real DATASET
DATA_DIR = "DATASETs/Task3_Datasets/"


def dominates(p1, p2):
    return (p1.x <= p2.x and p1.y <= p2.y
            and (p1.x < p2.x or p1.y < p2.y))


#3.1


def sequential_skyline(points):
    skylines = []

    for p in points:
        dominated = False

        for s in skylines:
            if dominates(s, p):                
                dominated = True
                break

        if dominated:
            continue

        skylines = [
            s for s in skylines
            if not dominates(p, s)        
            ]

        skylines.append(p)

    return skylines



#3.2


def bbs_skyline(tree):

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

            if not any(dominates(s, Point(0,mbr.x1,mbr.y1))
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
            if not any(dominates(s, entry)
                       for s in skylines):

                # 10. Add e to SKY
                skylines.append(entry)

    return skylines





#3.3


def split(points, num_bins):
    points = sorted(points, key=lambda p: p.x)
    size = (len(points) + num_bins - 1) // num_bins
    return [points[i:i + size] for i in range(0, len(points), size)]

def make_dc_trees(points, num_bins):
    bins = split(points, num_bins)
    trees = []
    for point_bin in bins:
        trees.append(RTree(B).build(point_bin))
    return trees

def dc_skyline(trees):
    candidates = []
    for tree in trees:
        candidates.extend(bbs_skyline(tree))

    candidates.sort(key=lambda p: p.x)

    skylines = []
    min_y = float("inf")

    for point in candidates:
        if point.y < min_y:
            skylines.append(point)
            min_y = point.y

    return skylines


def time_algorithm(name, algorithm, *args, **kwargs):
    start = time.perf_counter()
    result = algorithm(*args, **kwargs)
    elapsed = time.perf_counter() - start

    return result, elapsed

def main():

    points = load_points(DATA_DIR + DATASET + ".txt")
    print(f"Dataset: {DATASET} ({len(points)} points), B={B}")

    t0 = time.perf_counter()
    tree = RTree(B).build(points)
    print(f"R-tree built in {time.perf_counter() - t0:.1f}s (height {tree.height()})")
    t0 = time.perf_counter()
    dc_trees = make_dc_trees(points,2)
    print(f"DC R-trees built in {time.perf_counter() - t0:.1f}s (height {dc_trees[0].height()})")

    sequential, seq_time = time_algorithm(
        "Sequential Scan",
        sequential_skyline,
        points
    )

    bbs, bbs_time = time_algorithm(
        "BBS",
        bbs_skyline,
        tree
    )

    dc, dc_time = time_algorithm(
        "DC",
        dc_skyline,
        dc_trees
    )

    with open("Task3_output.txt", "w") as f:
        def output(text=""):
            print(text)
            print(text, file=f)

        output("Skylines:")
        output(f"seq: {sequential}")
        output(f"bbs: {bbs}")
        output(f"dc:  {dc}")

        sequential.sort(key=lambda p: (p.x, p.y))
        bbs.sort(key=lambda p: (p.x, p.y))
        dc.sort(key=lambda p: (p.x, p.y))

        output("\nSorted:")
        output(f"seq: {sequential}")
        output(f"bbs: {bbs}")
        output(f"dc:  {dc}")

        output("\nTimes:")
        output(f"seq time: {seq_time:.6f} seconds")
        output(f"bbs time: {bbs_time:.6f} seconds")
        output(f"dc time: {dc_time:.6f} seconds")




if __name__ == "__main__":
    main()
