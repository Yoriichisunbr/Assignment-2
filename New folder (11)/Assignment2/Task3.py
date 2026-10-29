import sys
import time
import heapq

from rtree import RTree, Node, Point, load_points, load_sample_file

#DATASET = "Datasets/Task3_Datasets/city1"
DATASET = "Datasets/Sample_data/skyline_search"
B = 5


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


def bbs_skyline_verbose(tree):

    skylines = []
    heap = []
    counter = 0
    step = 0

    def log(message):
        nonlocal step
        step += 1
        print(f"\n[{step}] {message}")

    def heap_contents():
        return [
            (priority, entry.mbr if isinstance(entry, Node)
             else (entry.x, entry.y))
            for priority, _, entry in sorted(heap)
        ]

    def skyline_contents():
        return [(p.x, p.y) for p in skylines]

    # 1. Insert root MBR into H, sorted by mindist to origin
    heapq.heappush(
        heap,
        (tree.root.mbr.x1 + tree.root.mbr.y1, counter, tree.root)
    )
    counter += 1

    log(f"Initialise H with root MBR: {tree.root.mbr}")
    print("H:", heap_contents())
    print("SKY:", skyline_contents())

    # 2. SKY = empty

    while heap:
        # 4. Remove first entry e from H
        _, _, entry = heapq.heappop(heap)

        log(f"Remove entry from H: {entry}")
        print("H after removal:", heap_contents())
        print("SKY:", skyline_contents())

        # 5. If e is the MBR of a node u
        if isinstance(entry, Node):
            # 6. If no skyline point dominates the lower-left corner
            mbr = entry.mbr
            corner = Point(0, mbr.x1, mbr.y1)

            dominators = [
                s for s in skylines if dominates(s, corner)
            ]

            log(f"Check node MBR: {mbr}")
            print(f"Lower-left corner: ({mbr.x1}, {mbr.y1})")
            print("Dominating skyline points:",
                  [(p.x, p.y) for p in dominators])

            if not dominators:

                # 7. Visit u and insert its children into H
                log(f"EXPAND node: {mbr}")

                for child in entry.entries:
                    child_mbr = child.mbr
                    priority = child_mbr.x1 + child_mbr.y1

                    heapq.heappush(
                        heap,
                        (priority, counter, child)
                    )
                    counter += 1

                    log(f"INSERT child into H: {child}")
                    print("Priority:", priority)
                    print("H:", heap_contents())

            else:
                log(f"PRUNE node: {mbr} is dominated")

        else:
            # 8–9. e is a point; check whether any skyline point dominates it
            dominators = [
                s for s in skylines if dominates(s, entry)
            ]

            log(f"CHECK point: ({entry.x}, {entry.y})")
            print("Dominating skyline points:",
                  [(p.x, p.y) for p in dominators])

            if not dominators:

                # 10. Add e to SKY
                skylines.append(entry)

                log(f"ADD point to SKY: ({entry.x}, {entry.y})")
                print("SKY:", skyline_contents())

            else:
                log(f"DISCARD point: ({entry.x}, {entry.y}) is dominated")

    log("SEARCH COMPLETE")
    print("Final skyline:", skyline_contents())

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
    left_candidates = bbs_skyline(trees[0])
    right_candidates = bbs_skyline(trees[1])

    right_candidates.sort(key=lambda p: p.x)

    skylines = left_candidates.copy()
    min_y = min(p.y for p in left_candidates)

    for point in right_candidates:
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

    points = load_points(DATASET + ".txt")
    print(f"Dataset: {DATASET} ({len(points)} points), B={B}")

    t0 = time.perf_counter()
    tree = RTree(B,None,True).build(points)
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
        bbs_skyline_verbose,
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

        sequential.sort(key=lambda p: (p.x, p.y))
        bbs.sort(key=lambda p: (p.x, p.y))
        dc.sort(key=lambda p: (p.x, p.y))

        output("\nSorted skylines:")
        output(f"seq: {sequential}")
        output(f"bbs: {bbs}")
        output(f"dc:  {dc}")

        output("\nTimes:")
        output(f"seq time: {seq_time:.6f} seconds")
        output(f"bbs time: {bbs_time:.6f} seconds")
        output(f"dc time: {dc_time:.6f} seconds")




if __name__ == "__main__":
    main()
