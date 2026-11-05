import sys
from rtree import RTree, Point

BASE_DIR = sys.path[0]
def load_points(filename):
    points = []

    with open(filename, "r") as file:
        for line in file:
            values = line.strip().split()

            if len(values) == 3:
                point_id = int(values[0])
                x = float(values[1])
                y = float(values[2])

                points.append((point_id, x, y))

    return points

shops = load_points(BASE_DIR + "/shop.txt")
queries = load_points(BASE_DIR + "/query_points_BaB.txt")


print("Total shops:", len(shops))
print("Total queries:", len(queries))
print("First shop:", shops[0])
print("First query:", queries[0])


import math
import time

def sequential_scan(query, shops):
    nearest_shop = None
    min_distance = float("inf")

    for shop in shops:
        distance = math.sqrt(
            (query[1] - shop[1]) ** 2 +
            (query[2] - shop[2]) ** 2
        )

        if distance < min_distance:
            min_distance = distance
            nearest_shop = shop

    return nearest_shop, min_distance



# Run Sequential Scan for all 200 queries
print("\nSEQUENTIAL SCAN RESULTS")

start_time = time.perf_counter()

results = []

for query in queries:
    nearest, distance = sequential_scan(query, shops)
    results.append((query[0], nearest, distance))

end_time = time.perf_counter()

running_time = end_time - start_time

# Display and save results
with open("task1_sequential_results.txt", "w") as file:
    for query_id, nearest, distance in results:
        output = (
            f"Query {query_id}: "
            f"id={nearest[0]}, "
            f"x={nearest[1]}, "
            f"y={nearest[2]}, "
            f"distance={distance:.4f}"
        )
        print(output)
        file.write(output + "\n")

    file.write(
        f"\nTotal running time: "
        f"{running_time:.6f} seconds\n"
    )

print(f"\nTotal running time: {running_time:.6f} seconds")
print("Total queries processed:", len(results))



# TEST SHARED R-TREE

test_points = [
    Point(shop[0], shop[1], shop[2])
    for shop in shops[:10]
]

tree = RTree(B=4)
tree.build(test_points)

print("\nR-TREE TEST RESULTS")
print("R-tree points:", tree.check())
print("R-tree height:", tree.height())


import heapq

def bab_nearest(query, tree):
    qx, qy = query[1], query[2]

    best_point = None
    best_distance = float("inf")

    # Priority queue: (MINDIST, unique number, node or point)
    queue = []
    counter = 0

    heapq.heappush(queue, (0.0, counter, tree.root))

    while queue:
        min_dist, _, entry = heapq.heappop(queue)

        # Prune entries that cannot improve the best result
        if min_dist > best_distance:
            break

        if isinstance(entry, Point):
            if min_dist < best_distance:
                best_distance = min_dist
                best_point = entry

        elif entry.is_leaf:
            for point in entry.entries:
                distance = math.hypot(qx - point.x, qy - point.y)
                if distance <= best_distance:
                    counter += 1
                    heapq.heappush(
                        queue, (distance, counter, point)
                    )

        else:
            for child in entry.entries:
                distance = child.mbr.min_dist(qx, qy)
                if distance <= best_distance:
                    counter += 1
                    heapq.heappush(
                        queue, (distance, counter, child)
                    )

    return best_point, best_distance


print("\nBRANCH-AND-BOUND TEST")

# Build R-tree using all shops
all_points = [
    Point(shop[0], shop[1], shop[2])
    for shop in shops
]

full_tree = RTree(B=4)
full_tree.build(all_points)

print("R-tree built with", full_tree.size, "shops")

for query in queries[:5]:
    nearest, distance = bab_nearest(query, full_tree)

    print(
        f"Query {query[0]}: "
        f"id={nearest.id}, "
        f"x={nearest.x}, "
        f"y={nearest.y}, "
        f"distance={distance:.4f}"
    )


# Run Branch-and-Bound for all 200 queries
print("\nBRANCH-AND-BOUND RESULTS")

bab_results = []

start_time = time.perf_counter()

for query in queries:
    nearest, distance = bab_nearest(query, full_tree)
    bab_results.append((query[0], nearest, distance))

end_time = time.perf_counter()
bab_running_time = end_time - start_time

# Display results after timing has finished
for query_id, nearest, distance in bab_results:
    print(
        f"Query {query_id}: "
        f"id={nearest.id}, "
        f"x={nearest.x}, "
        f"y={nearest.y}, "
        f"distance={distance:.4f}"
    )

print(f"\nBaB running time: {bab_running_time:.6f} seconds")
print("BaB queries processed:", len(bab_results))



# METHOD 3: DIVIDE-AND-CONQUER

# Find dataset boundaries
min_x = min(shop[1] for shop in shops)
max_x = max(shop[1] for shop in shops)
min_y = min(shop[2] for shop in shops)
max_y = max(shop[2] for shop in shops)

mid_x = (min_x + max_x) / 2
mid_y = (min_y + max_y) / 2

print("\nDIVIDE-AND-CONQUER PARTITIONS")
print("X midpoint:", mid_x)
print("Y midpoint:", mid_y)

# Divide shops into four quadrants
quadrants = [[], [], [], []]

for shop in shops:
    point = Point(shop[0], shop[1], shop[2])

    if point.x < mid_x and point.y < mid_y:
        quadrants[0].append(point)
    elif point.x < mid_x and point.y >= mid_y:
        quadrants[1].append(point)
    elif point.x >= mid_x and point.y < mid_y:
        quadrants[2].append(point)
    else:
        quadrants[3].append(point)

for i, quadrant in enumerate(quadrants):
    print(f"Quadrant {i + 1}: {len(quadrant)} shops")

print("Total shops partitioned:", sum(len(q) for q in quadrants))


# Build one R-tree for each quadrant
print("\nBUILDING FOUR R-TREES")

quadrant_trees = []

for i, quadrant in enumerate(quadrants):
    tree = RTree(B=4)
    tree.build(quadrant)
    quadrant_trees.append(tree)

    print(
        f"Quadrant {i + 1}: "
        f"{tree.size} points, "
        f"height = {tree.height()}"
    )

print("All four R-trees built successfully!")


# METHOD 3: DIVIDE-AND-CONQUER SEARCH

def divide_and_conquer_nearest(query, trees):
    best_point = None
    best_distance = float("inf")

    # Search each of the four R-trees
    for tree in trees:
        if tree.size == 0:
            continue

        nearest, distance = bab_nearest(query, tree)

        if nearest is not None and distance < best_distance:
            best_point = nearest
            best_distance = distance

    return best_point, best_distance


# Run all 200 queries
print("\nDIVIDE-AND-CONQUER RESULTS")

dc_results = []

start_time = time.perf_counter()

for query in queries:
    nearest, distance = divide_and_conquer_nearest(
        query, quadrant_trees
    )
    dc_results.append((query[0], nearest, distance))

end_time = time.perf_counter()
dc_running_time = end_time - start_time

# Print results outside the timed section
for query_id, nearest, distance in dc_results:
    print(
        f"Query {query_id}: "
        f"id={nearest.id}, "
        f"x={nearest.x}, "
        f"y={nearest.y}, "
        f"distance={distance:.4f}"
    )

print(f"\nDivide-and-Conquer running time: {dc_running_time:.6f} seconds")
print("Divide-and-Conquer queries processed:", len(dc_results))



# VERIFY ALL THREE METHODS

print("\nVERIFYING ALL THREE METHODS")

mismatches = 0

for query in queries:
    seq_point, seq_distance = sequential_scan(query, shops)
    bab_point, bab_distance = bab_nearest(query, full_tree)
    dc_point, dc_distance = divide_and_conquer_nearest(
        query, quadrant_trees
    )

    if not (
        math.isclose(seq_distance, bab_distance, abs_tol=1e-9)
        and math.isclose(seq_distance, dc_distance, abs_tol=1e-9)
    ):
        mismatches += 1
        print(f"Mismatch for query {query[0]}")

print("Total queries verified:", len(queries))
print("Distance mismatches:", mismatches)

if mismatches == 0:
    print("SUCCESS: All three methods agree!")



# FINAL COMBINED OUTPUT FILE

output_file = BASE_DIR + "/Task1_Final_Results.txt"

with open(output_file, "w") as file:

    file.write("TASK 1: NEAREST NEIGHBOR SEARCH\n")
    file.write("Dataset: Shop\n")
    file.write("Total shops: 150000\n")
    file.write("Total queries: 200\n\n")

    # METHOD 1
    file.write("=" * 50 + "\n")
    file.write("METHOD 1: SEQUENTIAL SCAN\n")
    file.write("=" * 50 + "\n")

    for query_id, nearest, distance in results:
        file.write(
            f"Query {query_id}: "
            f"id={nearest[0]}, "
            f"x={nearest[1]}, "
            f"y={nearest[2]}, "
            f"distance={distance:.4f}\n"
        )

    file.write(
        f"Running time: {running_time:.6f} seconds\n\n"
    )

    # METHOD 2
    file.write("=" * 50 + "\n")
    file.write("METHOD 2: BRANCH-AND-BOUND\n")
    file.write("=" * 50 + "\n")

    for query_id, nearest, distance in bab_results:
        file.write(
            f"Query {query_id}: "
            f"id={nearest.id}, "
            f"x={nearest.x}, "
            f"y={nearest.y}, "
            f"distance={distance:.4f}\n"
        )

    file.write(
        f"Running time: {bab_running_time:.6f} seconds\n\n"
    )

    # METHOD 3
    file.write("=" * 50 + "\n")
    file.write("METHOD 3: DIVIDE-AND-CONQUER\n")
    file.write("=" * 50 + "\n")

    for query_id, nearest, distance in dc_results:
        file.write(
            f"Query {query_id}: "
            f"id={nearest.id}, "
            f"x={nearest.x}, "
            f"y={nearest.y}, "
            f"distance={distance:.4f}\n"
        )

    file.write(
        f"Running time: {dc_running_time:.6f} seconds\n"
    )

print("\nFINAL OUTPUT FILE CREATED:")
print(output_file)

