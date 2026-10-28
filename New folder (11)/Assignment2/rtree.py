"""
rtree.py - Shared R-tree module for COMP3210/6210 Assignment 2

This one file is used by Task 1 (BaB), Task 2 (BF) and Task 3 (BBS).
It only does the R-tree part: nodes, MBRs, inserting points, handling
overflow, splitting nodes and fixing the MBRs/parents afterwards.
The search algorithms themselves are NOT in here, each task has its own.

Construction rules (the same for every task):
  - Choose subtree: pick the child whose MBR perimeter grows the least
    when the new point is added (Week 8 choose-subtree). Tie -> the
    smallest MBR (smaller area, then smaller perimeter). Still tied ->
    the first one in the list.
  - Overflow: a node overflows when it has more than B entries.
  - Split: sort the B+1 entries by x, and separately by y. For every
    allowed cut position (each side needs at least min_fill entries)
    work out perimeter(group1) + perimeter(group2). Use the cut with the
    smallest total. Tie -> x axis before y axis, then the smaller cut.
  - After a split the new node is added to the parent. If the parent
    overflows it is split too. If the root splits, a new root is made.

Only standard library modules are used (math, sys).
"""

import math
import sys


# ---------------------------------------------------------------------------
# Reading data files
# ---------------------------------------------------------------------------

def load_points(path):
    """Read a data file with lines 'id x y' and return a list of Points.
    Lines that are not 3 numbers (headers, blank lines) are skipped."""
    points = []  # list to collect every point we read
    with open(path, "r") as f:
        # go through the file line by line
        for line in f:
            parts = line.split()  # split on any whitespace
            # only keep lines that have exactly id, x, y
            if len(parts) != 3:
                continue
            try:
                pid = int(parts[0])      # id is a whole number
                x = float(parts[1])      # x coordinate
                y = float(parts[2])      # y coordinate
            except ValueError:
                # header line like "ID X Y" -> not numbers, skip it
                continue
            points.append(Point(pid, x, y))
    return points


def load_sample_file(path):
    """Read one of the Sample_Data files.
    These have a 'Facility Location' part and maybe a 'Query Point' part.
    Returns (data_points, query_points). query_points is [] if there are none."""
    data, queries = [], []
    current = data  # start by filling the data list
    with open(path, "r") as f:
        # read each line of the sample file
        for line in f:
            # once we reach the query section, switch to the query list
            if line.lower().startswith("query"):
                current = queries
                continue
            parts = line.split()
            # skip anything that isn't 'id x y'
            if len(parts) != 3:
                continue
            try:
                current.append(Point(int(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue  # header row
    return data, queries



# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

class Point:
    """A single data point (a facility / home). Stored in leaf nodes."""

    __slots__ = ("id", "x", "y")  # saves memory with 150k-300k points

    def __init__(self, pid, x, y):
        self.id = pid  # unique id from the file
        self.x = x     # x coordinate
        self.y = y     # y coordinate

    def get_mbr(self):
        """A point's MBR is just a rectangle with zero width and height."""
        return MBR(self.x, self.y, self.x, self.y)

    @property
    def mbr(self):
        return MBR(self.x, self.y, self.x, self.y)

    def __repr__(self):
        return f"p{self.id}({self.x:g},{self.y:g})"


class MBR:
    """Minimum Bounding Rectangle, stored as bottom-left (x1,y1)
    and top-right (x2,y2)."""

    __slots__ = ("x1", "y1", "x2", "y2")

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1  # min x
        self.y1 = y1  # min y
        self.x2 = x2  # max x
        self.y2 = y2  # max y

    def perimeter(self):
        """Perimeter = 2 * (width + height). Used for choosing subtrees
        and for splitting (works even when width or height is 0)."""
        return 2 * ((self.x2 - self.x1) + (self.y2 - self.y1))

    def area(self):
        """Width * height. Used to break ties in choose-subtree."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def union(self, other):
        """Return a NEW MBR that covers both this MBR and the other one."""
        return MBR(min(self.x1, other.x1), min(self.y1, other.y1),
                   max(self.x2, other.x2), max(self.y2, other.y2))

    def perimeter_increase(self, x, y):
        """How much the perimeter would grow if point (x,y) was added."""
        # new box after stretching to include the point
        nx1 = min(self.x1, x)
        ny1 = min(self.y1, y)
        nx2 = max(self.x2, x)
        ny2 = max(self.y2, y)
        new_perim = 2 * ((nx2 - nx1) + (ny2 - ny1))
        return new_perim - self.perimeter()

    def min_dist(self, x, y):
        """Smallest Euclidean distance from point (x,y) to this rectangle.
        0 if the point is inside. (Geometry helper the NN tasks can use.)"""
        # gap along x: 0 if x is between x1 and x2
        if x < self.x1:
            dx = self.x1 - x
        elif x > self.x2:
            dx = x - self.x2
        else:
            dx = 0.0
        # gap along y: same idea
        if y < self.y1:
            dy = self.y1 - y
        elif y > self.y2:
            dy = y - self.y2
        else:
            dy = 0.0
        return math.sqrt(dx * dx + dy * dy)

    def __repr__(self):
        return f"[({self.x1:g},{self.y1:g}),({self.x2:g},{self.y2:g})]"


def mbr_of_entries(entries, is_leaf):
    """Work out the MBR that covers a list of entries.
    Leaf entries are Points, internal entries are Nodes."""
    # leaf: read coordinates straight from the points
    if is_leaf:
        xs = [p.x for p in entries]
        ys = [p.y for p in entries]
        return MBR(min(xs), min(ys), max(xs), max(ys))
    # internal: combine the children's MBRs
    return MBR(min(c.mbr.x1 for c in entries), min(c.mbr.y1 for c in entries),
               max(c.mbr.x2 for c in entries), max(c.mbr.y2 for c in entries))


class Node:
    """One R-tree node.
    Leaf node     -> entries is a list of Point objects.
    Internal node -> entries is a list of child Node objects."""

    _counter = 0  # used to give nodes readable names (N1, N2, ...)

    def __init__(self, is_leaf, parent=None):
        Node._counter += 1
        self.name = f"N{Node._counter}"  # name for printing / report diagrams
        self.is_leaf = is_leaf           # True if this node stores points
        self.entries = []                # points (leaf) or child nodes
        self.parent = parent             # parent node, None for the root
        self.mbr = None                  # MBR covering all entries

    def update_mbr(self):
        """Recalculate this node's MBR from its entries."""
        # empty node has no MBR yet
        if not self.entries:
            self.mbr = None
        else:
            self.mbr = mbr_of_entries(self.entries, self.is_leaf)

    def __repr__(self):
        kind = "Leaf" if self.is_leaf else "Internal"
        return f"{self.name}({kind}, {len(self.entries)} entries, MBR={self.mbr})"

    # ---------------------------------------------------------------------------
    # The R-tree
    # ---------------------------------------------------------------------------

class RTree:
        """R-tree for 2D points with a configurable max node capacity B."""

        def __init__(self, B=4, min_fill=None, verbose=False):
            # B must be at least 2 or a split makes no sense
            if B < 2:
                raise ValueError("B (max node capacity) must be at least 2")
            self.B = B  # max entries allowed in one node
            # min entries per node after a split, default is 40% of B (rounded up)
            if min_fill is None:
                min_fill = max(1, math.ceil(0.4 * B))
            self.min_fill = min_fill
            self.verbose = verbose  # True -> print every step (for the report)
            self.size = 0  # how many points are in the tree
            Node._counter = 0  # restart node names for each new tree
            self.root = Node(is_leaf=True)  # start with one empty leaf as root

        # -- logging ---------------------------------------------------------
        def _log(self, msg):
            """Print a message only when verbose mode is on."""
            if self.verbose:
                print(msg)

        # -- building --------------------------------------------------------
        def build(self, points):
            """Insert every point one by one (same order as the file)."""
            # loop over all the points and insert each one
            for p in points:
                self.insert(p)
            return self

        def insert(self, point):
            """Insert one point into the tree."""
            self._log(f"\n--- Insert {point} ---")
            leaf = self._choose_leaf(point)  # 1. find the best leaf
            leaf.entries.append(point)  # 2. put the point in it
            self.size += 1
            self._log(f"  placed in {leaf.name}")
            # 3. if the leaf now has too many entries, deal with the overflow
            if len(leaf.entries) > self.B:
                self._handle_overflow(leaf)
            else:
                # no overflow, just stretch the MBRs on the way up
                self._expand_upwards(leaf, point.x, point.y)
            if self.verbose:
                self.print_tree()

        def _choose_leaf(self, point):
            """Go down from the root, at each level picking the child whose
            perimeter grows the least. Returns the leaf we end up at."""
            node = self.root
            # keep going down until we hit a leaf
            while not node.is_leaf:
                best_child = None
                best_inc = None
                best_area = None
                best_perim = None
                # check every child of this node
                for child in node.entries:
                    inc = child.mbr.perimeter_increase(point.x, point.y)
                    area = child.mbr.area()
                    perim = child.mbr.perimeter()
                    # pick smaller increase; if the increase is the same,
                    # favour the smaller MBR (area first, then perimeter)
                    if (best_child is None or inc < best_inc or
                            (inc == best_inc and (area, perim) < (best_area, best_perim))):
                        best_child = child
                        best_inc = inc
                        best_area = area
                        best_perim = perim
                self._log(f"  choose subtree: {node.name} -> {best_child.name} "
                          f"(perimeter increase {best_inc:g})")
                node = best_child
            return node

        def _adjust_upwards(self, node):
            """Update the MBR of this node and every ancestor up to the root."""
            # walk up the parent links
            while node is not None:
                node.update_mbr()
                node = node.parent


        def _expand_upwards(self, node, x, y):
            while node is not None:
                if node.mbr is None:
                    node.mbr = MBR(x, y, x, y)
                else:
                    mbr = node.mbr
                    if x < mbr.x1:
                        mbr.x1 = x
                    if y < mbr.y1:
                        mbr.y1 = y
                    if x > mbr.x2:
                        mbr.x2 = x
                    if y > mbr.y2:
                        mbr.y2 = y

                node = node.parent

        def _handle_overflow(self, node):
            """Split an overflowing node. If that makes the parent overflow,
            keep splitting up the tree. Makes a new root if the root splits."""
            # keep going while the current node is over capacity
            while node is not None and len(node.entries) > self.B:
                self._log(f"  OVERFLOW at {node.name}: {len(node.entries)} entries > B={self.B}")
                new_node = self._split(node)  # node keeps group 1, new_node gets group 2

                # case 1: we just split the root -> grow the tree by one level
                if node.parent is None:
                    new_root = Node(is_leaf=False)
                    new_root.entries = [node, new_node]
                    node.parent = new_root
                    new_node.parent = new_root
                    new_root.update_mbr()
                    self.root = new_root
                    self._log(f"  root was split -> new root {new_root.name} "
                              f"with children {node.name}, {new_node.name}")
                    return

                # case 2: normal node -> put the new sibling into the parent
                parent = node.parent
                new_node.parent = parent
                parent.entries.append(new_node)
                self._log(f"  {new_node.name} added to parent {parent.name} "
                          f"(now {len(parent.entries)} entries)")
                # move up and check whether the parent overflows now
                node = parent

            # whatever node we stopped at is fine, so fix MBRs from there up
            self._adjust_upwards(node)

        def _split(self, node):
            entries = node.entries
            n = len(entries)
            m = self.min_fill

            if node.is_leaf:
                def keys_x(e):
                    return (e.x, e.y)

                def keys_y(e):
                    return (e.y, e.x)
            else:
                def keys_x(e):
                    return (e.mbr.x1, e.mbr.x2)

                def keys_y(e):
                    return (e.mbr.y1, e.mbr.y2)

            best = None

            for axis_name, key in (("x", keys_x), ("y", keys_y)):
                ordered = sorted(entries, key=key)

                prefix = []
                suffix = []

                x1 = y1 = float("inf")
                x2 = y2 = float("-inf")

                for e in ordered:
                    if node.is_leaf:
                        ex1 = ex2 = e.x
                        ey1 = ey2 = e.y
                    else:
                        ex1, ey1 = e.mbr.x1, e.mbr.y1
                        ex2, ey2 = e.mbr.x2, e.mbr.y2

                    x1 = min(x1, ex1)
                    y1 = min(y1, ey1)
                    x2 = max(x2, ex2)
                    y2 = max(y2, ey2)

                    prefix.append(2 * ((x2 - x1) + (y2 - y1)))

                x1 = y1 = float("inf")
                x2 = y2 = float("-inf")

                for e in reversed(ordered):
                    if node.is_leaf:
                        ex1 = ex2 = e.x
                        ey1 = ey2 = e.y
                    else:
                        ex1, ey1 = e.mbr.x1, e.mbr.y1
                        ex2, ey2 = e.mbr.x2, e.mbr.y2

                    x1 = min(x1, ex1)
                    y1 = min(y1, ey1)
                    x2 = max(x2, ex2)
                    y2 = max(y2, ey2)

                    suffix.append(2 * ((x2 - x1) + (y2 - y1)))

                suffix.reverse()

                for k in range(m, n - m + 1):
                    cost = prefix[k - 1] + suffix[k]

                    if best is None or cost < best[0]:
                        best = (cost, axis_name, k, ordered)

            cost, axis_name, k, ordered = best
            group1 = ordered[:k]
            group2 = ordered[k:]

            node.entries = group1
            node.update_mbr()

            new_node = Node(is_leaf=node.is_leaf)
            new_node.entries = group2

            if not node.is_leaf:
                for child in group2:
                    child.parent = new_node

            new_node.update_mbr()

            self._log(f"  SPLIT {node.name} on {axis_name}-axis at position {k} "
                      f"(total perimeter {cost:g})")
            self._log(f"    {node.name}: {self._entry_names(group1)}  MBR={node.mbr}")
            self._log(f"    {new_node.name}: {self._entry_names(group2)}  MBR={new_node.mbr}")

            return new_node
    
       
        # -- helpers for printing / checking ------------------------------------
        @staticmethod
        def _entry_names(entries):
            """Short text list of entries, e.g. 'p1 p4 p7' or 'N2 N3'."""
            return " ".join(f"p{e.id}" if isinstance(e, Point) else e.name for e in entries)

        def height(self):
            """Number of levels (a tree with only a root leaf has height 1)."""
            h = 1
            node = self.root
            # follow the first child down to a leaf
            while not node.is_leaf:
                node = node.entries[0]
                h += 1
            return h

        def print_tree(self, node=None, depth=0, file=sys.stdout):
            """Print the tree with indentation, one node per line."""
            # start from the root on the first call
            if node is None:
                node = self.root
            indent = "    " * depth
            print(f"{indent}{node.name} {'[leaf]' if node.is_leaf else '[internal]'} "
                  f"MBR={node.mbr}: {self._entry_names(node.entries)}", file=file)
            # internal node -> print its children one level deeper
            if not node.is_leaf:
                for child in node.entries:
                    self.print_tree(child, depth + 1, file)

        def check(self):
            """Sanity check used while testing: every MBR is tight, no node
            is over capacity, parent links are right and all leaves are at the
            same depth. Returns the number of points found."""
            leaf_depths = set()
            count = 0
            stack = [(self.root, 1)]
            # depth-first walk over every node
            while stack:
                node, d = stack.pop()
                assert len(node.entries) <= self.B, f"{node.name} over capacity"
                # MBR stored must equal the MBR worked out from the entries
                expect = mbr_of_entries(node.entries, node.is_leaf)
                assert (node.mbr.x1, node.mbr.y1, node.mbr.x2, node.mbr.y2) == \
                       (expect.x1, expect.y1, expect.x2, expect.y2), f"{node.name} MBR wrong"
                if node.is_leaf:
                    leaf_depths.add(d)
                    count += len(node.entries)
                else:
                    # every child must point back to this node
                    for c in node.entries:
                        assert c.parent is node, f"{c.name} has wrong parent"
                        stack.append((c, d + 1))
            assert len(leaf_depths) == 1, "leaves are at different depths"
            assert count == self.size, "point count mismatch"
            return count
