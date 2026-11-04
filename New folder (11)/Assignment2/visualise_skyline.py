import argparse
import heapq
from pathlib import Path
from types import MethodType

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from rtree import RTree, Node, Point, load_sample_file


# Standard minimisation skyline dominance: smaller/equal in both dimensions.
def dominates(a, b):
    return a.x <= b.x and a.y <= b.y and (a.x < b.x or a.y < b.y)


def point_label(point):
    return f"p{point.id}"


def node_label(node):
    return f"{node.name}\n{node.mbr}"


def tree_nodes(tree):
    nodes = []
    stack = [(tree.root, 0)]

    while stack:
        node, depth = stack.pop()
        nodes.append((node, depth))
        if not node.is_leaf:
            stack.extend((child, depth + 1) for child in reversed(node.entries))

    return nodes


def draw_spatial_state(
        points, tree=None, title="", output_path=None,
        highlighted_point=None, skyline=None, pruned_nodes=None,
        examined_entry=None, partitions=None, viewport_points=None,
        discarded_entry=None, dominating_point=None,
        heap_entries=None, event_text=None,
        discarded_points=None, discarded_nodes=None,
        extra_trees=None, active_side=None, side_skylines=None,
        screening_status=None, screening_min_y=None):

    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.25], wspace=0.12)
    ax = fig.add_subplot(grid[0, 0])
    heap_ax = fig.add_subplot(grid[0, 1])

    viewport_points = viewport_points if viewport_points is not None else points
    xs = [p.x for p in viewport_points]
    ys = [p.y for p in viewport_points]

    if xs and ys:
        x_span = max(xs) - min(xs) or 1
        y_span = max(ys) - min(ys) or 1
        pad_x = max(1.2, x_span * 0.12)
        pad_y = max(1.2, y_span * 0.12)
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)

    if partitions:
        for x, label in partitions:
            ax.axvline(x, linestyle="--", linewidth=1.5, color="grey")
            ax.text(
                x, ax.get_ylim()[1] - 0.1, label,
                ha="left", va="top", fontsize=9
            )

    all_trees = []
    if tree is not None:
        all_trees.append(("main", tree))
    if extra_trees:
        all_trees.extend(extra_trees)

    nodes = []
    for side, current_tree in all_trees:
        nodes.extend((node, depth, side) for node, depth in tree_nodes(current_tree))

    node_offsets = {}
    for index, (node, depth, side) in enumerate(nodes):
        if node.mbr is None:
            continue

        mbr = node.mbr
        is_discarded_node = node.name in (discarded_nodes or set())
        is_active = active_side is None or side == active_side

        edge = "red" if is_discarded_node else (
            "tab:orange" if is_active and active_side is not None else "black"
        )

        rect = Rectangle(
            (mbr.x1, mbr.y1), mbr.x2 - mbr.x1, mbr.y2 - mbr.y1,
            fill=is_discarded_node,
            facecolor="red" if is_discarded_node else "none",
            edgecolor=edge,
            linewidth=3 if is_discarded_node else max(0.8, 2.2 - depth * 0.35),
            linestyle="-" if not node.is_leaf else "--",
            alpha=0.22 if is_discarded_node else (
                0.9 if is_active else 0.45
            ),
            zorder=2
        )
        ax.add_patch(rect)

        # Alternate label positions to reduce collisions between nested MBRs.
        offsets = [(4, 5), (4, -12), (-25, 5), (-25, -12), (4, 17), (-25, 17)]
        offset = offsets[index % len(offsets)]
        node_offsets[node.name] = offset

        ax.annotate(
            node.name, (mbr.x1, mbr.y2), xytext=offset,
            textcoords="offset points", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
            zorder=7
        )

    skyline_ids = {p.id for p in (skyline or [])}
    pruned_nodes = pruned_nodes or set()
    discarded_point_ids = set(discarded_points or set())

    if discarded_entry is not None and isinstance(discarded_entry, Point):
        discarded_point_ids.add(discarded_entry.id)

    # Cycle label positions so labels near MBR corners do not all sit top-right.
    point_offsets = [(5, 5), (5, -12), (-20, 5), (-20, -12), (5, 15), (-20, 15)]

    for index, point in enumerate(points):
        if point.id in discarded_point_ids:
            color, size, edge, linewidth = "red", 100, "darkred", 1.8
        elif dominating_point is not None and point.id == dominating_point.id:
            color, size, edge, linewidth = "gold", 125, "black", 2.0
        elif point.id in skyline_ids:
            color, size, edge, linewidth = "tab:green", 75, "white", 0.7
        elif highlighted_point is not None and point.id == highlighted_point.id:
            color, size, edge, linewidth = "tab:orange", 90, "black", 1.0
        else:
            color, size, edge, linewidth = "tab:blue", 35, "white", 0.5

        ax.scatter(
            point.x, point.y, color=color, s=size,
            edgecolors=edge, linewidths=linewidth, zorder=5
        )

        ax.annotate(
            point_label(point), (point.x, point.y),
            xytext=point_offsets[index % len(point_offsets)],
            textcoords="offset points", fontsize=8, zorder=8,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7)
        )

    if examined_entry is not None:
        if isinstance(examined_entry, Point):
            ax.scatter(
                examined_entry.x, examined_entry.y, s=180,
                facecolors="none", edgecolors="tab:orange",
                linewidths=2, zorder=9
            )
        elif isinstance(examined_entry, Node) and examined_entry.mbr is not None:
            m = examined_entry.mbr
            ax.add_patch(Rectangle(
                (m.x1, m.y1), m.x2 - m.x1, m.y2 - m.y1,
                fill=False, edgecolor="tab:orange", linewidth=3, zorder=9
            ))

    if isinstance(discarded_entry, Node) and discarded_entry.mbr is not None:
        m = discarded_entry.mbr
        ax.add_patch(Rectangle(
            (m.x1, m.y1), m.x2 - m.x1, m.y2 - m.y1,
            fill=False, edgecolor="red", linewidth=3,
            linestyle="--", zorder=10
        ))

    if screening_status:
        for point, status in screening_status:
            color = "green" if status == "accepted" else "red"
            ax.scatter(
                point.x, point.y, s=190, facecolors="none",
                edgecolors=color, linewidths=2.5, zorder=10
            )

    if screening_min_y is not None:
        ax.axhline(
            screening_min_y, color="purple", linestyle=":",
            linewidth=1.5, alpha=0.8
        )
        ax.text(
            ax.get_xlim()[1], screening_min_y,
            f" min_y = {screening_min_y:g}",
            color="purple", ha="right", va="bottom", fontsize=9
        )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.2)

    heap_ax.axis("off")

    if heap_entries is not None:
        heap_ax.set_title("BBS min-heap", fontsize=12, pad=12)
        lines = []

        for index, item in enumerate(heap_entries):
            priority, _, entry = item

            if isinstance(entry, Node):
                description = f"{entry.name}  [{entry.mbr}]"
            else:
                description = f"p{entry.id} ({entry.x:g}, {entry.y:g})"

            lines.append(f"{index + 1:>2}. {description}\n     key = {priority:g}")

        if not lines:
            lines = ["(empty)"]

        heap_ax.text(
            0.02, 0.98, "\n\n".join(lines),
            va="top", ha="left", fontsize=8.5,
            family="monospace", wrap=True,
            bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#bbbbbb"),
            transform=heap_ax.transAxes
        )

        if event_text:
            heap_ax.text(
                0.02, 0.02, event_text,
                va="bottom", ha="left", fontsize=9,
                color="darkred" if discarded_entry is not None else "black",
                wrap=True, transform=heap_ax.transAxes
            )
    else:
        heap_ax.set_visible(False)

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")

    plt.close(fig)


def entry_mbr(entry):
    """Return the MBR for either a Point or an internal Node entry."""
    return entry.mbr


def entry_sort_key(entry, axis):
    if isinstance(entry, Point):
        if axis == "x":
            return (entry.x, entry.y)
        return (entry.y, entry.x)

    if axis == "x":
        return (entry.mbr.x1, entry.mbr.x2)
    return (entry.mbr.y1, entry.mbr.y2)


def entry_description(entry):
    if isinstance(entry, Point):
        return f"p{entry.id}"
    return entry.name


def combined_mbr(entries):
    mbrs = [entry_mbr(entry) for entry in entries]
    return (
        min(m.x1 for m in mbrs), min(m.y1 for m in mbrs),
        max(m.x2 for m in mbrs), max(m.y2 for m in mbrs)
    )


def mbr_perimeter(bounds):
    x1, y1, x2, y2 = bounds
    return 2 * ((x2 - x1) + (y2 - y1))


def format_mbr(bounds):
    x1, y1, x2, y2 = bounds
    return f"[({x1:g},{y1:g}),({x2:g},{y2:g})]"


def split_evaluation_diagram(node, min_fill, output_path, B):
    """Create a standalone table of all legal R-tree split candidates."""
    entries = list(node.entries)
    count = len(entries)
    rows = []
    best_index = None
    best_score = float("inf")

    for axis in ("x", "y"):
        ordered = sorted(entries, key=lambda entry: entry_sort_key(entry, axis))

        for cut in range(min_fill, count - min_fill + 1):
            first = ordered[:cut]
            second = ordered[cut:]

            first_mbr = combined_mbr(first)
            second_mbr = combined_mbr(second)

            first_perimeter = mbr_perimeter(first_mbr)
            second_perimeter = mbr_perimeter(second_mbr)
            score = first_perimeter + second_perimeter

            rows.append([
                f"{axis.upper()}: " + " ".join(entry_description(e) for e in ordered),
                f"{cut} | {count - cut}",
                f"{{{', '.join(entry_description(e) for e in first)}}}"
                f"{format_mbr(first_mbr)},{first_perimeter:g}",
                f"{{{', '.join(entry_description(e) for e in second)}}}"
                f"{format_mbr(second_mbr)},{second_perimeter:g}",
                f"{score:g}"
            ])

            # Preserve the R-tree's tie preference: X before Y, then smaller cut.
            if score < best_score:
                best_score = score
                best_index = len(rows) - 1

    if not rows:
        return

    row_height = 0.72
    fig_height = max(3.0, 1.45 + len(rows) * row_height)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    ax.axis("off")

    ax.set_title(
        f"Split evaluation — overflow of {node.name} "
        f"({count} entries, maximum {B})",
        fontsize=15, fontweight="bold", loc="left", pad=18
    )

    table = ax.table(
        cellText=rows,
        colLabels=[
            "Sort axis / order", "Cut",
            "S1 (MBR, perimeter)", "S2 (MBR, perimeter)", "Sum"
        ],
        cellLoc="left", colLoc="left", loc="center",
        colWidths=[0.23, 0.08, 0.27, 0.27, 0.10]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.75)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#888888")
        cell.set_linewidth(0.7)

        if row == 0:
            cell.set_facecolor("#d9e2f3")
            cell.set_text_props(weight="bold")
        elif row - 1 == best_index:
            cell.set_facecolor("#e2f0d9")
            if col == 4:
                cell.get_text().set_text(f"{rows[best_index][4]}  ← best")
                cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("white")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Split evaluation saved: {output_path}")


def construction_diagrams(points, output_dir, B=5):
    output_dir = Path(output_dir) / "construction"
    output_dir.mkdir(parents=True, exist_ok=True)

    tree = RTree(B=B, verbose=True)
    step = 0

    for index, point in enumerate(points, 1):
        print(f"\n=== CONSTRUCTION INSERT {index}/{len(points)}: {point} ===")
        original_overflow = tree._handle_overflow

        def handle_overflow_with_diagrams(self, node):
            nonlocal step
            step += 1
            count = len(node.entries)

            draw_spatial_state(
                points[:index], tree=self,
                title=f"Before split: {node.name} has {count} entries (max {B})",
                output_path=output_dir / f"{index:03}_{step:03}_{node.name}_before_split.png",
                highlighted_point=point, viewport_points=points
            )

            print(f"SPLIT: {node.name} has {count} entries (max {B})")

            split_evaluation_diagram(
                node, self.min_fill,
                output_dir / f"{index:03}_{step:03}_{node.name}_split_evaluation.png",
                B
            )

            original_overflow(node)
            step += 1

            draw_spatial_state(
                points[:index], tree=self,
                title=f"After split: {node.name} overflow handled; new subspaces shown",
                output_path=output_dir / f"{index:03}_{step:03}_{node.name}_after_split.png",
                highlighted_point=point, viewport_points=points
            )

        tree._handle_overflow = MethodType(handle_overflow_with_diagrams, tree)
        tree.insert(point)
        tree._handle_overflow = original_overflow

        draw_spatial_state(
            points[:index], tree=tree,
            title=f"R-tree construction: insert p{point.id} ({index}/{len(points)})",
            output_path=output_dir / f"{index:03}_{point.id}_insert.png",
            highlighted_point=point, viewport_points=points
        )

    tree.check()
    tree.print_tree()
    return tree


def heap_label(heap):
    result = []

    for priority, _, entry in sorted(heap):
        if isinstance(entry, Node):
            result.append(f"{entry.name} MBR={entry.mbr} priority={priority:g}")
        else:
            result.append(f"p{entry.id} ({entry.x:g},{entry.y:g}) priority={priority:g}")

    return result


def bbs_skyline_visual(
        tree, points, output_dir, prefix="", viewport_points=None,
        extra_trees=None, partitions=None, active_side=None,
        side_skylines=None):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skyline = []
    heap = []
    counter = 0
    step = 0
    pruned = set()
    discarded_points = set()
    discarded_nodes = set()

    heapq.heappush(
        heap, (tree.root.mbr.x1 + tree.root.mbr.y1, counter, tree.root)
    )
    counter += 1

    def snapshot(event, entry=None, discarded_entry=None,
                 dominating_point=None, reason=None):

        nonlocal step
        step += 1

        filename = output_dir / f"{prefix}{step:03}_{event}.png"

        print(f"\n[BBS {step}] {event}")
        print("H:", heap_label(heap))
        print("SKY:", [(p.id, p.x, p.y) for p in skyline])

        if reason:
            print("Reason:", reason)

        print("Save:", filename)

        draw_spatial_state(
            points, tree=tree,
            title=f"BBS step {step}: {event}\nHeap entries: {len(heap)} | SKY: {[p.id for p in skyline]}",
            output_path=filename, skyline=skyline,
            pruned_nodes=pruned, examined_entry=entry,
            viewport_points=viewport_points or points,
            discarded_entry=discarded_entry,
            dominating_point=dominating_point,
            heap_entries=list(heap), event_text=reason,
            discarded_points=discarded_points,
            discarded_nodes=discarded_nodes,
            extra_trees=extra_trees, active_side=active_side,
            side_skylines=side_skylines,
            partitions=partitions
        )

    snapshot("initial_heap", tree.root)

    while heap:
        _, _, entry = heapq.heappop(heap)

        if isinstance(entry, Node):
            mbr = entry.mbr
            corner = Point(-1, mbr.x1, mbr.y1)
            dominator = next(
                (s for s in skyline if dominates(s, corner)), None
            )

            if dominator is not None:
                pruned.add(entry.name)
                discarded_nodes.add(entry.name)

                reason = f"{entry.name} pruned: dominated by p{dominator.id}."

                snapshot(
                    "prune_" + entry.name, entry,
                    entry, dominator, reason
                )
                continue

            for child in entry.entries:
                child_mbr = child.mbr
                heapq.heappush(
                    heap,
                    (child_mbr.x1 + child_mbr.y1, counter, child)
                )
                counter += 1

            snapshot("expand_" + entry.name, entry)

        else:
            dominator = next(
                (s for s in skyline if dominates(s, entry)), None
            )

            if dominator is not None:
                discarded_points.add(entry.id)

                reason = (
                    f"p{entry.id} ({entry.x:g}, {entry.y:g}) discarded: "
                    f"dominated by p{dominator.id} "
                    f"({dominator.x:g}, {dominator.y:g})."
                )

                snapshot(
                    f"discard_p{entry.id}", entry,
                    entry, dominator, reason
                )
                continue

            skyline.append(entry)

            reason = (
                f"p{entry.id} added to skyline: "
                "no skyline points dominate it."
            )

            snapshot(f"add_p{entry.id}", entry, reason=reason)

    print("\nFinal skyline:", [(p.id, p.x, p.y) for p in skyline])
    return skyline


def dc_diagrams(points, output_dir, B=5):
    output_dir = Path(output_dir) / "divide_and_conquer"
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_points = sorted(points, key=lambda p: (p.x, p.y, p.id))
    midpoint = (len(sorted_points) + 1) // 2

    left_points = sorted_points[:midpoint]
    right_points = sorted_points[midpoint:]

    if left_points and right_points:
        boundary = (left_points[-1].x + right_points[0].x) / 2
    else:
        boundary = sorted_points[-1].x if sorted_points else 0

    partitions = [(boundary, "partition boundary")]

    print("\n=== DIVIDE AND CONQUER PARTITION ===")
    print("X-sorted points:", [(p.id, p.x, p.y) for p in sorted_points])
    print("Left:", [(p.id, p.x, p.y) for p in left_points])
    print("Right:", [(p.id, p.x, p.y) for p in right_points])

    draw_spatial_state(
        points,
        title="Divide-and-conquer: X-sorted points and two subspaces",
        output_path=output_dir / "001_partition.png",
        partitions=partitions,
        viewport_points=points
    )

    left_tree = RTree(B=B, verbose=True).build(left_points)
    right_tree = RTree(B=B, verbose=True).build(right_points)

    left_tree.check()
    right_tree.check()

    left_tree.print_tree()
    right_tree.print_tree()

    # Both trees remain visible in every D&C diagram.
    trees = [("left", left_tree), ("right", right_tree)]

    left_skyline = bbs_skyline_visual(
        left_tree, points, output_dir / "left",
        prefix="left_", viewport_points=points,
        extra_trees=[("right", right_tree)],
        partitions=partitions, active_side="main"
    )

    # Keep the completed left skyline visible while processing the right side.
    right_skyline = bbs_skyline_visual(
        right_tree, points, output_dir / "right",
        prefix="right_", viewport_points=points,
        extra_trees=[("left", left_tree)],
        partitions=partitions, active_side="main",
        side_skylines={"left": left_skyline}
    )

    # -----------------------------------------------------------------------
    # Merge: 1D-dominance screening, matching dc_skyline().
    # -----------------------------------------------------------------------

    right_candidates = sorted(right_skyline, key=lambda p: p.x)
    skylines = left_skyline.copy()
    min_y = min(p.y for p in left_skyline) if left_skyline else float("inf")

    print("\n=== D&C DOMINANCE SCREENING ===")
    print("Left candidates:", [(p.id, p.x, p.y) for p in left_skyline])
    print("Right candidates (sorted by X):",
          [(p.id, p.x, p.y) for p in right_candidates])
    print("Initial min_y:", min_y)

    # Show the starting candidate sets.
    draw_spatial_state(
        points,
        title=f"Dominance screening: initial candidates | min_y = {min_y:g}",
        output_path=output_dir / "003_screening_start.png",
        skyline=left_skyline+right_candidates,
        partitions=partitions,
        viewport_points=points,
        extra_trees=trees
    )

    screening_status = []

    for index, point in enumerate(right_candidates, 1):
        # Find the point responsible for rejection.
        dominator = None

        if point.y >= min_y:
            dominator = next(
                (p for p in skylines if p.y <= point.y),
                None
            )

        if point.y < min_y:
            skylines.append(point)
            min_y = point.y
            screening_status.append((point, "accepted"))

            reason = (
                f"p{point.id} accepted: y={point.y:g} < min_y. "
                f"Add to skyline; update min_y = {min_y:g}."
            )
            highlighted = point
            rejected = None

        else:
            screening_status.append((point, "rejected"))

            reason = (
                f"p{point.id} rejected: y={point.y:g} >= min_y. "
                f"Dominated by p{dominator.id}."
                if dominator is not None
                else f"p{point.id} rejected: y={point.y:g} >= min_y."
            )
            highlighted = None
            rejected = point

        print(reason)

        draw_spatial_state(
            points,
            title=(
                f"Dominance screening {index}/{len(right_candidates)}: "
                f"p{point.id}"
            ),
            output_path=output_dir / f"003_screening_{index:03}_p{point.id}.png",
            skyline=skylines,
            highlighted_point=highlighted,
            discarded_entry=rejected,
            dominating_point=dominator,
            partitions=partitions,
            viewport_points=points,
            extra_trees=trees,
            screening_status=screening_status,
            screening_min_y=min_y,
            event_text=reason
        )

    draw_spatial_state(
        points,
        title=f"D&C final skyline: {[p.id for p in skylines]}",
        output_path=output_dir / "004_combined_final_skyline.png",
        skyline=skylines,
        partitions=partitions,
        viewport_points=points,
        extra_trees=trees,
        screening_status=screening_status,
        screening_min_y=min_y
    )

    print("\nCombined local skylines:",
          [(p.id, p.x, p.y) for p in left_skyline + right_skyline])
    print("Final skyline:", [(p.id, p.x, p.y) for p in skylines])

    return left_tree, right_tree, left_skyline, right_skyline, skylines


def main():
    parser = argparse.ArgumentParser(
        description="Generate R-tree construction, BBS, and divide-and-conquer diagrams."
    )

    parser.add_argument(
        "data_file", nargs="?",
        default="datasets/Sample_Data/skyline_search.txt",
        help="Path to skyline_search.txt"
    )
    parser.add_argument("--output", default="skyline_diagrams")
    parser.add_argument("--B", type=int, default=5)

    args = parser.parse_args()

    data_file = Path(args.data_file)

    if not data_file.exists():
        raise FileNotFoundError(
            f"Could not find {data_file}. Pass the path to skyline_search.txt."
        )

    points, _ = load_sample_file(data_file)

    if not points:
        raise ValueError(f"No data points were loaded from {data_file}")

    print(f"Loaded {len(points)} points from {data_file}; B={args.B}")

    output_dir = Path(args.output)

    tree = construction_diagrams(points, output_dir, B=args.B)

    bbs_skyline_visual(
        tree, points, output_dir / "bbs",
        viewport_points=points
    )

    dc_diagrams(points, output_dir, B=args.B)

    print(f"\nDiagrams saved under: {output_dir.resolve()}")


if __name__ == "__main__":
    main()