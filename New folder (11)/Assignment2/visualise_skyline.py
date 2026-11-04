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


def draw_spatial_state(points, tree=None, title="", output_path=None,
                       highlighted_point=None, skyline=None, pruned_nodes=None,
                       examined_entry=None, partitions=None, viewport_points=None,
                       discarded_entry=None, dominating_point=None,
                       heap_entries=None, event_text=None,
                       discarded_points=None, discarded_nodes=None):
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
            ax.text(x, ax.get_ylim()[1] - 0.1, label, ha="left", va="top", fontsize=9)

    nodes = tree_nodes(tree) if tree is not None else []
    node_offsets = {}
    for index, (node, depth) in enumerate(nodes):
        if node.mbr is None:
            continue
        mbr = node.mbr
        is_discarded_node = node.name in (discarded_nodes or set())
        rect = Rectangle(
            (mbr.x1, mbr.y1), mbr.x2 - mbr.x1, mbr.y2 - mbr.y1,
            fill=is_discarded_node,
            facecolor="red" if is_discarded_node else "none",
            edgecolor="red" if is_discarded_node else "black",
            linewidth=3 if is_discarded_node else max(0.8, 2.2 - depth * 0.35),
            linestyle="-" if not node.is_leaf else "--",
            alpha=0.22 if is_discarded_node else max(0.35, 0.9 - depth * 0.08), zorder=2
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
    discarded_id = discarded_entry.id if isinstance(discarded_entry, Point) else None
    if discarded_id is not None:
        discarded_point_ids.add(discarded_id)

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

        ax.scatter(point.x, point.y, color=color, s=size, edgecolors=edge,
                   linewidths=linewidth, zorder=5)
        ax.annotate(
            point_label(point), (point.x, point.y),
            xytext=point_offsets[index % len(point_offsets)],
            textcoords="offset points", fontsize=8, zorder=8,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7)
        )

    if examined_entry is not None:
        if isinstance(examined_entry, Point):
            ax.scatter(examined_entry.x, examined_entry.y, s=180,
                       facecolors="none", edgecolors="tab:orange",
                       linewidths=2, zorder=9)
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
            fill=False, edgecolor="red", linewidth=3, linestyle="--", zorder=10
        ))

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
        heap_ax.text(0.02, 0.98, "\n\n".join(lines), va="top", ha="left",
                     fontsize=8.5, family="monospace", wrap=True,
                     bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#bbbbbb"),
                     transform=heap_ax.transAxes)
        if event_text:
            heap_ax.text(0.02, 0.02, event_text, va="bottom", ha="left",
                         fontsize=9, color="darkred" if discarded_entry is not None else "black",
                         wrap=True, transform=heap_ax.transAxes)
    else:
        heap_ax.set_visible(False)

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


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
                highlighted_point=point, viewport_points=points,
                event_text=None
            )
            print(f"SPLIT: {node.name} has {count} entries (max {B})")
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


def bbs_skyline_visual(tree, points, output_dir, prefix="", viewport_points=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skyline = []
    heap = []
    counter = 0
    step = 0
    pruned = set()
    discarded_points = set()
    discarded_nodes = set()

    heapq.heappush(heap, (tree.root.mbr.x1 + tree.root.mbr.y1, counter, tree.root))
    counter += 1

    def snapshot(event, entry=None, discarded_entry=None, dominating_point=None, reason=None):
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
            output_path=filename, skyline=skyline, pruned_nodes=pruned,
            examined_entry=entry, viewport_points=viewport_points or points,
            discarded_entry=discarded_entry, dominating_point=dominating_point,
            heap_entries=list(heap), event_text=reason,
            discarded_points=discarded_points, discarded_nodes=discarded_nodes
        )

    snapshot("initial_heap", tree.root)

    while heap:
        _, _, entry = heapq.heappop(heap)
        snapshot("remove_" + (entry.name if isinstance(entry, Node) else f"p{entry.id}"), entry)

        if isinstance(entry, Node):
            mbr = entry.mbr
            corner = Point(-1, mbr.x1, mbr.y1)
            dominator = next((s for s in skyline if dominates(s, corner)), None)

            if dominator is not None:
                pruned.add(entry.name)
                discarded_nodes.add(entry.name)
                reason = f"{entry.name} discarded: lower-left corner ({corner.x:g}, {corner.y:g}) is dominated by p{dominator.id} ({dominator.x:g}, {dominator.y:g})."
                snapshot("prune_" + entry.name, entry, entry, dominator, reason)
                continue

            for child in entry.entries:
                child_mbr = child.mbr
                heapq.heappush(heap, (child_mbr.x1 + child_mbr.y1, counter, child))
                counter += 1
            snapshot("expand_" + entry.name, entry)

        else:
            dominator = next((s for s in skyline if dominates(s, entry)), None)
            if dominator is not None:
                discarded_points.add(entry.id)
                reason = f"p{entry.id} ({entry.x:g}, {entry.y:g}) discarded: dominated by p{dominator.id} ({dominator.x:g}, {dominator.y:g})."
                snapshot(f"discard_p{entry.id}", entry, entry, dominator, reason)
                continue

            skyline.append(entry)
            snapshot(f"add_p{entry.id}", entry)

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

    print("\n=== DIVIDE AND CONQUER PARTITION ===")
    print("X-sorted points:", [(p.id, p.x, p.y) for p in sorted_points])
    print("Left:", [(p.id, p.x, p.y) for p in left_points])
    print("Right:", [(p.id, p.x, p.y) for p in right_points])

    draw_spatial_state(
        points, title="Divide-and-conquer: X-sorted points and two subspaces",
        output_path=output_dir / "001_partition.png",
        partitions=[(boundary, "partition boundary")], viewport_points=points
    )

    left_tree = RTree(B=B, verbose=True).build(left_points)
    right_tree = RTree(B=B, verbose=True).build(right_points)
    left_tree.check()
    right_tree.check()

    left_tree.print_tree()
    right_tree.print_tree()

    left_skyline = bbs_skyline_visual(
        left_tree, left_points, output_dir / "left", prefix="left_", viewport_points=points
    )
    right_skyline = bbs_skyline_visual(
        right_tree, right_points, output_dir / "right", prefix="right_", viewport_points=points
    )

    candidates = left_skyline + right_skyline
    final_skyline = [
        point for point in candidates
        if not any(other is not point and dominates(other, point) for other in candidates)
    ]

    draw_spatial_state(
        points, title=f"Divide-and-conquer final skyline: {[p.id for p in final_skyline]}",
        output_path=output_dir / "002_combined_final_skyline.png",
        skyline=final_skyline, viewport_points=points
    )

    print("\nCombined local skylines:", [(p.id, p.x, p.y) for p in candidates])
    print("Final skyline:", [(p.id, p.x, p.y) for p in final_skyline])

    return left_tree, right_tree, left_skyline, right_skyline, final_skyline


def main():
    parser = argparse.ArgumentParser(
        description="Generate R-tree construction, BBS, and divide-and-conquer diagrams."
    )
    parser.add_argument(
        "data_file", nargs="?", default="datasets/Sample_Data/skyline_search.txt",
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
    bbs_skyline_visual(tree, points, output_dir / "bbs", viewport_points=points)
    dc_diagrams(points, output_dir, B=args.B)

    print(f"\nDiagrams saved under: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
