#!/usr/bin/env python3
"""Render the NPA inference kernel ablation paper from local benchmark JSON.

The repository does not assume a LaTeX toolchain is installed on developer
machines. This script writes an arXiv-style .tex source and also renders a PDF
from the same benchmark summaries via reportlab.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

GPU_3D = ROOT / "target" / "bench_gpu_kernel_ablation_3d.json"
GPU_2D = ROOT / "target" / "bench_gpu_kernel_ablation_2d.json"
GPU_2D_CAPACITY = ROOT / "target" / "bench_gpu_tiled_capacity_2d.json"
GPU_AUTO_AFTER = ROOT / "target" / "bench_gpu_auto_after.json"
GPU_BVH = ROOT / "target" / "bench_gpu_bvh_ablation.json"
GPU_BUILT_BVH = ROOT / "target" / "bench_gpu_gpu_bvh_ablation.json"
GPU_LBVH = ROOT / "target" / "bench_gpu_lbvh_ablation.json"
GPU_MORTON_LBVH = ROOT / "target" / "bench_gpu_morton_lbvh_ablation.json"
SPATIAL = ROOT / "target" / "bench_spatial_index_perf.json"


@dataclass(frozen=True)
class GpuRow:
    preset: str
    particles: int
    geometry: str
    mode: str
    ms: float
    overflow: int
    resolved: str
    capacity: int


@dataclass(frozen=True)
class SpatialRow:
    preset: str
    particles: int
    geometry: str
    strategy: str
    candidates: float
    exact: float
    nodes: float
    active_bins: int
    max_occupancy: int


def load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def gpu_rows(paths: Iterable[Path]) -> list[GpuRow]:
    rows: list[GpuRow] = []
    for path in paths:
        for row in load_json(path):
            if row.get("returncode", 0) != 0:
                continue
            try:
                rows.append(
                    GpuRow(
                        preset=str(row["preset"]),
                        particles=int(row["particles"]),
                        geometry=str(row["requested_geometry"]),
                        mode=str(row["requested_mode"]),
                        ms=float(row["avg_step_ms"]),
                        overflow=int(row.get("grid_overflow_count", 0)),
                        resolved=str(row["neighbor_mode"]),
                        capacity=int(row.get("bucket_capacity", 0)),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def spatial_rows(path: Path) -> list[SpatialRow]:
    rows: list[SpatialRow] = []
    for row in load_json(path):
        try:
            rows.append(
                SpatialRow(
                    preset=str(row.get("requested_preset") or row["preset"]).lower(),
                    particles=int(row["particles"]),
                    geometry=str(row.get("requested_geometry") or row["geometry"]).lower(),
                    strategy=str(row["strategy"]),
                    candidates=float(row["candidates_per_particle"]),
                    exact=float(row["exact_neighbors_per_particle"]),
                    nodes=float(row.get("node_visits_per_particle", 0.0)),
                    active_bins=int(row["active_bins"]),
                    max_occupancy=int(row["max_bin_occupancy"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def mode_label(mode: str) -> str:
    return {
        "auto": "auto",
        "linked-list": "linked",
        "fixed-buckets:128": "fixed-128",
        "fixed-buckets:256": "fixed-256",
        "tiled-fixed-buckets:128": "tiled-128",
        "tiled-fixed-buckets:256": "tiled-256",
        "tiled-fixed-buckets:512": "tiled-512",
        "tiled-fixed-buckets:1024": "tiled-1024",
        "tiled-fixed-buckets:2048": "tiled-2048",
        "gpu-bvh:8": "gpu-bvh-8",
        "gpu-bvh:16": "gpu-bvh-16",
        "gpu-bvh:32": "gpu-bvh-32",
        "gpu-lbvh:8": "gpu-lbvh-8",
        "gpu-lbvh:16": "gpu-lbvh-16",
        "gpu-lbvh:32": "gpu-lbvh-32",
        "gpu-morton-lbvh:8": "morton-lbvh-8",
        "gpu-morton-lbvh:16": "morton-lbvh-16",
        "gpu-morton-lbvh:32": "morton-lbvh-32",
        "sorted-cells": "sorted",
    }.get(mode, mode)


def fmt_ms(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    return f"{value:.2f}"


def fmt_x(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    return f"{value:.2f}x"


def table_best_exact(rows: list[GpuRow], preset: str, particles: int, geometries: list[str]) -> list[list[str]]:
    out = [["geometry", "best exact mode", "ms/step", "resolved layout"]]
    for geometry in geometries:
        candidates = [
            row
            for row in rows
            if row.preset == preset
            and row.particles == particles
            and row.geometry == geometry
            and row.overflow == 0
        ]
        if not candidates:
            out.append([geometry, "--", "--", "--"])
            continue
        best = min(candidates, key=lambda row: row.ms)
        out.append([geometry, mode_label(best.mode), fmt_ms(best.ms), best.resolved])
    return out


def auto_speedups(old_rows: list[GpuRow], new_rows: list[GpuRow]) -> list[list[str]]:
    old_auto = {
        (row.preset, row.particles, row.geometry): row
        for row in old_rows
        if row.mode == "auto" and row.overflow == 0
    }
    new_auto = {
        (row.preset, row.particles, row.geometry): row
        for row in new_rows
        if row.mode == "auto" and row.overflow == 0
    }
    selected = [
        ("growing-2d", 4096, "dense"),
        ("growing-2d", 4096, "line"),
        ("growing-2d", 8192, "dense"),
        ("growing-2d", 8192, "line"),
        ("texture-2d", 4096, "uniform"),
        ("texture-2d", 4096, "line"),
        ("texture-2d", 8192, "uniform"),
        ("texture-2d", 8192, "line"),
        ("growing-3d-gs", 4096, "line"),
        ("growing-3d-gs", 8192, "line"),
        ("growing-3d-gs", 8192, "torus"),
    ]
    out = [["preset", "N", "geometry", "old auto", "new auto", "speedup", "new layout"]]
    for key in selected:
        old = old_auto.get(key)
        new = new_auto.get(key)
        speedup = old.ms / new.ms if old and new and new.ms > 0.0 else None
        out.append(
            [
                key[0],
                str(key[1]),
                key[2],
                fmt_ms(old.ms if old else None),
                fmt_ms(new.ms if new else None),
                fmt_x(speedup),
                new.resolved if new else "--",
            ]
        )
    return out


def spatial_summary(rows: list[SpatialRow]) -> list[list[str]]:
    selected = [
        ("growing-3d-gs", 4096, "dense"),
        ("growing-3d-gs", 4096, "line"),
        ("growing-3d-gs", 4096, "torus"),
        ("growing-2d", 4096, "dense"),
        ("texture-2d", 4096, "dense"),
    ]
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row.preset, row.particles, row.geometry)].append(row)
    out = [["preset", "N", "geometry", "strategy", "cand./p", "exact/p", "nodes/p"]]
    for key in selected:
        candidates = by_key.get(key, [])
        for strategy in ["hash-grid", "tile-blocks", "bvh"]:
            row = next((item for item in candidates if item.strategy == strategy), None)
            if row is None:
                continue
            out.append(
                [
                    key[0],
                    str(key[1]),
                    key[2],
                    strategy,
                    f"{row.candidates:.1f}",
                    f"{row.exact:.1f}",
                    f"{row.nodes:.1f}",
                ]
            )
    return out


def bvh_wgpu_summary(rows: list[GpuRow]) -> list[list[str]]:
    selected = [
        ("growing-3d-gs", 4096, "dense"),
        ("growing-3d-gs", 4096, "line"),
        ("growing-3d-gs", 8192, "dense"),
        ("growing-3d-gs", 8192, "line"),
        ("growing-2d", 4096, "dense"),
        ("growing-2d", 4096, "line"),
        ("growing-2d", 8192, "dense"),
        ("growing-2d", 8192, "line"),
    ]
    out = [
        [
            "preset",
            "N",
            "geometry",
            "best non-BVH",
            "CPU-BVH",
            "GPU-BVH",
            "GPU-LBVH",
            "Morton-LBVH",
            "best BVH ratio",
        ]
    ]
    for key in selected:
        candidates = [
            row
            for row in rows
            if (row.preset, row.particles, row.geometry) == key and row.overflow == 0
        ]
        cpu_bvh = [row for row in candidates if row.mode.startswith("bvh")]
        gpu_bvh = [row for row in candidates if row.mode.startswith("gpu-bvh")]
        gpu_lbvh = [row for row in candidates if row.mode.startswith("gpu-lbvh")]
        gpu_morton_lbvh = [row for row in candidates if row.mode.startswith("gpu-morton-lbvh")]
        non_bvh = [row for row in candidates if not row.mode.startswith("bvh")]
        non_bvh = [row for row in non_bvh if not row.mode.startswith("gpu-bvh")]
        non_bvh = [row for row in non_bvh if not row.mode.startswith("gpu-lbvh")]
        non_bvh = [row for row in non_bvh if not row.mode.startswith("gpu-morton-lbvh")]
        if not non_bvh or (not cpu_bvh and not gpu_bvh and not gpu_lbvh and not gpu_morton_lbvh):
            continue
        best_cpu_bvh = min(cpu_bvh, key=lambda row: row.ms) if cpu_bvh else None
        best_gpu_bvh = min(gpu_bvh, key=lambda row: row.ms) if gpu_bvh else None
        best_gpu_lbvh = min(gpu_lbvh, key=lambda row: row.ms) if gpu_lbvh else None
        best_gpu_morton_lbvh = (
            min(gpu_morton_lbvh, key=lambda row: row.ms) if gpu_morton_lbvh else None
        )
        best_non_bvh = min(non_bvh, key=lambda row: row.ms)
        best_bvh = min(
            [
                row
                for row in [best_cpu_bvh, best_gpu_bvh, best_gpu_lbvh, best_gpu_morton_lbvh]
                if row is not None
            ],
            key=lambda row: row.ms,
        )
        ratio = best_bvh.ms / best_non_bvh.ms if best_non_bvh.ms > 0.0 else None
        out.append(
            [
                key[0],
                str(key[1]),
                key[2],
                f"{mode_label(best_non_bvh.mode)} {fmt_ms(best_non_bvh.ms)}",
                f"{mode_label(best_cpu_bvh.mode)} {fmt_ms(best_cpu_bvh.ms)}"
                if best_cpu_bvh
                else "--",
                f"{mode_label(best_gpu_bvh.mode)} {fmt_ms(best_gpu_bvh.ms)}"
                if best_gpu_bvh
                else "--",
                f"{mode_label(best_gpu_lbvh.mode)} {fmt_ms(best_gpu_lbvh.ms)}"
                if best_gpu_lbvh
                else "--",
                f"{mode_label(best_gpu_morton_lbvh.mode)} {fmt_ms(best_gpu_morton_lbvh.ms)}"
                if best_gpu_morton_lbvh
                else "--",
                fmt_x(ratio),
            ]
        )
    return out


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def latex_table(rows: list[list[str]], caption: str, label: str) -> str:
    cols = "l" * len(rows[0])
    body = ["\\begin{table}[t]", "\\centering", "\\small", f"\\begin{{tabular}}{{{cols}}}", "\\toprule"]
    body.append(" & ".join(latex_escape(cell) for cell in rows[0]) + " \\\\")
    body.append("\\midrule")
    for row in rows[1:]:
        body.append(" & ".join(latex_escape(cell) for cell in row) + " \\\\")
    body.extend(["\\bottomrule", "\\end{tabular}", f"\\caption{{{latex_escape(caption)}}}", f"\\label{{{label}}}", "\\end{table}"])
    return "\n".join(body)


def tex_document(
    best_3d: list[list[str]],
    best_growing_2d: list[list[str]],
    best_texture_2d: list[list[str]],
    speedups: list[list[str]],
    spatial: list[list[str]],
    bvh_wgpu: list[list[str]],
) -> str:
    return rf"""\documentclass[10pt,twocolumn]{{article}}
\usepackage[margin=0.72in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{hyperref}}
\usepackage{{microtype}}
\title{{Spatial Index Ablations for GPU Neural Particle Automata Inference}}
\author{{burn\_automata contributors}}
\date{{June 2026}}
\begin{{document}}
\maketitle
\begin{{abstract}}
Neural Particle Automata (NPA) inference is dominated by local neighbor
traversal rather than neural matrix-vector arithmetic once particle counts reach
interactive 2D and 3D regimes. We benchmark the direct WGPU inference backend in
\texttt{{burn\_automata}} across linked lists, scalar fixed buckets,
active-cell tiled fixed buckets, sorted cell ranges, CPU structural
prototypes for tile blocks and median-split BVH traversal, an executable
CPU-rebuilt/WGPU-traversed BVH mode, a fully GPU-built fixed-order BVH
baseline, a GPU-built sorted-cell LBVH-style baseline, and a GPU Morton-order
bitonic-sort LBVH baseline. The main result is
that 2D NPA inference benefits strongly from active-cell tiled buckets when
bucket capacity is selected from measured occupancy, while broad 3D clouds
remain best served by exact linked lists on the tested GPU. Collapsed
low-dimensional 3D distributions are the exception: a tiled active-cell path
reduces the adversarial line case by roughly 2.7x. The current WGPU BVH path is
exact and useful for ablation, but its CPU rebuild/readback/upload cost keeps it
behind the best grid/tiled modes at 4K--8K particles. The no-readback GPU-built
fixed-order BVH removes CPU rebuild cost but performs worse because it lacks
spatial ordering. The sorted-cell GPU-LBVH baseline improves the pathological
3D line case over fixed-order GPU BVH, but scan/scatter/tree/traversal overhead
still keeps it behind the active-cell tiled grid path at 4K--8K particles.
Morton ordering improves broad-cloud BVH coherence relative to sorted-cell
ordering, but the current bitonic sort overhead is too high for it to become an
auto-mode candidate.
\end{{abstract}}

\section{{Introduction}}
NPAs update each particle from compact local features gathered from a spatial
neighborhood. This locality creates a natural opportunity for spatial indexing,
but the best GPU data structure depends on dimension, occupancy, overflow
behavior, and workgroup utilization. The library now keeps the neighbor
strategy modular through \texttt{{WgpuNeighborMode}} for executable WGPU paths
and \texttt{{SpatialStrategyKind}} for CPU structural studies.

\section{{Kernel Strategies}}
The linked-list path stores a per-cell atomic head and a per-particle next
pointer. It is exact and memory efficient, but every particle launches its own
neighbor traversal. Fixed buckets replace linked traversal with a dense
\(\mathrm{{cells}}\times\mathrm{{capacity}}\) slab. This is exact only when the
overflow counter is zero. The active-cell tiled variant reuses the same slab,
stores the non-empty cells and indirect dispatch dimensions during binning, and
uses workgroup memory for neighbor tiles. Sorted cells build exact contiguous
cell ranges with a count, prefix scan, and scatter. The BVH prototype is a CPU
median-split AABB tree used to study 3D candidate behavior before committing to
a GPU LBVH rebuild and traversal implementation.

The CPU-rebuilt WGPU BVH mode stores packed median-split nodes and leaf particle
indices in the existing grid storage buffer, then performs density and update
traversal in WGSL. To keep resident multi-step rollouts exact before a GPU LBVH
builder exists, the backend reads current GPU positions, rebuilds the BVH on
CPU, uploads the packed tree, and then runs WGPU traversal every step. The
`GpuBvh` baseline initializes a complete binary tree and reduces AABBs entirely
on GPU, avoiding readback but using fixed particle order rather than Morton
order. The `GpuLbvh` baseline reuses the GPU sorted-cell count/scan/scatter
ordering, builds leaves over that spatially coherent order, then reduces the
tree bottom-up on GPU. It is not a Karras radix-tree LBVH yet, but it measures
the cost and benefit of spatially ordered GPU BVH traversal without host
transfer.
The `GpuMortonLbvh` baseline generates Morton keys from clamped grid
coordinates, bitonic-sorts `(key, particle)` pairs on GPU, builds leaves over
that order, and reuses the same bottom-up tree reduction and traversal.

\section{{Experimental Setup}}
Benchmarks were run with the release \texttt{{gpu\_wgpu}} backend. Each row
reports average milliseconds per simulation step. Overflowed fixed-bucket rows
are excluded from best-exact selections because truncating cells changes NPA
states and rendered output. The structural BVH and tile-block rows are CPU
oracle measurements and should be read as candidate-count evidence, not direct
GPU timing.

{latex_table(speedups, "Post-change auto-mode speedups versus the previous auto resolver. All rows have zero final bucket overflow.", "tab:auto")}

{latex_table(bvh_wgpu, "Executable WGPU BVH ablation. Ratios above 1.0 mean the best BVH variant is slower than the best exact non-BVH mode for the same row.", "tab:bvhgpu")}

{latex_table(best_growing_2d, "Best exact growing-2D WGPU rows from the ablation and tiled-capacity sweeps.", "tab:growing2d")}

{latex_table(best_texture_2d, "Best exact texture-2D WGPU rows from the ablation and tiled-capacity sweeps.", "tab:texture2d")}

{latex_table(best_3d, "Best exact growing-3D Gaussian-splat rows from the WGPU ablation and post-change auto sweeps.", "tab:3d")}

{latex_table(spatial, "CPU spatial strategy oracle. Candidate counts include exact radius filtering opportunities before the neural update.", "tab:spatial")}

\section{{Analysis}}
The 2D conclusion is direct: active-cell tiled buckets provide large speedups
when the capacity is selected from observed occupancy with headroom. The
post-change auto resolver therefore routes 2D particle and periodic grids to
\texttt{{TiledFixedCellBuckets}} and uses a tiled-specific capacity rule
\(\mathrm{{nextpow2}}(2m+64)\), where \(m\) is the initial maximum cell
occupancy. This prevented overflow in the tested seed, dense, line, uniform,
plane, and torus distributions.

The 3D conclusion is more nuanced. Broad 3D clouds have low maximum occupancy
and enough active cells that per-particle linked traversal remains competitive.
Forcing tiled workgroups in those cases adds barriers and duplicated
cell-block work without enough reuse. Collapsed 3D line distributions are
different: active cells are sparse and occupancy is high, so the new auto
resolver switches to tiled buckets only when
\(\mathrm{{nonempty}}\times 32 \leq N\) and \(m \geq 64\). This preserves broad
3D behavior while improving adversarial collapsed cases.

BVH is now measured in four ways. The CPU oracle often reduces candidates for
thin or low-dimensional manifolds, the CPU-rebuilt/WGPU-traversed backend
confirms that BVH can beat linked lists on collapsed 3D lines, the fully-GPU
fixed-order BVH shows that eliminating readback alone is not enough, and the
sorted GPU-LBVH baseline shows that spatial ordering helps the worst fixed-order
3D line case but does not yet beat active-cell tiled grids, and the Morton
GPU-LBVH baseline shows better broad-cloud BVH coherence but excessive bitonic
sort overhead. A production GPU LBVH path still needs cheaper ordering, better
hierarchy construction, and stackless or short-stack traversal every step
because NPA particles move.

\section{{Library Organization}}
The implementation keeps executable strategies and analysis strategies separate.
\texttt{{burn\_automata::gpu}} owns WGPU layouts, BVH traversal, and auto-mode resolution.
\texttt{{burn\_automata\_kernels::spatial}} owns CPU analysis for hash grids,
tile blocks, and BVH. CLI commands \texttt{{bench}} and \texttt{{bench-spatial}}
emit parseable JSON/CSV for regression tests and papers. This separation makes
it straightforward to add a future GPU LBVH or cooperative-neighbor storage
without hard-coding BVH assumptions into model configs.

\section{{Limitations and Next Steps}}
The `GpuBvh` path is GPU-built but fixed-order. The `GpuLbvh` path is
GPU-built and spatially sorted by existing grid cell order, but it is not yet a
full Morton radix-tree LBVH and remains slower than tiled grids on the measured
4K--8K rows. The `GpuMortonLbvh` path validates GPU Morton ordering but uses
multi-dispatch bitonic sort, which is intentionally simple and not production
efficient. The CPU-rebuilt BVH path remains useful as a traversal-quality upper
baseline for median-split trees. The tiled WGPU path still uses capped buckets,
so exactness requires an overflow counter check. The next high-value kernel
experiments are: (1) radix-sort or persistent Morton ordering plus stackless
traversal for 3D collapsed/surface-like clouds; (2)
cooperative sorted-cell traversal that amortizes scan/scatter over multiple NPA
steps or combines density and update staging; and (3) per-step overflow
telemetry in long rollouts, not only final-state overflow.

\section{{Reproducibility}}
\begin{{verbatim}}
cargo build -p burn_automata --release --features gpu_wgpu
scripts/bench_gpu_matrix.py --no-build --output target/bench_gpu_auto_after.json \
  --preset growing-3d-gs --preset growing-2d --preset texture-2d \
  --particles 4096 --particles 8192 --steps 8 --repeats 2 \
  --geometry seed --geometry dense --geometry line --geometry uniform \
  --geometry plane --geometry torus --mode auto
scripts/bench_spatial_index.py --no-build --output target/bench_spatial_index_perf.json
scripts/bench_gpu_matrix.py --no-build --output target/bench_gpu_bvh_ablation.json \
  --preset growing-3d-gs --preset growing-2d --particles 4096 --particles 8192 \
  --steps 6 --repeats 2 --geometry dense --geometry line --geometry torus \
  --geometry uniform --mode auto --mode linked-list --mode tiled-fixed-buckets:128 \
  --mode tiled-fixed-buckets:512 --mode sorted-cells \
  --mode bvh:8 --mode bvh:16 --mode bvh:32 --mode bvh:64
scripts/bench_gpu_matrix.py --no-build --output target/bench_gpu_gpu_bvh_ablation.json \
  --preset growing-3d-gs --preset growing-2d --particles 4096 --particles 8192 \
  --steps 6 --repeats 2 --geometry dense --geometry line --geometry torus \
  --geometry uniform --mode auto --mode linked-list --mode tiled-fixed-buckets:512 \
  --mode sorted-cells --mode bvh:16 \
  --mode gpu-bvh:8 --mode gpu-bvh:16 --mode gpu-bvh:32
scripts/bench_gpu_matrix.py --no-build --output target/bench_gpu_lbvh_ablation.json \
  --preset growing-3d-gs --preset growing-2d --particles 4096 --particles 8192 \
  --steps 6 --repeats 2 --geometry dense --geometry line --geometry torus \
  --geometry uniform --mode auto --mode linked-list --mode tiled-fixed-buckets:512 \
  --mode sorted-cells --mode bvh:16 --mode gpu-bvh:16 \
  --mode gpu-lbvh:16 --mode gpu-lbvh:32
scripts/bench_gpu_matrix.py --no-build --output target/bench_gpu_morton_lbvh_ablation.json \
  --preset growing-3d-gs --particles 4096 --particles 8192 --steps 4 \
  --repeats 1 --geometry dense --geometry line --geometry torus --geometry uniform \
  --mode auto --mode gpu-bvh:16 --mode gpu-lbvh:16 --mode gpu-morton-lbvh:16
scripts/render_kernel_ablation_paper.py
\end{{verbatim}}

\begin{{thebibliography}}{{9}}
\bibitem{{nca}} A. Mordvintsev et al. Growing Neural Cellular Automata.
\bibitem{{lbvh}} T. Karras. Maximizing Parallelism in the Construction of BVHs, HPG 2012.
\bibitem{{npa}} Self-organizing Neural Particle Automata reference implementation and model zoo.
\end{{thebibliography}}
\end{{document}}
"""


def pdf_story(title: str, tables: list[tuple[str, list[list[str]]]], output: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(output),
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )
    story = [
        Paragraph(title, styles["Title"]),
        Paragraph(
            "A reproducible ablation report for burn_automata WGPU inference. "
            "The accompanying LaTeX source is docs/kernel_ablations.tex.",
            styles["BodyText"],
        ),
        Spacer(1, 0.15 * inch),
        Paragraph("Summary", styles["Heading1"]),
        Paragraph(
            "2D inference now defaults to active-cell tiled fixed buckets with "
            "adaptive capacity. Broad 3D remains linked-list based; collapsed "
            "3D high-occupancy cases switch to tiled buckets. BVH now has both "
            "a CPU structural oracle and an executable CPU-rebuilt/WGPU-traversed "
            "inference path for future GPU LBVH comparison.",
            styles["BodyText"],
        ),
    ]
    for caption, rows in tables:
        story.extend([Spacer(1, 0.12 * inch), Paragraph(caption, styles["Heading2"])])
        table = Table(rows, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9AA4B2")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
                ]
            )
        )
        story.append(table)
    story.extend(
        [
            Spacer(1, 0.18 * inch),
            Paragraph("Interpretation", styles["Heading1"]),
            Paragraph(
                "The current evidence supports tiled kernels as the default 2D "
                "inference path and as a 3D fallback for collapsed high-occupancy "
                "distributions. CPU-rebuilt BVH is a useful traversal-quality "
                "baseline, fixed-order GPU BVH is too incoherent, sorted GPU-LBVH "
                "helps the worst fixed-order line case, and Morton GPU-LBVH improves "
                "broad-cloud BVH coherence but still pays too much sort overhead.",
                styles["BodyText"],
            ),
        ]
    )
    doc.build(story)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    old_gpu = gpu_rows([GPU_3D, GPU_2D])
    tiled_capacity = gpu_rows([GPU_2D_CAPACITY])
    new_auto = gpu_rows([GPU_AUTO_AFTER])
    bvh_gpu = gpu_rows([GPU_BVH])
    gpu_built_bvh = gpu_rows([GPU_BUILT_BVH])
    gpu_lbvh = gpu_rows([GPU_LBVH])
    gpu_morton_lbvh = gpu_rows([GPU_MORTON_LBVH])
    combined_gpu = (
        old_gpu
        + tiled_capacity
        + new_auto
        + bvh_gpu
        + gpu_built_bvh
        + gpu_lbvh
        + gpu_morton_lbvh
    )
    spatial = spatial_rows(SPATIAL)

    best_3d = table_best_exact(
        combined_gpu,
        "growing-3d-gs",
        8192,
        ["dense", "line", "plane", "torus", "uniform"],
    )
    best_growing_2d = table_best_exact(
        combined_gpu,
        "growing-2d",
        8192,
        ["dense", "line", "plane", "torus", "uniform"],
    )
    best_texture_2d = table_best_exact(
        combined_gpu,
        "texture-2d",
        8192,
        ["dense", "line", "plane", "torus", "uniform"],
    )
    speedups = auto_speedups(old_gpu, new_auto)
    spatial_table = spatial_summary(spatial)
    bvh_gpu_table = bvh_wgpu_summary(combined_gpu)

    tex = tex_document(
        best_3d,
        best_growing_2d,
        best_texture_2d,
        speedups,
        spatial_table,
        bvh_gpu_table,
    )
    tex_path = DOCS / "kernel_ablations.tex"
    tex_path.write_text(tex)

    pdf_path = DOCS / "kernel_ablations.pdf"
    pdf_story(
        "Spatial Index Ablations for GPU Neural Particle Automata Inference",
        [
            ("Auto-mode speedups", speedups),
            ("Executable WGPU BVH ablation", bvh_gpu_table),
            ("Best exact growing-2D rows", best_growing_2d),
            ("Best exact texture-2D rows", best_texture_2d),
            ("Best exact growing-3D rows", best_3d),
            ("CPU spatial strategy oracle", spatial_table),
        ],
        pdf_path,
    )
    print(f"wrote {tex_path.relative_to(ROOT)}")
    print(f"wrote {pdf_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
