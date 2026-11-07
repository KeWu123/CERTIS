"""Plot complete CERTIS evidence without using unfinished live results.

Run with --panel-a-only for the verified 32-cell certificate preview.  Without
that flag, a final two-panel figure is drawn only when native_live_analysis.json
exists, has status=pass, and contains the complete declared 32-run matrix.
No raw record, paper source, frozen driver, or earlier figure is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from PIL import Image
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
METHODS = ("batched_right", "adaptive_continuous", "separate", "joint")
SEEDS = (101, 202)
NAVY = "#233B5D"
GREEN = "#387B64"
GRAY = "#B9BDC3"
INK = "#222222"
CANVAS_POINTS = (514.0722045898438, 183.60000610351562)
WIDTH, HEIGHT = (value / 72 for value in CANVAS_POINTS)
MIN_FONT = 8.0
PNG_DPI = 600
STYLES = {
    "batched_right": dict(label="Exact", facecolor="white", hatch=""),
    "adaptive_continuous": dict(label="Per-side", facecolor=GRAY, hatch="//"),
    "separate": dict(label="Separate", facecolor=GREEN, hatch="\\\\"),
    "joint": dict(label="Joint", facecolor=NAVY, hatch=".."),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_record(path):
    path = Path(path).resolve()
    try:
        name = path.relative_to(WORKSPACE).as_posix()
    except ValueError:
        name = path.name
    return dict(path=name, sha256=sha(path), bytes=path.stat().st_size)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_sharp(directory):
    summary_path = directory / "summary.json"
    bound_path = directory / "fractional_bound.json"
    summary, bound = read_json(summary_path), read_json(bound_path)
    assert summary["status"] == bound["status"] == "pass"
    assert bound["secondary_post_hoc_diagnostic"]
    assert len(summary["rows"]) == len(bound["rows"]) == 32
    identify = lambda row: (row["case_id"], int(row["budget"]), int(row["seed"]))
    bounds = {identify(row): row for row in bound["rows"]}
    assert len(bounds) == 32
    points = []
    for row in summary["rows"]:
        other = bounds.pop(identify(row))
        upper = Fraction(other["separate_fractional_upper_bound"])
        integer_upper = upper.numerator // upper.denominator
        assert integer_upper == other["separate_integer_upper_bound"]
        assert row["initial_joint_discard"] == other["joint_greedy_discard"]
        assert row["initial_separate_discard"] == other["separate_greedy_discard"]
        assert row["raw"]["joint"]["sha256"] == other["raw_joint_sha256"]
        assert row["initial_joint_support"] <= 0
        advantage = max(0, row["initial_joint_discard"] - integer_upper)
        assert advantage == other["proven_cardinality_advantage"]
        saved = row["separate_requests"] - row["joint_requests"]
        assert saved == row["request_gap"]
        points.append(dict(
            case_id=row["case_id"], key=row["key"], budget=int(row["budget"]),
            seed=int(row["seed"]), proven_extra_discard_lower_bound=advantage,
            complete_requests_saved=saved, joint_initial_discard=row["initial_joint_discard"],
            separate_greedy_initial_discard=row["initial_separate_discard"],
            separate_fractional_upper_bound=str(upper), separate_integer_upper_bound=integer_upper,
            joint_requests=row["joint_requests"], separate_requests=row["separate_requests"],
            raw_joint_sha256=row["raw"]["joint"]["sha256"],
            raw_separate_sha256=row["raw"]["separate"]["sha256"],
        ))
    assert not bounds and len({point["case_id"] for point in points}) == 16
    assert {point["seed"] for point in points} == {101}
    assert Counter((point["key"], point["budget"]) for point in points) == {
        (key, budget): 8 for key in ("l1", "l2") for budget in (125, 500)
    }
    proven = [point for point in points if point["proven_extra_discard_lower_bound"] > 0]
    assert len(proven) == bound["cells_with_proven_cardinality_advantage"]
    overlap = Counter((point["proven_extra_discard_lower_bound"], point["complete_requests_saved"])
                      for point in points)
    panel = dict(
        rows=points, cells=32, related_queries=16, audit_seeds=[101], budgets=[125, 500],
        proven_cells=len(proven), proven_cells_with_more_requests=sum(
            point["complete_requests_saved"] < 0 for point in proven),
        exact_coordinate_overlaps=[dict(x=x, y=y, count=count) for (x, y), count in overlap.items()
                                   if count > 1],
        transformations=[
            "x = max(0, joint_initial_discard - floor(separate_fractional_upper_bound))",
            "y = separate_complete_requests - joint_complete_requests",
            "All 32 source cells retained at their exact coordinates; no jitter or smoothing.",
        ],
        zero_x_meaning="No strict advantage proved by this relaxation; not equality of optima.",
        replication="16 related native queries x two audit budgets, seed 101; not 32 independent workloads.",
        oracle_scope="Archived immutable Llama-3.3-70B labels; complete paid request counts from replay. "
                     "These counts are not measured inference latency.",
    )
    return panel, [source_record(summary_path), source_record(bound_path)]


def read_live(path):
    report = read_json(path)
    assert report["status"] == "pass", "Live analysis has not passed; refusing panel (b)"
    rows = report["rows"]
    assert len(rows) == 32, "Panel (b) requires all 32 actual live runs"
    cases = {row["case_id"] for row in rows}
    assert len(cases) == 4
    assert {row["key"] for row in rows} == {"l1", "l2"}
    assert {int(row["seed"]) for row in rows} == set(SEEDS)
    assert {row["method"] for row in rows} == set(METHODS)
    assert {(row["case_id"], row["method"], int(row["seed"])) for row in rows} == {
        (case, method, seed) for case in cases for method in METHODS for seed in SEEDS
    }
    assert report["all_paid_model_labels_rechecked"] and report["returned_first_sql"]
    for row in rows:
        assert row["live_outputs_used_for_policy"]
        assert row["all_live_labels_equal_fixed_reference"]
        assert math.isfinite(row["complete_seconds"]) and row["complete_seconds"] > 0
        assert row["audit_budget"] == (0 if row["method"] == "batched_right" else 500)
    bars = []
    for key in ("l1", "l2"):
        group = [row for row in rows if row["key"] == key]
        assert len(group) == 16 and len({row["case_id"] for row in group}) == 2
        exact = sum(row["complete_seconds"] for row in group if row["method"] == "batched_right")
        for method in METHODS:
            selected = [row for row in group if row["method"] == method]
            assert len(selected) == 4
            total = sum(row["complete_seconds"] for row in selected)
            bars.append(dict(key=key, method=method, pooled_complete_seconds=total,
                             exact_pooled_complete_seconds=exact, relative_complete_time=total / exact,
                             runs=4, related_queries=2, audit_seeds=list(SEEDS)))
    return dict(
        bars=bars, rows=rows, live_binding=report["binding"], executions=32,
        related_queries=4, audit_seeds=list(SEEDS), raw_sha256=report.get("raw_sha256", {}),
        transformations=[
            "For each l1/l2 group, sum complete_seconds for each method over both queries and seeds.",
            "Normalize that sum by exact right-first sum on the same group (ratio of pooled costs).",
            "No confidence interval or significance claim; seeds are repetitions of related queries.",
        ],
        scope="Actual paid single-slot Qwen inference plus audit, planning, bookkeeping and returned SQL; "
              "reference acquisition/validation, startup and warmup excluded. This is not replay latency.",
    ), source_record(path)


def draw_sharp(ax, panel):
    for key, marker in (("l1", "o"), ("l2", "s")):
        for budget in (125, 500):
            rows = [row for row in panel["rows"] if (row["key"], row["budget"]) == (key, budget)]
            ax.scatter(
                [row["proven_extra_discard_lower_bound"] for row in rows],
                [row["complete_requests_saved"] for row in rows],
                s=27, marker=marker, edgecolors=NAVY,
                facecolors="white" if budget == 125 else NAVY,
                linewidths=0.9, zorder=4,
            )
    max_x = max(row["proven_extra_discard_lower_bound"] for row in panel["rows"])
    ys = [row["complete_requests_saved"] for row in panel["rows"]]
    span = max(ys) - min(ys)
    ax.set_xlim(-0.055 * max_x, 1.075 * max_x)
    ax.set_ylim(min(ys) - 0.09 * span, max(ys) + 0.10 * span)
    ax.axvline(0, color="#8A8A8A", linestyle=(0, (2, 2)), linewidth=0.75, zorder=1)
    ax.axhline(0, color="#555555", linewidth=0.8, zorder=2)
    ax.set_xlabel("Proven extra discards (lower bound)", labelpad=4)
    ax.set_ylabel("Complete requests saved\n(separate minus joint)", labelpad=3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.5, zorder=0)
    handles = [Line2D([], [], linestyle="", marker=marker, markersize=4.6,
                      markerfacecolor="white" if budget == 125 else NAVY,
                      markeredgecolor=NAVY, markeredgewidth=0.9,
                      label=f"{key}, {budget}/side")
               for key, marker in (("l1", "o"), ("l2", "s")) for budget in (125, 500)]
    ax.legend(handles=handles, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.015),
              borderaxespad=0, columnspacing=1.1, handletextpad=0.4, handlelength=0.9,
              labelspacing=0.25, frameon=False, fontsize=8)
    ax.text(0.97, 0.96, f"{panel['proven_cells']}/32 proved", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=INK)


def draw_live(ax, panel):
    width = 0.17
    centers = (0.0, 1.1)
    for index, method in enumerate(METHODS):
        style = STYLES[method]
        ratios = [next(row["relative_complete_time"] for row in panel["bars"]
                       if (row["key"], row["method"]) == (key, method)) for key in ("l1", "l2")]
        positions = [center + (index - 1.5) * width for center in centers]
        ax.bar(positions, ratios, width=width * 0.89, facecolor=style["facecolor"],
               edgecolor="#333333", linewidth=0.65, hatch=style["hatch"], zorder=3)
    max_ratio = max(row["relative_complete_time"] for row in panel["bars"])
    ax.set_ylim(0, max(1.15, max_ratio * 1.14))
    ax.set_xlim(-0.53, 1.63)
    ax.axhline(1, color="#666666", linewidth=0.8, linestyle=(0, (3, 2)), zorder=4)
    ax.set_xticks(centers, ["l1 (coarse)", "l2 (fine)"])
    ax.set_xlabel("Native category keys", labelpad=4)
    ax.set_ylabel("Complete time / exact", labelpad=3)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.5, zorder=0)
    handles = [Patch(facecolor=STYLES[method]["facecolor"], edgecolor="#333333",
                     linewidth=0.65, hatch=STYLES[method]["hatch"], label=STYLES[method]["label"])
               for method in METHODS]
    ax.legend(handles=handles, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.015),
              borderaxespad=0, columnspacing=1.1, handletextpad=0.4, handlelength=1.3,
              labelspacing=0.25, frameon=False, fontsize=8)


def export(fig, directory, stem, provenance, force):
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / (stem + suffix) for suffix in (".pdf", ".png")]
    data_path = directory / "plot_data.json"
    caption_path = directory / "caption_stub.txt"
    existing = [path for path in paths + [data_path, caption_path] if path.exists()]
    if existing and not force:
        raise FileExistsError("Refusing to overwrite existing outputs; use --force: " + str(existing[0]))
    fig.canvas.draw()
    bounds = fig.get_tightbbox(fig.canvas.get_renderer())
    canvas_width, canvas_height = map(float, fig.get_size_inches())
    assert bounds.x0 >= 0 and bounds.y0 >= 0, "Text extends below or left of the canvas"
    assert bounds.x1 <= canvas_width and bounds.y1 <= canvas_height, "Text extends beyond the canvas"
    visible_text = [text for text in fig.findobj(matplotlib.text.Text)
                    if text.get_visible() and text.get_text()]
    assert min(text.get_fontsize() for text in visible_text) >= MIN_FONT
    pdf, png = paths
    fig.savefig(pdf, facecolor="white", edgecolor="white", transparent=False, bbox_inches=None,
                metadata={"Title": "CERTIS certification and complete-cost evidence", "Author": ""})
    fig.savefig(png, dpi=PNG_DPI, facecolor="white", edgecolor="white", transparent=False, bbox_inches=None)
    pdf_page = PdfReader(pdf).pages[0]
    pdf_inches = [float(pdf_page.mediabox.width) / 72, float(pdf_page.mediabox.height) / 72]
    assert all(abs(actual - target) < 1e-6 for actual, target in zip(pdf_inches, fig.get_size_inches()))
    font_subtypes = sorted({str(font.get_object().get("/Subtype"))
                            for font in pdf_page["/Resources"]["/Font"].get_object().values()})
    assert "/Type3" not in font_subtypes
    with Image.open(png) as picture:
        raster = dict(pixels=list(picture.size), mode=picture.mode,
                      dpi=list(picture.info.get("dpi", ())), opaque=picture.convert("RGBA").getextrema()[3] == (255, 255))
    assert raster["opaque"]
    expected = [round(inches * PNG_DPI) for inches in fig.get_size_inches()]
    assert all(abs(actual - target) <= 1 for actual, target in zip(raster["pixels"], expected))
    provenance["export"] = dict(
        canvas_inches=list(map(float, fig.get_size_inches())), minimum_font_points=MIN_FONT,
        full_two_panel_canvas_points=list(CANVAS_POINTS),
        rendered_minimum_font_points=min(text.get_fontsize() for text in visible_text),
        content_bounds_inches=[bounds.x0, bounds.y0, bounds.x1, bounds.y1],
        pdf_fonttype=42, pdf_font_subtypes=font_subtypes, pdf_page_inches=pdf_inches,
        png_dpi=PNG_DPI, raster=raster,
        files={path.name: dict(sha256=sha(path), bytes=path.stat().st_size) for path in paths},
        software=dict(python=platform.python_version(), matplotlib=matplotlib.__version__),
        publisher_status="Requested paper layout; no automatic publisher-compliance certification.",
    )
    caption_path.write_text(provenance["caption_stub"] + "\n", encoding="utf-8")
    data_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)
    print(json.dumps(dict(status="pass", panels=provenance["panels_drawn"],
                         outputs=[str(path) for path in paths + [data_path, caption_path]]), ensure_ascii=False))


def main(args):
    sharp, sources = read_sharp(args.sharp_results.resolve())
    live = None
    if not args.panel_a_only and args.live_analysis.is_file():
        live, record = read_live(args.live_analysis.resolve())
        sources.append(record)
    if args.require_live and live is None:
        raise FileNotFoundError("A complete, passed native_live_analysis.json is required")
    final = live is not None
    style = {
        "font.family": "DejaVu Sans", "font.size": 8.2,
        "axes.labelsize": 8.2, "axes.titlesize": 9.2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.65, "axes.edgecolor": "#555555", "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
        "hatch.linewidth": 0.5, "axes.unicode_minus": True,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2 if final else 1, figsize=(WIDTH if final else WIDTH / 2, HEIGHT))
        axes = [axes] if not final else list(axes)
        fig.subplots_adjust(left=0.106 if final else 0.212, right=0.986,
                            top=0.758, bottom=0.26, wspace=0.40)
        draw_sharp(axes[0], sharp)
        if final:
            draw_live(axes[1], live)
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
        for ax, title in zip(axes, ("(a) Sharpness and paid requests", "(b) Native live complete cost")):
            fig.text(ax.get_position().x0, 0.969, title, fontsize=9.0, fontweight="bold", va="top")
        left = axes[0].get_position().x0
        fig.text(left, 0.085, "x = 0: inconclusive, not equal.", fontsize=8, va="center")
        fig.text(left, 0.025, "Extra discards need not save requests.", fontsize=8, va="center")
        if final:
            right = axes[1].get_position().x0
            fig.text(right, 0.085, "4 related queries; 2 audit seeds each.", fontsize=8, va="center")
            fig.text(right, 0.025, "Pooled actual cost; exact = 1.", fontsize=8, va="center")
        caption = (
            "(a) Archived Llama-3.3-70B fixed-label replay: all 32 initial-cache cells from "
            "16 related native DBpedia queries, audit budgets 125/500 per side and seed 101. "
            "The x coordinate is max(0, G_joint - floor(U_separate)), a proved lower bound "
            "on extra certifiable initial discards; x = 0 is inconclusive, not equality. "
            "The y coordinate counts complete paid requests saved by joint versus separate greedy. "
            "Circles/squares denote l1/l2; hollow/solid markers denote audits 125/500. "
            "A cardinality advantage need not reduce adaptive requests. "
        )
        if final:
            caption += (
                "(b) Fresh Qwen2.5-7B Q4_K_M actual paid-label closed loop: all four related "
                "native queries, four methods and seeds 101/202 (32 measured runs). Bars are "
                "pooled complete seconds divided by pooled exact right-first seconds within "
                "each l1/l2 group. Cost includes real paid inference, audit, planning, bookkeeping "
                "and returned-first SQL; reference construction/validation, startup and warmup "
                "are excluded. The resident single-slot oracle executes logical batches serially. "
                "The panels use distinct oracles; seeds are repeated audits, not independent workloads."
            )
        else:
            caption += "The fresh-Qwen actual-latency panel is omitted from this preview."
        provenance = dict(
            status="pass", panels_drawn=["a", "b"] if final else ["a"],
            panel_b_status="drawn_from_complete_passed_analysis" if final else (
                "omitted_by_panel_a_only" if args.panel_a_only else "not_drawn_no_complete_live_analysis"),
            sources=sources + [source_record(__file__)], panel_a=sharp, panel_b=live,
            caption_stub=caption,
            design=dict(
                minimum_font_points=MIN_FONT, background="opaque white", colors=dict(joint=NAVY, separate=GREEN),
                panel_a_encodings="All points navy; circle=l1, square=l2; hollow=125/side, solid=500/side.",
                panel_b_encodings="Zero-based bars; white Exact, gray // Per-side, green backslash Separate, navy dots Joint.",
                uncertainty="None drawn. Related queries and repeated audit seeds do not provide independent replications.",
            ),
            alt_text="All 32 native certificate cells plot a proved lower bound on extra initial discards against complete "
                     "requests saved. Some positive cardinality gains increase requests; zero x is inconclusive. "
                     + ("The second panel compares pooled actual complete latency with exact right-first, separately for "
                        "coarse and fine keys, on four related native queries with two audit seeds each."
                        if final else "The live-cost panel is not drawn because no complete passed analysis was supplied."),
        )
        subdirectory = "final" if final else "preview_panel_a"
        stem = "fig_submission_evidence" if final else "fig_panel_a"
        export(fig, args.output_dir.resolve() / subdirectory, stem, provenance, args.force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sharp-results", type=Path,
                        default=WORKSPACE / "reproduction/sharp_value/results")
    parser.add_argument("--live-analysis", type=Path, default=HERE / "native_live_analysis.json")
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE / "outputs/submission_figures")
    parser.add_argument("--panel-a-only", action="store_true")
    parser.add_argument("--require-live", action="store_true")
    parser.add_argument("--force", action="store_true")
    main(parser.parse_args())
