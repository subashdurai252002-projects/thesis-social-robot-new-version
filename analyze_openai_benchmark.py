#!/usr/bin/env python3

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


RESULT_ROOT = Path(
    "results/task_f_openai_benchmark"
)

QUALITY_FILE = (
    RESULT_ROOT / "openai_quality_review.csv"
)

MODELS = [
    "gpt-6-luna",
    "gpt-6.1-terra",
    "gpt-6.1-sol",
    "gpt-6-astra",
]

SHORT_NAMES = {
    "gpt-6-luna": "Luna",
    "gpt-6.1-terra": "Terra",
    "gpt-6.1-sol": "Sol",
    "gpt-6-astra": "Astra",
}

COLORS = {
    "gpt-6-luna": "#8ea6bb",
    "gpt-6.1-terra": "#29928d",
    "gpt-6.1-sol": "#376899",
    "gpt-6-astra": "#203f68",
}


def read_csv(path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    fields = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def median_time(rows):
    return statistics.median(
        float(row["response_time_seconds"])
        for row in rows
    )


def main():
    result_files = sorted(
        RESULT_ROOT.glob(
            "openai_full_results_*.csv"
        )
    )

    if not result_files:
        raise FileNotFoundError(
            "No full OpenAI result file found."
        )

    detail_file = result_files[-1]

    detail_rows = read_csv(detail_file)
    quality_rows = read_csv(QUALITY_FILE)

    if len(detail_rows) != 108:
        raise RuntimeError(
            f"Expected 108 results, found "
            f"{len(detail_rows)}."
        )

    if len(quality_rows) != 36:
        raise RuntimeError(
            f"Expected 36 quality rows, found "
            f"{len(quality_rows)}."
        )

    quality_map = {
        (
            row["model"],
            row["prompt_id"],
        ): row
        for row in quality_rows
    }

    final_rows = []

    for model in MODELS:
        model_rows = [
            row
            for row in detail_rows
            if row["model"] == model
        ]

        category_medians = {}

        for category in [
            "short",
            "medium",
            "complex",
        ]:
            selected = [
                row
                for row in model_rows
                if row["category"] == category
            ]

            category_medians[category] = (
                median_time(selected)
            )

        reviews = [
            row
            for row in quality_rows
            if row["model"] == model
        ]

        quality_scores = [
            int(row["quality_score"])
            for row in reviews
        ]

        suitable_count = sum(
            row["suitable_for_speech"]
            == "yes"
            for row in reviews
        )

        times = [
            float(
                row["response_time_seconds"]
            )
            for row in model_rows
        ]

        successful = sum(
            row["status"] == "success"
            for row in model_rows
        )

        final_rows.append(
            {
                "model": model,
                "requests": len(model_rows),
                "successful": successful,
                "errors":
                    len(model_rows) - successful,
                "short_median_seconds":
                    round(
                        category_medians["short"],
                        3,
                    ),
                "medium_median_seconds":
                    round(
                        category_medians["medium"],
                        3,
                    ),
                "complex_median_seconds":
                    round(
                        category_medians["complex"],
                        3,
                    ),
                "overall_median_seconds":
                    round(
                        statistics.median(times),
                        3,
                    ),
                "overall_mean_seconds":
                    round(
                        statistics.mean(times),
                        3,
                    ),
                "maximum_seconds":
                    round(max(times), 3),
                "requests_over_10_seconds":
                    sum(
                        value > 10
                        for value in times
                    ),
                "mean_input_tokens":
                    round(
                        statistics.mean(
                            int(
                                row[
                                    "input_tokens"
                                ]
                            )
                            for row in model_rows
                        ),
                        1,
                    ),
                "mean_output_tokens":
                    round(
                        statistics.mean(
                            int(
                                row[
                                    "output_tokens"
                                ]
                            )
                            for row in model_rows
                        ),
                        1,
                    ),
                "mean_quality_score":
                    round(
                        statistics.mean(
                            quality_scores
                        ),
                        2,
                    ),
                "speech_suitable_questions":
                    f"{suitable_count}/9",
                "speech_suitable_percent":
                    round(
                        100
                        * suitable_count
                        / len(reviews),
                        1,
                    ),
            }
        )

    comparison_file = (
        RESULT_ROOT
        / "openai_final_comparison.csv"
    )

    write_csv(
        comparison_file,
        final_rows,
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    categories = [
        "Short",
        "Medium",
        "Complex",
    ]

    x_positions = list(
        range(len(categories))
    )

    width = 0.18

    fig, axis = plt.subplots(
        figsize=(8, 4.8)
    )

    for index, row in enumerate(final_rows):
        values = [
            row["short_median_seconds"],
            row["medium_median_seconds"],
            row["complex_median_seconds"],
        ]

        positions = [
            value
            + (
                index
                - (len(final_rows) - 1) / 2
            )
            * width
            for value in x_positions
        ]

        axis.bar(
            positions,
            values,
            width=width,
            label=SHORT_NAMES[row["model"]],
            color=COLORS[row["model"]],
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(categories)
    axis.set_ylabel(
        "Median response time (seconds)"
    )
    axis.set_title(
        "OpenAI Response Time by Question Complexity"
    )
    axis.legend()
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    latency_plot = (
        RESULT_ROOT
        / "openai_median_latency_by_complexity.png"
    )

    fig.savefig(
        latency_plot,
        dpi=200,
    )

    plt.close(fig)

    fig, axis = plt.subplots(
        figsize=(7, 4.8)
    )

    for row in final_rows:
        model = row["model"]

        axis.scatter(
            row["overall_median_seconds"],
            row["mean_quality_score"],
            s=100,
            color=COLORS[model],
        )

        axis.annotate(
            SHORT_NAMES[model],
            (
                row[
                    "overall_median_seconds"
                ],
                row["mean_quality_score"],
            ),
            xytext=(6, 5),
            textcoords="offset points",
        )

    axis.set_xlabel(
        "Overall median response time (seconds)"
    )
    axis.set_ylabel(
        "Mean manual quality score (1–5)"
    )
    axis.set_title(
        "OpenAI Quality versus Response Time"
    )
    axis.grid(alpha=0.25)

    fig.tight_layout()

    quality_plot = (
        RESULT_ROOT
        / "openai_quality_vs_latency.png"
    )

    fig.savefig(
        quality_plot,
        dpi=200,
    )

    plt.close(fig)

    grouped = defaultdict(list)

    for row in detail_rows:
        grouped[
            (
                row["model"],
                row["prompt_id"],
            )
        ].append(row)

    fig, axis = plt.subplots(
        figsize=(7.5, 4.8)
    )

    for model in MODELS:
        points = []

        for (
            group_model,
            prompt_id,
        ), rows in grouped.items():
            if group_model != model:
                continue

            input_tokens = statistics.mean(
                int(row["input_tokens"])
                for row in rows
            )

            response_median = median_time(rows)

            points.append(
                (
                    input_tokens,
                    response_median,
                )
            )

        points.sort()

        axis.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            label=SHORT_NAMES[model],
            color=COLORS[model],
        )

    axis.set_xlabel(
        "Mean input tokens per question"
    )
    axis.set_ylabel(
        "Median response time (seconds)"
    )
    axis.set_title(
        "Input Length versus OpenAI Response Time"
    )
    axis.legend()
    axis.grid(alpha=0.25)

    fig.tight_layout()

    input_plot = (
        RESULT_ROOT
        / "openai_input_tokens_vs_latency.png"
    )

    fig.savefig(
        input_plot,
        dpi=200,
    )

    plt.close(fig)

    fig, axis = plt.subplots(
        figsize=(7, 4.5)
    )

    names = [
        SHORT_NAMES[row["model"]]
        for row in final_rows
    ]

    percentages = [
        row["speech_suitable_percent"]
        for row in final_rows
    ]

    bars = axis.bar(
        names,
        percentages,
        color=[
            COLORS[row["model"]]
            for row in final_rows
        ],
    )

    axis.set_ylim(0, 110)
    axis.set_ylabel(
        "Speech-suitable answers (%)"
    )
    axis.set_title(
        "Suitability for Spoken Robot Responses"
    )
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        percentages,
    ):
        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 2,
            f"{value:.1f}%",
            ha="center",
        )

    fig.tight_layout()

    suitability_plot = (
        RESULT_ROOT
        / "openai_speech_suitability.png"
    )

    fig.savefig(
        suitability_plot,
        dpi=200,
    )

    plt.close(fig)

    print("Analysis complete.")
    print("Input results:", detail_file)
    print("Comparison:", comparison_file)
    print("Latency plot:", latency_plot)
    print("Quality plot:", quality_plot)
    print("Input-length plot:", input_plot)
    print(
        "Speech-suitability plot:",
        suitability_plot,
    )

    print()
    print("FINAL MODEL COMPARISON")

    for row in final_rows:
        print(
            row["model"],
            "| median:",
            row["overall_median_seconds"],
            "| quality:",
            row["mean_quality_score"],
            "| speech:",
            row["speech_suitable_questions"],
        )


if __name__ == "__main__":
    main()
