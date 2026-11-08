#!/usr/bin/env python3

import argparse
import csv
import statistics
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI


PROMPT_FILE = Path(
    "benchmark_data/openai/prompts.csv"
)

RESULT_ROOT = Path(
    "results/task_f_openai_benchmark"
)

MODELS = [
    "gpt-6-luna",
    "gpt-6.1-terra",
    "gpt-6.1-sol",
    "gpt-6-astra",
]

REPETITIONS = 3

INSTRUCTIONS = (
    "You are a friendly service robot. "
    "Answer accurately and clearly for spoken interaction. "
    "Use no more than three short sentences. "
    "Do not use Markdown lists or headings."
)


def load_prompts():
    with PROMPT_FILE.open(
        newline="",
        encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fields):
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


def get_usage(response, field):
    usage = getattr(response, "usage", None)

    if usage is None:
        return 0

    return getattr(usage, field, 0) or 0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Test only Q01 once with each model.",
    )

    args = parser.parse_args()

    prompts = load_prompts()
    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.smoke_test:
        prompts = prompts[:1]
        repetitions = 1
        run_type = "smoke"
    else:
        repetitions = REPETITIONS
        run_type = "full"

    run_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    detail_path = RESULT_ROOT / (
        f"openai_{run_type}_results_{run_id}.csv"
    )

    summary_path = RESULT_ROOT / (
        f"openai_{run_type}_summary_{run_id}.csv"
    )

    fields = [
        "run_id",
        "repetition",
        "prompt_id",
        "category",
        "model",
        "question",
        "quality_requirements",
        "response_text",
        "response_time_seconds",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "status",
        "error",
        "manual_quality_score",
        "manual_notes",
    ]

    client = OpenAI()
    rows = []

    total_requests = (
        len(prompts)
        * len(MODELS)
        * repetitions
    )

    request_number = 0

    print("Run type:", run_type)
    print("Models:", len(MODELS))
    print("Questions:", len(prompts))
    print("Repetitions:", repetitions)
    print("Total requests:", total_requests)

    for repetition in range(
        1,
        repetitions + 1,
    ):
        for prompt in prompts:
            for model_name in MODELS:
                request_number += 1

                print()
                print("=" * 70)
                print(
                    f"Request {request_number}/"
                    f"{total_requests}"
                )
                print(
                    model_name,
                    prompt["prompt_id"],
                    f"repetition {repetition}",
                )

                started = time.perf_counter()

                try:
                    response = client.responses.create(
                        model=model_name,
                        reasoning={
                            "effort": "low",
                        },
                        instructions=INSTRUCTIONS,
                        input=prompt["question"],
                        max_output_tokens=400,
                    )

                    elapsed = (
                        time.perf_counter()
                        - started
                    )

                    response_text = (
                        response.output_text.strip()
                    )

                    input_tokens = get_usage(
                        response,
                        "input_tokens",
                    )

                    output_tokens = get_usage(
                        response,
                        "output_tokens",
                    )

                    total_tokens = get_usage(
                        response,
                        "total_tokens",
                    )

                    status = "success"
                    error = ""

                    print(
                        "Response time:",
                        round(elapsed, 3),
                        "seconds",
                    )
                    print("Answer:", response_text)

                except Exception as exception:
                    elapsed = (
                        time.perf_counter()
                        - started
                    )

                    response_text = ""
                    input_tokens = 0
                    output_tokens = 0
                    total_tokens = 0
                    status = "error"
                    error = str(exception)

                    print("ERROR:", error)

                rows.append(
                    {
                        "run_id": run_id,
                        "repetition": repetition,
                        "prompt_id":
                            prompt["prompt_id"],
                        "category":
                            prompt["category"],
                        "model": model_name,
                        "question":
                            prompt["question"],
                        "quality_requirements":
                            prompt[
                                "quality_requirements"
                            ],
                        "response_text":
                            response_text,
                        "response_time_seconds":
                            round(elapsed, 3),
                        "input_tokens":
                            input_tokens,
                        "output_tokens":
                            output_tokens,
                        "total_tokens":
                            total_tokens,
                        "status": status,
                        "error": error,
                        "manual_quality_score": "",
                        "manual_notes": "",
                    }
                )

                write_csv(
                    detail_path,
                    rows,
                    fields,
                )

    summary_rows = []

    for model_name in MODELS:
        for category in [
            "short",
            "medium",
            "complex",
            "all",
        ]:
            selected = [
                row
                for row in rows
                if row["model"] == model_name
                and (
                    category == "all"
                    or row["category"] == category
                )
            ]

            successful = [
                row
                for row in selected
                if row["status"] == "success"
            ]

            if not selected:
                continue

            times = [
                row["response_time_seconds"]
                for row in successful
            ]

            if times:
                mean_time = statistics.mean(times)
                median_time = statistics.median(times)
                minimum_time = min(times)
                maximum_time = max(times)
            else:
                mean_time = 0
                median_time = 0
                minimum_time = 0
                maximum_time = 0

            summary_rows.append(
                {
                    "run_id": run_id,
                    "model": model_name,
                    "category": category,
                    "requests": len(selected),
                    "successful":
                        len(successful),
                    "errors":
                        len(selected)
                        - len(successful),
                    "mean_response_seconds":
                        round(mean_time, 3),
                    "median_response_seconds":
                        round(median_time, 3),
                    "minimum_response_seconds":
                        round(minimum_time, 3),
                    "maximum_response_seconds":
                        round(maximum_time, 3),
                    "mean_input_tokens":
                        round(
                            statistics.mean(
                                row["input_tokens"]
                                for row in successful
                            ),
                            1,
                        )
                        if successful
                        else 0,
                    "mean_output_tokens":
                        round(
                            statistics.mean(
                                row["output_tokens"]
                                for row in successful
                            ),
                            1,
                        )
                        if successful
                        else 0,
                }
            )

    summary_fields = list(
        summary_rows[0].keys()
    )

    write_csv(
        summary_path,
        summary_rows,
        summary_fields,
    )

    print()
    print("=" * 70)
    print("Benchmark complete.")
    print("Detailed results:", detail_path)
    print("Summary results:", summary_path)


if __name__ == "__main__":
    main()
