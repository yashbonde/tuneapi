# Copyright © 2024- Frello Technology Private Limited

# model APIs
from tuneapi.apis.model_tune import TuneModel
from tuneapi.apis.model_openai import Openai
from tuneapi.apis.model_anthropic import Anthropic
from tuneapi.apis.model_groq import Groq
from tuneapi.apis.model_mistral import Mistral
from tuneapi.apis.model_gemini import Gemini

# projectX APIs
from tuneapi.apis.threads import ThreadsAPI

# other imports
import os
import random
from time import time
from typing import List, Optional

# other tuneapi modules
import tuneapi.types as tt
import tuneapi.utils as tu


def test_models(thread: str | tt.Thread, models: Optional[List[str]] = None):
    """
    Runs thread on all the models and prints the time taken and response.
    """
    if os.path.exists(thread):
        thread = tt.Thread.from_dict(tu.from_json(thread))

    # get all the models
    models_to_test = [TuneModel, Openai, Anthropic, Groq, Mistral, Gemini]
    if models and models != "all":
        models_to_test = []
        for m in models:
            models_to_test.append(globals()[m])

    # run all in a loop
    for model in models_to_test:
        print(tu.color.bold(f"[{model.__name__}]"), end=" ", flush=True)
        try:
            st = time()
            m = model()
            out = m.chat(thread)
            et = time()
            print(
                tu.color.blue(f"[{et-st:0.2f}s]"),
                tu.color.green(f"[SUCCESS]", True),
                out,
            )
        except Exception as e:
            et = time()
            print(
                tu.color.blue(f"[{et-st:0.2f}s]"),
                tu.color.red(f"[ERROR]", True),
                str(e),
            )
            continue


def benchmark_models(
    thread: str | tt.Thread,
    models: Optional[List[str]] = "all",
    n: int = 20,
    max_threads: int = 5,
    o: str = "benchmark.csv",
):
    """
    Benchmarks a thread on all the models and saves the time taken and response in a CSV file and creates matplotlib
    histogram chart with latency and char count distribution. Runs `n` iterations for each model.

    It requires `matplotlib>=3.8.2` and `pandas>=2.2.0` to be installed.
    """

    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        tu.logger.error(
            "This is a special CLI helper function. If you want to use this then run: pip install matplotlib>=3.8.2 pandas>=2.2.0"
        )
        raise ImportError("Please install the required packages")

    # if this is a JSON then load the thread
    if os.path.exists(thread):
        thread = tt.Thread.from_dict(tu.from_json(thread))

    # get all the models
    models_to_test = [TuneModel, Openai, Anthropic, Groq, Mistral, Gemini]
    if models and models != "all":
        models_to_test = []
        for m in models:
            models_to_test.append(globals()[m])

    # function to perform benchmarking
    def _bench(thread, model):
        try:
            st = time()
            m = model()
            out = m.chat(thread)
            return model.__name__, time() - st, out, False
        except Exception as e:
            return model.__name__, time() - st, str(e), True

    # threaded map and get the results
    inputs = []
    for m in models_to_test:
        for _ in range(n):
            inputs.append((thread, m))
    random.shuffle(inputs)
    print(f"Total combinations: {len(inputs)}")
    results = tu.threaded_map(
        fn=_bench,
        inputs=inputs,
        pbar=True,
        safe=False,
        max_threads=max_threads,
    )
    model_wise_errors = {}
    all_results = []
    for r in results:
        name, time_taken, out, error = r
        if error:
            model_wise_errors.setdefault(name, 0)
            model_wise_errors[name] += 1
        else:
            all_results.append(
                {
                    "model": name,
                    "time": time_taken,
                    "response": out,
                }
            )
    n_errors = sum(model_wise_errors.values())
    if n_errors:
        print(
            tu.color.red(f"{n_errors} FAILED", True)
            + f" ie. {n_errors/len(inputs)*100:.2f}% failure rate"
        )
    n_success = len(inputs) - n_errors
    print(
        tu.color.green(f"{n_success} SUCCESS", True)
        + f" ie. {n_success/len(inputs)*100:.2f}% success rate"
    )

    # create the report and save it
    df = pd.DataFrame(all_results)
    print("Created the benchmark report at:", tu.color.bold(o))
    df.to_csv(o, index=False)

    # create the histogram
    fig, axs = plt.subplots(3, 1, figsize=(15, 10))
    latency_by_models = {}
    char_count_by_models = {}
    for res in all_results:
        latency_by_models.setdefault(res["model"], []).append(res["time"])
        char_count_by_models.setdefault(res["model"], []).append(len(res["response"]))

    # histogram for latency
    axs[0].hist(
        latency_by_models.values(),
        bins=20,
        alpha=0.7,
        label=list(latency_by_models.keys()),
    )
    axs[0].set_title("Latency Distribution (lower is better)")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Frequency")
    axs[0].legend()

    # histogram for character count
    axs[1].hist(
        char_count_by_models.values(),
        bins=20,
        alpha=0.7,
        label=list(char_count_by_models.keys()),
    )
    axs[1].set_title("Character Count Distribution")
    axs[1].set_xlabel("Count")
    axs[1].set_ylabel("Frequency")
    axs[1].legend()
    plt.tight_layout()

    # bar graph for success and failure rate
    axs[2].bar(
        model_wise_errors.keys(),
        model_wise_errors.values(),
        color="red",
        label="Failed",
    )
    axs[2].set_title("Failure Rate (lower is better)")
    axs[2].set_xlabel("Model")
    axs[2].set_ylabel("Count")
    axs[2].legend()

    # save the plot
    print("Created the benchmark plot at:", tu.color.bold("benchmark.png"))
    plt.savefig("benchmark.png")
