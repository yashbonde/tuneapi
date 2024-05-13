# Copyright © 2024- Frello Technology Private Limited

from fire import Fire


def main():
    from tuneapi.apis import test_models, benchmark_models

    Fire(
        {
            "models": {
                "test": test_models,
                "benchmark": benchmark_models,
            },
        }
    )


if __name__ == "__main__":
    main()
