# Copyright © 2024- Frello Technology Private Limited


class Evals:
    """
    A simple class containing different evaluation metrics. Each function is self explanatory and returns a JSON logic
    object.
    """

    # small things like unit tests
    def exactly(x):
        return {"==": [{"var": "response"}, x]}

    def contains(x):
        return {"in": [{"var": "response"}, x]}

    def is_function(name: str, **kwargs: dict):
        if not kwargs:
            return {"==": [{"var": "response.name"}, name]}
        args = []
        for k, v in kwargs.items():
            args.append({"==": [{"var": f"response.arguments.{k}"}, v]})
        return {"and": [{"==": [{"var": "response.name"}, name]}, *args]}
