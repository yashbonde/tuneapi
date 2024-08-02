# Copyright © 2023- Frello Technology Private Limited
# Original author: Yash Bonde
# Original license: Apache-2.0 License
# Original source: github.com/NimbleBoxAI/ChainFury

import inspect
from typing import Any, List, Union, Dict


class Var:
    def __init__(
        self,
        name: str = "",
        description: str = "",
        required: bool = False,
        type: Union[str, List["Var"]] = "",
        additionalProperties: Union[List["Var"], "Var"] = [],
    ):
        if not type:
            raise ValueError("type cannot be empty")

        self.type = type
        self.additionalProperties = additionalProperties
        self.required = required
        self.name = name
        self.description = description

    def __repr__(self) -> str:
        x = f"Var('{'*' if self.required else ''}{self.name}', type='{self.type}'"
        if self.additionalProperties:
            x += f", additionalProperties={self.additionalProperties}"
        x += ")"
        return x

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this Var to a dictionary that can be JSON serialised and sent to the client.

        Returns:
            Dict[str, Any]: The serialised Var.
        """
        d: Dict[str, Any] = {"type": self.type}
        if type(self.type) == list and len(self.type) and type(self.type[0]) == Var:
            d["type"] = [x.to_dict() for x in self.type]
        if self.additionalProperties:
            if isinstance(self.additionalProperties, Var):
                d["additionalProperties"] = self.additionalProperties.to_dict()
            else:
                d["additionalProperties"] = self.additionalProperties
        if self.required:
            d["required"] = self.required
        if self.name:
            d["name"] = self.name
        if self.description:
            d["description"] = self.description
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Var":
        """Deserialise a Var from a dictionary.

        Args:
            d (Dict[str, Any]): The dictionary to deserialise from.

        Returns:
            Var: The deserialised Var.
        """
        type_val = d.get("type")
        additional_properties_val = d.get("additionalProperties", [])
        required_val = d.get("required", False)
        name_val = d.get("name", "")
        description_val = d.get("description", "")

        if isinstance(type_val, list):
            type_val = [
                Var.from_dict(x) if isinstance(x, dict) else x for x in type_val
            ]
        elif isinstance(type_val, dict):
            type_val = Var.from_dict(type_val)

        items_val = [Var.from_dict(x) if isinstance(x, dict) else x for x in items_val]
        additional_properties_val = (
            Var.from_dict(additional_properties_val)
            if isinstance(additional_properties_val, dict)
            else additional_properties_val
        )

        var = cls(
            type=type_val,
            items=items_val,
            additionalProperties=additional_properties_val,
            required=required_val,
            name=name_val,
            description=description_val,
        )
        return var


def pyannotation_to_json_schema(
    x: Any,
    allow_any: bool,
    allow_exc: bool,
    allow_none: bool,
    *,
    trace: bool = False,
    is_return: bool = False,
) -> Var:
    """Function to convert the given annotation from python to a Var which can then be JSON serialised and sent to the
    clients.

    Args:
        x (Any): The annotation to convert.
        allow_any (bool): Whether to allow the `Any` type.
        allow_exc (bool): Whether to allow the `Exception` type.
        allow_none (bool): Whether to allow the `None` type.
        trace (bool, optional): Adds verbosity the schema generation. Defaults to False.

    Returns:
        Var: The converted annotation.
    """
    if isinstance(x, type):
        if trace:
            print("t0")

        if x == Any:
            return Var(type="any")
        elif x == str:
            return Var(type="string")
        elif x == int or x == float:
            return Var(type="number")
        elif x == bool:
            return Var(type="boolean")
        elif x == bytes:
            return Var(type="string", format="byte")
        elif x == list:
            return Var(type="array", items=[Var(type="string")])
        elif x == dict:
            return Var(type="object", additionalProperties=Var(type="string"))

        # there are some types that are unique to the fury system
        elif x == type(None) and allow_none:
            return Var(type="null", required=False, show=False)
        else:
            if is_return:
                raise ValueError(
                    f"i0-r: Unsupported type: {x}. Is your output annotated? Write like ... foo() -> Dict[str, str]"
                )
            else:
                raise ValueError(
                    f"i0: Unsupported type: {x}. Some of your inputs are not annotated. Write like ... foo(x: str)"
                )
    elif isinstance(x, str):
        if trace:
            print("t1")
        return Var(type="string")
    elif hasattr(x, "__origin__") and hasattr(x, "__args__"):
        if trace:
            print("t2")
        if x.__origin__ == list:
            if trace:
                print("t2.1")
            return Var(
                type="array",
                items=[
                    pyannotation_to_json_schema(
                        x=x.__args__[0],
                        allow_any=allow_any,
                        allow_exc=allow_exc,
                        allow_none=allow_none,
                        trace=trace,
                    )
                ],
            )
        elif x.__origin__ == dict:
            if len(x.__args__) == 2 and x.__args__[0] == str:
                if trace:
                    print("t2.2")
                return Var(
                    type="object",
                    additionalProperties=pyannotation_to_json_schema(
                        x=x.__args__[1],
                        allow_any=allow_any,
                        allow_exc=allow_exc,
                        allow_none=allow_none,
                        trace=trace,
                    ),
                )
            else:
                raise ValueError(f"i2: Unsupported type: {x}")
        elif x.__origin__ == tuple:
            if trace:
                print("t2.3")
            return Var(
                type="array",
                items=[
                    pyannotation_to_json_schema(
                        x=arg,
                        allow_any=allow_any,
                        allow_exc=allow_exc,
                        allow_none=allow_none,
                        trace=trace,
                    )
                    for arg in x.__args__
                ],
            )
        elif x.__origin__ == Union:
            # Unwrap union types with None type
            types = [arg for arg in x.__args__ if arg is not None]
            if len(types) == 1:
                if trace:
                    print("t2.4")
                return pyannotation_to_json_schema(
                    x=types[0],
                    allow_any=allow_any,
                    allow_exc=allow_exc,
                    allow_none=allow_none,
                    trace=trace,
                )
            else:
                if trace:
                    print("t2.5")
                return Var(
                    type=[
                        pyannotation_to_json_schema(
                            x=typ,
                            allow_any=allow_any,
                            allow_exc=allow_exc,
                            allow_none=allow_none,
                            trace=trace,
                        )
                        for typ in types
                    ]
                )
        else:
            raise ValueError(f"i3: Unsupported type: {x}")
    elif isinstance(x, tuple):
        if trace:
            print("t4")
        return Var(
            type="array",
            items=[
                Var(type="string"),
                pyannotation_to_json_schema(
                    x=x[1],
                    allow_any=allow_any,
                    allow_exc=allow_exc,
                    allow_none=allow_none,
                    trace=trace,
                ),
            ]
            * len(x),
        )
    elif x == Any and allow_any:
        if trace:
            print("t5")
        return Var(type="string")
    else:
        if trace:
            print("t6")
        raise ValueError(f"i4: Unsupported type: {x}")


def func_to_vars(func: object, log_trace: bool = False) -> List[Var]:
    """
    Extracts the signature of a function and converts it to an array of Var objects.

    Args:
        func (Callable): The function to extract the signature from.

    Returns:
        List[Var]: The array of Var objects.
    """
    signature = inspect.signature(func)  # type: ignore
    fields = []
    for param in signature.parameters.values():
        schema = pyannotation_to_json_schema(
            param.annotation,
            allow_any=False,
            allow_exc=False,
            allow_none=False,
            trace=log_trace,
        )
        schema.required = param.default is inspect.Parameter.empty
        schema.name = param.name
        schema.placeholder = (
            str(param.default) if param.default is not inspect.Parameter.empty else ""
        )
        if not schema.name.startswith("_"):
            schema.show = True
        fields.append(schema)
    return fields


if __name__ == "__main__":
    schema = func_to_vars(pyannotation_to_json_schema, log_trace=False)
    print([x.to_dict() for x in schema])
