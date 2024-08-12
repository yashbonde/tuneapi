# Copyright © 2024- Frello Technology Private Limited

# this file contains the code for the assistants API

from functools import cache
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict

import tuneapi.utils as tu
import tuneapi.types as tt

from tuneapi.endpoints.common import get_sub


@dataclass
class GenerationConfig:
    model_id: str
    temperature: float


class AssistantsAPI:
    def __init__(
        self,
        tune_org_id: str = None,
        tune_api_key: str = None,
        base_url: str = "https://studio.tune.app/",
    ):
        self.tune_org_id = tune_org_id or tu.ENV.TUNEORG_ID()
        self.tune_api_key = tune_api_key or tu.ENV.TUNEAPI_TOKEN()
        self.base_url = base_url
        if not self.tune_api_key:
            raise ValueError("Either pass tune_api_key or set Env var TUNEAPI_TOKEN")
        self.sub = get_sub(
            base_url + "v1/assistants/", self.tune_org_id, self.tune_api_key
        )

    def list_assistants(self, limit: int = 10, order: str = "desc"):
        out = self.sub(params={"limit": limit, "order": order})
        return out["data"]

    def create(
        self,
        model_name: str,
        instructions: str,
        function_model: GenerationConfig,
        response_model: GenerationConfig = None,
        user_filters: Dict[str, str] = None,
        tools: List[tt.Tool] = [],
    ):
        data = {
            "model_name": model_name,
            "instructions": instructions,
            "function_model": asdict(function_model),
            "response_model": (
                asdict(response_model if response_model else function_model)
            ),
            "tools": [
                {"type": "user_function", "function": t.to_dict()} for t in tools
            ],
            "user_filters": user_filters,
        }
        print(tu.to_json(data))
        return self.sub("POST", json=data)

    def delete(self, assistant_id: str):
        return self.sub.u(assistant_id)("DELETE")

    def get(self, assistant_id: str):
        return self.sub.u(assistant_id)()

    def modify(
        self,
        assistant_id: str,
        *,
        instructions: str = "",
        assistant_name: str = "",
        function_model: GenerationConfig = None,
        response_model: GenerationConfig = None,
        user_filters: Dict[str, str] = None,
        tools: List[tt.Tool] = []
    ):
        assistant_update = {}
        update_mask = ""
        if instructions:
            assistant_update["instructions"] = instructions
            update_mask += "instructions,"
        if assistant_name:
            assistant_update["assistant_name"] = assistant_name
            update_mask += "assistant_name,"
        if function_model:
            assistant_update["function_model"] = asdict(function_model)
            update_mask += "function_model,"
        if response_model:
            assistant_update["response_model"] = asdict(response_model)
            update_mask += "response_model,"
        if user_filters:
            assistant_update["user_filters"] = user_filters
            update_mask += "user_filters,"
        if tools:
            assistant_update["tools"] = [
                {"type": "user_function", "data": t.to_dict()} for t in tools
            ]
            update_mask += "tools,"
        update_mask = update_mask[:-1]

        if not assistant_update:
            raise ValueError("No fields to update")
        print(assistant_update)

        return self.sub.u(assistant_id)(
            "PATCH",
            json={
                "assistant": assistant_update,
                "update_mask": update_mask,
            },
        )
