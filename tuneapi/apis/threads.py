# Copyright © 2024- Frello Technology Private Limited

import json
import requests
from functools import cache
from typing import Optional, Dict, Any, Tuple, List

from tuneapi.utils import (
    ENV,
    SimplerTimes,
    from_json,
    to_json,
    Subway,
    logger,
    get_random_string,
    get_snowflake,
)
from tuneapi import types as tt


@cache
def get_sub(
    tune_org_id: Optional[str] = None,
    tune_api_key: Optional[str] = None,
) -> Subway:

    sess = Subway._get_session()
    sess.headers.update({"x-tune-key": tune_api_key})
    if tune_org_id:
        sess.headers.update({"X-Organization-Id": tune_org_id})
    return Subway("https://studio.rc.tune.app/v1/", sess)


class ThreadsAPI:
    def __init__(
        self,
        tune_org_id: Optional[str] = None,
        tune_api_key: Optional[str] = None,
    ):
        self.tune_org_id = tune_org_id or ENV.TUNE_ORG_ID("")
        self.tune_api_key = tune_api_key or ENV.TUNE_API_KEY()
        if not tune_api_key:
            raise ValueError("Either pass tune_api_key or set Env var TUNE_API_KEY")
        self.sub = get_sub(self.tune_org_id, self.tune_api_key)

    def set_token_and_org_id(
        self,
        tune_org_id: Optional[str] = None,
        tune_api_key: Optional[str] = None,
    ):
        self.tune_org_id = tune_org_id
        self.tune_api_key = tune_api_key
        self.sub = get_sub(self.tune_org_id, self.tune_api_key).threads

    # APIs
    def put_thread(
        self,
        thread: tt.Thread,
    ) -> tt.Thread:
        if not thread.title:
            thread.title = "autogen_" + str(get_snowflake())

        if thread.id:
            # validate this thread exists
            self.get_thread(thread.id)
            logger.warning(
                "Can only update title and metadata, messages will be ignored"
            )
            fn = self.sub.threads.u(thread.id)
            out = fn(
                "post",
                json={
                    "title": thread.title,
                    "metadata": thread.meta,
                },
            )
        else:
            logger.warning("Creating new thread")
            out = self.sub.threads(
                "post",
                json={
                    "title": thread.title,
                    "metadata": thread.meta,
                    "messages": [
                        m.to_dict(format="api", meta=False) for m in thread.chats
                    ],
                },
            )
            thread.id = out["id"]
        return thread

    def get_thread(
        self,
        thread_id: str,
        messages: bool = False,
    ) -> tt.Thread:
        # GET /threads/{thread_id}
        fn = self.sub.threads.u(thread_id)
        data = fn()
        data.pop("object")
        meta = data.pop("metadata", {})
        for k, v in meta.items():
            if k not in data:
                data[k] = v
        thread = tt.Thread(**data)

        if messages:
            # GET /threads/{thread_id}/messages
            fn = self.sub.threads.u(thread_id).messages
            data = fn()
            for m in data["data"]:
                text_items = list(filter(lambda x: x["type"] == "text", m["content"]))
                if len(text_items) != 1:
                    raise ValueError(f"Can handle one text only got: {len(text_items)}")
                msg = tt.Message(
                    value=text_items[0]["text"]["value"],
                    role=m["role"],
                    **{
                        "id": m["id"],
                        "createdAt": m["createdAt"],
                        "metadata": m["metadata"],
                    },
                )
                thread.append(msg)
        return thread

    def list_threads(
        self,
        limit: int = 20,
        order: Optional[str] = "desc",
        after: Optional[str] = None,
        before: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ) -> List[tt.Thread]:
        # GET /threads
        fn = self.sub.threads
        params = {
            "limit": limit,
            "order": order,
        }
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        if dataset_id:
            params["datasetId"] = dataset_id
        data = fn(params=params)
        all_threads = []
        for x in data["data"]:
            x.pop("object")
            meta = x.pop("metadata", {})
            for k, v in meta.items():
                if k not in data:
                    x[k] = v
            all_threads.append(tt.Thread(**x))
        return all_threads

    def fill_thread_messages(self, thread: tt.Thread):
        # GET /threads/{thread_id}/messages
        fn = self.sub.threads.u(thread.id).messages
        data = fn()
        for m in data["data"]:
            text_items = list(filter(lambda x: x["type"] == "text", m["content"]))
            if len(text_items) != 1:
                raise ValueError(f"Can handle one text only got: {len(text_items)}")
            msg = tt.Message(
                value=text_items[0]["text"]["value"],
                role=m["role"],
                **{
                    "id": m["id"],
                    "createdAt": m["createdAt"],
                    "metadata": m["metadata"],
                },
            )
            thread.append(msg)
        return thread
