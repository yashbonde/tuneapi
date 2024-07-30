# Copyright © 2024- Frello Technology Private Limited

from functools import cache
from typing import Optional, List

import tuneapi.utils as tu
from tuneapi import types as tt


@cache
def get_sub(
    base_url,
    tune_org_id: str,
    tune_api_key: str,
) -> tu.Subway:

    sess = tu.Subway._get_session()
    sess.headers.update({"x-tune-key": tune_api_key})
    if tune_org_id:
        sess.headers.update({"X-Organization-Id": tune_org_id})
    print("::::::", sess.headers)
    return tu.Subway(base_url, sess)


class ThreadsAPI:
    def __init__(
        self,
        tune_org_id: Optional[str] = None,
        tune_api_key: Optional[str] = None,
        base_url: str = "https://studio.tune.app/v1/",
    ):
        self.tune_org_id = tune_org_id or tu.ENV.TUNE_ORG_ID()
        self.tune_api_key = tune_api_key or tu.ENV.TUNE_API_KEY()
        if not tune_api_key:
            raise ValueError("Either pass tune_api_key or set Env var TUNE_API_KEY")
        self.sub = get_sub(base_url, self.tune_org_id, self.tune_api_key)

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
            thread.title = "thread_" + tu.get_snowflake()

        body = {
            "title": thread.title,
            "messages": [m.to_dict(format="api", meta=False) for m in thread.chats],
        }
        m = {}
        if thread.meta:
            m["thread_meta"] = thread.meta
        if thread.evals:
            m["evals"] = thread.evals
        if m:
            body["metadata"] = {"meta": tu.to_json(m, tight=True)}

        tu.logger.info("Creating new thread")
        out = self.sub.threads("post", json=body)
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
        meta = data.get("metadata", {})
        if meta == None:
            meta = {}
        extra = {}
        for k, v in meta.items():
            if k not in data:
                extra[k] = v
        thread = tt.Thread(
            id=data["id"],
            title=data["title"],
            evals=meta.get("evals", {}),
            **extra,
        )

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
