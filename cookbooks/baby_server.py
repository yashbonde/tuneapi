# Copyright © 2024- Frello Technology Private Limited
# MIT License
# Author: @yashbonde
#
# Steps to make your own super dev setup using Tune Chat + Studio proxy
#
# [1] pip install tuneapi>=0.4.8, fastapi, pydantic, uvicorn
#
# [2] create an ngrok account and setup a static domain
#
# [3] on one terminal run ngrok proxy, it'd be something like:
#     $ ngrok http --domain=lively-sacred-dog.ngrok-free.app 8000
#     Remember to only use your static domain
#
# [4] go onto Tune Studio and deploy a new model using Custom subdomain
#     once created you cannot change it, so be super careful. Call the model
#     something like `<name>-mbp-local` so it's easy to find
#
# [5] start the server in another terminal like `python baby_server.py`
#
# [5] Go to Playground and start using the model

from typing import List
from pydantic import BaseModel

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

import tuneapi.types as tt
import tuneapi.apis as ta
import tuneapi.utils as tu

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    temperature: float
    max_tokens: int = 1024
    stream: bool = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    choices: List[ChatCompletionResponseChoice]


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, data: ChatCompletionRequest):
    # API validations
    if data.temperature < 0.0 or data.temperature > 1.0:
        raise HTTPException(400, "Temperature must be between 0 and 1")

    # call tune model
    # All your logic goes here
    model = ta.TuneModel(id="rohan/mixtral-8x7b-inst-v0-1-32k")
    thread = tt.Thread.from_dict({"chats": [m.model_dump() for m in data.messages]})
    print(thread)

    # return the response
    if data.stream:
        stream_resp = model.stream_chat(
            thread,
            temperature=data.temperature,
            max_tokens=data.max_tokens,
        )
        api_resp = tu.generator_to_api_events(
            model=model.tune_model_id,
            generator=stream_resp,
        )
        return StreamingResponse(api_resp, media_type="text/event-stream")
    else:
        output = model.chat(thread)
        response = ChatCompletionResponse(
            id=f"chatcmpl-{tu.get_snowflake()}",
            object="chat.completion",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(content=output, role="assistant"),
                    finish_reason="stop",
                )
            ],
        )
        return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
