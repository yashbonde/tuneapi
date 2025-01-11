.. tuneapi documentation master file, created by
   sphinx-quickstart on Tue Jul 30 17:14:39 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

tuneapi documentation
=====================

Welcome to the documentation for ``tuneapi`` package. This combines the most commonly used python utilities across Tune AI
into a single MIT-licensed package. It contains 3 major submodules:
- ``tuneapi.apis``: Contains all the APIs that are used to interact with the Tune AI services and LLM providers
- ``tuneapi.types``: Contains all the types that are used generally
- ``tuneapi.utils``: Contains all the utility functions that are used across the Tune AI codebase. This is pretty interesting
- ``tuneapi.endpoints``: Contains all the API endpoints for the Tune Studio


Prompt for Chat
---------------

Since ``tuneapi`` is a new and unpopular package most LLMs will not be able to generate code based on it. However you can
paste the following code snippet in the prompt to generate the code for LLM API calls.

.. code-block:: markdown
    
    You are going to use `tuneapi` package to use the LLM endpoint. Use structured generation for logical parts of
    the process. Here's an example on how to use `tuneapi`:

    ```python
    from tuneapi import tt, ta

    # define a thread which is a collection of messages with system, user and assistant messages
    thread = tt.Thread(
        tt.system(...),   # add optional system message here
        tt.human(...),    # add user message here
        tt.assistant(...) # for assistant response
    )

    # define a model
    model = ta.Gemini() # other LLMs: Openai, Anthropic, Groq, TuneModel, Mistral

    # get the response
    resp: str = model.chat(thread)

    # You can also generate structured response for better control
    from pydantic import BaseModel, Field
    from typing import List, Optional

    class MathProblem(BaseModel):
        a: int = Field(..., description="First number")
        b: int = Field(..., description="Second number")
        operator: str = Field(..., description="Operator")
        hint: Optional[str] = Field(None, description="Hint for the problem")

    class MathTest(BaseModel):
        title: str = Field(..., description="Title of the test")
        problems: List[MathProblem] = Field(..., description="List of math problems")

    # define a thread which is a collection of messages
    thread = tt.Thread(
        tt.human("Give me 5 problems for KG-1 class"),
        schema=MathTest
    )

    # get structured output
    resp: MathTest = model.chat(thread)
    ```


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   source/tuneapi.apis
   source/tuneapi.types
   source/tuneapi.utils
   source/tuneapi.endpoints
   changelog

