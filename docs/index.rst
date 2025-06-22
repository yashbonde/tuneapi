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

    # define a model
    model = ta.Gemini() # other LLMs: Openai, Anthropic, Groq, TuneModel, Mistral

    # pass strings to models for better dev-ex
    out: str = model.chat("who are you?")

    # define a thread which is a collection of messages with system, user and assistant messages
    thread = tt.Thread(
        tt.system(...),   # add optional system message here
        tt.human(...),    # add user message here
        tt.assistant(...) # for assistant response
    )

    # get the response
    resp: str = model.chat(thread)

    # You can also generate structured response for better control

    class MathProblem(tt.BM):
        a: int = tt.Field(..., description="First number")
        b: int = tt.Field(..., description="Second number")
        operator: str = tt.Field(..., description="Operator")
        hint: str | None = tt.Field(None, description="Hint for the problem")

    class MathTest(tt.BM):
        title: str = tt.Field(..., description="Title of the test")
        problems: list[MathProblem]  # only list of other BaseModel is allowed

    # define a thread which is a collection of messages
    thread = tt.Thread(tt.human("Give me 5 problems for KG-1 class"), schema=MathTest)

    # get structured output
    resp: MathTest = model.chat(thread)

    # get multiple results in parallel
    resp: list[MathTest] = model.distributed_chat([thread for _ in range(5)])
    ```

Structured generation
---------------------

.. epigraph::

    Types and Logic is the two parts of programming.


With structured generation you can get ``pydantic.BaseModel`` objects from ``tt.ModelInterface.chat`` and
``tt.ModelInterface.chat_async`` methods. The currect limitation is that keys cannot have another ``BaseModel`` as value
only ``List[BaseModel]`` is allowed.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   changelog

