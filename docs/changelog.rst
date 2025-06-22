Changelog
=========

This package is already used in production at Tune AI, please do not wait for release ``1.x.x`` for stability, or expect
to reach ``1.0.0``. We **do not follow the general rules** of semantic versioning, and there can be breaking changes between
minor versions.

All relevant steps to be taken will be mentioned here.

8.0.12
------

- Add code in Anthropic API to fix structured generation.

8.0.8
-----

- Adds ``pydantic.BaseModel`` as a native datastructure in ``tuneapi.types`` and can be used as ``tt.BM``.
- [Breaking] the properties in a tool call are now abstracted as a seperate ``Prop`` class and equivalent changes
  will have to be done
- Fixes issue in turbo methods where progress bars were not closed properly.
- ``tuneapi.utils.randomness.get_random_string`` now returns ``n`` unique strings so caller doesn't have to worry
  about uniqueness

8.0.7
-----

- Add replace ``requests.Session`` with ``httpx.Client`` in all the ``tuneapi.apis`` models
- ``tuneapi.types.chats.ModelInterface`` is a class now instead of a protocol so must be initialised like
  ``super().__init__()``
- Make client common for a single instance of ``tuneapi.types.chats.ModelInterface`` instead of multuple clients. As is
  suggested in the httpx documentation, never create multiple clients in the ``hotloop``. This has increased the speed
  of ``distributed_chat`` by 8.6% in our benchmarks. It also means when running 100s of prompts in parallel in a server
  this will reduce the chance of API failure.

8.0.6
-----

- Standardise all the implementations of the ``tuneapi.types.chats.ModelInterface``

8.0.5
-----

- Adding support for Batches API in Openai and Anthropic models
  - Have not tested function calling on batches API
  - Have not tested structured generation on batches API
- Removed dependency on ``openai`` package for audio transcribing by figuring out uploading of files
- Added new ``Ollama`` class which works with ``OpenAIProtocol``

8.0.4
-----

- Bug fixes

8.0.3
-----

- Fix bug in ``tools`` that was causing ever increasing number of tools in the ``Thread`` object
- OpenAI protocol abstracted away as a ``OpenAIProtocol`` class in ``tuneapi.apis.openai``. This is to make it easier to
  add new endpoints in the future.

8.0.2
-----

- Added usage tracking for OpenAI and Anthropic

8.0.1
-----

- Typo so now we are in 8.x.x series
- Fix bug in structured generation for ``Openai``.

0.8.0
-----

- Adding multimodal support for OpenAI and Gemini endpoints.
- Added new types for the new endpoints, etc.
- New utils function ``file_size`` to get the size of the file in bytes.


0.7.4
-----

- Remove ``model_tune.py``, ``model_groq.py`` and ``model_mistral.py`` and subclass those models from ``Openai``.


0.7.3
-----

- Fix bug in image modality for Gemini
- Fix bug in structured generation for OpenAI by setting ``strict=True`` in the API.
- Fix bug in structured generation for Gemini
- Abstracted making input structure for ``Anthropic``, ``OpenAI`` and ``Gemini`` using ``_process_input`` method
- Abstracted outputs for ``Anthropic``, ``OpenAI`` and ``Gemini`` using ``_process_output`` method


0.7.2
-----

- Add image modality for Gemini and OpenAI to complement Anthropic.


0.7.1
-----

- Add structured genration support for Gemini and OpenAI APIs. You can jsut pass ``schema`` to ``Thread``. ``model.chat``
  will take care of it automatically. Here's an example:

  .. code-block:: python

      from tuneapi import tt, ta
      from pydantic import BaseModel
      from typing import List, Optional, Dict, Any

      class MedicalRecord(BaseModel):
          date: str
          diagnosis: str
          treatment: str

      class Dog(BaseModel):
          name: str
          breed: str
          records: Optional[List[MedicalRecord]] = None

      class Dogs(BaseModel):
          dogs: List[Dog]

      model = ta.Gemini()
      out: Dogs = model.chat(tt.Thread(
          tt.human("""
              At the Sunny Paws Animal Clinic, we keep detailed records of all our furry patients. Today, we saw a few dogs.
              There was 'Buddy,' a golden retriever, who visited on '2023-10-26' and was diagnosed with a 'mild ear infection,'
              which we treated with 'ear drops.' Then, there was 'Luna,' a playful beagle, who came in on '2023-10-25' for a
              'routine check-up,' and no treatment was needed, but we also had her back on '2023-10-28' with a 'upset tummy'
              which we treated with 'bland diet and probiotics.' Finally, a third dog named 'Rocky', a small terrier mix,
              showed up on '2023-10-29' with a small 'cut on his paw,' we cleaned it and used an 'antibiotic ointment'. We
              also have 'Daisy,' a fluffy poodle, who doesn't have any medical records yet, thankfully!
          """),
          schema=Dogs,
      ))

      for dog in out.dogs:
          print(f"Dog: {dog.name}, Breed: {dog.breed}")
          if dog.records:
              for record in dog.records:
                  print(f"  Date: {record.date}, Diagnosis: {record.diagnosis}, Treatment: {record.treatment}")
          else:
              print("  No medical records on file.")
          print()

- Add ``pydantic`` as a dependency in the package.

0.7.0
-----

- All models now have ``<model>.distributed_chat_async`` that can be used in servers without blocking the main event
  loop. This will give a much needed UX improvement to the entire system.

0.6.3
-----

- ``<model>.distributed_chat`` now takes in args that are passed to the ``post_logic``.


0.6.2
-----

- New set of utils in ``tuneapi.utils`` called ``prompt`` to help with the basics of prompting.

0.6.1
-----

- Package now uses ``fire==0.7.0``

0.6.0
-----

- ``distributed_chat`` functionality in ``tuneapi.apis.turbo`` support. In all APIs search for ``model.distributed_chat()``
  method. This enables **fault tolerant LLM API calls**.
- Moved ``tuneapi.types.experimental`` to ``tuneapi.types.evals``

0.5.13
------

- ``tuneapi.types.ModelInterface`` has an ``extra_headers`` attribute in it.

0.5.12
------

- Remove code to sanitize assistant message in for Tune and OpenAI LLM APIs.

0.5.11
------

- Fix bug where ``parallel_tool_calls`` was sent even for non tool calls.

0.5.10
------

- Remove redundant prints.

0.5.9
-----

- By default set the value ``parallel_tool_calls`` in OpenAI to ``False``.

0.5.8
-----

- If you have ``numpy`` installed in your environment, then ``tuneapi.utils.randomness.reservoir_sampling`` will honour
  the seed value. If you do not have ``numpy`` installed, then the seed value will be ignored.
- Fix Bug in Gemini API body for functions with no parameters.

0.5.7
-----

- Implement ``extra_headers`` via ``__init__`` as well.

0.5.6
-----

- Remove protobuf as a dependency in because bunch of other packages break. The functions are still present

0.5.5
-----

- In all implmenetations of ``tuneapi.types.chats.ModelInterface`` add new input to the API endpoints called ``extra_headers``
  which is a dictionary to update the outgoing headers.

0.5.4
-----

- Standardise ``tuneapi.types.chats.ModelInterface`` to have ``model_id``, ``api_token`` added to the base class.

0.5.3
-----

- Fix bug in Tune proxy API where incorrect variable ``stop_sequence`` was sent instead of the correct ``stop`` causing
  incorrect behaviour.
- bump dependency to ``protobuf>=5.27.3``
- remove ``__version__`` from tuneapi package
- remove CLI entrypoint in ``pyproject.toml``

0.5.2
-----

- Add ability to upload any file using ``tuneapi.endpoints.FinetuningAPI.upload_dataset_file`` to support the existing
  way to uploading using threads.

0.5.1
-----

- Fix bug in the endpoints module where error was raised despite correct inputs

0.5.0 **(breaking)**
--------------------

In this release we have moved all the Tune Studio specific API out of ``tuneapi.apis`` to ``tuneapi.endpoints`` to avoid
cluttering the ``apis`` namespace.

.. code-block:: diff

    - from tuneapi import apis as ta
    + from tuneapi import endpoints as te
    ...
    - ta.ThreadsAPI(...)
    + te.ThreadsAPI(...)

- Add support for finetuning APIs with ``tuneapi.endpoints.FinetuningAPI``
- Primary environment variables have been changed from ``TUNE_API_KEY`` to ``TUNEAPI_TOKEN`` and from ``TUNE_ORG_ID``
  to ``TUNEORG_ID``, if you were using these please update your environment variables
- Removed CLI methods ``test_models`` and ``benchmark_models``, if you want to use those, please copy the code from
  `this commit <https://github.com/NimbleBoxAI/tuneapi/blob/2fabdae461f4187621fe8ffda73a58a5ab7485b0/tuneapi/apis/__init__.py#L26>`_

0.4.18
------

- Fix bug where function response was tried to be deserialised to the JSON and then sent to the different APIs.

0.4.17
------

- Fix error in ``tuneapi.utils.serdeser.to_s3`` function where content type key was incorrect

0.4.16
------

- Adding support for python 3.12
- Adding ``tool`` as a valid role in ``tuneapi.types.chats.Message``

0.4.15
------

- When there is an error in the model API, we used to print the error message. Now we are returning the error message
  in the response.

0.4.14
------

- Fix bug where a loose ``pydantic`` import was present

0.4.13
------

- Bug fixes in JSON deserialisation

0.4.12
------

- Fix bug in Threads API where incorrect structure was sent by client
- Add images support for Anthropic API
- Add ``Message.images`` field to store all images