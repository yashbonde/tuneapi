Changelog
=========

This package is already used in production at Tune AI, please do not wait for release ``1.x.x`` for stability, or expect
to reach ``1.0.0``. We do not follow the general rules of semantic versioning, and there can be breaking changes between
minor versions.

All relevant steps to be taken will be mentioned here.

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