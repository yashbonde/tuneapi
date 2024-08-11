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

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   source/tuneapi.apis
   source/tuneapi.types
   source/tuneapi.utils
   source/tuneapi.endpoints
   changelog

