# https://stackoverflow.com/a/2145605
.PHONY: setup upload_to_pypi upload_to_npm build_frontend serve

# This takes care of setting up the project for the first time
setup:
	@if ! command -v uv > /dev/null 2>&1; then \
		echo "Installing uv package manager..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi

	@echo "Setting up virtual environment..."
	uv venv
	uv sync

# This upload the package to PyPI. This can only be run by me.
# Before bumping version make sure pyproject.toml and docs/conf.py are updated.
# Also add updates to the changelog.
upload_to_pypi:
	@echo "Building package..."
	uv sync
	uv build

	@echo "Uploading package to PyPI... "
	uv run -- twine upload dist/*

	@echo "Cleaning up..."
	rm -rf dist

# This uploads the package to npm. This can only be run by me.
upload_to_npm:
	@echo "Bumping patch version and publishing to npm..."
	cd ts && npm version patch
	cd ts && npm publish
	
	@echo "Package published to npm!"

docs:
	@echo "Building documentation..."
	uv run sphinx-build -b html docs/ docs/_build/html/

serve_docs:
	@echo "Serving documentation..."
	cd docs/_build/html/ && python -m http.server