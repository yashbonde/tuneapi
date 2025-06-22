# https://stackoverflow.com/a/2145605
.PHONY: setup upload_to_pypi build_frontend serve

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
upload_to_pypi:
	@echo "Building package..."
	uv sync
	uv build
	
	@echo "Uploading package to PyPI... "
	uv run -- twine upload dist/*

	@echo "Cleaning up..."
	rm -rf dist

docs:
	@echo "Building documentation..."
	uv run sphinx-build -b html docs/ docs/_build/html/

serve_docs:
	@echo "Serving documentation..."
	cd docs/_build/html/ && python -m http.server