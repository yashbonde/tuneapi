# https://stackoverflow.com/a/2145605
.PHONY: setup upload_to_pypi upload_to_npm docs docs_python docs_typescript docs_serve docs_clean

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

# Build all documentation (Python + TypeScript)
docs:
	@echo "Building all documentation..."
	@$(MAKE) docs_python
	@$(MAKE) docs_typescript
	@$(MAKE) docs_combine
	@echo "✓ All documentation built successfully!"

# Build Python documentation with Sphinx
docs_python:
	@echo "Building Python documentation with Sphinx..."
	cd docs && uv run sphinx-apidoc -o source/ ../tuneapi
	uv run sphinx-build -b html docs/ docs/_build/html/
	@echo "✓ Python docs built at docs/_build/html/"

# Build TypeScript documentation with TypeDoc
docs_typescript:
	@echo "Building TypeScript documentation with TypeDoc..."
	cd ts && npm install && npm run docs
	@echo "✓ TypeScript docs built at ts/docs/"

# Combine all documentation into single site
docs_combine:
	@echo "Combining documentation..."
	@mkdir -p _site
	@cp docs_landing/index.html _site/
	@mkdir -p _site/python
	@cp -r docs/_build/html/* _site/python/
	@mkdir -p _site/typescript
	@cp -r ts/docs/* _site/typescript/
	@touch _site/.nojekyll
	@echo "✓ Combined documentation at _site/"

# Serve documentation locally
docs_serve:
	@echo "Serving documentation at http://localhost:8000"
	@cd _site && python -m http.server

# Clean all documentation builds
docs_clean:
	@echo "Cleaning documentation builds..."
	@rm -rf docs/_build
	@rm -rf ts/docs
	@rm -rf _site
	@echo "✓ Documentation cleaned!"