.PHONY: generate check run

generate:
	.venv/bin/python -m examples.generate
	.venv/bin/python -m examples.certify
	.venv/bin/python -m examples.certify_transformer
	.venv/bin/python -m examples.certify_more
	.venv/bin/python -m examples.sampling
	.venv/bin/python -m examples.transpile_random

check:
	.venv/bin/python -m examples.generate --check
	.venv/bin/python -m examples.certify --check
	.venv/bin/python -m examples.certify_transformer --check
	.venv/bin/python -m examples.certify_more --check
	.venv/bin/python -m examples.sampling --check
	.venv/bin/python -m examples.transpile_random --check
	lake build
	.venv/bin/python -m pytest -q

run:
	lake build JaxLean.Run
	lake env lean JaxLean/Run.lean

.PHONY: docs docs-serve
docs:
	.venv/bin/python -m jaxlean.docs build

docs-serve: docs
	.venv/bin/python -m jaxlean.docs serve
