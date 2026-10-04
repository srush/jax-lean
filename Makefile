.PHONY: generate check check-generated check-lean test test-lean check-full run

generate:
	.venv/bin/python -m examples.generate
	.venv/bin/python -m examples.certify
	.venv/bin/python -m examples.certify_transformer
	.venv/bin/python -m examples.certify_more
	.venv/bin/python -m examples.certify_noether
	.venv/bin/python -m examples.certify_puzzles
	.venv/bin/python -m examples.sampling
	.venv/bin/python -m examples.transpile_random

check: check-generated check-lean test

check-generated:
	.venv/bin/python -m examples.generate --check
	.venv/bin/python -m examples.certify --check
	.venv/bin/python -m examples.certify_transformer --check
	.venv/bin/python -m examples.certify_more --check
	.venv/bin/python -m examples.certify_noether --check
	.venv/bin/python -m examples.certify_puzzles --check
	.venv/bin/python -m examples.sampling --check
	.venv/bin/python -m examples.transpile_random --check

check-lean:
	lake build

test:
	.venv/bin/python -m pytest -q

test-lean: check-lean
	.venv/bin/python -m pytest -q -m lean

check-full: check-generated check-lean
	.venv/bin/python -m pytest -q -m ""

run:
	lake build JaxLean.Run
	lake env lean JaxLean/Run.lean

.PHONY: docs docs-serve
docs:
	.venv/bin/python -m jaxlean.docs build

docs-serve: docs
	.venv/bin/python -m jaxlean.docs serve
