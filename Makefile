.PHONY: generate check check-generated check-lean test test-lean check-full run

generate:
	.venv/bin/python -m examples.generate

check: check-generated check-lean test

check-generated:
	.venv/bin/python -m examples.generate --check

check-lean:
	lake build JaxLean JaxLeanExamples

test:
	.venv/bin/python -m pytest -q

test-lean: check-lean
	.venv/bin/python -m pytest -q -m lean

check-full: check-generated check-lean
	.venv/bin/python -m pytest -q -m ""

run:
	lake build examples.basics.proofs.Run
	lake env lean examples/basics/proofs/Run.lean

.PHONY: docs docs-serve
docs:
	PYTHONPATH=tools/verso/src .venv/bin/python -m jaxlean_verso build

docs-serve: docs
	PYTHONPATH=tools/verso/src .venv/bin/python -m jaxlean_verso serve

.PHONY: docs-watch
docs-watch:
	PYTHONPATH=tools/verso/src .venv/bin/python -m jaxlean_verso.watch
