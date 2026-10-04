.PHONY: generate check run

generate:
	.venv/bin/python -m examples.generate

check:
	.venv/bin/python -m examples.generate --check
	lake build
	.venv/bin/python -m pytest -q

run:
	lake build JaxLean.Run
	lake env lean JaxLean/Run.lean
