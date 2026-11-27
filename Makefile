export PYTHONPATH := src

.PHONY: test smoke run lint
test:
	python -m pytest -q
smoke:
	python -m polarity.run --subsample 1500 --seeds 1 --out /tmp/polarity-smoke
run:
	python -m polarity.run
lint:
	ruff check .
