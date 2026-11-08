.PHONY: all install test run clean

PYTHON = python
PYTEST = pytest

all: install run test

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) src/universe.py
	$(PYTHON) src/optimizers.py
	$(PYTHON) src/backtester.py
	$(PYTHON) src/tear_sheet_plots.py

test:
	$(PYTEST) -v

clean:
	rm -rf __pycache__ .pytest_cache
