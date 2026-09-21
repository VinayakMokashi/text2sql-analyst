# Common tasks. On Windows without `make`, run the commands after each target by hand.
PY ?= python

.PHONY: install data index app ask eval test lint

install:
	$(PY) -m pip install -e ".[dev]"

data:
	$(PY) scripts/download_chinook.py

index:
	$(PY) -m text2sql index

app:
	$(PY) -m streamlit run app/streamlit_app.py

ask:
	$(PY) -m text2sql ask "Which 5 genres generate the most revenue?"

eval:
	$(PY) eval/run_eval.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src app scripts eval tests
