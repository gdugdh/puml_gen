PYTHON ?= python
INPUT ?= input/synthetic_data.json
OUTDIR ?= output

.PHONY: install generate

install:
	$(PYTHON) -m pip install -r requirements.txt

generate:
	$(PYTHON) -m src.main --input $(INPUT) --outdir $(OUTDIR)
