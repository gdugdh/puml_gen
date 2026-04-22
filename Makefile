PYTHON ?= python
INPUT ?= synthetic_data/basic_fastapi_ideal_ir.json
OUTDIR ?= output
DIAGRAM_MODE ?= route

.PHONY: install generate generate-route generate-service

install:
	$(PYTHON) -m pip install -r requirements.txt

generate:
	$(PYTHON) -m src.main --input $(INPUT) --outdir $(OUTDIR) --diagram-mode $(DIAGRAM_MODE)

generate-route:
	$(PYTHON) -m src.main --input $(INPUT) --outdir $(OUTDIR) --diagram-mode route

generate-service:
	$(PYTHON) -m src.main --input $(INPUT) --outdir $(OUTDIR) --diagram-mode service
