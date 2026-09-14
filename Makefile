.PHONY: sync test smoke inspect lint

sync:
	uv sync --extra dev

test:
	uv run pytest

smoke:
	bash scripts/smoke_reference.sh

inspect:
	uv run medjepa-inspect --config configs/smoke_ijepa_synthetic.yaml --output reports/ijepa_batch.png
	uv run medjepa-inspect --config configs/smoke_lejepa_synthetic.yaml --output reports/lejepa_views.png

lint:
	uv run ruff check src solutions tests
