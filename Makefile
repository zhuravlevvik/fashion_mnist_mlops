PYTEST ?= .venv/bin/pytest
UV ?= uv

.PHONY: lab2-up lab2-test lab2-down

lab2-up:
	docker compose -f docker-compose.yml -f compose.lab2.yml up --build -d

lab2-test:
	$(UV) run --env-file .env $(PYTEST) -m integration tests/integration/test_lab2_hbase.py -v

lab2-down:
	docker compose -f docker-compose.yml -f compose.lab2.yml down