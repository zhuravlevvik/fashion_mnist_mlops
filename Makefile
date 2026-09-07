PYTEST ?= .venv/bin/pytest
UV ?= uv
PYTHON ?= .venv/bin/python

.PHONY: lab2-up lab2-test lab2-down secrets-init lab3-up lab3-test

lab2-up:
	docker compose -f docker-compose.yml -f compose.lab2.yml up --build -d

lab2-test:
	$(UV) run --env-file .env $(PYTEST) -m integration tests/integration/test_lab2_hbase.py -v

lab2-down:
	docker compose -f docker-compose.yml -f compose.lab2.yml down

secrets-init:
	$(PYTHON) scripts/bootstrap_ansible_vault.py

lab3-up:
	docker compose -f docker-compose.yml -f compose.lab2.yml -f compose.lab3.yml up --build -d

lab3-test:
	$(PYTEST) -m integration tests/integration/test_lab2_hbase.py tests/integration/test_lab3_vault.py -v