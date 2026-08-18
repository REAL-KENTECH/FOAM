.PHONY: compile test validate verify smoke

compile:
	python -m compileall -q foam_experiments optimizers tools vit.py submission.py

test: compile
	python -m pytest -q

validate: compile
	python tools/validate_configs.py configs/vit --world-size 4
	@for sweep in configs/sweeps/*.yaml; do \
		python tools/run_sweep.py --sweep "$$sweep" --dry-run >/dev/null; \
	done

verify: test validate

smoke: verify
	bash scripts/run_smoke_tests.sh
