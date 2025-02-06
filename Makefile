# Makefile

define HELP_MESSAGE
zbot-unit-tests

# Installing

1. Create a new Conda environment: `conda create --name zbot-unit-tests python=3.11`
2. Activate the environment: `conda activate zbot-unit-tests`
3. Install the package: `make install-dev`

# Running Tests

1. Run autoformatting: `make format`
2. Run static checks: `make static-checks`
3. Run unit tests: `make test`

endef
export HELP_MESSAGE

all:
	@echo "$$HELP_MESSAGE"
.PHONY: all

# ------------------------ #
#       Static Checks      #
# ------------------------ #

py-files := $(shell find . -name '*.py')

format:
	@black $(py-files)
	@ruff format $(py-files)
.PHONY: format

static-checks:
	@black --diff --check $(py-files)
	@ruff check $(py-files)
	@mypy --install-types --non-interactive $(py-files)
.PHONY: lint

# ------------------------ #
#        Unit tests        #
# ------------------------ #

test-00:
	python -m zbot_unit_tests.test_00
.PHONY: test-00

test-01:
	python -m zbot_unit_tests.test_01
.PHONY: test-01

test-02:
	python -m zbot_unit_tests.test_02
.PHONY: test-02

test: test-00 test-01
.PHONY: test
