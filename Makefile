# Clearboard's Makefile
#
# /!\ /!\ /!\ /!\ /!\ /!\ /!\ DISCLAIMER /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\
#
# This Makefile is only meant to be used for DEVELOPMENT purpose.
#
# PLEASE DO NOT USE IT FOR YOUR CI/PRODUCTION/WHATEVER...
#
# /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\
#
# Note to developpers:
#
# While editing this file, please respect the following statements:
#
# 1. Every variable should be defined in the ad hoc VARIABLES section with a
#    relevant subsection
# 2. Every new rule should be defined in the ad hoc RULES section with a
#    relevant subsection depending on the targeted service
# 3. Rules should be sorted alphabetically within their section
# 4. When a rule has multiple dependencies, you should:
#    - duplicate the rule name to add the help string (if required)
#    - write one dependency per line to increase readability and diffs
# 5. .PHONY rule statement should be written after the corresponding rule

# ==============================================================================
# VARIABLES

include env.d/development

BOLD := \033[1m
RESET := \033[0m

# -- Docker
COMPOSE              = docker-compose
COMPOSE_RUN          = $(COMPOSE) run --rm
COMPOSE_RUN_APP      = $(COMPOSE_RUN) app

# We run linters on the lambda code using the Python dev dependencies installed in the "app" image:
COMPOSE_RUN_LAMBDA   = $(COMPOSE_RUN) "--volume=$(PWD)/src/terraform/lambda-enhance/:/lambda" app

# ==============================================================================
# RULES

default: help

# -- Project

bootstrap: ## Prepare Docker images for the project
bootstrap: \
	env.d/development \
	env.d/lambda \
	env.d/terraform \
	build \
	run
.PHONY: bootstrap

# -- Docker/compose

build: ## build the app and lambda docker images
	@$(COMPOSE) build app
	@$(COMPOSE) build lambda
.PHONY: build

down: ## Stop and remove containers, networks, images, and volumes
	@$(COMPOSE) down
.PHONY: down

lambda: ## start the lambda container using Docker
	@$(COMPOSE) up lambda
.PHONY: lambda

logs: ## display app logs (follow mode)
	@$(COMPOSE) logs -f app
.PHONY: logs

run: ## start the development server using Docker
	@$(COMPOSE) up -d app
.PHONY: run

stop: ## stop the development server using Docker
	@$(COMPOSE) stop
.PHONY: stop

# -- Back-end

check-black:  ## Run the black tool in check mode only (won't modify files)
	@echo "$(BOLD)Checking black$(RESET)"
	@$(COMPOSE_RUN_APP) black --check clearboard 2>&1
	@$(COMPOSE_RUN_LAMBDA) black --check /lambda 2>&1
.PHONY: check-black

lint:  ## Run all linters (isort, black, flake8, pylint)
lint: \
	lint-isort \
	lint-black \
	lint-flake8 \
	lint-pylint \
	lint-bandit
.PHONY: lint

lint-black:  ## Run the black tool and update files that need to
	@echo "$(BOLD)Running black$(RESET)"
	@$(COMPOSE_RUN_APP) black clearboard
	@$(COMPOSE_RUN_LAMBDA) black /lambda
.PHONY: lint-black

lint-flake8:  ## Run the flake8 tool
	@echo "$(BOLD)Running flake8$(RESET)"
	@$(COMPOSE_RUN_APP) flake8 clearboard
	@$(COMPOSE_RUN_LAMBDA) flake8 /lambda
.PHONY: lint-flake8

lint-isort:  ## automatically re-arrange python imports in code base
	@echo "$(BOLD)Running isort$(RESET)"
	@$(COMPOSE_RUN_APP) isort clearboard --atomic
	@$(COMPOSE_RUN_LAMBDA) isort /lambda --atomic
.PHONY: lint-isort

lint-pylint:  ## Run the pylint tool
	@echo "$(BOLD)Running pylint$(RESET)"
	@$(COMPOSE_RUN_APP) pylint --rcfile=.pylintrc clearboard
	@$(COMPOSE_RUN_LAMBDA) pylint --rcfile=.pylintrc /lambda
.PHONY: lint-pylint

lint-bandit: ## lint back-end python sources with bandit
	@echo "$(BOLD)Running bandit$(RESET)"
	@$(COMPOSE_RUN_APP) bandit -c .banditrc -qr clearboard
	@$(COMPOSE_RUN_LAMBDA) bandit -c .banditrc -qr /lambda
.PHONY: lint-bandit

# -- Misc

env.d/development:
	cp env.d/development.dist env.d/development

env.d/lambda:
	cp env.d/lambda.dist env.d/lambda

env.d/terraform:
	cp env.d/terraform.dist env.d/terraform

help:  ## Show this help
	@echo "$(BOLD)Clearboard Makefile$(RESET)"
	@echo "Please use 'make $(BOLD)target$(RESET)' where $(BOLD)target$(RESET) is one of:"
	@grep -h ':\s\+##' Makefile | column -tn -s# | awk -F ":" '{ print "  $(BOLD)" $$1 "$(RESET)" $$2 }'
.PHONY: help
