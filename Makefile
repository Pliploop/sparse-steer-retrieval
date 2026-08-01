.DEFAULT_GOAL := help

test:  ## Run the test suite
	pytest tests/

check-lint:  ## Check if the files are linted properly
	autoflake -r --quiet --check-diff --remove-all-unused-imports --ignore-init-module-imports --remove-duplicate-keys --remove-unused-variables .
	black -l 120 --diff --color .
	isort -c --profile black -l 120 .

lint:  ## Lint files
	autoflake -r --quiet --in-place --remove-all-unused-imports --ignore-init-module-imports --remove-duplicate-keys --remove-unused-variables .
	black -l 120 .
	isort --profile black -l 120 .

help:  ## Show help message
	@IFS=$$'\n' ; \
	help_lines=(`fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##/:/'`); \
	printf "%s\n\n" "Usage: make [task]"; \
	printf "%-25s %s\n" "task" "help" ; \
	printf "%-25s %s\n" "------" "----" ; \
	for help_line in $${help_lines[@]}; do \
		IFS=$$':' ; \
		help_split=($$help_line) ; \
		help_command=`echo $${help_split[0]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		help_info=`echo $${help_split[2]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		printf '\033[36m'; \
		printf "%-25s %s" $$help_command ; \
		printf '\033[0m'; \
		printf "%s\n" $$help_info; \
	done
