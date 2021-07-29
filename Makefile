BRANCH := $(shell git rev-parse --symbolic-full-name --abbrev-ref HEAD)

# by default, we'll bump the "build" part of the version, for non-releases
PART = build

# if the current version is a release and not a dev build, bump the patch part instead
VERSION := $(shell grep -Po '(?<=current_version = )[\w\d\.]+' .bumpversion.cfg)
ifeq (,$(findstring dev,$(VERSION)))
	ifeq ($(PART),build)
		PART = patch
    endif
endif


# Minimal makefile for Sphinx documentation

SPHINXOPTS	=
SPHINXBUILD = sphinx-build
SOURCEDIR   = docs_source
BUILDDIR	= _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

docs:
	-rm -rf _build/
	make html
	aws s3 sync --delete _build/html/ s3://docs.pewresearch.tech/pewanalytics/

python_lint_errors:
	# stop the build if there are Python syntax errors or undefined names
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=.git,__pycache__,build,dist

python_lint_quality:
	flake8 . --exit-zero --statistics --count --show-source --max-line-length=127 --ignore=E201,E202,E221,E251,E501,E722 --exclude=.git,__pycache__,build,dist

github_lint_flake8:
	flake8 . --max-line-length 127 --ignore=E201,E202,E221,E251,E501,E722 --exclude=.git,__pycache__,build,dist | reviewdog -reporter=github-pr-check -f=flake8

python_test:
	python3 -m unittest tests

python_build:
	python3 setup.py sdist bdist_wheel

bump:
	git checkout $(BRANCH)
	git pull origin $(BRANCH)
	bumpversion --commit $(PART)

sync_branch:
	git checkout $(BRANCH)
	git pull origin $(BRANCH)
	git push origin $(BRANCH)

release:
	git checkout $(BRANCH)
	git pull origin $(BRANCH)
	bumpversion --commit --tag release
	git push origin $(BRANCH) --follow-tags

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
