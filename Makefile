# simple makefile to simplify repetitive build env management tasks under posix

PYTHON ?= python3
PIP ?= pip
PYTEST ?= py.test

# this is default target, it should always be first in this Makefile
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  init               to install all required dependencies only"
	@echo "  install            to install all the library and required dependencies"
	@echo "  install-develop    to install the library in development mode"
	@echo "  install-user       to install the library only for the current user"
	@echo "  install-dev-local  to install the library for development locally"
	@echo "  egg                to build EGG file"
	@echo "  wheel              to build WHEEL file"
	@echo "  clean              to clean the installation"
	@echo "  test               to run unit tests"
	@echo "  lint               to check Pylint sources"
	@echo "  all                to run all targets"


init:
	$(PIP) install --no-cache-dir -r requirements.txt

install:
	$(PYTHON) setup.py install

install-develop:
	$(PYTHON) setup.py develop

install-user:
	$(PYTHON) setup.py install --user

# install-test:
# 	$(PIP) install .[test]

egg:
	$(PYTHON) setup.py bdist_egg

wheel:
	$(PYTHON) setup.py bdist_wheel

clean:
	$(PYTHON) setup.py clean --all

test:
	$(PYTHON) -m unittest discover

lint:
	$(PYTHON) -m pylint --rcfile=.pylintrc -j 0 --exit-zero folder/

all:
	clean lint tests wheel egg

tox:
	tox