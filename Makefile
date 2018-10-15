MAKE   = make
PYTHON = python

.PHONY: clean cleandocs docs opendocs

help:
	@echo "Usage: \`make <target>' where <target> is one or more of"
	@echo "  clean     delete intermediate work product and start fresh"
	@echo "  cleandocs delete cached HTML documentation and start fresh"
	@echo "  docs      build HTML documentation using Sphinx (incremental)"
	@echo "  opendocs  open local HTML documentation in browser"

clean:
	find . -type f -name \*.pyc -exec rm {} \;
	find . -type f -name .DS_Store -exec rm {} \;
	rm -rf dist .coverage

cleandocs:
	$(MAKE) -C docs clean

docs:
	$(MAKE) -C docs html

opendocs:
	open docs/build/html/index.html
