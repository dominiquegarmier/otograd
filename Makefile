OPAM_EXEC := opam exec --
DATA_DIR ?= data/mnist
TRAIN_LIMIT ?= 512
TEST_LIMIT ?= 128
EPOCHS ?= 3
LEARNING_RATE ?= 0.25

.PHONY: help build data demo train test clean

help:
	@printf '\n'
	@printf 'otograd targets\n'
	@printf '  make data   - download and unpack MNIST into %s\n' "$(DATA_DIR)"
	@printf '  make demo   - run the tiny scalar autograd demo\n'
	@printf '  make train  - run the MNIST single-output linear training example\n'
	@printf '  make test   - run the autograd smoke tests\n'
	@printf '  make build  - compile the project\n'
	@printf '\n'
	@printf 'Variables you can override:\n'
	@printf '  DATA_DIR=%s TRAIN_LIMIT=%s TEST_LIMIT=%s EPOCHS=%s LEARNING_RATE=%s\n' \
	  "$(DATA_DIR)" "$(TRAIN_LIMIT)" "$(TEST_LIMIT)" "$(EPOCHS)" "$(LEARNING_RATE)"
	@printf '\n'

build:
	@printf '\n==> building\n'
	@$(OPAM_EXEC) dune build

data:
	@printf '\n==> downloading mnist into %s\n' "$(DATA_DIR)"
	@sh scripts/download_mnist.sh "$(DATA_DIR)"

demo: build
	@printf '\n==> running demo\n'
	@$(OPAM_EXEC) dune exec ./bin/main.exe -- demo

train: data build
	@printf '\n==> training single-output linear model on mnist\n'
	@$(OPAM_EXEC) dune exec ./bin/main.exe -- mnist "$(DATA_DIR)" "$(TRAIN_LIMIT)" "$(TEST_LIMIT)" "$(EPOCHS)" "$(LEARNING_RATE)"

test: build
	@printf '\n==> running smoke tests\n'
	@$(OPAM_EXEC) dune build ./test/test_main.exe
	@_build/default/test/test_main.exe

clean:
	@printf '\n==> cleaning build outputs\n'
	@rm -rf _build
