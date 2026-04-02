# otograd

Very small OCaml autograd and MNIST demo.

```sh
make help                    # show all targets
make build                   # build the project
make data                    # download and unpack MNIST into data/mnist
make demo                    # run the scalar autograd demo
make train                   # train the minimal MNIST model
make test                    # run the smoke tests
make clean                   # remove _build

make train TRAIN_LIMIT=1024 TEST_LIMIT=256 EPOCHS=5 LEARNING_RATE=0.1
```
