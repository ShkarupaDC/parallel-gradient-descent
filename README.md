# Serial and parallel SGD for linear regression

## Installation

Before building a C++ project you have to install [Boost](https://www.boost.org/) and [Xtensor](https://xtensor.readthedocs.io/en/latest/) libs using your package manager, for example. Then you should run the following commands

```sh
mkdir build && cd build
cmake ..
cmake --build .
```

To use a demo notebook or generate random data for linear regression, you have to create a virtual environment and install dependencies in the [demo](/demo) folder using the following commands

```sh
virtualenv --python /path/to/python3.9 .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Linear regression

To run linear regression, you should use an executable file created in [build/bin](/build/bin/) folder. It provides the following options for configuration (look at [examples](/examples) folder for config file examples):

```
CLI options:
  -h [ --help ]                        produce help message
  -c [ --config ] arg                  path to config file

Algorithm options:
  -i [ --input-path ] arg              path to input NPY file
  -t [ --target-path ] arg             path to target NPY file
  -e [ --eval-path ] arg               path to evaluation NPY file
  -o [ --out-path ] arg (=output.npy)  path to output NPY file
  -p [ --parallel ]                    wether to use parallel or serial SGD
  -n [ --num-epochs ] arg (=1000)      number of training epochs
  -l [ --lr ] arg (=0.001)             learning rate
  -w [ --weight-decay ] arg (=0.01)    L2 regularization lambda term
  --normalize                          wether to normalize input
  --num-threads arg (=12)              number of threads to use for parallel
                                       SGD
  --num-step-epochs arg (=1)           number of epochs to compute in each
                                       thread before weight sharing
```

### Examples

```sh
./build/bin/gradient_descent --parallel --num-threads 8 --num-step-epochs 12 --config configs/config.cfg
```

## Demo

Random data for experiments and benchmarking can be generated via [demo/gen_data.py](/demo/gen_data.py) script that provides the following configuration options

```
Tool arguments:
  -h, --help          show this help message and exit
  -t , --n-train      number of train samples (default: 1000)
  -e , --n-eval       number of evaluation samples (default: 100)
  -f , --n-features   number of per sample features (default: 10)
  -s , --seed         seed that make data generation deterministic (default: None)
  -o , --out-dir      path to directory where generated data will be stored (default: ./data)
```

### Examples

```sh
python demo/gen_data.py --n-train 10000 --n-eval 1000 --n-features 20
```

## Pipeline

The complete pipeline can be found in [demo notebook](/demo/notebooks/demo.ipynb)
