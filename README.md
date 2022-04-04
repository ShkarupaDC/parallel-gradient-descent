# Serial and parallel SGD for linear regression

## Installation

Before building a C++ project you have to install [Boost](https://www.boost.org/) and [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) libs using your package manager, for example. Then you should run the following commands

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
  -i [ --input-path ] arg              path to input CSV file
  -t [ --target-path ] arg             path to target CSV file
  -e [ --eval-path ] arg               path to evaluation CSV file
  -o [ --out-path ] arg (=output.csv)  path to output CSV file
  --cost-path arg (=cost.csv)          path to cost CSV file
  -p [ --parallel ]                    wether to use parallel or serial SGD
  -n [ --num-epochs ] arg (=1000)      number of training epochs
  -l [ --lr ] arg (=0.001)             learning rate
  -w [ --weight-decay ] arg (=0.01)    L2 regularization lambda term
  --normalize                          wether to normalize input
  --num-threads arg (=11)              number of threads to use for parallel
                                       SGD
  --num-step-epochs arg (=1)           number of epochs to compute in each
                                       thread before weight sharing
```

### Example

```sh
./build/bin/gradient_descent --parallel --num-threads 8 --num-step-epochs 12 --config configs/config.cfg
```

## Demo

Random data for experiments and benchmarking can be generated via [demo/gen_data.py](/demo/gen_data.py) script that provides the following configuration options

```
Data generator arguments:
  -h, --help            show this help message and exit
  -t , --num-train      number of train samples (default: 1000)
  -e , --num-eval       number of evaluation samples (default: 0)
  -f , --num-features   number of per sample features (default: 10)
  -s , --seed           seed that make data generation deterministic (default: None)
  -o , --out-dir        path to directory where generated data will be stored (default: ./data)
```

### Example

```sh
python demo/gen_data.py --num-train 10000 --num-eval 1000 --num-features 20
```

## Benchmark

Serial and parallel implementations of SGD can be compared using benchmark provided in the [demo](/demo/) folder. It has the following options

```
Benchmark arguments:
  binary                path to C++ gradient descent project executable
  -h, --help            show this help message and exit
  -c , --config         path to benchmark JSON config file (default: None)
  -s , --serial-config
                        path to serial SGD INI config file (default: None)
  -p , --parallel-config
                        path to parallel SGD INI config file (default: None)
  -t , --temp-dir       path to temporary dir that will be removed after extection (default: benchmark_data)
```

### Example

```sh
python demo/benchmark.py build/bin/gradient_descent --serial-config examples/serial_config.cfg --parallel-config examples/parallel_config.cfg --config examples/benchmark_config.json
```

## Pipeline

The complete pipeline can be found in [demo notebook](/demo/notebooks/demo.ipynb)
