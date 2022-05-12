import shutil
from pathlib import Path
import json
from typing import Optional

from src.benchmark.utils import (
    Params,
    SgdConfigParser,
    parse_option,
    run_command,
    run_sgd_multiple,
    get_sgd_callback,
    clear_options,
    dict_to_str,
)
from src.benchmark.parser import get_parser

CLEAR_OPTIONS = ["eval-path", "out-path", "cost-path"]
X_PATH = "x_train.csv"
Y_PATH = "y_train.csv"
GEN_DATA = "demo/gen_data.py"


def process_sgd_config(path: Optional[str], parent_dir: str) -> Optional[str]:
    if path is None:
        return None
    parser = SgdConfigParser()
    parser.read(path)

    parser.config = clear_options(parser.config, CLEAR_OPTIONS)
    new_path = Path(parent_dir).joinpath(Path(path).name).as_posix()

    parser.write(new_path)
    return new_path


def main() -> None:
    args = get_parser().parse_args()
    if args.serial_config is None and args.parallel_config is None:
        return

    parent_dir = Path(args.temp_dir)
    parent_dir.mkdir(parents=True)

    config_dir = parent_dir.joinpath("configs")
    config_dir.mkdir()

    args.parallel_config = process_sgd_config(args.parallel_config, config_dir)
    args.serial_config = process_sgd_config(args.serial_config, config_dir)

    if args.config is not None:
        with open(args.config, "r") as file:
            config = json.load(file)

        sgd_set = parse_option(config, "sgd")
        data_set = parse_option(config, "data")
    else:
        sgd_set, data_set = [], []

    def run_experiments(data_params: Params = {}) -> None:
        parallel_sgd = get_sgd_callback(
            args, sgd_set, data_params, parallel=True
        )
        if args.serial_config is not None:
            callback = None if args.parallel_config is None else parallel_sgd
            run_sgd_multiple(
                args, params=data_params, callback=callback, parallel=False
            )
        else:
            if args.parallel_config is not None:
                parallel_sgd()

    if data_set:
        for idx, gen_data_params in enumerate(data_set):
            data_dir = parent_dir / str(idx)
            data_dir.mkdir(exist_ok=True)

            path_option = ["--out-dir", data_dir.as_posix()]
            executable = ["python", GEN_DATA]
            run_command(executable, gen_data_params, path_option)

            data_params = {
                "input-path": (data_dir / X_PATH).as_posix(),
                "target-path": (data_dir / Y_PATH).as_posix(),
            }
            print("Data params:", dict_to_str(gen_data_params))
            run_experiments(data_params)
            shutil.rmtree(data_dir.as_posix())
    else:
        run_experiments()
    shutil.rmtree(parent_dir.as_posix())


if __name__ == "__main__":
    main()
