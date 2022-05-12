from pathlib import Path
from argparse import (
    ArgumentParser,
    ArgumentDefaultsHelpFormatter,
    ArgumentTypeError,
    SUPPRESS,
)
from typing import Callable, Optional

SGD_CONFIG_EXTS = [".ini", ".cfg"]
BINARY_EXTS = ["", ".exe"]
BENCHMARK_CONFIG_EXTS = [".json"]


def make_file_type(exts: Optional[list[str]] = None) -> Callable[[str], str]:
    def file_type(path: str) -> str:
        in_path = Path(path)
        if (
            in_path.exists()
            and in_path.is_file()
            and (exts is None or in_path.suffix in exts)
        ):
            return path
        raise ArgumentTypeError("Invalid file path")

    return file_type


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        usage=SUPPRESS,
        description="SGD benchmark",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "binary",
        type=make_file_type(BINARY_EXTS),
        help="path to C++ gradient descent project executable",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=make_file_type(BENCHMARK_CONFIG_EXTS),
        metavar="",
        help="path to benchmark JSON config file",
    )
    parser.add_argument(
        "-s",
        "--serial-config",
        type=make_file_type(SGD_CONFIG_EXTS),
        metavar="",
        help="path to serial SGD INI config file",
    )
    parser.add_argument(
        "-p",
        "--parallel-config",
        type=make_file_type(SGD_CONFIG_EXTS),
        metavar="",
        help="path to parallel SGD INI config file",
    )
    parser.add_argument(
        "-t",
        "--temp-dir",
        type=str,
        default="benchmark_data",
        metavar="",
        help="path to temporary dir that will be removed after extection",
    )
    return parser
