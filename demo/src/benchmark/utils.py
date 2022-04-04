import itertools
import subprocess
import re
import io
from configparser import ConfigParser
from argparse import Namespace
from typing import Any, Callable, Optional, Union

TIME_REGEX = re.compile("(?<=: )\d+(?= \S+$)")
IGNORE_PARAMS = ["input-path", "target-path"]


Params = dict[str, Any]
Options = list[str]
Callback = Callable[[Optional[Params]], None]


def dict_to_str(
    items: dict, ignore_keys: list[Any] = [], join_str: str = ", "
) -> str:
    return f"{join_str}".join(
        f"{name}={value}"
        for name, value in items.items()
        if name not in ignore_keys
    )


class SgdConfigParser:
    section = "section"

    def __init__(self) -> None:
        self._config = ConfigParser()

    @property
    def config(self) -> dict:
        return self._config[self.section]

    @config.setter
    def config(self, section: dict) -> None:
        self._config[self.section] = section

    def read(self, path: str) -> None:
        with open(path) as file:
            self._config.read_string(f"[{self.section}]\n" + file.read())

    def write(self, path: str) -> None:
        buffer = io.StringIO()
        self._config.write(buffer)
        buffer.seek(0)
        next(buffer)
        with open(path, "w") as file:
            file.write(buffer.read())


def clear_options(config: dict[str], options: list[str]) -> ConfigParser:
    for option in options:
        if option in config:
            config.pop(option)
    return config


def get_set_from_space(items: dict[str, list[Any]]) -> list[Params]:
    values = items.values()
    keys = items.keys()
    if keys and values:
        combinations = [
            dict(zip(keys, combination))
            for combination in itertools.product(*values)
        ]
        return combinations
    return []


def parse_option(config: dict[str, Any], option: str) -> list[Params]:
    if option not in config:
        return []
    subconfig: dict[str, Any] = config[option]

    param_set = subconfig.get("param_set", [])
    param_space = subconfig.get("param_space", {})

    space_set = get_set_from_space(param_space)
    param_set.extend(space_set)
    return param_set


def parse_duration(stdout: str) -> float:
    match = TIME_REGEX.search(stdout)
    if match:
        str_time = match.group(0)
        return float(str_time)
    raise Exception(f"Failed to parse stdout: {stdout}")


def run_command(
    executable: Union[str, list[str]],
    params: Optional[Params] = None,
    options: Optional[Options] = None,
    stdout: bool = False,
) -> Optional[str]:
    command = [executable] if isinstance(executable, str) else executable
    if params is not None:
        param_options = itertools.chain.from_iterable(
            [[f"--{name}", str(value)] for name, value in params.items()]
        )
        command += param_options
    if options is not None:
        command += options
    result = subprocess.run(command, capture_output=stdout)
    return result.stderr.decode("utf-8") if stdout else None


def format_output(
    name: str, duration: float, params: Optional[Params] = None
) -> str:
    params_str = (
        dict_to_str(params, IGNORE_PARAMS) if params is not None else ""
    )
    return f"{name} ({params_str}): {duration}"


def run_sgd_once(
    args: Namespace, params: Params = {}, parallel: bool = False
) -> None:
    name = "parallel" if parallel else "serial"
    config_path = getattr(args, f"{name}_config")

    config_option = ["--config", config_path]
    stdout = run_command(args.binary, params, config_option, stdout=True)

    duration = parse_duration(stdout)
    print(format_output(name.title(), duration, params))


def run_sgd_multiple(
    args: Namespace,
    param_set: list[Params] = [],
    params: Params = {},
    callback: Optional[Callback] = None,
    parallel: bool = False,
) -> None:
    if param_set:
        for run_params in param_set:
            run_sgd_once(args, dict(params, **run_params), parallel)
            if callback is not None:
                callback(params)
    else:
        run_sgd_once(args, params, parallel)
        if callback is not None:
            callback()


def get_sgd_callback(
    args: Namespace,
    param_set: list[Params] = [],
    params: Params = {},
    callback: Optional[Callback] = None,
    parallel: bool = False,
) -> Callback:
    def sgd_callback(parent_params: Optional[Params] = {}) -> None:
        all_params = dict(params, **parent_params)
        run_sgd_multiple(
            args,
            param_set,
            params=all_params,
            callback=callback,
            parallel=parallel,
        )

    return sgd_callback
