import argparse


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage=argparse.SUPPRESS,
        description="Generate train and test data for linear regression model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser._optionals.title = "Tool arguments"
    parser.add_argument(
        "-t",
        "--n-train",
        type=int,
        metavar="",
        default=1000,
        help="number of train samples",
    )
    parser.add_argument(
        "-e",
        "--n-eval",
        type=int,
        metavar="",
        default=100,
        help="number of evaluation samples",
    )
    parser.add_argument(
        "-f",
        "--n-features",
        type=int,
        default=10,
        metavar="",
        help="number of per sample features",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        metavar="",
        help="seed that make data generation deterministic",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default="./data",
        metavar="",
        help="path to directory where generated data will be stored",
    )
    return parser
