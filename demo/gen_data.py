import numpy as np
from pathlib import Path

from src.regression import generate_data
from src.parser import get_parser

FILENAMES = ["x_train", "x_eval", "y_train", "y_eval"]


def main() -> None:
    args = get_parser().parse_args()

    data_split = generate_data(
        n_train=args.n_train,
        n_eval=args.n_eval,
        n_features=args.n_features,
        seed=args.seed,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename, data in zip(FILENAMES, data_split):
        np.savetxt(str(out_dir / f"{filename}.csv"), data, delimiter=",")


if __name__ == "__main__":
    main()
