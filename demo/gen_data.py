import numpy as np
from pathlib import Path

from src.gen_data.regression import generate_data
from src.gen_data.parser import get_parser

FILENAMES = ["x_train", "y_train", "x_eval", "y_eval"]


def main() -> None:
    args = get_parser().parse_args()

    data_split = generate_data(
        num_train=args.num_train,
        num_eval=args.num_eval,
        num_features=args.num_features,
        seed=args.seed,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename, data in zip(FILENAMES, data_split):
        np.savetxt(str(out_dir / f"{filename}.csv"), data, delimiter=",")


if __name__ == "__main__":
    main()
