import numpy as np
import os

from src.regression import generate_data
from src.parser import get_parser


def main() -> None:
    args = get_parser().parse_args()

    data_split, coef = generate_data(
        n_train=args.n_train,
        n_eval=args.n_eval,
        n_features=args.n_features,
        seed=args.seed,
    )
    names = ["x_train", "x_eval", "y_train", "y_eval"]
    os.makedirs(args.out_dir, exist_ok=True)

    for name, data in zip(names, data_split):
        save_path = os.path.join(args.out_dir, name)
        print(data.shape)
        np.save(save_path, data)

    coef_path = os.path.join(args.out_dir, "coef")
    np.save(coef_path, coef)


if __name__ == "__main__":
    main()
