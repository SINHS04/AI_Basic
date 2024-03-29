import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=False, help="Using wandb, True or False"
    )

    parser.add_argument(
        "-b", "--batch_size", type=int, default=2_048, help="Batch size(int, default=2_048)"
    )

    parser.add_argument(
        "-e", "--epochs", type=int, default=10_000, help="Number of Traning(int, default=10_000)"
    )

    parser.add_argument(
        "-r", "--learning_rate", type=float, default=1e-3, help="Learning rate(float, default=1e-3)"
    )

    parser.add_argument(
        "-v", "--validation_intervals", type=int, default=10, help="Number of training epochs between validations(int, default=10)"
    )

    parser.add_argument(
        "-p", "--early_stop_patience", type=int, default=10, help="Number of waitings to early stop(int, default=10)"
    )

    parser.add_argument(
        "-d", "--early_stop_delta", type=float, default=0.00001, help="Early stop delta(float, default=0.00001)"
    )

    return parser