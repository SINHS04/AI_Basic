import argparse

def get_parse():
    """
    Required arguments.
    wandb, batch, epoch, lr, validation intervals, earlystop patience, early stop delta and so on
    """
    p = argparse.ArgumentParser()

    p.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=False, help="Using wandb, default False"
    )

    p.add_argument(
        "-c", "--choice", type=str, default=None, help="Kind of job you want, must set, train or test"
    )

    p.add_argument(
        "-b", "--batch_size", type=int, default=1_024, help="Mini batch size, default 1_024"
    )

    p.add_argument(
        "-e", "--epochs", type=int, default=10_000, help="Epochs, default 10_000"
    )

    p.add_argument(
        "-r", "--learning_rate", type=float, default=1e-3, help="Learning rate, default 1e-3"
    )

    p.add_argument(
        "-v", "--validation_intervals", type=int, default=200, help="Validation intervals, default 200"
    )

    p.add_argument(
        "-p", "--early_stop_patience", type=int, default=10, help="Early stop patience, default 10"
    )

    p.add_argument(
        "-d", "--early_stop_delta", type=float, default=1e-5, help="Early stop delta, default 1e-5"
    )

    p.add_argument(
        "-w", "--weight_decay", type=float, default=1e-4, help="Weight decay, default 1e-4"
    )

    return p
