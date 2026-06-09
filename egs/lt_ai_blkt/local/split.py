import argparse
import logging
import os
import random

from datasets import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="""Input file.
        """,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output dir",
    )
    args = parser.parse_args()
    logging.info(f"Splitting data in {args.input}")

    random.seed(42)

    trc, dc, tc = 0, 0, 0
    dev_p = 0.05

    with open(args.input, "r", encoding="utf-8") as f:
        with tqdm(desc="Splitting file") as pbar:
            with open(os.path.join(args.output_dir, "dev.txt"), "w", encoding="utf-8") as f_dev, \
                    open(os.path.join(args.output_dir, "test.txt"), "w", encoding="utf-8") as f_test, \
                    open(os.path.join(args.output_dir, "train.txt"), "w", encoding="utf-8") as f_train:
                for i, line in enumerate(f):
                    r = random.random()
                    if r < dev_p:
                        f_dev.write(line)
                        dc += 1
                    elif r < 2 * dev_p:
                        f_test.write(line)
                        tc += 1
                    else:
                        f_train.write(line)
                        trc += 1
                    pbar.update(1)
    logging.info(f"Train: {trc}, Dev: {dc}, Test: {tc} lines")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    logging.info(f"Starting")

    main()

    logging.info("Done")
