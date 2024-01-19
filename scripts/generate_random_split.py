from argparse import ArgumentParser
import random
import numpy as np

import pandas as pd

parser = ArgumentParser()
parser.add_argument("--num_labels", "-nl", default=40)
parser.add_argument(
    "--csv", default="isic_challenge/ISBI2016_ISIC_Part3B_Training_GroundTruth.csv"
)

SEED = 98123  # for reproducibility
random.seed(SEED)
args = parser.parse_args()

df = pd.read_csv(args.csv)

# obtain random subset

labelled_index = random.choices(list(range(len(df))), k =40)
unlabelled_index = [ i for i in range(len(df)) if i not in labelled_index]
labelled = df.iloc[labelled_index]
unlabelled = df.iloc[unlabelled_index]

from pathlib import Path

labelled_csv_path = Path(args.csv).parent / str(Path(args.csv).stem + "_labelled.csv")
unlabelled_csv_path = (
    Path(args.csv).parent / str(Path(args.csv).stem + "_unlabelled.csv")
)
labelled.to_csv(header=None, path_or_buf=str(labelled_csv_path))
unlabelled.to_csv(header=None, path_or_buf=str(unlabelled_csv_path))
