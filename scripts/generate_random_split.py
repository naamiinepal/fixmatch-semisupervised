from argparse import ArgumentParser
import random
import numpy as np
from pathlib import Path

import pandas as pd

parser = ArgumentParser()
parser.add_argument("--num_labels", "-nl", default=40, type=int)
parser.add_argument(
    "--csv", default="isic_challenge/ISBI2016_ISIC_Part3B_Training_GroundTruth.csv"
)
parser.add_argument(
    "-sn",
    "--split_name",
    nargs="+",
    help="<Required> Set flag",
    default=["labelled", "unlabelled"],
)
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument('--balanced_split',action='store_true',default=False)

SEED = 98123  # for reproducibility
random.seed(SEED)
args = parser.parse_args()

df = pd.read_csv(args.csv, header=None)
classes = df[1].unique()
smallest_class_size = np.min([ (df[1] == c).sum() for c in classes])
if args.balanced_split:
    # if balanced split, then split the smallest class equally into two sets
    # the ulb classes will also be split equally of size equal to smallest class
    lb_samples_perclass = smallest_class_size // 2 
    ulb_samples_perclass = smallest_class_size // 2
else:    
    # split the first set as balanced set, everything else goes into ulb set  
    lb_samples_perclass = args.num_labels // len(classes)
    ulb_samples_perclass = None

lb_idx = []
ulb_idx = []

# get balanced samples per class in the first split
for c in classes:
    idx = np.where(df[1] == c)[0]
    np.random.shuffle(idx)
    lb_idx.extend(idx[:lb_samples_perclass])
    if args.balanced_split:
        ulb_idx.extend(idx[lb_samples_perclass: lb_samples_perclass+ulb_samples_perclass])
    else:
        ulb_idx.extend(idx[lb_samples_perclass:])
    if args.verbose:
        print(
            f"{c} labelled {lb_samples_perclass} unlabelled {len(df[1]) - lb_samples_perclass}"
        )
if args.verbose:
    print(f"Total {len(df[1])} labelled {len(lb_idx)} unlabelled {len(ulb_idx)}")

# obtain random subset
labelled_index = lb_idx
unlabelled_index = ulb_idx
# labelled_index = random.choices(list(range(len(df))), k=args.num_labels)
# unlabelled_index = [i for i in range(len(df)) if i not in labelled_index]

for l in labelled_index:
    assert l not in unlabelled_index

labelled = df.iloc[labelled_index].sort_values(0)  # sort by first column
unlabelled = df.iloc[unlabelled_index].sort_values(0)


labelled_csv_path = Path(args.csv).parent / str(
    Path(args.csv).stem + f"_{args.split_name[0]}.csv"
)
unlabelled_csv_path = Path(args.csv).parent / str(
    Path(args.csv).stem + f"_{args.split_name[1]}.csv"
)
labelled.to_csv(header=None, index=False, path_or_buf=str(labelled_csv_path))
unlabelled.to_csv(header=None, index=False, path_or_buf=str(unlabelled_csv_path))
