from argparse import ArgumentParser

import pandas as pd
from pathlib import Path


parser = ArgumentParser()

parser.add_argument('-c','--csvs', nargs='+', help='<Required> Set flag', required=True)
parser.add_argument('-s','--stem',help='file name of merged csv',required=True)
args = parser.parse_args()

df_list = [ pd.read_csv(c,header=None) for c in args.csvs]
merged_df = df_list[0]

for df in df_list[1:]:
    merged_df = pd.concat([merged_df,df],ignore_index=True)

merged_csv_path = Path(args.csvs[0]).parent / str(args.stem + '.csv')
merged_df.to_csv(path_or_buf=str(merged_csv_path),header=None,index=False) # save in the same director as the first csv