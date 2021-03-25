import pandas as pd
# from selection import basic_info
# from selection import final_selection

# From zero:
# 1.) first download all (core,nice,-more) into dictionary
from download import new_fetch
bseqs = new_fetch(do_write=False, max_seq_length=100)  # = default length
# or equivalently load already downloaded:
# from save_new_bfiles import bseqs
# 2.) remove Mersene seq:
bseqs.pop('A000043')
# 3.) select small:
from selection import select_small
# filter accordingly:
seqs = select_small(bseqs)

# Then header, body. 
df = pd.DataFrame(seqs)
df_sorted = df.sort_index(axis=1)
print(df_sorted.head())
csv_filename = "oeis_selection.csv"
df_sorted.to_csv(csv_filename, index=False)
check = pd.read_csv(csv_filename)
print("Read file from csv:")
print(check)

