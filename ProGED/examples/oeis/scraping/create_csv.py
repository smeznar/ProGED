import pandas as pd
# from selection import basic_info
# from selection import final_selection

# From zero:
# 1.) first download all (core,nice,-more) into dictionary
from download import new_fetch
bseqs = new_fetch(do_write=False, max_seq_length=50)  # = default length
# bseqs = new_fetch(0, 3, do_write=False, max_seq_length=50)  # = default length
# or equivalently load already downloaded:
# from save_new_bfiles import bseqs
# 2.) remove Mersene seq:
bseqs.pop('A000043')
# bseqs.pop('A000043', None)
# 3.) select small:
from selection import select_small
# filter accordingly:
seqs = select_small(bseqs, has_titles=1)

# Then header, body. 
df = pd.DataFrame(seqs)
df_sorted = df.sort_index(axis=1)
print(df_sorted.head())
csv_filename = "oeis_selection.csv"
# df_sorted.to_csv(csv_filename, index=False)

# after download:
check = pd.read_csv(csv_filename)
print("Read file from csv:")
print(check)

