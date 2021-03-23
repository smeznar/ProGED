import pandas as pd
# from selection import final_selection as seqs
from selection import basic_info

# From zero:
# 1.) first download all (core,nice,-more) into dictionary
from download import new_fetch
bseqs = new_fetch(do_write=False, max_seq_length=100)  # = default length
# or equivalent:
# from save_new_bfiles import bseqs
# 2.) remove Mersene seq:
bseqs.pop('A000043')
# 3.) select small:
from selection import select_small
# filter accordingly only first 50 terms:
seqs = select_small(bseqs, head_length=50)



# basic_info(seqs)
# First sort way:
# seqs_sorted = {id:seqs[id] for id in sorted(list(seqs))}
# print([(id,seq) for id,seq in seqs.items()][:4])

# Then header, body. 
df = pd.DataFrame(seqs)
print(df.head())
# print(df.tail())
# print(df.describe())
# print(df.columns)
print(df.sort_index(axis=1).head())
df_sorted = df.sort_index(axis=1)
csv_filename = "oeis_selection.csv"
df_sorted.to_csv(csv_filename, index=False)
check = pd.read_csv(csv_filename)
print("Read file from csv:")
print(check)

