import pandas as pd
from selection import final_selection as seqs
from selection import basic_info

# basic_info(seqs)
# First sort:
seqs_sorted = {id:seqs[id] for id in sorted(list(seqs))}
# print([(id,seq) for id,seq in seqs.items()][:4])

# Then header, body. 
df = pd.DataFrame(seqs)
print(df.head())
# print(df.tail())
# print(df.describe())
# print(df.columns)
# print(df.sort_index(axis=1).head())
df_sorted = df.sort_index(axis=1)
csv_filename = "oeis_selection.csv"
df_sorted.to_csv(csv_filename, index=False)
check = pd.read_csv(csv_filename, index_col=None)
print("Read file from csv:")
print(check)

