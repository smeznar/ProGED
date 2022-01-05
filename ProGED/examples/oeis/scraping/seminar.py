import pandas as pd
csv_filename = "oeis_dmkd.csv"
check = pd.read_csv(csv_filename)

print("Read file from csv:")
print(check)
