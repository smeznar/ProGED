import re
import sys

from tee_this import create_log

print(sys.argv, len(sys.argv))
file_name = sys.argv[1] if len(sys.argv)>=2 else "bests_druga.txt"
threshold_order = int(sys.argv[2]) if len(sys.argv)>=3 else 2
all_lines = int(sys.argv[3]) if len(sys.argv)>=4 else "uncounted"

threshold = 10**(-1*threshold_order)
create_log("threshold-" + file_name, with_random=False) 
with open(file_name, "r") as file:
    string = file.read()
print_models = re.findall("Final score:\n[^w]*", string)
# print(print_models)
lines = re.findall("model:.*error:.[\d\.e\-]+", print_models[0])
filtered_lines = [line for line in lines if float(re.findall("error:.([\d\.e\-]+)", line)[0]) <= threshold]
# print(filtered_lines)
for i in filtered_lines:
    print(i)
print(len(lines), "regexed lines")
print(all_lines, "all of them")

