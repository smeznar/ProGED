import re

from tee_this import create_log

threshold = 10**(-1)
with open("bests_druga.txt", "r") as file:
    string = file.read()
print(string)
print(string[:10])
list = re.findall("dsa", "dsadsadsa")
list = re.match("dsa", "dsadsadsa")
print(list.group())
print(list)
print_models = re.findall("Final score:[.]", string)
print_models = re.findall("Final score:\n[^w]*", string)
print(print_models)
# lines = re.findall(".*", print_models[0])
# print(len(lines))
lines = re.findall("model:.*error:.[\d\.e\-]+", print_models[0])
# lines = re.findall("model:.*error:.+", print_models[0])

print(len(lines))
# for line in lines:
#     error = re.findall("error:.([\d\.e\-]+)", line)[0]
#     # print(error)
#     if float(error) <= threshold:
#         print(line)
filtered_lines = [line for line in lines if float(re.findall("error:.([\d\.e\-]+)", line)[0]) <= threshold]
print(filtered_lines)
print(len(filtered_lines))
# print(lines)
for i in filtered_lines:
    print(i)


print(1082-883, "vseh")

