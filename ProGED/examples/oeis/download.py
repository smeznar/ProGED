import pyoeis
c = pyoeis.OEISClient()
look = c.lookup_by(
        "",
        "keyword:core keyword:more",
        max_seqs=300,
        list_func=True)
print(len(look), "len(look)")

for i in look:
    #print(i.unsigned_list)
    print(len(i.unsigned_list))
import time
time1 = time.perf_counter()
look_core = c.lookup_by(
        "",
        "keyword:core",
        max_seqs=300,
        list_func=True)
print("time used for core:", time1-time.perf_counter())

print(len(look_core), "len(look)")
for i in look_core:
    #print(i.unsigned_list)
    print(len(i.unsigned_list))
edited = [(i.id, i.name, i.unsigned_list) for i in look_core]
f = open("saved_core.py", "w")
f.write("core_list = " + repr(edited) + "\n")
f.close()
#import saved_core
   
#print(sum([len(i.unsigned_list) for i in look_core])/len(look_core), "mean")

