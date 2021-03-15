from saved_bfiled import core_bfiled_dict as bseqs 
more = ("A000109", "A000112", "A000609", "A002106", "A003094", "A005470", "A006966")
ids_nomore = list(set(bseqs).difference(set(more)))
print("number of nomore seqs:", len(ids_nomore))
nomore = {id: bseqs[id] for id in ids_nomore}
# bseqs = nomore
print("number of all seqs:", len(bseqs))
lens = [len(bseqs[id]) for id in bseqs]
print("min, max number of terms in seqs:", min(lens), max(lens))
# print(lens)
less = [(id, len(bseqs[id])) for id in bseqs if len(bseqs[id])<100]
def key_fun (i):
    return i[1]
less.sort(key=key_fun)
print(less)
for id, j in less:
    print("oeis.org/" + id + '/b' + id[1:] + '.txt', j)
    maxi = max([abs(term) for term in bseqs[id]])
    max_str = "1e16" if maxi>1e16 else float(maxi)
    print(max_str)
    if id in more:
        print(f"but {id} is in more, so ...")
# print([max([abs(i) for i in seq]) for _id, seq in bseqs])
for i in more:
    print(i)


ids = [
    "A000043",
    "A000001",
    "A000105",
    "A000798",
    "A055512",
    ]
for id in ids:
    print("oeis.org/" + id + '/b' + id[1:] + '.txt')

# to generate Mersene oeis:
# from sympy import isprime, prime
# for n in range(1, 10**2*2*2*2):
#     if isprime(2**prime(n)-1):
#         print((n, prime(n)), end=', ')
# AnsMy: Mersene out (they are hard (keyword))

# # Are there more with enough terms? No! (max 47 terms)
# print(10**5-3)
# maxs = [max([abs(term) for term in bseqs[id]]) for id in more]
# lens = [len(bseqs[id]) for id in more]
# print(maxs)
# print(lens)
# print("id", "max", "len")
# for id in more:
#     # print("oeis.org/" + id + '/b' + id[1:] + '.txt', seq)
#     maxi = max([abs(term) for term in bseqs[id]])
#     leni = len(bseqs[id])
#     max_str = "1e16" if maxi>1e16 else float(maxi)
#     print(id, max_str, leni)
# # AnsMy: all more go out.


### Answer by this bfiles: in dataset we keep: 
###     *keyword:core keyword:nice 
###     * and delete all sequences with absolute terms > 1e16.
###     * and delete sequence Mersene A000043, since only 47 terms (in hard keyword)
###             because we need at least 47 or should I say 100 terms (as it turns out).
###     ** it turns out all other remaining sequences in core have at least 100 terms. Very good.