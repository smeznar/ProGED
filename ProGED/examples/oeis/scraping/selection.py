from save_new_bfiles import bseqs
# Selection begins with select small (or 2 lines before).

def abs_max(seq):
    return max([abs(term) for term in seq])
def basic_info(seqs):
    """Print basic info of the dataset."""
    print("number of all seqs:", len(seqs))
    lens = [len(seqs[id]) for id in seqs]
    maxs = [abs_max(seq) for _, seq in seqs.items()]
    print("min, max number of terms in seqs:", min(lens), max(lens))
    print("max(1e16 limit) absolute value in seqs:",
         "{0:e}".format(float(max(maxs))))
    # print(lens)  # print(maxs) too big to print
    print("first two sequences:", [(i, seqs[i]) for i in list(seqs)[:2]])
    return
# basic_info(bseqs)

def less_len(seqs, max_len=100):
    """Make sure, which sequences have less than 100 terms. 
    Also see their max value.
    """
    less = [(id, len(seqs[id])) for id in seqs if len(seqs[id])<max_len]
    def key_fun (i):
        return i[1]
    less.sort(key=key_fun)
    print("number of seqs with less than 100 terms:", len(less))
    # print("all seqs with less than 100 terms:", less)
    print("all seqs with less than 100 terms, printed with their:\n"
        "        bfile url         | len | max ")
    for id, j in less:
        maxi = max([abs(term) for term in bseqs[id]])
        max_str = 1e16 if maxi>1e16 else float(maxi)
        print("oeis.org/" + id + '/b' + id[1:] + '.txt', j, max_str)
        # if id in more:
        #     print(f"but {id} is in more, so ...")
    return
# less_len(bseqs)

def print_big(seqs):
    """Mostly useless."""
    maxs = [abs_max(seq) for _id, seq in seqs.items()]
    avoid_overflow = [1e16 if maxi>1e16 else float(maxi) for maxi in maxs]
    print(avoid_overflow)
    return
# print_big(bseqs)


### start selection ###
# def abs_max(seq):
#     return max([abs(term) for term in seq])

# remove Mersene seq:
bseqs.pop('A000043')

def select_small(seqs, head_length=50):
    small = {id: seq for id, seq in seqs.items() if abs_max(seq[:head_length])<1e16}
    return small
# small = select_small(bseqs)
if __name__ == "__main__":
    # execute only if run as a script

    # filter accordingly only first 50 terms:
    small = select_small(bseqs, head_length=50)
    print("selected:")
    basic_info(small)
    print(len(small), small.popitem())
    
# For use outside of this file.
# final_selection = select_small(bseqs)







### Answer by bfiles_old: in dataset we keep: 
###     *keyword:core keyword:nice 
###     * and delete all sequences with absolute terms > 1e16.
###     * and delete sequence Mersene A000043, since only 47 terms (in hard keyword)
###             because we need at least 47 or should I say 100 terms (as it turns out).
###     ** it turns out all other remaining sequences in core have at least 100 terms. Very good.
