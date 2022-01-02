from save_new_bfiles import bseqs
# Selection begins with select small (or 2 lines before).
# New download:
from saved_new_bfile2 import bseqs

def abs_max(seq):
    return max([abs(term) for term in seq])
def power_order(number):
    bigger = [order for order in range(-100, 100) if number > 10**order]
    return bigger[-1]

def basic_info(seqs, has_titles: int = 1, display_len: int = 100):
    """Print basic info of the dataset."""
    print("number of all seqs:", len(seqs))
    lens = [len(seqs[id][has_titles:]) for id in seqs]
    maxs = [abs_max(seq[has_titles:]) for _, seq in seqs.items()]
    print("min, max number of terms in seqs:", min(lens), max(lens))
    print("max(1e16 limit) absolute value in seqs:",
          f"10e{power_order(max(maxs))}")
         # f"{max(maxs):e}")
    # print(lens)  # print(maxs) too big to print
    print("first two sequences:", [(i, seqs[i][:display_len]) for i in list(seqs)[:2]])
    return
# basic_info(bseqs)
# basic_info(bseqs, has_titles=1, display_len=100)
# 1/0

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
# bseqs.pop('A000043')

def select_small(seqs: dict, head_length=10**16, has_titles=0):
    small = {id_: seq[has_titles:head_length+has_titles] for id_, seq in seqs.items() 
            if abs_max(seq[has_titles:head_length+has_titles]) < 1e16}
    return small
# small = select_small(bseqs)

if __name__ == "__main__":
    # execute only if run as a script

    # filter accordingly only first 100 terms:
    # small = select_small(bseqs, head_length=100)
    # small = select_small(bseqs, head_length=50)

    basic_info(bseqs)

    # 1/0
    # small = select_small(bseqs, head_length=10, has_titles=1)
    # small = select_small(bseqs, head_length=1000, has_titles=1)
    # print('\n'*10)
    small = select_small(bseqs, head_length=1000, has_titles=1)
    print("selected:")
    basic_info(small, display_len=20)
    bseqs.pop('A000043')
    small = select_small(bseqs, head_length=1000, has_titles=1)
    print('\n'*10, 'just popped')
    basic_info(small, display_len=20)
    print(len(small))
    # print(small.popitem())
    
# For use outside of this file.
# final_selection = select_small(bseqs)







### Answer by bfiles_old: in dataset we keep: 
###     *keyword:core keyword:nice 
###     * and delete all sequences with absolute terms > 1e16.
###     * and delete sequence Mersene A000043, since only 47 terms (in hard keyword)
###             because we need at least 47 or should I say 100 terms (as it turns out).
###     ** it turns out all other remaining sequences in core have at least 100 terms. Very good.
