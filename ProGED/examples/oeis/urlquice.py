from urllib import request
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import re
from numpy import number

txt = """# Greetings from The On-Line Encyclopedia of Integer Sequences! http://oeis.org/

Search: keyword:core
Showing 1-10 of 178

%I A000040 M0652 N0241
%S A000040 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,
%T A000040 97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,
%U A000040 181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271
%N A000040 The prime numbers.
%C A000040 See A065091 for comments, formulas etc. concerning only odd primes. For all information concerning prime powers, see A000961. For contributions concerning "almost primes" see A002808.
%C A000040 A number n is prime if (and only if) it is greater than 1 and has no positive divisors except 1 and n.
%C A000040 A natural number is prime if and only if it has exactly two (positive) divisors.
%C A000040 A prime has exactly one proper positive divisor, 1.
%C A000040 The paper by Kaoru Motose start
"""
txt2 = """ A038566 A054424 gives mapping to Stern-Brocot tree.
%Y A038566 Row sums give rationals A111992(n)/A069220(n), n>=1.
%Y A038566 A112484 (primes, rows n >=3).
%K A038566 nonn,frac,core,nice,tabf
%O A038566 1,4
%A A038566 _N. J. A. Sloane_
%E A038566 More terms from _Erich Friedman_
%E A038566 Offset corrected by _Max Alekseyev_, Apr 26 2010

%I A000058 M0865 N0331
%S A000058 2,3,7,43,1807,3263443,10650056950807,113423713055421844361000443,
%T A000058 12864938683278671740537145998360961546653259485195807
%N A000058 Sylvester's sequence: a(n+1) = a(n)^2 - a(n) + 1, with a(0) = 2.
%C A000058 Also c"""
txt2 += txt

def fetch(start=0, end=1e10):
    search=request.urlopen('https://oeis.org/search?q=keyword:core&fmt=text')
    header_length = 1000  # number of characters in header
    header = search.read(header_length).decode()
    # print(header)
    total = int(re.findall("Search: .+\nShowing \d+-\d+ of (\d+)\n", header)[0])
    print(total)
    seqs = []
    # number_of_sequences = 30  # ... test
    cropped = min(total, end)
    beginning = max(0, cropped-start)
    number_of_sequences = beginning
    print(f"total number of sequences found: {total}\n"
        f"start: {start}, end: {end}, "
        f"sequences of intersection:{number_of_sequences}")

    for i in range((number_of_sequences-1)//10+1):  # 178 -> 10-20, 20-30, ..., 170-180
        # print(i)
        search=request.urlopen(
            f'https://oeis.org/search?q=keyword:core&fmt=text&start={start+i*10}')
        print(i, "after wget")
        page = search.read().decode()
        # print(i, "after decode")
        stu_lines = re.findall(
            r"%S (A\d{6}) (.+)\n(%T A\d{6} (.*)\n)*(%U A\d{6} (.*)\n)*", page)
        # ten_seqs = [(seq[0], seq[1]+seq[3]+seq[5]) for seq in stu_lines]
        ten_seqs = [(seq[0], [int(a_n) for a_n in (seq[1]+seq[3]+seq[5]).split(",")])
            for seq in stu_lines]
        # print(ten_seqs)
        if len(ten_seqs)<10: print("\n\n\n less than 10 !!!!\n\n\n")
        print(len(ten_seqs))
        seqs += [ten_seqs]
        # print(i, "after regex ")
        
    seqs_flatten = []
    for ten_seq in seqs:
        seqs_flatten += ten_seq
    print("all seqs:", len(seqs_flatten))
    file = open("saved_core.py", "w")
    file.write("core_list = " + repr(seqs_flatten) + "\n"
               + "core_unflatten = " + repr(seqs) + "\n")
    file.close()
    # import saved_core
    print("fetch ended")
    return
# fetch(130, 190)
# fetch()
# print("end of fetch")

# page = txt2
# # stu_lines = re.findall( r"%S (A\d{6}) (.+)\n%T A\d{6} (.*)\n%U A\d{6} (.*)\n", page)
# stu_lines = re.findall( r"%S (A\d{6}) (.+)\n(%T A\d{6} (.*)\n)*(%U A\d{6} (.*)\n)*", page)
# stu_seqs = [(stu_line[0], stu_line[1]+stu_line[3]+stu_line[5]) for stu_line in stu_lines]
# print(stu_seqs)
# t_line = stu_lines[0][2]
# t_line = stu_lines[0][3]
# print(t_line)
# t_seq = re.findall(r"%[STU] A\d{6} (.+)\n", t_line)
# print(t_seq)
# stu_seqs = [[(i, re.findall(r"%[STU] A\d{6} (.+)\n", i)
#   for i in stu_line[1:]]
#     for stu_line in stu_lines]
# print(stu_seqs)




import saved_core
# import saved_core110up as saved_core
seqs_flat = saved_core.core_list
seqs = saved_core.core_unflatten

# # print(len(list(set(seqs_flatten))))
# # print(seqs)
print("number of packets of seqs", len(seqs))
print("number of seqs in seqs:", sum(len(tseqs) for tseqs in seqs))
print("unique ids:")
ids = [seq[0] for seq in seqs_flat]
print(len(ids))
print(len(list(set(ids))))
print("end")
print(seqs_flat[-4:])


## --- Length of sequence analisis: --- ##
#
# result = 0
# for ten_seqs in seqs:
#     result += 10
#     print(f"in results: {result-10}-{result}")
#     print(len(ten_seqs))
# for seq_pair in seqs_flat:
#     print("number of terms in sequence:", len(seq_pair[1]))
import numpy as np
# print(np.mean(np.array([len(seq_pair[1]) for seq_pair in seqs_flat])))
lens = np.array([len(seq_pair[1]) for seq_pair in seqs_flat])
print(lens)
# print(sum(lens<10))
# for i in range(10):
#     print(sum(lens<10*i), 10*i)
# print(len(lens))
thresholds = np.arange(max(lens))
amount_manys = list(map(lambda i: sum(lens>=i), thresholds))
# print(bigs)

## -- check those with more than 100 terms -- ##
seqs = seqs_flat
def check_centum():
    # seqs = seqs_flat
    print("100lens")
    # cents = [(seq[0], len(seq[1]), seq[1]) for seq in seqs if len(seq[1])>100]
    ids_cents = [seq[0] for seq in seqs if len(seq[1])>=23]
    # print(cents, "cents")
    ids_cents.sort()
    print(ids_cents)
    print("number of sequences lots", len(ids_cents))
    return ids_cents
ids_lots = check_centum()

## -- check those with terms bigger than 1e16 -- ##
seqs = seqs_flat
def pick_bignum():
    print("big numbers")
    # largs = [(seq[0], len(seq[1]), seq[1]) for seq in seqs if max(seq[1])>1e16]
    # ids_largs = [seq[0] for seq in seqs if max(seq[1])<=1e16.5]
    ids_largs = [seq[0] for seq in seqs if max(seq[1])<=10**(15.6)]
    ids_largs.sort()
    # print(largs, "largs")
    print(ids_largs)
    print("number of sequences large w/ numbers", len(ids_largs))
    return ids_largs
ids_largs = pick_bignum()
# assert check_centum() == pick_bignum(), "they do not perfectly match"
inter = set(ids_largs).intersection(set(ids_lots))
print("intersection:", inter, "len(intersection) : ", len(list(inter)))
dif_lots = set(ids_lots).difference(inter)
dif_largs = set(ids_largs).difference(inter)

seqs_dict = dict(seqs)
print("\n\n")
for i in [(id, len(dict(seqs)[id]), float(max(dict(seqs)[id])), dict(seqs)[id]) for id in dif_lots]:
    print(i)
print("\n\n")
for i in [(id, len(dict(seqs)[id]), float(max(dict(seqs)[id])), dict(seqs)[id]) for id in dif_largs]:
    print(i)
# print(seqs[:3], "seqs")
# print(dict(seqs[:3]))


# plot largs:
maxs = np.array([max(seq[1]) for seq in seqs])
# xs = np.array([10**i for i in range(int(np.log10(max(maxs))))])
# xs = [10**i for i in range(int(np.log10(max(maxs))))]
xs = np.array([float(10**i) for i in range(1, 300)])
# xs = np.array([i for i in range(1, 300)])
# nums = 10**xs
max_exp = sum(xs < max(maxs))
xs = xs[:max_exp]#[::-1]
# xs = float(xs)
# xs = xs[:15]
# # print(max_exp, max(maxs))
# print(np.log10(max(maxs)))
largs = list(map(lambda x: sum(maxs<=x), xs))
# largs = list(map(lambda x: x, xs))
# largs = list(map(lambda x: x**2, xs))
# xs = (-1)*xs
# print(xs)
def plot_largs(xs, largs):
    fig, ax = plt.subplots()
    # plt.grid(True)
    # plt.xlabel("size [int]")
    # plt.ylabel("number of sequences with largest term smaller than *threshold* (x)")
    ax.plot(xs, largs)
    print(len(xs))
    ax.set_xscale("log", base=10)
    ax.set_xlim(float(xs[-1]), xs[0])
    # ax.xaxis.set_major_locator(MultipleLocator(45*(xs[-1]-xs[0])/len(xs)))
    # ax.xaxis.set_major_formatter('{x:.0f}')
    # ax.xaxis.set_minor_locator(MultipleLocator(4*(xs[-1]-xs[0])/len(xs)))
    # ax.xaxis.set_minor_locator(xs)
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    min_mult = 3
    ax.set_xticks(xs[::3], minor=True)
    ax.set_xticks(xs[::5*3])
    # plt.xticks(xs[::3])
    # ax.set_yticks(np.array(list(set(largs[::10]))))
    # ax.set_yticks(np.array(list(set(largs[::2]))), minor=True)
    ax.set_yticks(np.arange(0, largs[-1], 20))
    ax.set_yticks(np.arange(0, largs[-1], 5), minor=True)
    print(np.array(list(set(largs[::10]))))
    # ax.xaxis.grid(True, which="minor")
    ax.axes.grid(True, which="minor")
    # ax.xaxis.grid(True, which="major", color="k", linewidth=1)
    ax.axes.grid(True, which="major", color="k", linestyle="-", linewidth=1)
    # ax.xaxis.grid(True)
    # ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=5, width=1)
    # ax.tick_params(which='minor', length=2, color='r')
    # plt.plot(xs, largs)
    # plt.xscale("symlog", base=10)
    # plt.xscale("log", base=10)
    # ax.xscale("log", base=10)  # Not working
    # plt.xscale("log", base=1.2)
    # plt.xlim(xs[-1], xs[0])
    # plt.xlim(min(xs)-1, max(xs)+1)
    # print(float(xs[-1]))
    # plt.xlim(0, float(xs[-1]))
    # plt.xlim(float(xs[-1]), xs[0])
    # plt.xlim(0, 10**100)
    # plt.xlim(0, 1e99)
    # plt.xlim(0, 1e99)
    # plt.xlim(10*100, 0)
    # plt.title("Sequences with small enough terms")
    # plt.xticks(xs[::3])
    # plt.yticks(np.array(set(largs[::10])))
    # plt.yticks([i for i in range(1, 178, 10)])
    plt.show()
    return
# plot_largs(xs, largs)

def plot_fews(xs, ys):
    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    ax.set_xticks(np.arange(0, xs[-1], 10))
    ax.set_xticks(np.arange(0, xs[-1], 20/10), minor=True)
    ax.set_yticks(np.arange(0, max(ys), 25))
    ax.set_yticks(np.arange(0, max(ys), 5), minor=True)
    ax.axes.grid(True, which="minor")
    ax.axes.grid(True, which="major", color="k", linestyle="-", linewidth=1)
    ax.set_xlabel("threshold")
    ax.set_ylabel("number of sequences with more than *threshold* terms")
    ax.set_title("Sequences with enough terms")
    plt.show()
    return
# plot_fews(thresholds, amount_manys)

def plot_alot(x, bigs):
    plt.grid(True)
    plt.xlabel("threshold")
    plt.ylabel("number of sequences with more than *threshold* terms")
    plt.plot(x, bigs)
    plt.title("Sequences with enough terms")
    plt.show()
    return
# plot_alot(x, bigs)
## EOF lengths analysis ##


## -- Check scrap against gzipped oeis database -- ##
def check_scrap():
    # seqs = seqs_flat
    file2 = open("stripped_oeis_database.txt", "r")
    # original = file2.read(10**9)
    original = file2.read()
    file2.close()
    print(original[:1000])
    for n, seq_pair in enumerate(seqs):
        seq_id = seq_pair[0]
        seq = seq_pair[1]
        # seq_id = "A000002"
        # compare = re.findall(seq_id+r".*\n", original)
        # compare = re.findall(seq_id+".[\d\-,]+\n", original)
        compare = re.findall(seq_id+".+\n", original)
        with_ = seq_id+" "
        str_seq = seq_id + " ,"
        for i in seq:
            str_seq += str(i)+","
        str_seq += "\n"
        str_seq = [str_seq]
        # print(compare, "this is what I found. Compare with:")
        # print(str_seq)
        bool = compare[0] == str_seq[0]
        assert compare[0] == str_seq[0]
        print(n, "Correct scrapping!!! found it" if bool else "scrapping FAILED!!!")
        assert len(compare[0]) == len(str_seq[0])
        print(len(compare[0]), len(str_seq[0]))
    print("Seems scrapping is correct. Checking is done.")
    return
# check_scrap()
print("end")
