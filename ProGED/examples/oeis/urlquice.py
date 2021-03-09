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
    search=request.urlopen('https://oeis.org/search?q=keyword:core%20keyword:nice&fmt=text')
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
            f'https://oeis.org/search?q=keyword:core%20keyword:nice&fmt=text&start={start+i*10}')
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
seqs = saved_core.core_list
seqs_unflatten = saved_core.core_unflatten

# # print(len(list(set(seqs_flatten))))
# # print(seqs)
print("number of packets of seqs", len(seqs_unflatten))
print("number of seqs in seqs:", sum(len(tseqs) for tseqs in seqs_unflatten))
print("unique ids:")
ids = [seq[0] for seq in seqs]
print(len(ids))
print(len(list(set(ids))))
print("last 4 seqs in seqs:", seqs[-4:])

# seqs = seqs_flat
## -- remove seq A000035 -- ##
print("Removing seqs by hand ... ")
seqs_dict = dict(seqs)
print("number of seqs (before removing):", len(list(seqs_dict)))
# seqs_dict = dict(seqs_flat)
seqs_to_remove = ["A000035"]
seqr = seqs_to_remove[0]
print("no error?", seqs_dict[seqr])
print("is the seq inside before removing:", seqs_to_remove[0] in seqs_dict)
[seqs_dict.pop(seq, None) for seq in seqs_to_remove]
print("is seq inside after removing:", seqs_to_remove[0] in seqs_dict)
print("number of seqs (now):", len(list(seqs_dict)))
# print(seqs_dict)
# print("error?", seqs_dict[seqr])


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
lens = np.array([len(seq_pair[1]) for seq_pair in seqs])
print(lens)
# print(sum(lens<10))
# for i in range(10):
#     print(sum(lens<10*i), 10*i)
# print(len(lens))
thresholds = np.arange(max(lens))
amount_manys = list(map(lambda i: sum(lens>=i), thresholds))
# print(bigs)

## -- check those with more than 100 terms -- ##
# seqs = seqs_flat
def check_centum():
    # seqs = seqs_flat
    print("100lens")
    # cents = [(seq[0], len(seq[1]), seq[1]) for seq in seqs if len(seq[1])>100]
    ids_cents = [id for id in seqs_dict if len(seqs_dict[id])>=23]
    # print(cents, "cents")
    ids_cents.sort()
    print(ids_cents)
    print("number of sequences lots", len(ids_cents))
    return ids_cents
ids_lots = check_centum()

## -- check those with terms bigger than 1e16 -- ##
# seqs = seqs_flat
def pick_bignum():
    print("big numbers")
    # largs = [(seq[0], len(seq[1]), seq[1]) for seq in seqs if max(seq[1])>1e16]
    # ids_largs = [seq[0] for seq in seqs if max(seq[1])<=1e16.5]
    ids_largs = [id for id in seqs_dict if max(seqs_dict[id])<=10**(15.6)]
    ids_largs.sort()
    # print(largs, "largs")
    print(ids_largs)
    print("number of sequences large w/ numbers", len(ids_largs))
    return ids_largs
ids_largs = pick_bignum()
# assert check_centum() == pick_bignum(), "they do not perfectly match"

## -- new metric (sum of digits) -- ##
def metric(seq):
    # print("calculating metric of the seq")
    # print(seq)
    # print([str(an) for an in seq])
    # print([2+len(str(an)) for an in seq])
    return sum([1+len(str(an)) for an in seq])
print("some metric:", metric(seqs_dict["A005036"]))

metrics = [metric(seqs_dict[seq]) for seq in seqs_dict]
metrics_bounds = (min(metrics), max(metrics))
print(metrics_bounds)
metrics_dict = {} 
for i in metrics:
    if str(i) in metrics_dict:
        metrics_dict[str(i)] += 1
    else:
        metrics_dict[str(i)] = 1
print(metrics_dict)
metric_histo = [metrics_dict.get(str(long), 0) 
    for long in range(metrics_bounds[0], metrics_bounds[1])]
# metric_histo = [long
#     for long in range(metrics_bounds[0], metrics_bounds[1]+1)]
print(metric_histo)
def plot_metric(xs, ys):
    plt.title("metric(length)")
    plt.ylabel("metric")
    plt.xlabel("length")
    plt.plot(xs, ys)
    plt.show()
    return
# plot_metric(np.arange(min(metrics), max(metrics)), metric_histo)

## -- new metric's lots -- ##
find_threshold = [sum(np.array(metrics)<=threshold)<=137 
    for threshold in range(min(metrics), max(metrics))]
threshold = min(metrics) + sum(find_threshold)
print("threshold found:", threshold)
metric_dict = {}
[metric_dict.update({i:seqs_dict[i]}) 
    for i in seqs_dict if metric(seqs_dict[i])<=threshold]
print(len(metric_dict))
print(sum(np.array(metrics)<=threshold))
keyword_more = ("A000109", "A000112", "A000609", "A002106", "A003094", "A005470", "A006966")
ids_metrics = [i for i in seqs_dict if metric(seqs_dict[i])<=threshold]
ids_metrics = [i for i in seqs_dict if metric(seqs_dict[i])>=250]
print("high metric", ids_metrics)
ids_metrics_spec = [i for i in seqs_dict if metric(seqs_dict[i]) in range(0, 195)]
ids_metrics_spec = list(set(ids_metrics_spec).difference(set({})))
print(len(ids_metrics_spec), ids_metrics_spec)
ids_metrics_spec = list(set(ids_metrics_spec).difference(set(keyword_more)))
print(len(ids_metrics_spec), ids_metrics_spec)
nice_ids = [(i, metric(seqs_dict[i]), len(seqs_dict[i])) for i in ids_metrics_spec]
print(len(nice_ids), nice_ids)
print(min(metrics), max(metrics))

# print("all metrics: \nthresh: metric: 300 terms: 23 max term: 1e16\n")
# for i in seqs_dict:
#     seq = seqs_dict[i]
#     print(i, "metric:", metric(seq), "terms:", len(seq), "max term:", "{0:e}".format(max(seq)))

# 1/0

inter = set(ids_largs).intersection(set(ids_lots))
print("intersection:", inter, "len(intersection) : ", len(list(inter)))
dif_lots = set(ids_lots).difference(inter)
dif_largs = set(ids_largs).difference(inter)
print(len(dif_lots))
print(len(dif_largs))
# inter_metric = set(inter).intersection(set(ids_metrics))
# print("inter \cap ids_metrics", len(inter_metric))


# seqs_dict = dict(seqs)
print("\n\n")
for i in [(id, 
            len(dict(seqs)[id]), 
            float(max(dict(seqs)[id])), 
            metric(seqs_dict[id]),
            dict(seqs)[id],
            ) for id in dif_lots]:
    print(i)
print("\n\n")
for i in [(id, 
            len(dict(seqs)[id]), 
            "{0:e}".format(float(max(dict(seqs)[id]))), 
            metric(seqs_dict[id]),
            dict(seqs)[id]) for id in dif_largs]:
    print(i)
# print(seqs[:3], "seqs")
# print(dict(seqs[:3]))

## -- checkout ids_largs -- ##
largs2 = [len(seqs_dict[i]) for i in ids_largs]
# print(seqs_dict)
print("number of enough terms", len(ids_largs))
largs2.sort()
print(largs2, "largs2")
# print(ids_largs, "ids_largs")
largs_max = min([len(seqs_dict[i]) for i in ids_largs])
print(largs_max, "largs_max")
print("nic")
ids_largs

ids_largs_nomore = list(set(ids_largs).difference(set(keyword_more)))
print("largs nomore:", len(ids_largs_nomore))
largs_nomore = [len(seqs_dict[i]) for i in ids_largs_nomore]
largs_nomore.sort()
print(largs_nomore)

# 1/0

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
    original = file2.read()
    file2.close()
    for n, seq_id in enumerate(seqs_dict):
        seq = seqs_dict[seq_id]
        # seq_id = "A000002"
        compare = re.findall(seq_id+".+\n", original)
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
print(len(list(seqs_dict)))

