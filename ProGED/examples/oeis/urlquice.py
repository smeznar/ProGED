from urllib import request
# import requests
# import json
# import random
# import shutil
import re

from numpy import number
#import os

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
x = np.arange(max(lens))
bigs = list(map(lambda i: sum(lens>=i), x))
# print(bigs)

# import matplotlib.pyplot as plt
# plt.grid(True)
# plt.xlabel("threshold")
# plt.ylabel("number of sequences with more than *threshold* terms")
# plt.plot(x, bigs)
# plt.title("Sequences with enough terms")
# plt.show()
## EOF lengths analysis ##

## -- Check scrap against gzipped oeis database -- ##
seqs = seqs_flat

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
print("end")
