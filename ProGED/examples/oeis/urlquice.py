from urllib import request
# import requests
# import json
# import random
# import shutil
import re
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

def fetch(start=0, end=1e6):
    search=request.urlopen('https://oeis.org/search?q=keyword:core&fmt=text')
    header_length = 1000  # number of characters in header
    header = search.read(header_length).decode()
    # print(header)
    number_of_sequences = int(re.findall("Search: .+\nShowing \d+-\d+ of (\d+)\n", header)[0])
    print(number_of_sequences)
    seqs = []
    # number_of_sequences = 30  # ... test
    number_of_sequences = max(number_of_sequences, number_of_sequences-start)
    for i in range((number_of_sequences-1)//10+1):  # 178 -> 10-20, 20-30, ..., 170-180
        # print(i)
        search=request.urlopen(
            f'https://oeis.org/search?q=keyword:core&fmt=text&start={start+i*10}')
        print(i, "after wget")
        page = search.read().decode()
        # print(i, "after decode")
        stu_lines = re.findall(
            r"%S (A\d{6}) (.+)\n%T A\d{6} (.*)\n%U A\d{6} (.*)\n", page)
        ten_seqs = [(seq[0], [int(a_n) for a_n in (seq[1]+seq[2]+seq[3]).split(",")])
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
    file = open("saved_core1.py", "w")
    file.write("core_list = " + repr(seqs_flatten) + "\n"
               + "core_unflatten = " + repr(seqs) + "\n")
    file.close()
    # import saved_core
    return
fetch(130, 140)

import saved_core
seqs_flat = saved_core.core_list
seqs = saved_core.core_unflatten

# print(len(list(set(seqs_flatten))))
# print(seqs)
# print(len(seqs))
print("number of seqs in seqs:", sum(len(tseqs) for tseqs in seqs))
print("unique ids:")
ids = [seq[0] for seq in seqs_flat]
print(len(ids))
print(len(list(set(ids))))
print("konec")

result = 0
for ten_seqs in seqs:
    result += 10
    print(f"in results: {result-10}-{result}")
    print(len(ten_seqs))
for seq_pair in seqs_flat:
    print("number of terms in sequence:", len(seq_pair[1]))
import numpy as np
print(np.mean(np.array([len(seq_pair[1]) for seq_pair in seqs_flat])))
lens = np.array([len(seq_pair[1]) for seq_pair in seqs_flat])
print(lens)
print(sum(lens<10))
for i in range(10):
    print(sum(lens<10*i), 10*i)
print(len(lens))
x = np.arange(max(lens))
bigs = list(map(lambda i: sum(lens>i), x))
print(type(bigs))
print(bigs)

import matplotlib.pyplot as plt
plt.grid(True)
plt.xlabel("threshold")
plt.ylabel("number of sequences with more than *threshold* terms")
plt.plot(x, bigs)
plt.title("Sequences with enough terms")
plt.show()
