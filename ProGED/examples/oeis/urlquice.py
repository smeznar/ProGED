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

search=request.urlopen('https://oeis.org/search?q=keyword:core&fmt=text')
header_length = 1000  # number of characters in header
header = search.read(header_length).decode()
# print(header)
number_of_sequences = int(re.findall("Search: .+\nShowing \d+-\d+ of (\d+)\n", header)[0])
print(number_of_sequences)
seqs = []
# number_of_sequences = 30  # ... test
for i in range((number_of_sequences-1)//10+1):  # 178 -> 10-20, 20-30, ..., 170-180
    print(i)
    search=request.urlopen(
        f'https://oeis.org/search?q=keyword:core&fmt=text&start={i*10}')
    print(i, "after wget")
    page = search.read().decode()
    print(i, "after decode")
    stu_lines = re.findall(
        r"%S (A\d{6}) (.+)\n%T A\d{6} (.+)\n%U A\d{6} (.+)\n", page)
    ten_seqs = [(seq[0], [int(a_n) for a_n in (seq[1]+seq[2]+seq[3]).split(",")])
        for seq in stu_lines]
    # print(ten_seqs)
    print(len(ten_seqs))
    seqs += ten_seqs
    print(i, "after print ")
    
print(seqs)
print(len(seqs))
ids = [seq[0] for seq in seqs]
print(len(list(set(ids))))
# print(list(set(ids)))

# file = open("saved_core.py", "w")
# file.write("core_list = " + repr(seqs) + "\n")
# file.close()
# import saved_core