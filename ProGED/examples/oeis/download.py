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
txt3 = """# Greetings from The On-Line Encyclopedia of Integer Sequences! http://oeis.org/

Search: keyword:core
Showing 11-20 of 178

%I A000142 M1675 N0659
%S A000142 1,1,2,6,24,120,720,5040,40320,362880,3628800,39916800,479001600,
%T A000142 6227020800,87178291200,1307674368000,20922789888000,355687428096000,
%U A000142 6402373705728000,121645100408832000,2432902008176640000,51090942171709440000,1124000727777607680000
%N A000142 Factorial numbers: n! = 1*2*3*4*...*n (order of symmetric group S_n, number of permutations of n letters).
%C A000142 The earliest publication that discusses this sequence appears to be the Sepher Yezirah [Book of Creation], circa AD 300. (See Knuth, also the Zeilberger link) - _N. J. A. Sloane_, Apr 07 2014
%C A000142 For n >= 1, a(n) is the number of n X n (0,1) matrices with each row and column containing exactly one entry equal to 1.
%C A000142 This sequence is the BinomialMean transform of A000354. (See A075271 for definition.) - _John W. Layman_, Sep 12 2002 [This is easily verified from the Paul Barry formula for A000354, by interchanging summations and using the formula: Sum_k (-1)^k C(n-i, k) = KroneckerDelta(i,n). - _David Callan_, Aug 31 2003]
%C A000142 Number of distinct subsets of T(n-1) elements with 1 element A, 2 elements B,..., n - 1 elements X (e.g., at n = 5, we consider the distinct subsets of ABBCCCDDDD and there are 5! = 120). - _Jon Perry_, Jun 12 2003
%C A000142 n! is the smallest number with that prime signature. E.g., 720 = 2^4 * 3^2 * 5. - _Amarnath Murthy_, Jul 01 2003
%C A000142 a(n) is the permanent of the n X n matrix M with M(i, j) = 1. - _Philippe Deléham_, Dec 15 2003
%C A000142 Given n objects of distinct sizes (e.g., areas, volumes) such that each object is sufficiently large to simultaneously contain all previous objects, then n! is the total number of essentially different arrangements using all n objects. Arbitrary levels of nesting of objects are permitted within arrangements. (This application of the sequence was inspired by considering leftover moving boxes.) If the restriction exists that each object is able or permitted to contain at most one smaller (but possibly nested) object at a time, the resulting sequence begins 1,2,5,15,52 (Bell Numbers?). Sets of nested wooden boxes or traditional nested Russian dolls come to mind here. - _Rick L. Shepherd_, Jan 14 2004
%C A000142 From _Michael Somos_, Mar 04 2004; edited by _M. F. Hasler_, Jan 02 2015: (Start)
%C A000142 Stirling transform of [2, 2, 6, 24, 120, ...] is A052856 = [2, 2, 4, 14, 76, ...].
%C A000142 Stirling transform of [1, 2, 6, 24, 120, ...] is A000670 = [1, 3, 13, 75, ...].
%C A000142 Stirling transform of [0, 2, 6, 24, 120, ...] is A052875 = [0, 2, 12, 74, ...].
%C A000142 Stirling transform of [1, 1, 2, 6, 24, 120, ...] is A000629 = [1, 2, 6, 26, ...].
%C A000142 Stirling transform of [0, 1, 2, 6, 24, 120, ...] is A002050 = [0, 1, 5, 25, 140, ...].
%C A000142 Stirling transform of (A165326*A089064)(1...) = [1, 0, 1, -1, 8, -26, 194, ...] is [1, 1, 2, 6, 24, 120, ...] (this sequence). (End)
%C A000142 First Eulerian transform of 1, 1, 1, 1, 1, 1... The first Eulerian transform transforms a sequence s to a sequence t by the formula t(n) = Sum_{k=0..n} e(n, k)s(k), where e(n, k) is a first-order Eulerian number [A008292]. - _Ross La Haye_, Feb 13 2005
%C A000142 Conjecturally, 1, 6, and 120 are the only numbers which are both triangular and factorial. - Christopher M. Tomaszewski (cmt1288(AT)comcast.net), Mar 30 2005
%C A000142 n! is the n-th finite difference of consecutive n-th powers. E.g., for n = 3, [0, 1, 8, 27, 64, ...] -> [1, 7, 19, 37, ...] -> [6, 12, 18, ...] -> [6, 6, ...]. - Bryan Jacobs (bryanjj(AT)gmail.com), Mar 31 2005
%C A000142 a(n+1) = (n+1)! = 1, 2, 6, ... has e.g.f. 1/(1-x)^2. - _Paul Barry_, Apr 22 2005
%C A000142 Write numbers 1 to n on a circle. Then a(n) = sum of the products of all n - 2 adjacent numbers. E.g., a(5) = 1*2*3 + 2*3*4 + 3*4*5 + 4*5*1 +5*1*2 = 120. - _Amarnath Murthy_, Jul 10 2005
%C A000142 The number of chains of maximal length in the power set of {1, 2, ..., n} ordered by the subset relation. - _Rick L. Shepherd_, Feb 05 2006
%C A000142 The number of circular permutations of n letters for n >= 0 is 1, 1, 1, 2, 6, 24, 120, 720, 5040, 40320, ... - Xavier Noria (fxn(AT)hashref.com), Jun 04 2006
%C A000142 a(n) is the number of deco polyominoes of height n (n >= 1; see definitions in the Barcucci et al. references). - _Emeric Deutsch_, Aug 07 2006
%C A000142 a(n) is the number of partition tableaux of size n. See Steingrimsson/Williams link for the definition. - _David Callan_, Oct 06 2006
%C A000142 Consider the n! permutations of the integer sequence [n] = 1, 2, ..., n. The i-th permutation consists of ncycle(i) permutation cycles. Then, if the Sum_{i=1..n!} 2^ncycle(i) runs from 1 to n!, we have Sum_{i=1..n!} 2^ncycle(i) = (n+1)!. E.g., for n = 3 we have ncycle(1) = 3, ncycle(2) = 2, ncycle(3) = 1, ncycle(4) = 2, ncycle(5) = 1, ncycle(6) = 2 and 2^3 + 2^2 + 2^1 + 2^2 + 2^1 + 2^2 = 8 + 4 + 2 + 4 + 2 + 4 = 24 = (n+1)!. - _Thomas Wieder_, Oct 11 2006
%C A000142 a(n) is the number of set partitions of {1, 2, ..., 2n - 1, 2n} into blocks of size 2 (perfect matchings) in which each block consists of one even and one odd integer. For example, a(3) = 6 counts 12-34-56, 12-36-45, 14-23-56, 14-25-36, 16-23-45, 16-25-34. - _David Callan_, Mar 30 2007
%C A000142 Consider the multiset M = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, ...] = [1, 2, 2, ..., n x 'n'] and form the set U (where U is a set in the strict sense) of all subsets N (where N may be a multiset again) of M. Then the number of elements |U| of U is equal to (n+1)!. E.g. for M = [1, 2, 2] we get U = [[], [2], [2, 2], [1], [1, 2], [1, 2, 2]] and |U| = 3! = 6. This observation is a more formal version of the comment given already by _Rick L. Shepherd_, Jan 14 2004. - _Thomas Wieder_, Nov 27 2007
%C A000142 For n >= 1, a(n) = 1, 2, 6, 24, ... are the positions corresponding to the 1's in decimal expansion of Liouville's constant (A012245). - _Paul Muljadi_, Apr 15 2008
%C A000142 Triangle A144107 has n! for row sums (given n > 0) with right border n! and left border A003319, the INVERTi transform of (1, 2, 6, 24, ...). - _Gary W. Adamson_, Sep 11 2008
%C A000142 Equals INVERT transform of A052186: (1, 0, 1, 3, 14, 77, ...) and row sums of triangle A144108. - _Gary W. Adamson_, Sep 11 2008
%C A000142 From _Abdullahi Umar_, Oct 12 2008: (Start)
%C A000142 a(n) is also the number of order-decreasing full transformations (of an n-chain).
%C A000142 a(n-1) is also the number of nilpotent order-decreasing full transformations (of an n-chain). (End)
%C A000142 n! is also the number of optimal broadcast schemes in the complete graph K_{n}, equivalent to the number of binomial trees embedded in K_{n} (see Calin D. Morosan, Information Processing Letters, 100 (2006), 188-193). - Calin D. Morosan (cd_moros(AT)alumni.concordia.ca), Nov 28 2008
%C A000142 Sum_{n >= 0} 1/a(n) = e. - _Jaume Oliver Lafont_, Mar 03 2009
%C A000142 Let S_{n} denote the n-star graph. The S_{n} structure consists of n S_{n-1} structures. This sequence gives the number of edges between the vertices of any two specified S_{n+1} structures in S_{n+2} (n >= 1). - _K.V.Iyer_, Mar 18 2009
%C A000142 Chromatic invariant of the sun graph S_{n-2}.
%C A000142 It appears that a(n+1) is the inverse binomial transform of A000255. - Timothy Hopper (timothyhopper(AT)hotmail.co.uk), Aug 20 2009
%C A000142 a(n) is also the determinant of an square matrix, An, whose coefficients are the reciprocals of beta function: a{i, j} = 1/beta(i, j), det(An) = n!. - _Enrique Pérez Herrero_, Sep 21 2009
%C A000142 The asymptotic expansions of the exponential integrals E(x, m = 1, n = 1) ~ exp(-x)/x*(1 - 1/x + 2/x^2 - 6/x^3 + 24/x^4 + ...) and E(x, m = 1, n = 2) ~ exp(-x)/x*(1 - 2/x + 6/x^2 - 24/x^3 + ...) lead to the factorial numbers. See A163931 and A130534 for more information. - _Johannes W. Meijer_, Oct 20 2009
%C A000142 Satisfies A(x)/A(x^2), A(x) = A173280. - _Gary W. Adamson_, Feb 14 2010
%C A000142 a(n) = A173333(n,1). - _Reinhard Zumkeller_, Feb 19 2010
%C A000142 a(n) = G^n where G is the geometric mean of the first n positive integers. - _Jaroslav Krizek_, May 28 2010
%C A000142 Increasing colored 1-2 trees with choice of two colors for the rightmost branch of nonleaves. - _Wenjin Woan_, May 23 2011
%C A000142 Number of necklaces with n labeled beads of 1 color. - _Robert G. Wilson v_, Sep 22 2011
%C A000142 The sequence 1!, (2!)!, ((3!)!)!, (((4!)!)!)!, ..., ((...(n!)!)...)! (n times) grows too rapidly to have its own entry. See Hofstadter.
%C A000142 The e.g.f. of 1/a(n) = 1/n! is BesselI(0, 2*sqrt(x)). See Abramowitz-Stegun, p. 375, 9.3.10. - _Wolfdieter Lang_, Jan 09 2012
%C A000142 a(n) is the length of the n-th row which is the sum of n-th row in triangle A170942. - _Reinhard Zumkeller_, Mar 29 2012
%C A000142 Number of permutations of elements 1, 2, ..., n + 1 with a fixed element belonging to a cycle of length r does not depend on r and equals a(n). - _Vladimir Shevelev_, May 12 2012
%C A000142 a(n) is the number of fixed points in all permutations of 1, ..., n: in all n! permutations, 1 is first exactly (n-1)! times, 2 is second exactly (n-1)! times, etc., giving (n-1)!*n = n!. - _Jon Perry_, Dec 20 2012
%C A000142 For n >= 1, a(n-1) is the binomial transform of A000757. See Moreno-Rivera. - _Luis Manuel Rivera Martínez_, Dec 09 2013
%C A000142 Each term is divisible by its digital root (A010888). - _Ivan N. Ianakiev_, Apr 14 2014
%C A000142 For m >= 3, a(m-2) is the number hp(m) of acyclic Hamiltonian paths in a simple graph with m vertices, which is complete except for one missing edge. For m < 3, hp(m)=0. - _Stanislav Sykora_, Jun 17 2014
%C A000142 a(n) = A245334(n,n). - _Reinhard Zumkeller_, Aug 31 2014
%C A000142 a(n) is the number of increasing forests with n nodes. - _Brad R. Jones_, Dec 01 2014
%C A000142 Sum_{n>=0} a(n)/(a(n + 1)*a(n + 2)) = Sum_{n>=0} 1/((n + 2)*(n + 1)^2*a(n)) = 2 - exp(1) - gamma + Ei(1) = 0.5996203229953..., where gamma = A001620, Ei(1) = A091725. - _Ilya Gutkovskiy_, Nov 01 2016
%C A000142 The factorial numbers can be calculated by means of the recurrence n! = (floor(n/2)!)^2 * sf(n) where sf(n) are the swinging factorials A056040. This leads to an efficient algorithm if sf(n) is computed via prime factorization. For an exposition of this algorithm see the link below. - _Peter Luschny_, Nov 05 2016
%C A000142 Treeshelves are ordered (plane) binary (0-1-2) increasing trees where the nodes of outdegree 1 come in 2 colors. There are n! treeshelves of size n, and classical Françon's bijection maps bijectively treeshelves into permutations. - _Sergey Kirgizov_, Dec 26 2016
%C A000142 Satisfies Benford's law [Diaconis, 1977; Berger-Hill, 2017] - _N. J. A. Sloane_, Feb 07 2017
%C A000142 a(n) = Sum((d_p)^2), where d_p is the number of standard tableaux in the Ferrers board of the integer partition p and summation is over all integer partitions p of n. Example: a(3) = 6. Indeed, the  partitions of 3 are [3], [2,1], and [1,1,1], having 1, 2, and 1 standard tableaux, respectively; we have 1^2 + 2^2 + 1^2 = 6. - _Emeric Deutsch_, Aug 07 2017
%C A000142 a(n) is the n-th derivative of x^n. - _Iain Fox_, Nov 19 2017
%C A000142 a(n) is the number of maximum chains in the n-dimensional Boolean cube {0,1}^n in respect to the relation "precedes". It is defined as follows: for arbitrary vectors u, v of {0,1}^n, such that u=(u_1, u_2, ..., u_n) and v=(v_1, v_2, ..., v_n), "u precedes v" if u_i <= v_i, for i=1, 2, ..., n. - _Valentin Bakoev_, Nov 20 2017
%C A000142 a(n) is the number of all shortest paths (for example, obtained by Breadth First Search) between the nodes (0,0,...,0) (i.e., the all-zero vector) and (1,1,...,1) (i.e., the all-ones vector) in the graph H_n, corresponding to the n-dimensional Boolean cube {0,1}^n. The graph is defined as H_n= (V_n, E_n), where V_n is the set of all vectors of {0,1}^n, and E_n contains edges formed by each pair adjacent vectors. - _Valentin Bakoev_, Nov 20 2017
%C A000142 a(n) is also the determinant of the symmetric n X n matrix M defined by M(i,j) = sigma(gcd(i,j)) for 1 <= i,j <= n. - _Bernard Schott_, Dec 05 2018
%C A000142 a(n) is also the number of inversion sequences of length n. A length n inversion sequence e_1, e_2, ..., e_n is a sequence of n integers such that 0 <= e_i < i. - _Juan S. Auli_, Oct 14 2019
%C A000142 Sum_{n >= 0} 1/(a(n)*(n+2)) = 1. - Multiplying the denominator by (n+2) in Jaume Oliver Lafont's entry above creates a telescoping sum. - _Fred Daniel Kline_, Nov 08 2020
%D A000142 M. Abramowitz and I. A. Stegun, eds., Handbook of Mathematical Functions, National Bureau of Standards Applied Math. Series 55, 1964 (and various reprintings), p. 833.
%D A000142 A. T. Benjamin and J. J. Quinn, Proofs that really count: the art of combinatorial proof, M.A.A. 2003, id. 125; also p. 90, ex. 3.
%D A000142 Diaconis, Persi, The distribution of leading digits and uniform distribution mod 1, Ann. Probability, 5, 1977, 72--81,
%D A000142 Douglas R. Hofstadter, Fluid concepts & creative analogies: computer models of the fundamental mechanisms of thought, Basic Books, 1995, pages 44-46.
%D A000142 A. N. Khovanskii. The Application of Continued Fractions and Their Generalizations to Problem in Approximation Theory. Groningen: Noordhoff, Netherlands, 1963. See p.141 (10.19)
%D A000142 D. E. Knuth, The Art of Computer Programming, Vol. 3, Section 5.1.2, p. 623. [From _N. J. A. Sloane_, Apr 07 2014]
%D A000142 J.-M. De Koninck & A. Mercier, 1001 Problèmes en Théorie Classique des Nombres, Problème 693 pp. 90, 297, Ellipses Paris 2004.
%D A000142 A. P. Prudnikov, Yu. A. Brychkov and O. I. Marichev, "Integrals and Series", Volume 1: "Elementary Functions", Chapter 4: "Finite Sums", New York, Gordon and Breach Science Publishers, 1986-1992.
%D A000142 R. W. Robinson, Counting arrangements of bishops, pp. 198-214 of Combinatorial Mathematics IV (Adelaide 1975), Lect. Notes Math., 560 (1976).
%D A000142 Sepher Yezirah [Book of Creation], circa AD 300. See verse 52.
%D A000142 N. J. A. Sloane, A Handbook of Integer Sequences, Academic Press, 1973 (includes this sequence).
%D A000142 N. J. A. Sloane and Simon Plouffe, The Encyclopedia of Integer Sequences, Academic Press, 1995 (includes this sequence).
%D A000142 D. Stanton and D. White, Constructive Combinatorics, Springer, 1986; see p. 91.
%D A000142 Carlo Suares, Sepher Yetsira, Shambhala Publications, 1976. See verse 52.
%D A000142 D. Wells, The Penguin Dictionary of Curious and Interesting Numbers, pp. 102 Penguin Books 1987.
%H A000142 N. J. A. Sloane, <a href="/A000142/b000142.txt">The first 100 factorials: Table of n, n! for n = 0..100</a>
%H A000142 M. Abramowitz and I. A. Stegun, eds., <a href="http://www.convertit.com/Go/ConvertIt/Reference/AMS55.ASP">Handbook of Mathematical Functions</a>, National Bureau of Standards, Applied Math. Series 55, Tenth Printing, 1972 [alternative scanned copy].
%H A000142 S. B. Akers and B. Krishnamurthy, <a href="http://dx.doi.org/10.1109/12.21148">A group-theoretic model for symmetric interconnection networks</a>, IEEE Trans. Comput., 38(4), April 1989, 555-566.
%H A000142 Masanori Ando, <a href="https://arxiv.org/abs/1504.04121">Odd number and Trapezoidal number</a>, arXiv:1504.04121 [math.CO], 2015.
%H A000142 David Applegate and N. J. A. Sloane, <a href="/A000142/a000142.txt.gz">Table giving cycle index of S_0 through S_40 in Maple format</a> [gzipped]
%H A000142 C. Banderier, M. Bousquet-Mélou, A. Denise, P. Flajolet, D. Gardy and D. Gouyou-Beauchamps, <a href="https://doi.org/10.1016/S0012-365X(01)00250-3">Generating Functions for Generating Trees</a>, Discrete Mathematics 246(1-3), March 2002, pp. 29-55.
%H A000142 Stefano Barbero, Umberto Cerruti, Nadir Murru, <a href="https://doi.org/10.1007/s11587-018-0389-5">On the operations of sequences in rings and binomial type sequences</a>, Ricerche di Matematica (2018), pp 1-17., also <a href="https://arxiv.org/abs/1805.11922">arXiv:1805.11922</a> [math.NT], 2018.
%H A000142 E. Barcucci, A. Del Lungo and R. Pinzani, <a href="https://doi.org/10.1016/0304-3975(95)00199-9">"Deco" polyominoes, permutations and random generation</a>, Theoretical Computer Science, 159, 1996, 29-42.
%H A000142 E. Barcucci, A. Del Lungo, R. Pinzani and R. Sprugnoli, <a href="http://www.emis.de/journals/SLC/opapers/s31barc.html">La hauteur des polyominos dirigés verticalement convexes</a>, Actes du 31e Séminaire Lotharingien de Combinatoire, Publ. IRMA, Université Strasbourg I (1993).
%H A000142 Jean-Luc Baril, Sergey Kirgizov, Vincent Vajnovszki, <a href="https://doi.org/10.1016/j.disc.2017.07.021">Patterns in treeshelves</a>, Discrete Mathematics, Vol. 340, No. 12 (2017), 2946-2954, arXiv:<a href="https://arxiv.org/abs/1611.07793">1611.07793 [cs.DM]</a>, 2016.
%H A000142 A. Berger and T. P. Hill, <a href="http://www.ams.org/publications/journals/notices/201702/rnoti-p132.pdf">What is Benford's Law?</a>, Notices, Amer. Math. Soc., 64:2 (2017), 132-134.
%H A000142 M. Bhargava, <a href="http://dx.doi.org/10.2307/2695734">The factorial function and generalizations</a>, Amer. Math. Monthly, 107 (Nov. 2000), 783-799.
%H A000142 Natasha Blitvić, Einar Steingrímsson, <a href="https://arxiv.org/abs/2001.00280">Permutations, moments, measures</a>, arXiv:2001.00280 [math.CO], 2020.
"""

from urllib import request
# import matplotlib.pyplot as plt
# from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import re
# from numpy import number


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

## -- old, i.e. stu lines -- ##
page = txt2
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

## -- new (NOT NEEDED, my bad), i.e. links %H line -- ##
# <a href="/A000142/b000142.txt">
# page = txt3
# h_line = re.findall( r"%H.+<a href=(\"/A\d{6}/b\d{6}.txt\")>", page)
# print(h_line, " \n... = h_line[0]")

# new not needed function (for archive right now):
# def fetch_with_bfiles(start=0, end=1e10, do_write=True):
#     search=request.urlopen('https://oeis.org/search?q=keyword:core%20keyword:nice&fmt=text')
#     header_length = 1000  # number of characters in header
#     header = search.read(header_length).decode()
#     # print(header)
#     total = int(re.findall("Search: .+\nShowing \d+-\d+ of (\d+)\n", header)[0])
#     print(total)
#     seqs = []
#     # number_of_sequences = 30  # ... test
#     cropped = min(total, end)
#     beginning = max(0, cropped-start)
#     number_of_sequences = beginning
#     print(f"total number of sequences found: {total}\n"
#         f"start: {start}, end: {end}, "
#         f"sequences of intersection:{number_of_sequences}")

#     for i in range((number_of_sequences-1)//10+1):  # 178 -> 10-20, 20-30, ..., 170-180
#         # print(i)
#         search = request.urlopen(
#             f'https://oeis.org/search?q=keyword:core%20keyword:nice&fmt=text&start={start+i*10}')
#         print(i, "after wget")
#         page = search.read().decode()
#         hlines = re.findall( r"%H.+<a href=\"(/A\d{6}/b\d{6}.txt)\">", page)
#         print(hlines, "\n ... = h_line")
#         ten_seqs = hlines
#         if len(ten_seqs)<10: print("\n\n\n less than 10 !!!!\n\n\n")
#         print(len(ten_seqs))
#         # seqs += [ten_seqs]
#         print(i, "after regex ")
#     return
# # fetch_with_bfiles(130, 140, do_write=False)
# # fetch_with_bfiles(do_write=False)
# # print("end of fetch")

## -- download b-files via saved sequences -- ##
import saved_core
seqs = saved_core.core_list
seqs_unflatten = saved_core.core_unflatten
dict_seqs = dict(seqs)
print(seqs[0])

def fetch_bfiles(start=0, end=180, do_write=True, max_seq_length=100):
    """Download extended sequences from bfiles of all sequences."""

    counter = start
    new_seqs ={}
    for id in list(dict_seqs)[start:end]:
        baddress = id + '/b' + id[1:] +'.txt'
        print(baddress)
        bfile = request.urlopen('https://oeis.org/'+baddress).read().decode()
        # new_seq = re.findall("(\d+) (\d+)\n", bfile)
        new_seq_check = re.findall(r"(\d+)([ \t]+)(-?\d+)\n", bfile)  # with a ^ 
        first_findall = new_seq_check[0]
        # print("first findall:", first_findall)
        first_space = first_findall[1]
        if len(first_space) > 1:
            print("\"" + first_space + '\"', len(first_space))
        first_index = first_findall[0]
        legal = ('-1', '0', '1')
        if first_index in legal:
            # print("all is well, really")
            pass
        else:
            print(f"first index is not in {legal}:", first_index)
        new_seq = [triple[2] for triple in new_seq_check[:max_seq_length]]
        # print(new_seq)
        new_seqs[id] = [int(term) for term in new_seq]
        # print(new_seqs[id])
        # print("len of new seq", len(new_seqs[id]))
        counter += 1
        if (counter%10) == 0: print(counter)

    print("all bfiled seqs:", len(new_seqs), "all original seqs:", len(dict_seqs))
    if do_write:
        file = open("saved_bfiled.py", "w")
        file.write("core_bfiled_dict = " + repr(new_seqs) + "\n")
        file.close()
        print("Sequences \"as are\" are written into the output file.")
    print("fetch of b-files ended")
    return
# fetch_bfiles(130, 140, do_write=False)
# fetch_bfiles(do_write=False)
# fetch_bfiles(80, 180, do_write=False)
# fetch_bfiles(do_write=False)
fetch_bfiles(80, 90)
# fetch_bfiles()


bfile = """0	1
1	1
2	3
3	6
4	13
5	24
6	48
7	86
8	160
9	282
10	500
11	859
12	1479
13	2485
14	4167
15	6879
16	11297
17	18334
18	29601
19	47330
20	75278
21	118794
"""

# new_seq_check = re.findall(r"(\d+)([ \t]+)(-?\d+)\n", bfile)  # with a ^ 
# print(new_seq_check)
