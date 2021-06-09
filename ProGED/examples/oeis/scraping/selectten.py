from urllib import request
import pandas as pd
import re
import time
import urllib.request

HTML_LENGTH = 10**6
# HTML_LENGTH = 10**9
# MUCH_IDS = 200
# MUCH_IDS = 50

 # The data u need
# user_agent = 'Mozilla/5.0 (X11; Linux x86_64; rv:78.0) Gecko/20100101 Firefox/78.0'
user_agent = 'Mozilla/5.0 (X11; Linux x86_64; rv:78.0) Gecko/20000101 Firefox/78.0'
# user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
id_ = "A000001"
# urlg = f"https://www.google.com/search?q=%22{id_}%22"
# urlb = f"https://www.bing.com/search?q=%22{id_}%22"
def get_page(url):
    headers = {'User-Agent':user_agent,}
    request = urllib.request.Request(url, None, headers) #The assembled request
    response = urllib.request.urlopen(request)
    page = response.read().decode()[:HTML_LENGTH]
    return page
# print(get_page(urlg))
# print(get_page(urlb))

def rank_gb(id_, goo=True):
    if goo:
        urlg = f"https://www.google.com/search?q=%22{id_}%22"
    urlb = f"https://www.bing.com/search?q=%22{id_}%22"
    if goo:
        pageg = get_page(urlg)
        resultsg = re.findall(r'''Približno (\d+)\.?(\d+) rez.''', pageg)
        resultsg = int(resultsg[0][0]+resultsg[0][1])
        print("number of results in Google: \n", resultsg)
    pageb = get_page(urlb)
    # print(pageb)
    resultsb = re.findall(r'''(\d+),?(\d+)? rezultati''', pageb)
    # print(resultsb)
    resultsb = int(resultsb[0][0]+resultsb[0][1])
    print("number of results in Bing: \n", resultsb)
    if goo:
        return resultsg, resultsb
    else:
        return  resultsb
# print(rank_gb(id_, goo=False))
# print(rank_gb("A018252", goo=False))
# print(rank_gb(id_))

def search_rank(top=20, goo=True, start=0, end=200):
    # ids = pd.read_csv("oeis_selection.csv").columns[:first_columns]
    ids = pd.read_csv("oeis_selection.csv").columns[start:end]
    print(ids, type(ids[0]))
    ranks = []
    for id_ in ids:
        print(id_)
        if goo:
            rankg, rankb = rank_gb(id_, goo=goo)
            print(f"{id_} has ranks Google:{rankg} Bing:{rankb}.")
        else:
            rankb = rank_gb(id_, goo=goo)
        print(f"{id_} has ranks Bing:{rankb}.")
        if goo:
            ranks += [(id_, rankg, rankb)]
        else:
            ranks += [(id_, rankb)]
        # print(ranks)
    # ranks = {id_: rank_gb(id_) for id_ in ids}
    bindex = 1
    if goo:
        ranksg = sorted(ranks, key=(lambda x: x[1]), reverse=True)
        bindex = 2
    ranksb = sorted(ranks, key=(lambda x: x[bindex]), reverse=True)
    if goo:
        return ranksg[:top], ranksb[:top]
    else:
        return ranksb[:top]
# print(search_rank(20, goo=False))



print(search_rank(20))
# refinement = 50
# [( for i in 200/refinementj
# for i in [(0,50), (50, 100), (100, 150), (150, 200)]:

# for i in [(50, 100), (100, 150), (150, 200)]:
# # for i in [(100, 150)]:
#     print(i)
#     if i[0]==0:
#         continue
#     print(search_rank(20, goo=False, start=i[0], end=i[1]))

# li = {"a": 221, "b": 2, "c": 3}
# li = [("b", 121), ("z", 2), ("c", 3)]
# print(li)
# sorted([(123, 23, 98), (1, 23,54), (32, 12, 53)], key=(lambda x: x[2]))

# print(sorted(li))

# print(time.perf_counter() - start)
