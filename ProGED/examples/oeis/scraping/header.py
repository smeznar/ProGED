from urllib import request
import pandas as pd
import re
import time

start = time.perf_counter()
def header(id_):
    """Download title from HTML of the sequence of given id."""
    
    search = request.urlopen(f"https://oeis.org/search?q=id%3a{id_}&fmt=data")
    header = search.read().decode()
    # print(header)
    # return header
# header = header("A000001")

# total = re.findall(r'''<a href=\"/A\d{6}\">A\d{6}</a>''', header)
    total = re.findall(r'''<a href=\"/A\d{6}\">A\d{6}</a>
                    
                    
                    <td width=5>
                    <td valign=top align=left>
                    ((.+\n)+)[ \t]+<td width=\d+>''', header)
    title = total[0][0]
    print(time.perf_counter(), id_, title)
    return title
# print(header("A000001"))


titles_filename = "titles.csv"
def download_headers(titles_filename=titles_filename):
    ids = pd.read_csv("oeis_selection.csv").columns
    titles = {id_: [header(id_)] for id_ in ids}
    for id_, title in titles.items():
        print(id_, title)

    df = pd.DataFrame(titles)
    df.to_csv(titles_filename, index=False)
    return
# download_headers(titles_filename)
print(time.perf_counter() - start)
# post download:
titles = pd.read_csv(titles_filename, index_col=0)
print(titles.shape)
print(titles)
for id_, title in titles.items():
    print(id_, len(title[0]))
print(min([len(title[0]) for _, title in titles.items()]))
