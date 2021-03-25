from urllib import request
import re
def header(id):
    """Download title from HTML of the sequence of given id."""
    
    search = request.urlopen(f'https://oeis.org/search?q=id:{id}&fmt=data')
    # header_length = 1000000  # number of characters in header.
    # header = search.read(header_length).decode()
    header = search.read().decode()
    # prin
    print(header)
    total = re.findall(r'''<a href=\"/A\d{6}\">A\d{6}</a>
                      
                        
                        <td width=5>
                        <td valign=top align=left>
                        ((.+\n)+)[ \t]+<td width=\d+>''', header)
    total = re.findall(r'''<a href=\"/A\d{6}\">A\d{6}</a>''', header)
                        
                        
                        # <td width=5>
                        # <td valign=top align=left>
                        # ((.+\n)+)[ \t]+<td width=\d+>''', header)
    print(total)
    return
header("A000001")

# print(total, len(total))
