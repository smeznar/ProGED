from save_new_bfiles import bseqs
import re

# basic info, preview:
seqs = bseqs
print(len(seqs), type(seqs), [len(seq) for _, seq in seqs.items()],
    [seq for _, seq in seqs.items()][0])  # preview
print(len(set(seqs)))


## -- Check scrap against gzipped oeis database -- ##
def check_scrap(seqs_dict=dict(seqs), is_bfile=False, max_seq_len=100):
    # seqs = seqs_flat
    file2 = open("stripped_oeis_database.txt", "r")
    original = file2.read()
    file2.close()
    counter = 1
    for id, seq in seqs_dict.items():
        # seq = seqs_dict[seq_id]
        # seq_id = "A000002"
        compare = re.findall(id+".+\n", original)
        str_seq = id + " ,"
        for term in seq:
            str_seq += str(term)+","
        str_seq += "\n"
        str_seq = [str_seq]
        orig, scraped = compare[0], str_seq[0]
        # print(orig, "this is what I found in original database. Compare with scraped:")
        # print(scraped)
        if not is_bfile:
            print("I can see, you are not using b-files. Those are more longer.")
            bool = orig == scraped
            assert orig == scraped
            print(counter, "Correct scrapping!!! found it" if bool else "scrapping FAILED!!!")
            assert len(orig) == len(scraped)
            print(len(orig), len(scraped))
        else:
            # print("Checking b-files! Let's see.")
            length = min(len(orig) - 2, max_seq_len)  # Avoid \n.
            orig_cut = orig[:length]
            scraped_cut = scraped[:length]
            assert len(orig_cut) == len(scraped_cut)
            assert orig_cut == scraped_cut
            bool = orig_cut == scraped_cut
            print(counter)
            # print(counter, "Correct scrapping!!! found it" if bool else "scrapping FAILED!!!")
            if not bool:
                print("scrapping FAILED!!!")
                print("orig:", orig_cut)
                print("scra:", scraped_cut)
                # print("orig:", orig, end="")
                # print("scra:", scraped, end="")
        counter += 1
    print("Seems scrapping is correct. Checking is done.")
    return
# check_scrap()
# check_scrap(bseqs, is_bfile=True)
check_scrap(bseqs, is_bfile=True, max_seq_len=200)  # returns error (max len set to 100)
print("end")