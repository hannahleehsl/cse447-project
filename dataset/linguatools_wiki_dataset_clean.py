# -*- coding: utf-8 -*-
'''
Created on Jan 25, 2026

@author: juneyang2005@gmail.com (@3la5t1c81rdy)

A utility python file to extract the necessary contents from the linguatools Wikipedia monolingual corpora
    .XML file

If run, proceeds with "produce a sample of the Wikipedia"

Otherwise provides functions to clean the resulting .XML dataset, such as:
    - raw text extraction
    - element removal
    - link reduction
    ... (WIP)
'''

####
READLEN = 1048576
HEAD = "<wikipedia" #root element "wikipedia"
TAIL = "</wikipedia>"
####

import pickle
import random
import sys
import os
import html
import gc

def collect_article_titles(path: os.path) -> list[str]:
    #takes a .XML file and exhaustively collects the titles of every <article> elements
    #    (their name attribute)
    
    T = []
    
    with open(path, "r", encoding="utf-8") as f:
        curr = f.read(READLEN)
        
        while True:
            if len(T) % 1000 == 0: 
                print("Processed articles: " + str(len(T)))
            
            while ("<article name=\"" not in curr) and ("</wikipedia>" not in curr):
                i = len(curr)
                curr += f.read(READLEN)
                assert i < len(curr)
            
            if "<article name=\"" not in curr:
                break #done; stopped curr extension because of (and only because of) </wikipedia>
                
            
            x = curr.find("<article name=\"") + len("<article name=\"")
            
            while curr.find("\">\n", x) == -1:
                i = len(curr)
                curr += f.read(READLEN)
                assert i < len(curr)
            
            cname = curr[x : curr.find("\">\n", x)]
            x = curr.find("\">\n", x) + 2 #include newline
        
            while curr.find("</article>\n", x) == -1:
                i = len(curr)
                curr += f.read(READLEN)
                assert i < len(curr)
            
            T.append(cname)
            
            curr = curr[curr.find("</article>\n", x) + 10:] #push curr forward
    
    print("Processed articles: " + str(len(T)))
    return T

def collect_article_with_titles(path: os.path, titles:set[str] = None) -> (dict[str:str], str):
    #takes a .XML file and exhaustively collects <article> elements, forming a dictionary
    #with keys their names.
    ###Only articles with its "name" in the <titles> set will be collected, unless titles is None.
    ###In this case, every article will be collected.
    #As per linguatools' specs, each article name is unique ... but in rare cases this seems to not hold.
        #(maybe update in page right as the dump was created?)
    
    #returns the dictionary containing (K=article name, V=raw text content) pairs,
    #as well as the language of the Wikipedia .XML file originates from.
    
    D = {}
    lang = None
    with open(path, "r", encoding="utf-8") as f:
        curr = f.read(READLEN)
        
        while "<wikipedia lang=\"" not in curr:
            i = len(curr)
            curr += f.read(READLEN)
            assert i < len(curr)
        x = curr.find("<wikipedia lang=\"") + len("<wikipedia lang=\"")
        lang = curr[x:curr.find("\"", x)]
        
        act_ct = 0
        while len(D) < len(titles):
            if act_ct % 1000 == 0: 
                print("Processed articles: " + str(act_ct) + " seen, " + str(len(D)) + " added")
                gc.collect()
            
            while ("<article name=\"" not in curr) and ("</wikipedia>" not in curr):
                i = len(curr)
                curr += f.read(READLEN)
                assert i < len(curr)
            
            if "<article name=\"" not in curr:
                break #done; stopped curr extension because of (and only because of) </wikipedia>
                
            
            x = curr.find("<article name=\"") + len("<article name=\"")
            
            while curr.find("\">\n", x) == -1:
                i = len(curr)
                curr += f.read(READLEN)
                assert i < len(curr)
            
            cname = curr[x : curr.find("\">\n", x)]
            x = curr.find("\">\n", x) + 2 #include newline
            
            while curr.find("</article>\n", x) == -1:
                i = len(curr)
                curr += f.read(READLEN)
                assert i < len(curr)
            
            
            if titles is None or cname in titles:
                #only add to D if cname is in titles or titles is unspecified
                D[cname] = curr[x:curr.find("</article>\n", x)]
            
            curr = curr[curr.find("</article>\n", x) + 10:] #push curr forward
            
            act_ct += 1
    print("Processed articles: " + str(act_ct) + " seen, " + str(len(D)) + " (out of " + str(len(titles)) + ") added")
    return (D, lang)

def write_articles(articles_dict: dict, lang: str, path: os.path) -> bool:
    #attempt to write, to <path>, the sampled/abridged .XML (in the same format as linguatools wiki dump)
    #... that only contain articles contained in <articles_dict>.
    # supply the wikipedia language with <lang> parameter.
    
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"<wikipedia lang=\"{lang}\">\n")
            for (i, k) in enumerate(articles_dict.keys()):
                if i % 1000 == 0:
                    print(f"Wrote {i} / {len(articles_dict)} articles")
                f.write(f"<article name=\"{k}\">")
                f.write(articles_dict[k])
                f.write(f"</article>\n")
            f.write(f"</wikipedia>\n")
            print(f"Wrote {len(articles_dict)} / {len(articles_dict)} articles")
    except:
        return False
    return True


#deprecated/unused.
def _sample_from_dict(articles_dict:dict, count:int) -> dict:
    #sample <count> sub-dictionary from <articles_dict>, without replacement
    #returned dictionary will contain exactly <count> (K,V) pairs
    
    keys = random.sample(list(articles_dict.keys()), k=count)
    
    D_prime = {}
    for k in keys:
        D_prime[k] = D[k]
    
    return D_prime

if __name__ == "__main__":
    # Requires two argument variables, input_path and output_path
    # reads the .XML file pointed by the input_path, cleans it, then outputs .XML containing exactly <article_count> number
    # of "article" elements
    # if -inspect is passed instead of output path, enters article inspection mode
    # otherwise, the result is stored as a .XML file in <output_path>
    
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <input_path> <article_count> <output_path | \"-inspect\">")
        print("Input should be a .XML file as downloaded (and extracted) from https://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/")
        quit()
    _, input_path, article_count, output_path = sys.argv
    
    if not (os.path.isfile(input_path) or len(input_path) < 4 or input_path[-4:].lower() != ".xml"):
        print(f"Usage: {sys.argv[0]} <input_path> <article_count> <output_path>")
        print("Input should be a *.XML* file as downloaded (and extracted) from https://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/")
        print(f"'{input_path}' is not a valid .XML file.")
        quit()
    try:
        article_count = int(article_count)
    except:
        print(f"Usage: {sys.argv[0]} <input_path> <article_count> <output_path>")
        print(f"'{article_count}' is not a valid article count.")
        quit()
    
    
    T = collect_article_titles(input_path)
    
    print(f"... collected {len(T)} article titles")
    
    if len(T) < article_count:
        print(f"Only found {len(T)} articles when {article_count} is required.")
        quit()
    
    K = list(set(T)) #ensure uniqueness
    
    T = set(random.sample(K, article_count)) #sample <article_count> titles -> set
    
    (D, lang) = collect_article_with_titles(input_path, T)
    if article_count > 1000:
        print("... sampled " + str(article_count) + " articles.")
    else:
        print("... sampled " + str(article_count) + " articles: " + str(D.keys()))
    
    while output_path == "-inspect" or output_path == "-I":
        k = input("article title: ")
        print(html.unescape(D[k]))
    
    if write_articles(D, lang, output_path):
        print(f"Successfully wrote a {len(D)} Wikipedia sample to '{output_path}'")
    else:
        print(f"Failed to write to '{output_path}'.")