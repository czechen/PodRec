#!/usr/bin/env python3

import csv
from nltk.corpus import words
import nltk

'''
Preprocessing script for final_podcasts_en.csv dataset
mainly filters out non-english podcasts, podcasts with no title or no description or no category
'''

#nltk.download('words')
#large corpus of english words
word_set = set(words.words())

def isEnglish(s):
	#check for non english characters in string s
	#then check if more than half of tokens in s aren not included in word_set
	try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
    	tokens = list(s.lower().split())
    	non_english_count = 0
    	token_count = 100
    	for token in tokens[:token_count]:
    		non_english_count += not token in word_set
    	if non_english_count/token_count < 0.5:
        	return True

with open('final_podcasts_en.csv',newline='') as cvs:
	with open('final_podcasts_en_processed.csv', 'w', newline='') as csvfile:
		rows = csv.reader(cvs)
		header = next(rows) #skip header
		file_writer = csv.writer(csvfile, delimiter=',')
		file_writer.writerow(header)
		num_of_rows = 1
		for row in rows:
			category = row[-2].split(',')
			description = row[4].lower()
			title = row[0].lower()
			rss = row[5]
			if category == [''] or description == '' or title == 'none' or not isEnglish(description):
				continue #ignore podcasts without categories or title and with too short description
			else:
				file_writer.writerow(row)
				num_of_rows += 1
print(num_of_rows)