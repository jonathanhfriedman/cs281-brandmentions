import praw
import re

def process(text):
    """
    Remove non-ascii characters, commas, and newlines.
    """
    text = ''.join(i for i in text if ord(i)<128)
    text = re.sub(r'[^\w]', '', text)
    return text.replace('\n', ' ').strip()

def scrape(outfile='mfadata.txt'):
	"""
	Scrape Reddit data and write it to outfile.
	"""
	r = praw.Reddit(user_agent='/r/malefashionadvice scraping script')
	subs_all = r.get_subreddit('malefashionadvice').get_top_from_all(limit=1000)
	subs_year = r.get_subreddit('malefashionadvice').get_top_from_year(limit=1000)
	seen = []
	for s in subs_all:
	    if s.name not in seen:
	        seen.append(s.name)
	        s.replace_more_comments(limit=None, threshold=5)
	        with open (outfile, 'w') as outfile:
	            outfile.write(process(s.title)+',')
	            for c in s.comments:
	                outfile.write(process(c.body)+',')
	            outfile.write('\n')
	        
    for s in subs_year:
        if s.name not in seen:
            seen.append(s.name)
            s.replace_more_comments(limit=None, threshold=5)
            outfile.write(process(s.title)+',')
            for c in s.comments:
                outfile.write(process(c.body)+',')
            outfile.write('\n')

