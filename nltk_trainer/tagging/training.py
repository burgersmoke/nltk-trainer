from nltk.tag import brill
from nltk.tag.brill_trainer import BrillTaggerTrainer
from nltk.tag.brill import Word, Pos

from nltk.tbl import Feature, Template

def train_brill_tagger(initial_tagger, train_sents, end, trace=0, **kwargs):
	bounds = [(1, end)]
	
	# call this to fetch templates directly
	# NOTE : This is the comment from the method below:
	#### Return 37 templates taken from the postagging task of the
	#### fntbl distribution http://www.cs.jhu.edu/~rflorian/fntbl/
	templates = brill.fntbl37()
	
	trainer = BrillTaggerTrainer(initial_tagger, templates,
		deterministic=True, trace=trace)
	return trainer.train(train_sents, **kwargs)