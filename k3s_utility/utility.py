from .config import Config
import math
import hashlib
import json
from .file import File
import os
import re
import urllib

class Utility():


	""" return the list with duplicate elements removed """
	@staticmethod
	def unique(a):
		return list(set(a))


	""" return the intersection of two lists """
	@staticmethod
	def intersect(a, b):
		return list(set(a) & set(b))


	""" return the union of two lists """
	@staticmethod
	def union(a, b):
		return list(set(a) | set(b))


	@staticmethod
	def diff(a, b):
		return (set(a) - set(b))


	@staticmethod
	def getStopWords():
		path = File.join(os.path.abspath(__file__ + "/../"), 'stopwords.txt')
		file = File(path)
		stopWords = re.split('[\n]', file.read())


		return stopWords


	@staticmethod
	def implode(terms, divider = ','):
		return divider.join(s for s in terms)


	@staticmethod
	def utlencode(params):
		return urllib.parse.urlencode(params)

	@staticmethod
	def debug(value):
		#print('---------------------------------------------------------------------------------------------')
		print(value)
		print('---------------------------------------------------------------------------------------------')
		return

		



