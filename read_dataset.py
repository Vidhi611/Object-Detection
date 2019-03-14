#Test in draft.py

import numpy as np
from math import sqrt
import os
import random
import pickle
import codecs
'''
calculate temporal intersection over union
'''
def calculate_IoU(i0, i1):
	union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
	inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
	iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
	return iou

'''
calculate the non Intersection part over Length ratia, make sure the input IoU is larger than 0
'''
def calculate_nIoL(base, sliding_clip):
	inter = (max(base[0], sliding_clip[0]), min(base[1], sliding_clip[1]))
	inter_l = inter[1]-inter[0]
	length = sliding_clip[1]-sliding_clip[0]
	nIoL = 1.0*(length-inter_l)/length
	return nIoL

class TrainingDataSet(object):

	def __init__(self, sliding_dir, it_path, batch_size):
		self.counter = 0
		self.batch_size = batch_size
		self.context_num = 1
		self.context_size = 128
		print("Reading training data list from "+it_path)
		cs = pickle.load(codecs.open(it_path,"rb",encoding="iso-8859-1"))
		#movie_length_info = pickle.load(codecs.open("./TACoS/video_allframes_info.pkl","rb",encoding="iso-8859-1"))
		self.clip_sentence_pairs = []
		for l in cs:
		    clip_name = l[0]
		    sent_vecs = l[1]
		    for sent_vec in sent_vecs:
		        self.clip_sentence_pairs.append((clip_name, sent_vec))

		movie_names_set = set()
		self.movie_clip_names = {}
		# read groundtruth sentence-clip pairs
		for k in range(len(self.clip_sentence_pairs)):
		    clip_name = self.clip_sentence_pairs[k][0]
		    movie_name = clip_name.split("_")[0]
		    if not movie_name in movie_names_set:
		        movie_names_set.add(movie_name)
		        self.movie_clip_names[movie_name] = []
		    self.movie_clip_names[movie_name].append(k)
		self.movie_names = list(movie_names_set)
		self.visual_feature_dim = 4096*3
		self.sent_vec_dim = 4800
		self.num_samples = len(self.clip_sentence_pairs)
		self.sliding_clip_path = sliding_dir
		print(str(len(self.clip_sentence_pairs))+" clip-sentence pairs are readed")

		# read sliding windows, and match them with the groundtruths to make training samples
		sliding_clips_tmp = os.listdir(self.sliding_clip_path)
		self.clip_sentence_pairs_iou = []
		for clip_name in sliding_clips_tmp:
			if clip_name.split(".")[2]=="npy":
				movie_name = clip_name.split("_")[0]
				for clip_sentence in self.clip_sentence_pairs:
					original_clip_name = clip_sentence[0] 
					original_movie_name = original_clip_name.split("_")[0]
					if original_movie_name==movie_name:
						start = int(clip_name.split("_")[1])
						end = int(clip_name.split("_")[2].split(".")[0])
						o_start = int(original_clip_name.split("_")[1]) 
						o_end = int(original_clip_name.split("_")[2].split(".")[0])
						iou = calculate_IoU((start, end), (o_start, o_end))
						if iou>0.5:
							nIoL=calculate_nIoL((o_start, o_end), (start, end))
							if nIoL<0.15:
								#movie_length = movie_length_info[movie_name.split(".")[0]]
								start_offset =o_start-start
								end_offset = o_end-end
								self.clip_sentence_pairs_iou.append((clip_sentence[0], clip_sentence[1], clip_name, start_offset, end_offset))
		self.num_samples_iou = len(self.clip_sentence_pairs_iou)
		#with open("TACoS_input.txt", "w") as text_file:
			#text_file.write("Purchase Amount: %s" % TotalAmount)
		print(str(len(self.clip_sentence_pairs_iou))+" iou clip-sentence pairs are readed")

train_csv_path = "./TACoS/train_clip-sentvec.pkl"
#test_csv_path = "./TACoS/test_clip-sentvec.pkl"
#test_feature_dir="./TACOS/Interval128_256_overlap0.8_c3d_fc6/"
train_feature_dir = "./TACOS/Interval64_128_256_512_overlap0.8_c3d_fc6/"
batch_size=56
TrainingDataSet(train_feature_dir,train_csv_path,batch_size)



