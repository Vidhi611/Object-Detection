'''Reading pascal dataset
Every image has width, height and depth
Each image has diff no of objects
Each object has name, bndbox(xmin,xmax,ymin,ymax)

reading annotations'''

import xml.etree.ElementTree as ET 
import os
import pickle
from PIL import Image

class PascalData():
	def __init__(self, image_path, annotation_path,newshape):
		#self.classes=classes_list
		self.images=[]
		self.image_files=[]
		self.annotations={}
		self.annotation_path=annotation_path
		self.imagesdir_path=image_path
		self.class_labels={'motorbike':0,'bottle':1,'bird':2,'cat':3,'aeroplane':4,'chair':5,'person':6,'diningtable':7,'boat':8,'train':9,'sofa':10,'bicycle':11,'bus':12,'horse':13,'tvmonitor':14,'cow':15,'pottedplant':16,'car':17,'dog':18,'sheep':19}
		self.newshape=newshape
		
		
	def readXmlfile(self):
		annotations_files = os.listdir(self.annotation_path)
		for file in annotations_files:
			#file_details={}
			#file=annotations_files[0]
			file_to_parse=self.annotation_path+file
			tree=ET.parse(file_to_parse)
			root=tree.getroot()
			img_filename=root.find('filename')
			#img_filename includes .jpg also
			#file_details.append(img_filename)
			self.image_files.append(img_filename.text)

			Objects=[]
			for element in root.findall('object'):
				Object=[]
				Object.append(element.find('name').text)
				Object.append(element.find('bndbox').find('xmin').text)
				Object.append(element.find('bndbox').find('ymin').text) 
				Object.append(element.find('bndbox').find('xmax').text)
				Object.append(element.find('bndbox').find('ymax').text)
				Objects.append(Object)
			#print(len(Objects))
			self.annotations[img_filename.text]=Objects
		#print(len(self.annotations))
		#pickle.dump(self.annotations, open("./PASCAL-VOC/VOC2012/Annotations.pkl", 'wb'))


	def buildDataset(self):
		#X(image), Y(class) from dataset
		self.readXmlfile()
		X=[]
		Y=[]
		for image_file,annotations in self.annotations.items():
			#append img to X
			#append img details array to Y in the form[class,xmin,xmax,ymin,ymax]
			#image_file=list(self.annotations.keys())[0]
			#annotations=self.annotations[image_file]
			filename=image_file.split(".")[0]
			img_path=self.imagesdir_path + image_file
			im = Image.open(img_path)
			nx, ny = im.size
			im2 = im.resize(self.newshape , Image.BICUBIC)
			im2.load()
			#print(im2.size)
			img_array = np.asarray( im2, dtype="int32" )
			#np.save(numpyfile_path + filename , x)
			#print(img_array)
			X.append(img_array)
			y=[]
			for annot in annotations:
				temp=[]
				class_label=self.class_labels[annot[0]]
				temp.append(class_label)
				temp.append(float(annot[1]))
				temp.append(float(annot[2]))
				temp.append(float(annot[3]))
				temp.append(float(annot[4]))
				y.append(temp)
				#print(y)
			Y.append(y)
			#print(Y)
		pickle.dump(X, open("./PASCAL-VOC/VOC2012/ProcessedDataImage.pkl", 'wb'))
		pickle.dump(Y, open("./PASCAL-VOC/VOC2012/ProcessedDataClasses.pkl", 'wb'))



ps=PascalData("./PASCAL-VOC/VOC2012/JPEGImages/","./PASCAL-VOC/VOC2012/Annotations/",(300,300))
ps.buildDataset()

