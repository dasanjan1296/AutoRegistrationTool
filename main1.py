import numpy as np
import cv2
#from matplotlib import pyplot as plt
import gdal
from gdalconst import *
import time
import sys
import os
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from second import *

inputimgloc = None
refimgloc = None
outimgloc = None
iplist = []
reflist = []


class MainWindow(QMainWindow):
	
	def __init__(self, parent=None):
		super(MainWindow, self).__init__(parent)
		self.main_app = mainapp(self) 
		self.setCentralWidget(self.main_app)
		self.setWindowTitle("[**BETA**] BATCH AUTOMATIC GEOREFERENCER v0.1.2")
		self.setGeometry(50,100,1000,700)

		file_quit_Action = QAction("&Quit", self)
		file_quit_Action.setShortcut("Ctrl+Q")
		file_quit_Action.setStatusTip('Leave The App')
		file_quit_Action.triggered.connect(self.close_application)

		help_about_Action = QAction("&About Us", self)
		help_about_Action.setShortcut("Ctrl+H")
		help_about_Action.setStatusTip('About the Developers')
		help_about_Action.triggered.connect(self.aboutDialog)

		self.statusBar()

		mainMenu = self.menuBar()
		#Assign mainMenu
		fileMenu = mainMenu.addMenu('&File')
		fileMenu.addAction(file_quit_Action)

		helpMenu = mainMenu.addMenu('&Help')
		helpMenu.addAction(help_about_Action)

	def aboutDialog(self):
		msg = QMessageBox()
		msg.setIcon(QMessageBox.Information)

		msg.setText("[**BETA**] Batch_Automatic_Georeferencer v0.1.2\nRelease: 01-July-2017\nThis program does automatic georeferencing by feature matching.\nFor more details, visit <site-name>")
		msg.setInformativeText("This program was developed in North Eastern Space Application Centre (NE-SAC), Shillong, India under supervision of Mr. Victor Saikhom and Mr. Siddhartha Bhuyan\nAs a part of Practice-School 1 program of BITS-Pilani.\n\nDevelopers: Akash Manna, Anjan Das, Amit Shukla")
		msg.setWindowTitle("ABOUT Batch_Automatic_Georeferencer")
		#msg.setDetailedText(detailedText)
		msg.setStandardButtons(QMessageBox.Close)
		msg.exec()


	def close_application(self):
		print("Thanks. I is now dead. I rests in peace. I hope I served you well.")
		sys.exit()


class mainapp(QWidget):
	def __init__(self, parent = None):
		super(mainapp, self).__init__(parent)
		
		wrapper = QVBoxLayout()

		data = QHBoxLayout()


		inpdata = QVBoxLayout()

		lblInput = QLabel("INPUT DATA:")
		inpdata.addWidget(lblInput)
		loadinpimg = QHBoxLayout()
		self.inpfil = QLineEdit()
		self.inpbtn = QPushButton("Add")
		self.inpbtn.clicked.connect(self.getinpfile)
		self.remInp = QPushButton("Remove")
		self.remInp.clicked.connect(self.removeInpItem)
		self.upInp = QPushButton("Move Up")
		self.upInp.clicked.connect(self.upInpItem)
		self.downInp = QPushButton("Move Down")
		self.downInp.clicked.connect(self.downInpItem)
		loadinpimg.addWidget(self.inpfil)
		loadinpimg.addWidget(self.inpbtn)
		loadinpimg.addWidget(self.remInp)
		loadinpimg.addWidget(self.upInp)
		loadinpimg.addWidget(self.downInp)
		inpdata.addLayout(loadinpimg)

		self.inpList = QListWidget()
		inpdata.addWidget(self.inpList)
		inpdata.addStretch(1)


		refdata = QVBoxLayout()
		lblRef = QLabel("REFERENCE DATA:")
		refdata.addWidget(lblRef)
		loadrefimg = QHBoxLayout()
		#loadrefimg.addStretch(1)
		self.reffil = QLineEdit()
		self.refbtn = QPushButton("Add")
		self.refbtn.clicked.connect(self.getreffile)
		self.remRef = QPushButton("Remove")
		self.remRef.clicked.connect(self.removeRefItem)
		self.upRef = QPushButton("Move Up")
		self.upRef.clicked.connect(self.upRefItem)
		self.downRef = QPushButton("Move Down")
		self.downRef.clicked.connect(self.downRefItem)
		loadrefimg.addWidget(self.reffil)
		loadrefimg.addWidget(self.refbtn)
		loadrefimg.addWidget(self.remRef)
		loadrefimg.addWidget(self.upRef)
		loadrefimg.addWidget(self.downRef)

		refdata.addLayout(loadrefimg)

		self.refList = QListWidget()
		refdata.addWidget(self.refList)
		refdata.addStretch(1)

		self.checkBox_showimg = QCheckBox('Show Matches for every pair', self)
		#checkBox.stateChanged.connect(self.show_matches)

		Horizontal3 = QHBoxLayout()
		self.projlbl = QLabel("Projection System: ")
		self.proj = QComboBox(self)
		self.proj.addItem("Reference Image")
		self.proj.addItem("Input Image")
		self.proj.addItem("WGS")
		self.proj.addItem("UTM")
		self.proj.currentIndexChanged.connect(self.projchange)
		self.projfil = QLineEdit()
		self.projfil.setEnabled(False)
		Horizontal3.addWidget(self.projlbl)
		Horizontal3.addWidget(self.proj)
		Horizontal3.addWidget(self.projfil)
		Horizontal3.addStretch(1)


		loadoutimg = QHBoxLayout()
		self.outbtn = QPushButton("SELECT OUTPUT FOLDER")
		self.outbtn.clicked.connect(self.getoutfile)

		self.outlbl = QLabel(" Output: ")
		self.outfil = QLineEdit()

		self.outformatlbl = QLabel("Output FORMAT: ")
		self.outformat = QComboBox(self)
		self.outformat.addItem("GTiff")
		self.outformat.addItem("HFA")
		self.outformat.addItem("ENVI")
		self.outformat.addItem("VRT")
		self.outformat.addItem("ELAS")
		self.outformat.addItem("NITF")
		self.outformat.addItem("GPKG")
		self.outformat.addItem("ERS")


		self.outintplbl = QLabel(" INTERPOLATION: ")
		self.outintp = QComboBox(self)
		self.outintp.addItem("near")
		self.outintp.addItem("bilinear")
		self.outintp.addItem("cubic")
		self.outintp.addItem("cubicspline")
		self.outintp.addItem("lanczos")
		self.outintp.addItem("average")
		self.outintp.addItem("mode")
		self.outintp.addItem("max")
		self.outintp.addItem("min")
		self.outintp.addItem("med")

		
		self.downscalelbl = QCheckBox("\tDownscale Images",self)
		self.downscalelbl.stateChanged.connect(self.changeTitle)

		self.downscale = QComboBox(self)
		self.downscale.addItem("2")
		self.downscale.addItem("4")
		self.downscale.addItem("8")
		self.downscale.setEnabled(False)


		self.outorderlbl = QLabel(" Poly Order: ")
		self.outorder = QComboBox(self)
		self.outorder.addItem("1")
		self.outorder.addItem("2")
		self.outorder.addItem("3")
		
		#self.outbtn = QPushButton("BROWSE")
		#self.outbtn.clicked.connect(self.getoutfile)
		
		loadoutimg.addWidget(self.outlbl)
		loadoutimg.addWidget(self.outfil)
		loadoutimg.addWidget(self.outbtn)
		loadoutimg.addWidget(self.outformatlbl)
		loadoutimg.addWidget(self.outformat)
		loadoutimg.addWidget(self.outorderlbl)
		loadoutimg.addWidget(self.outorder)
		loadoutimg.addWidget(self.outintplbl)
		loadoutimg.addWidget(self.outintp)

		

		outlayer2 = QHBoxLayout()

		self.minGCPlbl = QLabel("\tMinimum GCPs: ")
		self.minGCPs = QSpinBox()
		self.minGCPs.setValue(5)

		self.maxRMSElbl = QLabel("\tMax RMS Error permitted: ")
		self.maxRMSE = QDoubleSpinBox()
		self.maxRMSE.setDecimals(2)
		self.maxRMSE.setValue(0.75)
		self.maxRMSE.setSingleStep(0.05)

		self.bandlbl = QLabel("\tBand: ")
		self.band = QComboBox(self)
		self.band.addItem("3")
		self.band.addItem("2")
		self.band.addItem("1")

		outlayer2.addWidget(self.checkBox_showimg)
		outlayer2.addWidget(self.downscalelbl)
		outlayer2.addWidget(self.downscale)
		outlayer2.addWidget(self.minGCPlbl)
		outlayer2.addWidget(self.minGCPs)
		outlayer2.addWidget(self.maxRMSElbl)
		outlayer2.addWidget(self.maxRMSE)
		outlayer2.addWidget(self.bandlbl)
		outlayer2.addWidget(self.band)
		outlayer2.addStretch(1)


		self.runbtn = QPushButton("START REFERENCING")
		self.runbtn.clicked.connect(self.exex)

		self.warpbtn = QPushButton("Dummy Button that does nothing")
		self.warpbtn.clicked.connect(self.runwarp)

		data.addLayout(inpdata)
		data.addLayout(refdata)

		wrapper.addLayout(data)
		wrapper.addLayout(Horizontal3)
		wrapper.addLayout(loadoutimg)
		wrapper.addLayout(outlayer2)
		wrapper.addWidget(self.runbtn)
		wrapper.addWidget(self.warpbtn)
		
		self.setLayout(wrapper)

	def changeTitle(self, state):
		if state == Qt.Checked:
			self.downscale.setEnabled(True)
		else:
			self.downscale.setEnabled(False)
	
	
	def projchange(self, i):
		if(self.proj.currentText()=="WGS" or self.proj.currentText()=="UTM"):
			self.projfil.setEnabled(True)
		else:
			self.projfil.setEnabled(False)


	def exex(self):
		if(self.refList.count() == 0):
			print("NO REFERENCE FILE SPECIFIED.")
			showWarning("NO Referenced Image", "Please specify a referenced image.")
			return -1

		'''
		if(self.inpList.count()!=self.refList.count()):
			print("NO WAY SAME COUNT")
			showWarning("COUNT INCONSISTENCY", "The number of Input Images and Reference Images are different.\nPlease Check.")
			return -1
		'''

		if(self.outfil.text()==""):
			print("NO Output FOLDER specified.")
			showWarning("NO OUTPUT FOLDER SPECIFIED", "Please specify an output folder.")
			return -1

		inpitems = []
		refitems = [] 
		for index in range(self.inpList.count()):
			inpitems.append(self.inpList.item(index))
		for index in range(self.refList.count()):
			refitems.append(self.refList.item(index))
		inplocs = [i.text() for i in inpitems]
		reflocs = [i.text() for i in refitems]
		print(inplocs)
		print(reflocs)

		for i in range(len(inplocs)):
			
			inputimgtest = gdal.Open(inplocs[i])
			inputgt = inputimgtest.GetGeoTransform()
			#cols_x = inputimgtest.RasterXSize
			#rows_y = inputimgtest.RasterYSize
			inputimgtest = None
			inputcoord = [inputgt[0], inputgt[3]]
			print(inputcoord)
			refindex = 255

			for j in range(len(reflocs)):
				print(j)
				print(reflocs[j])
				refimgtest = gdal.Open(reflocs[j])
				refgt = refimgtest.GetGeoTransform()
				ref_xsize = refimgtest.RasterXSize
				ref_ysize = refimgtest.RasterYSize
				refimgtest = None
				refcoord0 = [refgt[0], refgt[3]]
				refcoord1 = [refgt[0] + refgt[1]*ref_xsize, refgt[3] + refgt[5]*ref_ysize]

				print(refcoord0, refcoord1)
				if(inputcoord[0]>min(refcoord0[0], refcoord1[0]) \
					and inputcoord[0]<max(refcoord0[0], refcoord1[0]) \
					and inputcoord[1]>min(refcoord0[1], refcoord1[1]) \
					and inputcoord[1]<max(refcoord0[1], refcoord1[1])):
					print("SCORED ***************************************", i, j)
					refindex = j
					break

			if(refindex==255):
				print("NO MATCHES FOUND FOR INPUTIMG", inputimgloc)
				showWarning("NO MATCHING REFERENCE FILE FOUND for " + inputimgloc, "Operation aborted for the file.")
				continue

			print (i, refindex)

			scaling = 1
			if(self.downscalelbl.isChecked()):
				scaling = int(self.downscale.currentText())
			print("DOWNSCALE OPTION >>>>>>>>>>>>>>>>>>> ", scaling)


			iplist, reflist = run(inplocs[i], 
				reflocs[refindex], 
				self.checkBox_showimg.isChecked(), 
				scaling, 
				int(self.band.currentText()))

			gdal_register(self.outfil.text(), 
				inplocs[i], 
				reflocs[refindex], 
				iplist, 
				reflist, 
				self.outformat.currentText(), 
				self.outorder.currentText(), 
				self.outintp.currentText(), 
				float(self.downscale.currentText()), 
				self.maxRMSE.value(), 
				self.minGCPs.value(),
				self.proj.currentText())

		#global inputimgloc, refimgloc
		#inputimgloc = self.inpfil.text()
		#refimgloc = self.reffil.text()
		#run(inputimgloc, refimgloc)

	def runwarp(self):
		print(self.outformat.currentText())

	def removeInpItem(self):
		for item in self.inpList.selectedItems():
			self.inpList.takeItem(self.inpList.row(item))

	def removeRefItem(self):
		for item in self.refList.selectedItems():
			self.refList.takeItem(self.refList.row(item))

	def getinpfile(self):
		fname = QFileDialog.getOpenFileName(self, 'Open file', 
		 'c:\\',"Image files (*.tif *.img)")
		global inputimgloc
		self.inpfil.setText(fname)
		self.inpList.addItem(fname)
		inputimgloc = fname
		#self.le.setPixmap(QPixmap(fname))

	def upInpItem(self):
		currentRow = self.inpList.currentRow()
		currentItem = self.inpList.takeItem(currentRow)
		self.inpList.insertItem(currentRow - 1, currentItem)
		self.inpList.setCurrentItem(currentItem)

	def downInpItem(self):
		currentRow = self.inpList.currentRow()
		currentItem = self.inpList.takeItem(currentRow)
		self.inpList.insertItem(currentRow + 1, currentItem)
		self.inpList.setCurrentItem(currentItem)

	def upRefItem(self):
		currentRow = self.refList.currentRow()
		currentItem = self.refList.takeItem(currentRow)
		self.refList.insertItem(currentRow - 1, currentItem)
		self.refList.setCurrentItem(currentItem)

	def downRefItem(self):
		currentRow = self.refList.currentRow()
		currentItem = self.refList.takeItem(currentRow)
		self.refList.insertItem(currentRow + 1, currentItem)
		self.refList.setCurrentItem(currentItem)		

	def getreffile(self):
		fname = QFileDialog.getOpenFileName(self, 'Open file', 
		 'c:\\',"Image files (*.tif *.img)")
		self.reffil.setText(fname)
		self.refList.addItem(fname)
		global refimgloc
		refimgloc = fname
		
	def getoutfile(self):
		fname = QFileDialog.getExistingDirectory(self, 'Select Folder for Output')
		self.outfil.setText(fname)
		#global outimgloc
		#outimgloc = fname



def showWarning(text="WARNING: ERROR", detailedText=""):
	msg = QMessageBox()
	msg.setIcon(QMessageBox.Critical)

	msg.setText(text)
	msg.setInformativeText(detailedText)
	msg.setWindowTitle("WARNING")
	#msg.setDetailedText(detailedText)
	msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
	msg.exec()

def run(inputimgloc, refimgloc, showMatches, rescalefac, band):    
	time_init = time.time()

	'''
	# CHECK WITH SAMPLE IMAGES 
	img1 = cv2.imread('clockt.jpg',0)          # queryImage
	img2 = cv2.imread('clockt1.jpg',0) # trainImage
	'''

	inputimg = gdal.Open(inputimgloc, GA_ReadOnly)
	if inputimg is None:
		print("FAILED TO IMPORT INPUT IMAGE\n")
		showWarning("INPUT FAILED", "failed to import "+inputimgloc)
		return -1
	else:
		print("INPUT IMAGE IMPORTED\n")

	
	refimg = gdal.Open(refimgloc, GA_ReadOnly)
	if refimg is None:
		print("FAILED TO IMPORT REFERENCE IMAGE\n")
		showWarning("INPUT FAILED", "failed to import "+refimgloc)
		return -1
	else:
		print("REFERENCE IMAGE IMPORTED\n")

	inpGeoTransform = inputimg.GetGeoTransform()
	refGeoTransform = refimg.GetGeoTransform()	
	inpXsize, inpYsize = inpGeoTransform[1], inpGeoTransform[5]
	refXsize, refYsize = refGeoTransform[1], refGeoTransform[5]
	refbyinp_x = abs(refXsize/inpXsize)
	refbyinp_y = abs(refYsize/inpYsize)
	print("IP SIZEs\t", inpXsize, inpYsize)
	print("REF SIZEs\t", refXsize, refYsize)
	print("RATIO\t\t", refbyinp_x, refbyinp_y)

	inputimg = inputimg.GetRasterBand(band).ReadAsArray()
	refimg = refimg.GetRasterBand(band).ReadAsArray()

	inputimg = cv2.convertScaleAbs(inputimg)
	refimg = cv2.convertScaleAbs(refimg)

	#inputimg = cv2.bitwise_not(inputimg)
	#refimg = cv2.bitwise_not(refimg)

	print("LOADING COMPLETE, FEATURE DETECTION STARTS\t", time.time()-time_init)
	heightinp, widthinp = inputimg.shape[:2]
	heightref, widthref = refimg.shape[:2]

	inputimg = cv2.resize(inputimg, (int(heightinp/rescalefac),int(widthinp/rescalefac)))
	refimg = cv2.resize(refimg, (int(heightref/rescalefac),int(widthref/rescalefac)))


	'''
	HISTOGRAM EQUALIZATION
	'''
	inputimg = cv2.equalizeHist(inputimg)
	refimg = cv2.equalizeHist(refimg)



	'''
	Initiate SIFT / SURF / ORB / AKAZE detector
	'''
	#sift = cv2.xfeatures2d.SURF_create()
	#sift = cv2.ORB_create()
	#sift = cv2.BRISK_create()
	sift = cv2.AKAZE_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(inputimg,None)
	kp2, des2 = sift.detectAndCompute(refimg,None)


	'''
	AGAST / BRISK + FREAK DETECTOR + DESCRIPTOR

	agast = cv2.BRISK_create()
	kp1 = agast.detect(inputimg)
	kp2 = agast.detect(refimg)
	freak = cv2.xfeatures2d.FREAK_create()
	kp1, des1 = freak.compute(inputimg, kp1)
	kp2, des2 = freak.compute(refimg, kp2)
	#des1 = np.float32(des1)
	#des2 = np.float32(des2)
	'''


	print("FEATURE AND DESCRIPTOR DETECTION COMPLETE\t\t", time.time()-time_init)

	'''
	BF MATCHER (BFMatcher(cv2.NORM_HAMMING) << AGAST+FREAK; void bracket for others)

	bf = cv2.BFMatcher(cv2.NORM_HAMMING)
	matches = bf.knnMatch(des1,des2, k=2)
	'''


	'''
	FLANN BASED MATCHER
	'''
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 5)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1, des2, k=2)



	'''
	GOOD - TEST (as per paper by LOWE)
	'''

	print("INITIAL MATCHES FOUND\t\t", time.time()-time_init)
	# Apply ratio test
	good = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append([m])

	# cv2.drawMatchesKnn expects list of lists as matches.

	'''
	RATIO - TEST (creates combinations of points, compares distances-ratio to the pixel-ratio)
	'''

	good1_1 = [] #RATIO - correct layer
	(ip, ref) = pointsFromMatches(kp1, kp2, good)
	hash_list = [0 for i in range(len(ip))]
	for i in range(len(ip)-1):
		for j in range(i+1, len(ip)):
			distratio = (distance(ip[i][0], ip[i][1], ip[j][0], ip[j][1]))/(distance(ref[i][0], ref[i][1], ref[j][0], ref[j][1]))
			#print(ip[i], ref[i], ip[j], ref[j], distratio)
			if (distratio>min(refbyinp_x,refbyinp_y) and distratio<max(refbyinp_y,refbyinp_x)):
				#print("YES")
				hash_list[i] += 1
				hash_list[j] += 1

	max_hash = max(hash_list)
	min_hash = min(hash_list)
	print("\nHASHLIST", hash_list)
	print(max_hash, 0.5*max_hash)
	for i in range(len(hash_list)):
		if (hash_list[i] > 0.5*max_hash):
			good1_1.append(good[i])


	'''
	CORRELATION - TEST
	'''

	padding = 50
	(ip, ref) = pointsFromMatches(kp1, kp2, good1_1)
	corr_list = [1 for i in range(len(ip))]
	good1 = [] #correlation layer
	pad_inp_x = int(padding*refbyinp_x)
	pad_inp_y = int(padding*refbyinp_y)

	for i in range(len(good1_1)):
		(inX, inY) = ip[i]
		(detX, detY) = ref[i]
		inX, inY = int(inX), int(inY)
		detX, detY = int(detX), int(detY)
		print(i, inX, inY, detX, detY)
		if (inX-pad_inp_x<0 or inY-pad_inp_y<0 or inX+pad_inp_x>=len(inputimg) or inY+pad_inp_y>=len(inputimg[0])) :
			continue
		if (detX-padding<0 or detY-padding<0 or detX+padding>=len(refimg) or detY+padding>=len(refimg[0])) :
			continue

		# CORRELATION LAYER
		template2 = refimg[detY-padding:detY+padding, detX-padding:detX+padding]
		template1 = inputimg[inY-pad_inp_y:inY+pad_inp_y, inX-pad_inp_x:inX+pad_inp_x]
		corrval = correlate(template1, template2)
		corr_list[i] = corrval

	max_corr = max(corr_list)
	#min_corr = min(corr_list)
	avg_corr = sum(corr_list)/len(corr_list)
	corr_thresh = (avg_corr+max_corr)/2
	for i in range(len(corr_list)):	
		if (corrval < corr_thresh):
			good1.append(good1_1[i][0])

	'''
	RANSAC ALGORITHM
	'''

	src_pts = np.float32([ kp1[m.queryIdx].pt for m in good1 ]).reshape(-1,1,2)
	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good1 ]).reshape(-1,1,2)

	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,3.0)

	print("HOMOGRAPHY FOUND\t\t", time.time()-time_init)
	good2 = []
	for i in range(len(mask)):
		if(mask[i][0] == 1):
			good2.append([good1[i]])



	'''
	OUTPUT - IMAGEs
	'''
	if(showMatches):
		img3 = cv2.drawMatchesKnn(inputimg,kp1,refimg,kp2,good2,None, flags=2)
		drawcv("img3", img3)

		img4 = cv2.drawMatchesKnn(inputimg,kp1,refimg,kp2,good1_1,None, flags=2)
		drawcv("img4", img4)


	#img5 = cv2.drawMatchesKnn(inputimg,kp1,refimg,kp2,good,None, flags=2)
	#drawcv("img5", img5)
	#global iplist
	#global reflist
	(iplist, reflist) = pointsFromMatches(kp1, kp2, good2)
	print(iplist)
	print(reflist)
	print("\n\nFINAL NUMBER OF TIE POINTS >>>>>>>>>>>>>>>>>>>>>> ", len(good2))

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return(iplist, reflist)


def drawGUI():
	app = QApplication(sys.argv)
	ex = MainWindow()
	ex.show()
	sys.exit(app.exec_())


drawGUI()

