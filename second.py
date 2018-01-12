import numpy as np
import cv2
#from matplotlib import pyplot as plt
import gdal
import osr
from gdalconst import *
import time
import sys
import os


def pixel2coord(ds, x, y):
	"""Returns global coordinates to pixel center using base-0 raster index"""
	'''
	GeoTransform() returns a tuple where (X, Y) are corner coordinates of the image indicating the origin of the array,
	i.e. data element[0,0]. But, which corner?
	If deltaX is positive, X is West. 
	Otherwise, X is East. 
	If deltaY is positive, Y is South. 
	Otherwise, Y is North.
	In other words, when both deltaX and deltaY is positive, (X, Y) is the lower-left corner of the image.

	It is also common to have positive deltaX but negative deltaY which indicates that (X, Y) is the top-left corner of the image.
	There are several standard notations for SRS, e.g. WKT, Proj4, etc.
	GetProjection() returns a WKT string.
	The meaning and unit-of-measurement for  X, Y, deltaX and deltaY are based on the spatial reference system (SRS) of the image.
	For example, if the SRS is Stereographic projection, they are in kilometres.
	'''
	ulx, xres, xskew, uly, yskew, yres = ds.GetGeoTransform()#(X, deltaX, rotation, Y, rotation, deltaY) = ds.GetGeoTransform()
	xp = xres * x + ulx #+ b * y 
	yp = yres * y + uly #d * x  
	return(xp, yp)

def distance(x1, y1, x2, y2):
	return ((x1-x2)**2+(y1-y2)**2)**(0.5)

def getExtension(extn):
	if (extn == "HFA"):
		return ".img"
	elif (extn == "ENVI"):
		return ".hdr"
	elif (extn == "VRT"):
		return ".vrt"
	elif (extn == "ELAS"):
		return ""
	elif (extn == "NITF"):
		return ".ntf"
	elif (extn == "GPKG"):
		return ".gpkg"
	elif (extn == "ERS"):
		return ".ers"
	else:
		return ".tif"

def gdal_register(outimgloc, 
	inputimgloc, 
	refimgloc, 
	iplist, 
	reflist, 
	outformat, 
	outorder, 
	outintp, 
	downscale, 
	tolerance, 
	min_gcps, 
	proj_type = "ref"):

	if len(iplist)==0:
		print("NO GCPs to translate with.")
		return

	extn = getExtension(outformat)

	command = "gdal_translate -of " + str(outformat)+ " "

	inputimg = gdal.Open(inputimgloc, GA_ReadOnly)
	if inputimg is None:
		print("FAILED TO IMPORT INPUT IMAGE")
	else:
		print("INPUT IMAGE IMPORTED")

	#refIMAGE = gdal.Open("112_53_a_22oct12.img", GA_ReadOnly)
	refimg = gdal.Open(refimgloc, GA_ReadOnly)
	if refimg is None:
		print("FAILED TO IMPORT REFERENCE IMAGE")
	else:
		print("REFERENCE IMAGE IMPORTED")

	print(proj_type)
	if(proj_type=="Reference Image"):
		inSRS = refimg.GetProjection()
		print("***PROJ SYSTEM: REF IMG")
	elif(proj_type=="Input Image"):
		inSRS = inputimg.GetProjection()
		print("***PROJ SYSTEM: INP IMG")
	else:
		inSRS = refimg.GetProjection()
		print("ELSE INVOKED *********")
	inSRS_converter = osr.SpatialReference()
	inSRS_converter.ImportFromWkt(inSRS)
	inSRS_forPyProj = inSRS_converter.ExportToProj4()
	adcmd = "-a_srs \"" + str(inSRS_forPyProj) + "\" "
	command = command + adcmd


	for i in range(len(iplist)):
		refx, refy = pixel2coord(refimg, downscale*reflist[i][0], downscale*reflist[i][1])
		adcmd = "-gcp " + str(int(downscale*iplist[i][0])) + " " + str(int(downscale*iplist[i][1])) + " " + str(refx) + " " + str(refy)+ " "
		command = command + adcmd
	command = command + " " + (inputimgloc) + " temp" + extn
	print(command)
	os.system(command)
	print("GCPs added Successfully.")

	print("WARP STARTS")

	filename = os.path.basename(inputimgloc)[:-4]
	cmd2 = "gdalwarp -of " + outformat + \
	" -r " + outintp + \
	" -order " + outorder + \
	" -refine_gcps " + str(tolerance) + " " + str(min_gcps) + \
	" -co COMPRESS=NONE temp" + extn + \
	" \"" + \
	str(outimgloc) + "/" + filename +  extn + "\"" 

	print(cmd2)
	os.system(cmd2)
	os.system("del temp.tif")
	#os.system("gdalinfo " + inputimgloc +" > log.txt")
	print("WARP COMPLETE")



def pointsFromMatches(kp1, kp2, matches):
	pairsOfKp1 = [i[0].queryIdx for i in matches]
	pairsOfKp2 = [i[0].trainIdx for i in matches]
	sP = cv2.KeyPoint_convert(kp1, pairsOfKp1)
	dP = cv2.KeyPoint_convert(kp2, pairsOfKp2)
	return sP, dP

def drawcv(imagename, img3):
	screen_res = 1920, 1080
	scale_width = screen_res[0] / img3.shape[1]
	scale_height = screen_res[1] / img3.shape[0]
	scale = min(scale_width, scale_height)
	window_width = int(img3.shape[1] * scale)
	window_height = int(img3.shape[0] * scale)

	cv2.namedWindow(imagename, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(imagename, window_width, window_height)

	cv2.imshow(imagename, img3)

def draw2cv(imagename, imgA, imgB):
	imgAx = len(imgA[0])
	imgAy = len(imgA)
	imgBx = len(imgB[0])
	imgBy = len(imgB)

	imgA = cv2.resize(imgA, (imgBx, imgBy))
	img = np.hstack((imgA, imgB))
	drawcv(imagename, img)
	#plt.imshow(img)
	#plt.show()

def correlate(img, template):
	w, h = template.shape[::-1]
	# All the 6 methods for comparison in a list
	#methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
	#            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
	#for meth in methods:
	meth = 'cv2.TM_SQDIFF_NORMED'
	method = eval(meth)

	# Apply template Matching
	res = cv2.matchTemplate(img,template,method)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	#print("correlate called")
	# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
	if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
		return min_val
	else:
		return max_val
	#bottom_right = (top_left[0] + w, top_left[1] + h)
	#cv2.rectangle(img,top_left, bottom_right, 255, 2)
	#drawcv(img)
