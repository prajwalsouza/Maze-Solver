"""
The Following program is designed to solve Mazes which are given as input in pictoral form. 

Prajwal DSouza
23rd June 2017

This was the second algorithm and was finished on June 27th. 

"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import time
import os

# The first three parameters must be changed based on the maze. 

pointdistance = 17
# If this value isn't set properly, there could be errors.
# This reduces the number of points on the image to be analyzed. 
# The Algorithm will move through only those points avoiding the obstacles. 
# Every point at distance equal to pointdistance is chosen for analysis. 

startingposition = (210,2246)
endingposition = (2799,3950)
# The Maze must be marked, like in the example maze. Ending and starting of the maze must be closed.
# But, the starting positions and ending positions must be specified and marked in the image for simplification. 


plotpointthickness = 1
# This is about how thick the line of the solution must be. 

startingposition = (startingposition[0] - (startingposition[0]%pointdistance),startingposition[1] - (startingposition[1]%pointdistance))
endingposition = (endingposition[0] - (endingposition[0]%pointdistance),endingposition[1] - (endingposition[1]%pointdistance))
# Starting and ending position is approximated to a point closest to the point that can be accessed by the algorithm. (based on point distance)



from Tkinter import Tk as ttk
from tkFileDialog import askopenfilename as askfile
print(" Select the img file.")
ttk().withdraw()  #Tkinter dialog box helps select the file
filename = askfile() 
print(" File Selected : %s" % filename) 
print("")
# This Selects the file name. 



try:
            os.stat("AlgorithmData")
except:
	os.mkdir("AlgorithmData")

base = os.path.basename(filename)

shortfilename = "Image Name - " + os.path.splitext(base)[0]

try:
            os.stat("AlgorithmData/" + shortfilename)
except:
	os.mkdir("AlgorithmData/" + shortfilename)

try:
            os.stat("AlgorithmData/" + shortfilename + "/Info")
except:
	os.mkdir("AlgorithmData/" + shortfilename + "/Info")


from shutil import copyfile
copyfile(filename, "AlgorithmData/" + shortfilename + "/InputImage" + os.path.splitext(base)[1])
# Create Image Directory and all the necessary folders. 




# Load an color image in grayscale
img = cv2.imread(filename,0)
incolorimg = cv2.imread(filename)

copyimg = img
# A copy is created.

height = copyimg.shape[0]
width = copyimg.shape[1]

print(" Specifics : ")
print("Width of the Image : %d " % width)
print("Height of the Image : %d " % height)
# Displaying the Height and width of the image. 


def Draw(image,brushsize,location,choice,color):
	ycor = location[0]
	xcor = location[1]
	for ky in range(ycor-brushsize,ycor+brushsize):
		for kx in range(xcor-brushsize,xcor+brushsize):
			if color != [0,140,255]:
				if ky < height and ky > 0 and kx < width and kx > 0 and copyimg[ky,kx] > 200:
					image[ky,kx] = color
			else:
				if ky < height and ky > 0 and kx < width and kx > 0:
					image[ky,kx] = color

# The Draw function, puts a point on the image with a given brush size. 

def checkConnectedness(point1,point2,gap,type):
	Connected = 1
	if type == 'East':
		for x in range(point1[1],point2[1]):
			if copyimg[point1[0],x] < 200:
				Connected = 0
				return Connected
				break
	if type == 'South':
		for y in range(point1[0],point2[0]):
			if copyimg[y,point1[1]] < 200:
				Connected = 0
				return Connected
				break
	if type == 'SouthEast':
		diagonaltrack = 0
		for x in range(point1[1],point2[1]):
			diagonaltrack = diagonaltrack + 1
			if copyimg[point1[0]+diagonaltrack,x] < 200:
				Connected = 0
				return Connected
				break
	if type == 'NorthEast':
		diagonaltrack = 0
		for x in range(point1[1],point2[1]):
			diagonaltrack = diagonaltrack - 1
			if copyimg[point1[0]+diagonaltrack,x] < 200:
				Connected = 0
				return Connected
				break

	return Connected


# There are 4 types of connectors possible. East, South, (Non Diagonal), and SouthEast, NorthEast, (Diagonals)
# This checks if two points in the algorithm can be connected by a line. 
# They cannot be connected of there is a black obstacle such as a maze wall in between them.

def DrawLine(point1,point2,gap,type,color,reverse):
	if reverse == 0:
		if type == 'East':
			for x in range(point1[1],point2[1]):
				newcolor = color
				Draw(incolorimg,1,(point1[0],x),1,newcolor)


		if type == 'South':
			for y in range(point1[0],point2[0]):
				newcolor = color
				Draw(incolorimg,1,(y,point1[1]),1,newcolor)

		if type == 'SouthEast':
			diagonaltrack = 0
			for x in range(point1[1],point2[1]):
				diagonaltrack = diagonaltrack + 1
				newcolor = color
				Draw(incolorimg,1,(point1[0]+diagonaltrack,x),1,newcolor)

		if type == 'NorthEast':
			diagonaltrack = 0
			for x in range(point1[1],point2[1]):
				diagonaltrack = diagonaltrack - 1
				newcolor = color
				Draw(incolorimg,1,(point1[0]+diagonaltrack,x),1,newcolor)
	if reverse == 1:
		if type == 'East':
			for x in range(point2[1],point1[1]):
				newcolor = color
				Draw(incolorimg,1,(point2[0],x),1,newcolor)

		if type == 'South':
			for y in range(point2[0],point1[0]):
				newcolor = color
				Draw(incolorimg,1,(y,point2[1]),1,newcolor)

		if type == 'SouthEast':
			diagonaltrack = 0
			for x in range(point2[1],point1[1]):
				diagonaltrack = diagonaltrack + 1
				newcolor = color
				Draw(incolorimg,1,(point2[0]+diagonaltrack,x),1,newcolor)

		if type == 'NorthEast':
			diagonaltrack = 0
			for x in range(point2[1],point1[1]):
				diagonaltrack = diagonaltrack - 1
				newcolor = color
				Draw(incolorimg,1,(point2[0]+diagonaltrack,x),1,newcolor)




# This draws lines between points, 8 types of lines to be precise.


Connectors = []

for y in range(2*pointdistance,height-pointdistance,pointdistance):
	for x in range(2*pointdistance,width-pointdistance,pointdistance):
		k = pointdistance
		point = (y,x)
		Draw(incolorimg,plotpointthickness,point,1,[255,191,0])
		if copyimg[y,x] > 200:
			Connectors.append([ (y,x), (y,x+k), (y+k,x+k), (y+k,x), (y-k,x+k)])

# This check for all the possible connections for the points and its immediate neighbours. 

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1000,800)
cv2.imshow('image',incolorimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Displaying the Image with all the points marked. 
# We need to make sure that the points arent spaced too far. they must be comparable to the maze path gap, or lesser. But, not too less for this particular algorithm.


Draw(incolorimg,3,endingposition,1,[0,255,0])
Draw(incolorimg,3,startingposition,1,[0,255,0])
# Marks the Starting and Ending positions. 


print("")
print(" Starting Position ")
print(startingposition)
print(" Ending Position ")
print(endingposition)
print(" ")


# This whole section displays the starting and ending positions. 

cropY = startingposition[0] - 100
cropYplusH =  startingposition[0] + 100
cropX = startingposition[1] - 100
cropXplusH = startingposition[1] + 100


if startingposition[0] - 100 < 1:
	cropY = 1

if startingposition[1] - 100 < 1:
	cropX = 1


if (startingposition[0] + 100) > (height - 1):
	cropYplusH = height - 1


if (startingposition[1] + 100) > (width - 1):
	cropXplusH = width - 1


crop_img = incolorimg[cropY:cropYplusH, cropX:cropXplusH] 
cv2.namedWindow('cropped',cv2.WINDOW_NORMAL)
cv2.resizeWindow('cropped', 800,800)
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
# Displayed Starting position.
cv2.destroyAllWindows()


cropY = endingposition[0] - 100
cropYplusH =  endingposition[0] + 100
cropX = endingposition[1] - 100
cropXplusH = endingposition[1] + 100


if endingposition[0] - 100 < 1:
	cropY = 1

if endingposition[1] - 100 < 1:
	cropX = 1


if (endingposition[0] + 100) > (height - 1):
	cropYplusH = height - 1


if (endingposition[1] + 100) > (width - 1):
	cropXplusH = width - 1



crop_img = incolorimg[cropY:cropYplusH, cropX:cropXplusH] 
cv2.namedWindow('cropped',cv2.WINDOW_NORMAL)
cv2.resizeWindow('cropped', 800,800)
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
# Displayed Ending position. 
cv2.destroyAllWindows()





ConnectorInfo = []
ConnectorInfoDict = {}

for pointdata in Connectors:

	point = pointdata[0]
	#Horizontal Type

	pointH = pointdata[1]
	checkCH = checkConnectedness(point,pointH,pointdistance,'East')

	#Diagonal down Type

	pointD = pointdata[2]
	checkCD = checkConnectedness(point,pointD,pointdistance,'SouthEast')

	#Vertical Type

	pointV = pointdata[3]
	checkCV = checkConnectedness(point,pointV,pointdistance,'South')

	#Diagonal up Type

	pointUP = pointdata[4]
	checkCDup = checkConnectedness(point,pointUP,pointdistance,'NorthEast')

	ConnectorInfo.append([point,checkCH,checkCD,checkCV,checkCDup])
	ConnectorInfoDict[point] = [point,checkCH,checkCD,checkCV,checkCDup]
	

# Checked all the possible connections using checkConnectedness function defined earlier.  
ConnectorData = []


c = 0
totalpoints = (height * width) / float(pointdistance**2) 


import time
init = time.time()
# We time the algorithm to estimate time remaining.

SurroundingData = {}
NumberofUnvisitedNeighbours = {}
NNeighbours = {}
TotalUnvisitedMap = {}
VisitorCount = {}
NewNeighbours = {}
VisitCountContainer0 = []
# Lot of arrays with different purposes. Some are unecessary, but, the previously developed algorithm using Dijkstra, needed them.


for y in range(2*pointdistance,height-pointdistance,pointdistance):
	for x in range(2*pointdistance,width-pointdistance,pointdistance):
		currentposition = (y,x)
		c = c + 1
		if c % 500 == 0:
			print("%f %s done." % ((c * 100 / float(totalpoints)),'%'))
			finaltime = time.time()
			diff = finaltime - init
			timeperiter = diff / 500
			timeremain = timeperiter*(totalpoints - c) / 60
			print(" Time remaining : %d min and %d sec" % (int(timeremain),int((timeremain*60)%60)))
			init = finaltime


		Npoint = (currentposition[0]-pointdistance,currentposition[1])
		NWpoint = (currentposition[0]-pointdistance,currentposition[1]-pointdistance)
		Wpoint = (currentposition[0],currentposition[1]-pointdistance)
		SWpoint = (currentposition[0]+pointdistance,currentposition[1]-pointdistance)
		Spoint = (currentposition[0]+pointdistance,currentposition[1])
		SEpoint = (currentposition[0]+pointdistance,currentposition[1]+pointdistance)
		Epoint = (currentposition[0],currentposition[1]+pointdistance)
		NEpoint = (currentposition[0]-pointdistance,currentposition[1]+pointdistance)

		dir1 = 0
		dir2 = 0
		dir3 = 0
		dir4 = 0
		dir5 = 0
		dir6 = 0
		dir7 = 0
		dir8 = 0

		try:
			info = ConnectorInfoDict[currentposition]
			dir1 = info[1]
			dir8 = info[2]
			dir7 = info[3]
			dir2 = info[4]
		except:
			None
		try:
			info = ConnectorInfoDict[Npoint]
			dir3 = info[3]
		except:
			None
		try:
			info = ConnectorInfoDict[NWpoint]
			dir4 = info[2]
		except:
			None
		try:
			info = ConnectorInfoDict[Wpoint]
			dir5 = info[1]
		except:
			None
		try:
			info = ConnectorInfoDict[SWpoint]
			dir6 = info[4]
		except:
			None

		Data = [currentposition,dir1,dir2,dir3,dir4,dir5,dir6,dir7,dir8]
		DirectionsforNeighbours = [Epoint,NEpoint,Npoint,NWpoint,Wpoint,SWpoint,Spoint,SEpoint]
		Neighbours = []
		for i in range(1,9):
			if Data[i] == 1:
				if copyimg[DirectionsforNeighbours[i-1]] > 200:
					Neighbours.append(DirectionsforNeighbours[i-1])



		if copyimg[currentposition] > 200:
			NNeighbours[currentposition] = Neighbours
			TotalUnvisitedMap[currentposition] = 1
			VisitorCount[currentposition] = 0
			NewNeighbours[currentposition] = []
			VisitCountContainer0.append(currentposition)


# So, we have the Data = [currentposition,dir1,dir2,dir3,dir4,dir5,dir6,dir7,dir8]
# and DirectionsforNeighbours = [Epoint,NEpoint,Npoint,NWpoint,Wpoint,SWpoint,Spoint,SEpoint]
# which means, for a current position, if dir1 = 1, implies that a line can be drawn between the current position and it's East neighbour. 
# dir2 = 0 implies that a line cannot be drawn between the current position and it's NorthEast neighbour. So on..


print(" ")
print(" Totally : %d points." % len(VisitorCount))
print(" ")

import random
currentnode = startingposition
Draw(incolorimg,plotpointthickness+1,currentnode,1,[0,0,255])
BranchNodes = []

foundEndnode = 0
notfound = 0
count = 0
Terminated = 0

VisitCountContainer1 = []
print(" Applying Minimum Tree Algorithm! (Prim's)")

c = 0
totalpoints = len(VisitorCount)


while len(BranchNodes) != 0 or notfound != 1:
	c = c + 1
	if c % 5000 == 0:
		print("About %f %s done." % ((c * 100 / float(totalpoints)),'%'))
		finaltime = time.time()
		diff = finaltime - init
		timeperiter = diff / 5000
		timeremain = timeperiter*(totalpoints - c) / 60
		print(" Time remaining : Less than %d min and %d sec" % (int(timeremain),int((timeremain*60)%60)))
		init = finaltime

	TotalUnvisitedMap[currentnode] = 0
	if currentnode != startingposition:
		Draw(incolorimg,plotpointthickness+1,currentnode,1,[0,191,0])
		if count > 0:
			Draw(incolorimg,plotpointthickness+1,currentnode,1,[0,140,244])
		else:
			if currentnode != startingposition and currentnode != endingposition:
				if VisitorCount[currentnode] == 0:
					VisitCountContainer0.remove(currentnode)
					VisitCountContainer1.append(currentnode)
				if VisitorCount[currentnode] == 1:
					VisitCountContainer1.remove(currentnode)
			VisitorCount[currentnode] = VisitorCount[currentnode] + 1
			
			if previousnode != currentnode:
				NewNeighbours[previousnode].append(currentnode)
				NewNeighbours[currentnode].append(previousnode)
	
	if currentnode == endingposition:
		print(" Ending Found! ")
		Draw(incolorimg,plotpointthickness+1,currentnode,1,[0,0,255])
		break

	notfound = 1
	countunvisitedN = 0
	for neighbour in NNeighbours[currentnode]:
		if TotalUnvisitedMap[neighbour] == 1:
			countunvisitedN = countunvisitedN + 1
		if TotalUnvisitedMap[neighbour] == 1 and notfound == 1:
			chosenNeighbour = neighbour
			notfound = 0


	if countunvisitedN > 0:
		if currentnode != startingposition and currentnode != endingposition:
			if VisitorCount[currentnode] == 0:
				VisitCountContainer0.remove(currentnode)
				VisitCountContainer1.append(currentnode)
			if VisitorCount[currentnode] == 1:
				VisitCountContainer1.remove(currentnode)
		VisitorCount[currentnode] = VisitorCount[currentnode] + 1


	if countunvisitedN > 1:
		BranchNodes.append(currentnode)
	

	if notfound == 1 and len(BranchNodes) > 0:
		breakflag = 0
		CopyofBranchNodes = BranchNodes[:]
		for node in CopyofBranchNodes:
			count = 0
			for neighbour in NNeighbours[node]:
				if TotalUnvisitedMap[neighbour] == 1:
					count = count + 1

			if len(BranchNodes) != 1:
				BranchNodes.remove(node)
			if count > 0:
				chosenNeighbour = node
				breakflag = 1
			
			if breakflag == 1:
				break
	else:
		count = 0


	if notfound == 0:
		cv2.line(incolorimg,(currentnode[1],currentnode[0]),(chosenNeighbour[1],chosenNeighbour[0]),(0,140,255),1)


	if currentnode == chosenNeighbour:
		ActivateTermination = ActivateTermination + 1
	if currentnode != chosenNeighbour:
		ActivateTermination = 0

	previousnode = currentnode
	currentnode = chosenNeighbour

	if ActivateTermination == 9:
		Terminated = 1
		print(" It seems that the ending node cannot be reached, try reducing point distances for clarity. \n If the doesn't help, There's no path that leads to the ending node. \n Sorry. \n (Just saying what Prajwal told me to tell, I really don't give a damn)")
		break



# The Minimal Tree Algorithm is applied and if the point distance is too large, the algorithm cannot reach the ending position and algorithm terminates.

cv2.imwrite("AlgorithmData/" + shortfilename + "/Info/Tree.png", incolorimg)
# Saving the Tree Data.
# Find this file after running the algorithm to see what it means. 


if Terminated == 0:

	for node in VisitCountContainer0:
		if node != startingposition and node != endingposition:
			del VisitorCount[node]
			del NewNeighbours[node]

	# Eliminates all nodes that have no connections. (with no visits in the tree/Isolated points)

	while len(VisitCountContainer1) != 0:
		currentnode = VisitCountContainer1[0]

		VisitorCount[NewNeighbours[currentnode][0]] = VisitorCount[NewNeighbours[currentnode][0]] - 1
		if VisitorCount[NewNeighbours[currentnode][0]] == 1:
			VisitCountContainer1.append(NewNeighbours[currentnode][0])
		if VisitorCount[NewNeighbours[currentnode][0]] == 0:
			if NewNeighbours[currentnode][0] in VisitCountContainer1:
				VisitCountContainer1.remove(NewNeighbours[currentnode][0])
		NewNeighbours[NewNeighbours[currentnode][0]].remove(currentnode)
		del VisitorCount[currentnode]
		del NewNeighbours[currentnode]
		del VisitCountContainer1[0]


	# This delete all the points other than Starting and Ending positions which have only one connection. 
	# And simulateously delete its connection to its neighbours.
	# This step generated more points with single connections and hence the above repeats till there are no points with single connections other than starting and ending nodes.
	

	import pickle

	pickle.dump(VisitorCount, open("AlgorithmData/" + shortfilename + "/Info/VisitCount.p", "wb" ))
	pickle.dump(NewNeighbours, open("AlgorithmData/" + shortfilename + "/Info/NewNeighbours.p", "wb" ))
	# Saving all the Important Data.


	solutioncolorimg = cv2.imread(filename)
	currentposition = endingposition
	while currentposition != startingposition:
		nextposition = NewNeighbours[currentposition][0]
		NewNeighbours[nextposition].remove(currentposition)
		cv2.line(solutioncolorimg,(currentposition[1],currentposition[0]),(nextposition[1],nextposition[0]),(0,140,255),pointdistance/2)
		currentposition = nextposition


	cv2.destroyAllWindows()
	img = cv2.imread(filename)
	cv2.destroyAllWindows()

	opacity = 0.6
	overlaypic = cv2.addWeighted(solutioncolorimg, opacity, img, 1 - opacity, 0)

	# This draws the solution.
	cv2.imwrite("AlgorithmData/" + shortfilename + "/Solution.png", overlaypic)
	cv2.destroyAllWindows()


	import subprocess
	path = os.path.abspath("AlgorithmData/" + shortfilename)
	subprocess.call("explorer " + path, shell=False)

	# Opens the folder.

# Thank you. 