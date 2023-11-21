"""
The Following program is designed to solve Mazes which are given as input in pictoral form. 

Prajwal DSouza
23rd June 2017

This was the first algorithm and was finished on June 25th. 

"""


# gap_factor = 0.28  by default.


import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import time
import os


def PointAt(from_point, direction, distance, max_width, max_height):
    global copyimg, darkthreshold, incolorimg, height, width
    
    if direction == 'N':
        return (max(from_point[0] - distance, 0), from_point[1])
    elif direction == 'S':
        return (min(from_point[0] + distance, max_height), from_point[1])
    elif direction == 'E':
        return (from_point[0], min(from_point[1] + distance, max_width))
    elif direction == 'W':
        return (from_point[0], max(from_point[1] - distance, 0))
    

def Draw(image, brushsize, location, choice, color):
    global copyimg, darkthreshold, incolorimg, height, width
    ycor, xcor = location
    y_range = np.arange(max(ycor - brushsize, 0), min(ycor + brushsize, height))
    x_range = np.arange(max(xcor - brushsize, 0), min(xcor + brushsize, width))

    yy, xx = np.meshgrid(y_range, x_range, indexing='ij')

    if np.array_equal(color, [0, 140, 255]):
        image[yy, xx] = color
    else:
        mask = copyimg[yy, xx] > darkthreshold
        image[yy[mask], xx[mask]] = color
     
def checkConnectedness(point1,point2,gap,type):
    global copyimg, darkthreshold, incolorimg, height, width
    
    if type == 'East':
        # Slice the row from point1 to point2 and check if all values are above the threshold
        return int(np.all(copyimg[point1[0], point1[1]:point2[1]] >= darkthreshold))

    if type == 'South':
        # Slice the column from point1 to point2 and check if all values are above the threshold
        return int(np.all(copyimg[point1[0]:point2[0], point1[1]] >= darkthreshold))

    if type in ['SouthEast', 'NorthEast']:
        # Create an index array for the diagonal
        x_indices = np.arange(point1[1], point2[1])
        if type == 'SouthEast':
            y_indices = np.arange(point1[0], point1[0] + len(x_indices))
        else: # NorthEast
            y_indices = np.arange(point1[0], point1[0] - len(x_indices), -1)

        # Check if all values in the diagonal are above the threshold
        return int(np.all(copyimg[y_indices, x_indices] >= darkthreshold))

    return 1

def DrawLine(point1,point2,gap,type,color,reverse):
	global copyimg, darkthreshold, incolorimg, height, width
    
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

def computeStartAndEndingPositions(points):
    points = np.array(points)
    # Calculate all pairwise distances
    distances = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=2))

    # Set the diagonal to -1 to ignore self-distances
    np.fill_diagonal(distances, -1)

    # Find the indices of the maximum distance
    max_dist_indices = np.unravel_index(np.argmax(distances), distances.shape)

    # Get the points corresponding to these indices
    point1 = points[max_dist_indices[0]]
    point2 = points[max_dist_indices[1]]

    return point1, point2

def solve_maze(filename, gap_factor = 0.28):
    global copyimg, darkthreshold, incolorimg, height, width
    print(" File Selected : %s" % filename) 
    # This Selects the file name. 

    # Load an color image in grayscale
    img = cv2.imread(filename,0)
    incolorimg = cv2.imread(filename)

    copyimg = img
    # A copy is created.


    # Another copy of image
    analysisCopy = copyimg

    # This is the image that will be analyzed.

    # The goal is to grazes across a line in the image and measures the variation of intensity of the pixels across the line. The line is chosen at random
    # The line is chosen at random, and the variation of intensity is measured.

    imageheight = copyimg.shape[0]
    imagewidth = copyimg.shape[1]


    gapAverages = []

    for lineTrial in range(0,1000):
        
        # Get intensity values of all pixels in the line
        lineType = random.choice(['vertical', 'horizontal'])
        if lineType == 'vertical':
            randomLineColumn = random.randint(0, imagewidth - 1)
            intensityValues = copyimg[:,randomLineColumn]
        else:
            randomLineRow = random.randint(0, imageheight - 1)
            intensityValues = copyimg[randomLineRow,:]

        # Finding average period (window for which the intensity remains high) with numpy 
        threshold = np.mean(intensityValues)
        

        high_intensity = np.where(intensityValues > threshold)[0]
        runs = np.diff(high_intensity)
        variationPoints = np.where(runs > 1)[0]
        variationPoints = variationPoints[1:-1]
        gaps = np.diff(variationPoints)
        # print("Gaps", gaps)
        if len(gaps) < 5:
            smallest_gaps = gaps
        else:
            sorted_arr = np.sort(gaps)
            smallest_gaps = sorted_arr[:5]
        average_gap = np.mean(smallest_gaps)
        

        # Check if type is nan 
        
        if not np.isnan(average_gap):
            gapAverages.append(average_gap)
            

    print("Average gap", np.mean(gapAverages))

    # The first three parameters must be changed based on the maze. 

    pointdistance = int(np.mean(gapAverages) * gap_factor)
    print("Point Distance :",pointdistance)
    # If this value isn't set properly, there could be errors.
    # This reduces the number of points on the image to be analyzed. 
    # The Algorithm will move through only those points avoiding the obstacles. 
    # Every point at distance equal to pointdistance is chosen for analysis. 


    plotpointthickness = 2
    # This is about how thick the line of the solution must be. 


    darkthreshold = 200







    height = copyimg.shape[0]
    width = copyimg.shape[1]

    print(" Specifics : ")
    print("Width of the Image : %d " % width)
    print("Height of the Image : %d " % height)
    # Displaying the Height and width of the image. 
    
    
    
    Connectors = []


    enterExitPoints = []

    for y in range(2*pointdistance,height-pointdistance,pointdistance):
        for x in range(2*pointdistance,width-pointdistance,pointdistance):
            k = pointdistance
            point = (y,x)
            Draw(incolorimg,plotpointthickness,point,1,[255,191,0])
            westCheck = checkConnectedness(point,(y, width - 1),k,'East')
            eastCheck = checkConnectedness((y, 1),point,k,'East')
            southCheck = checkConnectedness(point,(height - 1, x),k,'South')
            northCheck = checkConnectedness((1, x), point,k,'South')
            total = westCheck + eastCheck + southCheck + northCheck
    
            if copyimg[y,x] > darkthreshold and (westCheck + eastCheck == 2 or northCheck + southCheck == 2 or total >= 2) == False:
                Connectors.append([ (y,x), (y,x+k), (y+k,x+k), (y+k,x), (y-k,x+k)])
            else:
                Draw(incolorimg,plotpointthickness,point,1,[55,191,0])
    
            # Enter/Exit points
            criteria = (westCheck + eastCheck + southCheck + northCheck == 1) 
            if copyimg[y,x] > darkthreshold and criteria == True:
                enterExitPoints.append(point)
    
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1000,800)
    cv2.imshow('image',incolorimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    startAndExit = computeStartAndEndingPositions(enterExitPoints)
    startingposition = startAndExit[0]
    endingposition = startAndExit[1]

    print(" Starting Position : ")
    print(startingposition)

    print(" Ending Position : ")
    print(endingposition)

    # startingposition = (150, 150)
    # endingposition = (750, 1140)


    # startingposition = (570,740)
    # endingposition = (1500, 1600)

    # startingposition = (210,2246)
    # endingposition = (2799,3950)
    # The Maze must be marked, like in the example maze. Ending and starting of the maze must be closed.
    # But, the starting positions and ending positions must be specified and marked in the image for simplification. 




    startingposition = (startingposition[0] - (startingposition[0]%pointdistance),startingposition[1] - (startingposition[1]%pointdistance))
    endingposition = (endingposition[0] - (endingposition[0]%pointdistance),endingposition[1] - (endingposition[1]%pointdistance))
    # Starting and ending position is approximated to a point closest to the point that can be accessed by the algorithm. (based on point distance)





    Draw(incolorimg,3,endingposition,1,[0,255,0])
    Draw(incolorimg,3,startingposition,1,[0,255,0])

    print("")
    print(" Starting Position ")
    print(startingposition)
    print(" Ending Position ")
    print(endingposition)
    print(" ")

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
            if copyimg[currentposition] > darkthreshold:
                ConnectorData.append(Data)
                SurroundingData[currentposition] = Data



    # So, we have the Data = [currentposition,dir1,dir2,dir3,dir4,dir5,dir6,dir7,dir8]
    # and DirectionsforNeighbours = [Epoint,NEpoint,Npoint,NWpoint,Wpoint,SWpoint,Spoint,SEpoint]
    # which means, for a current position, if dir1 = 1, implies that a line can be drawn between the current position and it's East neighbour. 
    # dir2 = 0 implies that a line cannot be drawn between the current position and it's NorthEast neighbour. So on..


    print(" ")
    print(" Totally : %d points." % len(ConnectorData))





    allnodes = []
    distances = []
    pred = []

    DictionaryforNodes = {}


    for info in ConnectorData:
        point = info[0]
        allnodes.append(point)
        distances.append(float('inf'))
        DictionaryforNodes[point] = float('inf')
        pred.append('Nil')


    infinity = float('inf')


    visitednodes = []

    index = allnodes.index(startingposition)
    distances[index] = 0
    DictionaryforNodes[startingposition] = 0

    DictonaryUnvisitedNodes = DictionaryforNodes



    c = 0
    totalpoints = len(ConnectorData)

    currentnode = startingposition

    NeighbourData = {}

    print("")
    print(" Starting Djikstra! ")
    print("")

    TimeSticks = 0

    TimeData = []
    IterData = []

    showevery = 50

    if totalpoints > 5000:
        showevery = int(totalpoints / 100)
    while len(visitednodes) != len(allnodes):
        c = c + 1
        if c % showevery == 0:
            print("%f %s done." % ((c * 100 / float(totalpoints)),'%'))
            finaltime = time.time()
            diff = finaltime - init
            timeperiter = diff / showevery
            timeremain = timeperiter*(totalpoints - c) / 60
            print(" Time remaining : %d min and %d sec" % (int(timeremain),int((timeremain*60)%60)))
            init = finaltime
            TimeData.append(init)
            IterData.append(c)
            TimeSticks = 0
        if TimeSticks == 1:
            TimeStick1 = time.time()

        Draw(incolorimg,3,currentnode,1,[0,191,0])
        
        UnvisitedNeighbours = []
        Cost = []

        Npoint = (currentnode[0]-pointdistance,currentnode[1])
        NWpoint = (currentnode[0]-pointdistance,currentnode[1]-pointdistance)
        Wpoint = (currentnode[0],currentnode[1]-pointdistance)
        SWpoint = (currentnode[0]+pointdistance,currentnode[1]-pointdistance)
        Spoint = (currentnode[0]+pointdistance,currentnode[1])
        SEpoint = (currentnode[0]+pointdistance,currentnode[1]+pointdistance)
        Epoint = (currentnode[0],currentnode[1]+pointdistance)
        NEpoint = (currentnode[0]-pointdistance,currentnode[1]+pointdistance)

        directions = [Epoint,NEpoint,Npoint,NWpoint,Wpoint,SWpoint,Spoint,SEpoint]

        Neighbours = []
        if TimeSticks == 1:
            TimeStick2 = time.time()

        info = SurroundingData[currentnode]

        if TimeSticks == 1:
            TimeStick3 = time.time()
        for i in range(1,9):
            if TimeSticks == 1:
                TimeStick4 = time.time()
            if info[i] > 0:
                (y,x) = directions[i-1]
                if (y,x) in allnodes:
                    Neighbours.append((y,x))
                    
                if (y,x) not in visitednodes and (y,x) in allnodes:
                    UnvisitedNeighbours.append((y,x))
                    Cost.append(info[i])
                    indexcurrent = allnodes.index(currentnode)
                    index = allnodes.index((y,x))
                    if(distances[index] > (info[i] + distances[indexcurrent])):
                        distances[index] = info[i] + distances[indexcurrent]
                        DictonaryUnvisitedNodes[(y,x)] = distances[index]
                        pred[index] = currentnode
            if TimeSticks == 1:
                TimeStick5 = time.time()


        if TimeSticks == 1:
            TimeStick6 = time.time()


        visitednodes.append(currentnode)
        DictonaryUnvisitedNodes.pop(currentnode, 0)
        


        NeighbourData[currentnode] = Neighbours



        if len(DictonaryUnvisitedNodes) != 0:
            currentnode = min(DictonaryUnvisitedNodes, key=DictonaryUnvisitedNodes.get)





        if TimeSticks == 1:
            TimeStick7 = time.time()



        if TimeSticks == 1:
            TimeStick8 = time.time()
            TimeSticks = 0
            print(" Printing All Time Sticks.")
            print(" Loop 1 : %f" % (TimeStick2 - TimeStick1))
            print(" Loop 2 : %f" % (TimeStick3 - TimeStick2))
            print(" ILoop 3 : %f" % (TimeStick5 - TimeStick4))
            print(" Loop 4 : %f" % (TimeStick6 - TimeStick3))
            print(" Loop 5 : %f" % (TimeStick7 - TimeStick6))
            print(" Loop 6 : %f" % (TimeStick8 - TimeStick7))

        



    # Dump the Data
    # import pickle

    # pickle.dump(allnodes, open("AlgorithmData/" + shortfilename + "/Info/AllNodeData.p", "wb" ))
    # pickle.dump(distances, open("AlgorithmData/" + shortfilename + "/Info/Distances.p", "wb" ))
    # pickle.dump(pred, open("AlgorithmData/" + shortfilename + "/Info/Preds.p", "wb" ))
    # pickle.dump(NeighbourData, open("AlgorithmData/" + shortfilename + "/Info/NeighbourData.p", "wb" ))

    # print(" Saved all Important Data. ")


    #Draw the tree

    for index in range(0,len(pred)):
        node = pred[index]
        othernode = allnodes[index]
        if node != 'Nil':
            cv2.line(incolorimg,(othernode[1],othernode[0]),(node[1],node[0]),(0,140,250),2)



    cv2.destroyAllWindows()

    cv2.imwrite("AlgorithmData/" + shortfilename + "/Info/TreeData.png", incolorimg)


    # To Draw the solution

    solutioncolorimg = cv2.imread(filename)
    currentposition = endingposition

    while currentposition != startingposition:
        index = allnodes.index(currentposition)
        distance = distances[index]

        indices = [i for i, x in enumerate(distances) if x == (distance - 1)]

        for index in indices:
            node = allnodes[index]
            if node in NeighbourData[currentposition]:
                cv2.line(solutioncolorimg,(currentposition[1],currentposition[0]),(node[1],node[0]),(0,140,255),int(pointdistance/2))
                currentposition = node
                break



    img = cv2.imread(filename)

    opacity = 0.6
    overlaypic = cv2.addWeighted(solutioncolorimg, opacity, img, 1 - opacity, 0)

    cv2.imwrite("AlgorithmData/" + shortfilename + "/Solution.png", overlaypic)

    cv2.destroyAllWindows()



solve_maze('Test Images\simple-maze.jpg', gap_factor = 0.28)