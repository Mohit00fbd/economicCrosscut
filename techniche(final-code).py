import heapq
import numpy as np
import cv2
import math
import serial
import time

ser=serial.Serial(5)

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements)==0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority,item))

    def get(self):
        return heapq.heappop(self.elements)[1]

class SquareGrid:
    def __init__(self, width, height):
        self.width = numberOfNodesInOneEdge
        self.height = numberOfNodesInOneEdge
    def in_bounds(self, id):
        (x,y) = id
        return 0<= x < self.width and 0<= y <self.height

    def neighbors(self, id):
        (x,y) = id
        results = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)]
        results = filter(self.in_bounds, results)
        #print (x,y), results
        return results

cap = cv2.VideoCapture(1)

ret, img = cap.read()
    #img=cv2.imread('day1.jpg')
kernel = np.ones((7, 7),np.uint8)
dst = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
yuv = cv2.cvtColor(dst, cv2.COLOR_RGB2YUV)
lower = np.array([210, 124, 124]) 
upper = np.array([255, 150, 150])
mask = cv2.inRange(yuv, lower, upper)
              
ret, thresh = cv2.threshold(mask, 127, 255, 0)
_, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(yuv,contours,-1,(0,0,255),3)
j=0
whitecontourslist={}
for i in range(len(contours)):
                M = cv2.moments(contours[i])
                if all([(int(M['m00']))!= 0 ,25<int(cv2.arcLength(contours[i],True))<72, (cv2.contourArea(contours[i]))>90]) :
                                     cx = int(M['m10']/M['m00'])
                                     cy = int(M['m01']/M['m00'])
          
                                     #cv2.circle(mask, (cx,cy), 5, (0,0,0), -1)         
                                     whitecontourslist[j]=(i)    
                                     j+=1
print j
numberOfNodesInOneEdge = int(math.sqrt(j))
print whitecontourslist                                                 

n=0
wccl=[]
whiteCentroidCoordinates = {}
for j in range(numberOfNodesInOneEdge):
   for i in range(numberOfNodesInOneEdge):
          M= cv2.moments(contours[whitecontourslist[n]])
          centreWhiteX = int(M['m10']/M['m00'])
          centreWhiteY = int(M['m01']/M['m00'])
          whiteCentroidCoordinates [i,j]= (centreWhiteX, centreWhiteY)
          wccl.append((centreWhiteX,centreWhiteY))
          
          #cv2.circle(mask, whiteCentroidCoordinates[i,j], 5, (0,0,0), -1)
          
          n+=1
print n                        
print n
print n
print n
print n
print whiteCentroidCoordinates


    #blur = cv2.GaussianBlur(img,(5,5),0)
    #lower = np.array([180,170,215])
    #upper = np.array([235,230,235])
    #mask1 = cv2.inRange(blur, lower, upper)
    #ret, thresh1 =  cv2.threshold(mask1,127,255,0)
'''k=0
i=0
_, contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print len(contours)
whitecontourslist = {}
for i in range(len(contours)): 
      M = cv2.moments(contours[i])
      if all([int(M['m00'])!= 0, 25<cv2.arcLength(contours[i],True)<70 and cv2.contourArea(contours[i])>50]):
        #cx = int(M['m10']/M['m00'])
        #cy = int(M['m01']/M['m00'])
        #print "Centroid = ", cx, ", ",cy
        #cv2.circle(mask1, (cx,cy), 5, (0,0,0), -1)
        #print cv2.contourArea(contours[i])
        #print cv2.arcLength(contours[i],True)
        whitecontourslist[k]=(i)    
        k+=1
print k
numberOfNodesInOneEdge = int(math.sqrt(k))
print 'numberOfNodesInOneEdge =', int(math.sqrt(k))
print whitecontourslist

k=0
whiteCentroidCoordinates = {}
for j in range(numberOfNodesInOneEdge):
   for i in range(numberOfNodesInOneEdge):
      whiteMoment = cv2.moments(contours[whitecontourslist[k]])
      centreWhiteX = int(whiteMoment['m10']/whiteMoment['m00'])
      centreWhiteY = int(whiteMoment['m01']/whiteMoment['m00'])
      whiteCentroidCoordinates [i,j]= (centreWhiteX, centreWhiteY)
      print '(',i,',', j,')', whiteCentroidCoordinates[i,j]
      k+=1'''

###############################
#yellow
###############################
#img=cv2.imread('still.jpg')
#blur = cv2.GaussianBlur(img,(5,5),0)
param1 = [180,80,90]
param2 = [255,124,133]

lower = np.array(param1)
upper = np.array(param2)
mask1 = cv2.inRange(yuv, lower, upper)
ret, thresh1 =  cv2.threshold(mask1,127,255,0)

yellowcontourlist={}
k=0
i=0
_, contoursy, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print len(contoursy)
for i in range(len(contoursy)):
     #cv2.drawContours(yuv,contoursy,i,(255,255,255),3)
     M = cv2.moments(contoursy[i])  
     if int(M['m00'])!= 0 and cv2.contourArea(contoursy[i])>500:
       yellowcontourlist[k]=(i) 
       k+=1
yc = []
j=0
for j in range(k):
   M= cv2.moments(contoursy[yellowcontourlist[j]]) 
   cx = int(M['m10']/M['m00'])
   cy = int(M['m01']/M['m00'])
   yc.append((cx,cy))              
print 'coordinates of yellow lines'
print(yc)     
     
print k
###########################
#red
###########################
param1 = [130,80,135]
param2 = [255,195,225]

lower = np.array(param1)
upper = np.array(param2)
mask1 = cv2.inRange(yuv, lower, upper)
ret, thresh1 =  cv2.threshold(mask1,127,255,0)

redcontourlist={}
k=0
i=0
_, contoursr, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print len(contoursr)
for i in range(len(contoursr)): 
      M = cv2.moments(contoursr[i])
      if int(M['m00'])!= 0 and cv2.contourArea(contoursr[i])>500:
          redcontourlist[k]=(i)
          k+=1

rc = []
j=0
for j in range(k):
   M= cv2.moments(contoursr[redcontourlist[j]]) 
   cx = int(M['m10']/M['m00'])
   cy = int(M['m01']/M['m00'])
   rc.append((cx,cy))              
print 'coordinates of red lines'
print(rc)               
print k        
#############################
#green
#############################
param1 = [140,100,0]
param2 = [225,223,108]

lower = np.array(param1)
upper = np.array(param2)
mask1 = cv2.inRange(yuv, lower, upper)
ret, thresh1 =  cv2.threshold(mask1,127,255,0)


greencontourlist={}
k=0
i=0
_, contoursg, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print len(contoursg)
for i in range(len(contoursg)):
     #cv2.drawContours(mask1,contoursg,i,(255,255,255),3)
     M = cv2.moments(contoursg[i])  
     if int(M['m00'])!= 0 and cv2.contourArea(contoursg[i])>400:
       greencontourlist[k]=(i) 
       k+=1
gc = []
j=0
for j in range(k):
   M= cv2.moments(contoursg[greencontourlist[j]]) 
   cx = int(M['m10']/M['m00'])
   cy = int(M['m01']/M['m00'])
   gc.append((cx,cy))              
print 'coordinates of green lines'
print(gc)     
     
print k
##############################
#blue
##############################
param1 = [0,139,110]
param2 = [255,184,155]

lower = np.array(param1)
upper = np.array(param2)
mask = cv2.inRange(yuv, lower, upper)
ret, thresh =  cv2.threshold(mask,127,255,0)



bluecontourlist={}
k=0
i=0
_, contoursb, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print len(contoursb)
for i in range(len(contoursb)):
     #cv2.drawContours(mask,contours,i,(255,255,255),3)
     M = cv2.moments(contoursb[i])  
     if int(M['m00'])!= 0 and cv2.contourArea(contoursb[i])>200:
       bluecontourlist[k]=(i) 
       k+=1
bc = []
j=0
for j in range(k):
   M= cv2.moments(contoursb[bluecontourlist[j]]) 
   cx = int(M['m10']/M['m00'])
   cy = int(M['m01']/M['m00'])
   cv2.circle(mask, (cx,cy), 5, (0,0,0), -1)
   bc.append((cx,cy))              
print 'coordinates of blue lines'
print(bc)     
     
print k
##############################
def cost((a,b),(c,d)):
    (e,f) =  int((a+c)/2), int((b+d)/2)
    print 'ef'
    print (e,f)
    z=[]
    z=[(p,q) for p in range(e-25, e+25) for q in range(f-25, f+25)]
    for item in z:
        if item in rc:
            return 25
        elif item in bc:
            return 20
        elif item in gc:
            return 15
        elif item in yc:
            return 10
    return 1000
    
####################
def dijkstra_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from ={}
    cost_so_far= {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        """if current == goal:
            break"""

        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + cost(whiteCentroidCoordinates[current],whiteCentroidCoordinates[next])
            #print cost(wccl[current],whiteCentroidCoordinates[next])
            
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost
                frontier.put(next, priority)
                came_from[next] = current
    return came_from, cost_so_far

def reconstruct_path(came_from, start, goal):
    current = goal
    path = [current]
    while current!=start:
        current = came_from[current]
        path.append(current)
        cv2.circle(img,whiteCentroidCoordinates[current], 10, (255,255,255), -1)
    path.reverse()
    return path

start = (0,0)
goal = (numberOfNodesInOneEdge-1, numberOfNodesInOneEdge-1)
#start = (0,4)
#goal = (4,0)
graph = SquareGrid(numberOfNodesInOneEdge, numberOfNodesInOneEdge)
came_from, cost_so_far = dijkstra_search(graph, start, goal)
print "CAME FROM"
print came_from, "\n"
print "COST SO FAR"
print cost_so_far

path = reconstruct_path(came_from, start, goal)
print "PATH"
print path
print 'shortest path coooooooost'
print cost_so_far[goal]
print len(path)
print whiteCentroidCoordinates[0,0]
print whiteCentroidCoordinates[0,1]
print path[0]
cv2.circle(img,whiteCentroidCoordinates [goal], 10, (255,255,255), -1)
cv2.imshow('img', img)
cv2.waitKey(0)
########################################################
########################################################
res={}
print "directions for bot"
def ans_pat(pac1,pac2,current):
 for j in range(numberOfNodesInOneEdge):
  for i in range(numberOfNodesInOneEdge):
   if whiteCentroidCoordinates[i,j]==pac1:
 #if pac1 in [(x,y) for x in range(numberOfNodesInOneEdge) for y in range ((numberOfNodesInOneEdge))] : 
     ccx1=i
     ccy1=j
 for j in range(numberOfNodesInOneEdge):
  for i in range(numberOfNodesInOneEdge):
   if whiteCentroidCoordinates[i,j]==pac2:
 #if pac2 in [(x,y) for x in range(numberOfNodesInOneEdge) for y in range ((numberOfNodesInOneEdge))] : 
     ccx2=i
     ccy2=j
 resx=ccx2-ccx1
 resy=ccy2-ccy1
 if resx == 1:
  ans=6
  if(ans==current):
     #print 'move st.'
     res[z]=8
     
  else:
     print 'rotate 90* leftwards & then move st.'
     res[z]=6
     
#current=4
  return 6

 elif resx == -1:
  ans=4
  if(ans==current):
     print 'move st.'
     res[z]=8
      
  else:
     print 'rotate 90* rightwards & then move st.'
     res[z]=4
     
#current=6
  return 4

 elif resy == 1:
  ans=8
  if(ans==current):
     print 'move st.'
     res[z]=8
     
  elif(current==6):
     print 'rotate 90* rightwards & then move st.'
     res[z]=4
     
  elif(current==4):
     print 'rotate 90* leftwards & then move st.'
     res[z]=6
     
  else:
     print 'move st.'
     res[z]=8
     
#current=8
  return 8




current=0
for z in range(len(path)-1):        

  current=ans_pat(whiteCentroidCoordinates [path[z]],whiteCentroidCoordinates [path[z+1]],current)
  #print ins

print res

ser.write("8")
V=0
#####################################################################################################3
while(1):
     
   ret, frame = cap.read()
   kernel = np.ones((5,5),np.uint8)
   
   yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
   dst = cv2.morphologyEx(yuv, cv2.MORPH_OPEN, kernel)
   #if ret == False :
   #          break
   #else:
   lower = np.array([205, 105, 105]) 
   upper = np.array([225, 129, 129])
   mask = cv2.inRange(dst, lower, upper)
        
   ret, thresh = cv2.threshold(mask, 127, 255, 0)
   _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   #cv2.drawContours(yuv,contours,-1,(0,0,255),3)

   H=0
   whitecontourslist={}
   for L in range(len(contours)):
         M = cv2.moments(contours[L])
         if all([(int(M['m00']))!= 0, (cv2.contourArea(contours[L]))>900]) :
                 #cv2.drawContours(yuv,contours,L,(0,0,255),3)
                 CX = int(M['m10']/M['m00'])
                 CY = int(M['m01']/M['m00'])
                 #cv2.circle(mask, (CX,CY), 10, (255,255,255), 6) 
                 whitecontourslist[H]=(L)    
                 H+=1

                 W=[]
                 W=[(p,q) for p in range(CX-25,CX+25) for q in range(CY-25,CY+25)]
                 for item in W:
                    if item in wccl:
                          print 'stop'
                          ser.write("5")
                          time.sleep(2)
                          if res[V]==8:
                                ser.write("8")
                          elif res[V]==6:
                                ser.write("6")
                          else :
                                ser.write("4")
                          print 'next instruction given'
                          time.sleep(3.6)
                          V+=1
                          CX=0
                          CY=0

   print H
   
   #numberOfNodesInOneEdge = int(math.sqrt(H))
   #if H<25:
     # print 'found'     
   #ser.write("5")
   #ser.close()
 
   #cv2.imshow('mask', mask)
   #cv2.imshow('yuv', yuv)
   #k = cv2.waitKey(1) & 0xFF
   #if k == 27 :
   #    break   

   
   cv2.imshow('mask1', mask)
   cv2.waitKey(10)
                      

cap.release()    
cv2.destroyAllWindows()
