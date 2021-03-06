import cv2
import numpy as np
import imutils
import copy
import math

class Query_card:
    """Structure to store information about cards in the camera image."""

    def __init__(self):
        self.contour = [] # Contour of card
        self.width, self.height, self.x, self.y = 0, 0, 0, 0 # Width and height of card
        self.corner_pts = [] # Corner points of card
        self.center = [] # Center point of card
        self.warp = None # 300x300, warped image
        self.key_pts = None #Not currently used. Sift Points?
        self.descriptors = None #Same as above
        self.approx = [] #The point approximation

def preprocess_image(image,color=None):
  """Take video frame and prepare it to find contours (threshold, edge detection, or similar)"""
  if color == "red":
    gray = image[:,:,2]
    limit = 110
  elif color == "blue":
    gray = image[:,:,0]
    limit = 110
  elif color == "green":
    gray = image[:,:,1]
    limit = 110
  else:
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray,(1,1),1000)
    limit = 50

  blue = image[:,:,0]
  green = image[:,:,1]
  red = image[:,:,2]


  #blur = cv2.GaussianBlur(gray,(5,5),0)

  #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  #clahe_output = clahe.apply(gray)
  #thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
  #thresh = cv2.Canny(image,10,150,apertureSize=3)

  #Najma's thresholding below
  

  flag, thresh= cv2.threshold(gray,limit,255, cv2.THRESH_BINARY)#2nd arguement 50 works well

  #Display the image being used to find contours. Comment out for final display  
  cv2.imshow('canny',thresh)
  key = cv2.waitKey(1) & 0xFF
  if key == ord("q"):
    cam_quit = 1  

  return thresh

def e_distance(pt1,pt2):
  return math.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)

def preprocess_card(contour, image):
  """Use contour (or contours) of card to make a Query_card"""
  qCards = []
  contours = []
  peri = cv2.arcLength(contour,True)
  approx = cv2.approxPolyDP(contour,0.02*peri,True)
  cards = []
  distance = 100000
  pt = -1
  direction = 1
  #Check if the outline is a single card or multiple overlapping cards
  if len(approx) > 7 and len(approx)%4 is 0:
    corners = len(approx)
    overlappedCards = corners//4
    for i in range(overlappedCards):
      contours.append([])    
    #Find the intersection of two cards
    for i in range(corners//2):
      opposite = (i + corners//2)%corners
      temp_dist = e_distance(approx[i][0],approx[opposite][0])
      if temp_dist < distance:
        distance = temp_dist
        pt = i
    old_dist = 0
    index = 0
    #Add the points to each card
    for i in range(overlappedCards*2):
      pt1 = (pt + 2*i)%corners
      pt2 = (pt1 + 1)%corners
      pt3 = (pt2 + 1)%corners
      new_dist = e_distance(approx[pt1][0],approx[pt3][0])
      if new_dist > old_dist or old_dist is 0:
        direction *= -1
      else:
        index += direction
        if index < 0:
          index = overlappedCards-1
        elif index >= overlappedCards:
          index = 0
      contours[index].append(approx[pt1])
      contours[index].append(approx[pt2])
      contours[index].append(approx[pt3])
      old_dist = new_dist
  else:
    #If it's a single card, just append it
    size = cv2.contourArea(approx)
    #if size > 7500:
    #  y,x,w,h = cv2.boundingRect(approx)
    #  temp = image[x:x+w,y:y+h]
    #  gray = cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
    #  dummy,thresh = cv2.threshold(gray,175,255, cv2.THRESH_BINARY)
    #  cv2.imshow("big cards",thresh)
    #  cv2.waitKey(1)
    contours.append(cv2.approxPolyDP(contour,0.12*peri,True))
  for contour in contours:
    contour = np.array(contour)
    qCard = Query_card()
    qCard.contour = np.array(contour)
    peri = cv2.arcLength(np.array(contour),True)

    pts = approx
    s = np.sum(pts, axis = 2)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    cont = np.copy(approx)
    qCard.approx = np.array([tl,tr,bl,br])
    pts = np.float32(approx)
    qCard.corner_pts = pts


    y,x,w,h = cv2.boundingRect(contour)
    qCard.width, qCard.height, qCard.y, qCard.x = w,h,x,y
  
    if h/w > 2 or h/w < 1/3:
      pass #return None

    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    qCard.center = [cent_x, cent_y]
    if len(pts) >= 4:
      qCard.warp = flattener(image, pts, w, h)
      #orb =  cv2.ORB_create()
      qCard.tracker = cv2.TrackerKCF.create()
      qCard.tracker.init(image, (y,x,w,h))
      #qCard.key_pts, qCard.descriptors = orb.detectAndCompute(qCard.warp,None)
      qCards.append(qCard)
  return qCards

def flattener(image, pts, w, h):
  '''Get a color image of the card to be identified'''
  temp_rect = np.zeros((4,2),dtype="float32")
  
  #Determine points and how best to warp
  tl = pts[0]
  br = pts[2]
  tr = pts[1]
  bl = pts[3]

  if w <= 0.8*h:
    temp_rect[0] = tl
    temp_rect[1] = tr
    temp_rect[2] = br
    temp_rect[3] = bl

  if w >= 1.2*h:
    temp_rect[0] = bl
    temp_rect[1] = tl
    temp_rect[2] = tr
    temp_rect[3] = br

  if w > 0.8*h and w < 1.2*h:
    if pts[1][0][1] <= pts[3][0][1]:
      temp_rect[0] = pts[1][0]
      temp_rect[1] = pts[0][0]
      temp_rect[2] = pts[3][0]
      temp_rect[3] = pts[2][0]

    if pts[1][0][1] > pts[3][0][1]:
      temp_rect[0] = pts[0][0]
      temp_rect[1] = pts[3][0]
      temp_rect[2] = pts[2][0]
      temp_rect[3] = pts[1][0]

  maxWidth = 300
  maxHeight = 300

  # Warp image to maxWidth and maxHeight
  dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
  M = cv2.getPerspectiveTransform(temp_rect,dst)
  warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
  #warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
  
  return warp

def find_cards(thresh_image):
  '''Find cards in the provided image'''

  #The maximum area that could be a card.
  CARD_MAX_AREA = 15500
  #The minimum area that could be a card.
  CARD_MIN_AREA = 4200

  #Get countours from the image
  dummy,cnts,hier = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  index_sort = sorted(range(len(cnts)),key=lambda i : cv2.contourArea(cnts[i]),reverse=True)

  #If there are no contours, return
  if len(cnts) == 0:
    return [], []

  cnts_sort = []
  hier_sort = []
  cnt_is_card = np.zeros(len(cnts),dtype=int)
  
  for i in index_sort:
    cnts_sort.append(cnts[i])
    hier_sort.append(hier[0][i])

  #Check through the contours to find rectangular contours within the correct area
  for i in range(len(cnts_sort)):
    size = cv2.contourArea(cnts_sort[i])
 
    if size < CARD_MAX_AREA and size > CARD_MIN_AREA:
      peri = cv2.arcLength(cnts_sort[i],True)
      approx = cv2.approxPolyDP(cnts_sort[i],0.12*peri,True)
      if len(approx) == 4:
        cnt_is_card[i] = 1

  return cnts_sort, cnt_is_card

def compare_cards(card1, card2):
  inside = is_same_card(card1,card2)
  if inside > 0:
    return inside
  heightDiff = card1.height - card2.height
  widthDiff = card1.width - card2.width
  xDiff = card1.x - card2.x
  yDiff = card1.y - card2.y
  diff = np.max(np.abs([heightDiff,widthDiff,xDiff,yDiff]))
  if diff < 20:
    return 1
  return 0

def contains(card1, card2):
  '''Check if card 2 contains card 1 by comparing corner points'''

  if card1.y < card2.y or card1.y+card1.height > card2.y+card2.height:
    return False
  if card1.x < card2.x or card1.x+card1.width > card2.x+card2.width:
    return False
  return True

def is_same_card(card1, card2):
  '''Determine if two cards are the same card'''
  pts1 = card1.approx
  pts2 = card2.approx

  if contains(card1, card2):
    return 2
  if contains(card2, card1):
    return 1

  return 0

def compare_all_cards(all_cards, new_cards, image):

  #Check if any of the new cards found are duplicates of each other and only keep one
  temp = []
  keepme = False
  for i in range(len(new_cards)):
    #print("new_cards",len(new_cards))
    keepme = True
    for j in range(len(new_cards)):
      if i < j:
        val = compare_cards(new_cards[i],new_cards[j])
        if val is 1:
          new_cards[j] = new_cards[i]
          keepme = False
          break
        if val is 2:
          keepme = False
          break
    if keepme:
      temp.append(new_cards[i])
      keepme = False
  if keepme:
    temp.append(new_cards[-1])
  new_cards = temp

  #Of the old cards, only keep the ones we still have track of
  temp = []
  for i in range(len(all_cards)):
    #print("track cards",len(all_cards))
    ok, bbox = all_cards[i].tracker.update(image)
    if ok:
      all_cards[i].x,all_cards[i].y,all_cards[i].width,all_cards[i].height = bbox
      temp.append(all_cards[i])
    else:
      print("Lost card")
  all_cards = temp

  for i in range(len(new_cards)):
    keepme = True
    for j in range(len(all_cards)):
      val = compare_cards(new_cards[i],all_cards[j])
      if val is 2:
        keepme = False
        break
      if val is 1:
        keepme = False
        #all_cards[j] = new_cards[i]
    if keepme:
      all_cards.append(new_cards[i])

  temp = []
  keepme = False
  for i in range(len(all_cards)):
    #print("new_cards",len(new_cards))
    keepme = True
    for j in range(len(all_cards)):
      if i < j:
        val = compare_cards(all_cards[i],all_cards[j])
        if val is 1:
          all_cards[j] = all_cards[i]
          keepme = False
          break
        if val is 2:
          keepme = False
          break
    if keepme:
      temp.append(all_cards[i])
      keepme = False
  if keepme:
    temp.append(all_cards[-1])
  all_cards = temp
  return new_cards
  return all_cards

def get_contour(q):
  '''Create a contour from a queryCard'''
  return np.array([[q.x,q.y],[(q.x+q.width),q.y],[(q.x+q.width),(q.y+q.height)],[q.x,(q.y+q.height)]],dtype=np.int32)

all_cards = []

cap = cv2.VideoCapture("../2018 Magic World Championship Finals.mp4")

cap.set(cv2.CAP_PROP_POS_FRAMES, 4789)
cap.set(cv2.CAP_PROP_POS_FRAMES, 7000)
sec = 1/cap.get(cv2.CAP_PROP_FPS)

f=0
while cap.isOpened() and f<2000:
  f+=1
  ret, frame = cap.read()
  image = np.copy(frame)

  new_cards = []
  pre_proc = preprocess_image(frame)
  cnts_sort, cnt_is_card = find_cards(pre_proc)
  #if np.sum(cnts_sort) > 1:
  #  exit
  for i in range(len(cnts_sort)):
    if cnt_is_card[i] == 1:
      temp = preprocess_card(cnts_sort[i],frame)
      for card in temp:
        new_cards.append(card)

  pre_proc = preprocess_image(frame,"red")
  cnts_sort, cnt_is_card = find_cards(pre_proc)
  for i in range(len(cnts_sort)):
    if cnt_is_card[i]:
      temp = preprocess_card(cnts_sort[i],frame)
      for card in temp:
        new_cards.append(card)

  all_cards = compare_all_cards(all_cards,new_cards,image)
  temp_cnts = []
  color = (255,0,0)
  for i in range(len(all_cards)):
    temp_cnts.append(get_contour(all_cards[i]))
  cv2.drawContours(image,temp_cnts, -1, color, 2)
  r,g,b = color
  color = (g,b,r)

  cv2.imshow('Card Detector',image)
  key = cv2.waitKey(1) & 0xFF
  if key == ord("q"):
    cam_quit = 1  

cv2.destroyAllWindows()
#videostream.stop()
