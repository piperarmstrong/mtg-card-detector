import cv2
import numpy as np
import imutils
import copy

class Query_card:
    """Structure to store information about query cards in the camera image."""

    def __init__(self):
        self.contour = [] # Contour of card
        self.width, self.height, self.x, self.y = 0, 0, 0, 0 # Width and height of card
        self.corner_pts = [] # Corner points of card
        self.center = [] # Center point of card
        self.warp = None # 100x150, flattened, grayed, blurred image
        self.key_pts = None
        self.descriptors = None
        self.approx = []

def preprocess_image(image):

  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(5,5),0)

#  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#  clahe_output = clahe.apply(gray)
  thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

#  thresh = cv2.Canny(image,25,400)
#  lines = cv2.HoughLines(t,1,np.pi/180,200)

  #cv2.imshow('canny',thresh)
  key = cv2.waitKey(1) & 0xFF
  if key == ord("q"):
    cam_quit = 1  

  return thresh

def preprocess_card(contour, image):
  qCard = Query_card()
  qCard.contour = contour

  peri = cv2.arcLength(contour,True)
  approx = cv2.approxPolyDP(contour,0.12*peri,True)

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
  if len(pts) == 4: 
    qCard.warp = flattener(image, pts, w, h)
    orb =  cv2.ORB_create()
    qCard.tracker = cv2.TrackerKCF.create()
    qCard.tracker.init(image, (y,x,w,h))
    #qCard.key_pts, qCard.descriptors = orb.detectAndCompute(qCard.warp,None)
    #if qCard.descriptors is None:
    #  pass #return None
  return qCard

def flattener(image, pts, w, h):
  temp_rect = np.zeros((4,2),dtype="float32")
  
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

  dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
  M = cv2.getPerspectiveTransform(temp_rect,dst)
  warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
  #warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
  
  return warp

def find_cards(thresh_image):

  CARD_MAX_AREA = 7500
  CARD_MIN_AREA = 6000  

  cnts,hier = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  index_sort = sorted(range(len(cnts)),key=lambda i : cv2.contourArea(cnts[i]),reverse=True)
  if len(cnts) == 0:
    return [], []

  cnts_sort = []
  hier_sort = []
  cnt_is_card = np.zeros(len(cnts),dtype=int)
  
  for i in index_sort:
    cnts_sort.append(cnts[i])
    hier_sort.append(hier[0][i])

  for i in range(len(cnts_sort)):
    size = cv2.contourArea(cnts_sort[i])
 
    if size < CARD_MAX_AREA and size > CARD_MIN_AREA:
      peri = cv2.arcLength(cnts_sort[i],True)
      approx = cv2.approxPolyDP(cnts_sort[i],0.12*peri,True)
      if len(approx) == 4:
        cnt_is_card[i] = 1
        print(size)

  return cnts_sort, cnt_is_card

def draw_results(image, qCard):
  x = qCard.center[0]
  y = qCard.center[1]
  cv2.circle(image,(x,y),5,(255,0,0),-1)

  return image

def compare_cards(card1, card2):
  inside = is_same_card(card1,card2)
  if inside > 0:
    return inside
  heightDiff = card1.height - card2.height
  widthDiff = card1.width - card2.width
  xDiff = card1.x - card2.x
  yDiff = card1.y - card2.y
  diff = np.max(np.abs([heightDiff,widthDiff,xDiff,yDiff]))
  #print(heightDiff,widthDiff,xDiff,yDiff)
  #diff = np.abs(card1.corner_pts - card2.corner_pts)
  #print("diff",diff)
  if diff < 20:
    return 1
  return 0
  #return np.max(diff)
  
'''def is_same_card(card1, card2):
  pts1 = card1.approx
  pts2 = card2.approx
  
  if contains(card1, card2):
    return 1
  if contains(card2, card1):
    return 0
  
  return None''' 

def contains(card1, card2):
  if card1.approx[0][0][0] > card2.approx[0][0][0] and card1.approx[0][0][1] > card2.approx[0][0][1]:
    if card1.approx[1][0][0] < card2.approx[1][0][0] and card1.approx[1][0][1] > card2.approx[1][0][1]:
      if card1.approx[2][0][0] > card2.approx[2][0][0] and card1.approx[2][0][1] < card2.approx[2][0][1]:
        if card1.approx[3][0][0] < card2.approx[3][0][0] and card1.approx[3][0][1] < card2.approx[3][0][1]:    
          return True
  return False

def is_same_card(card1, card2):
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

  
  '''if len(new_cards) < 1:
    return (all_cards)
  these_cards = []
  num_cards = len(new_cards)
  
  #Deal with multiple outlines being assigned to a single card.
  #Keep the outer outline in case we've grabbed just the explanation section
  if num_cards > 1:
    cards = list(range(num_cards))

    for i in range(num_cards):
      for j in range(num_cards):
        if j > i:
          if is_same_card(new_cards[cards[i]],new_cards[cards[j]]) is 0:
            cards[i] = j
          if is_same_card(new_cards[i],new_cards[j]) is 1:
            cards[j] = i
    cards = set(cards)
    for i in cards:
      these_cards.append(new_cards[i])
    new_cards = these_cards   

  cards = np.zeros(shape=(len(new_cards),len(all_cards)))  

  # Check corner points of the old cards and new cards to see if any should correspond to each other.
  for i, new_card in enumerate(new_cards):
    for j, card in enumerate(all_cards):
      keep = compare_cards(new_card,card)
      cards[i][j] = keep

  keep_cards = []
  cards = np.transpose(cards)

  # If there is no card in the new set corresponding to the old set, keep the card in the old set
  #TODO: Check the area around these cards for interesting features that correspond to the card
  #If there aren't enough matching features, assume the card has been removed from play
  for i in range(len(all_cards)):
    m = np.min(cards[i])
    if m > 200:
      new_cards.append(all_cards[i])


  #Check again to see if there are cards inside other cards, and keep the larger card outline
  num_cards = len(new_cards)
  if num_cards > 1:
    these_cards = []
    cards = list(range(num_cards))

    for i in range(num_cards):
      for j in range(num_cards):
        if j > i:
          if is_same_card(new_cards[cards[i]],new_cards[cards[j]]) is 0:
            cards[i] = j
          if is_same_card(new_cards[i],new_cards[j]) is 1:
            cards[j] = i
    cards = set(cards)
    for i in cards:
      these_cards.append(new_cards[i])
    new_cards = these_cards'''
  #print(len(new_cards))
  #print(len(all_cards))
  return all_cards

def get_contour(q):
  return np.array([[q.x,q.y],[(q.x+q.width),q.y],[(q.x+q.width),(q.y+q.height)],[q.x,(q.y+q.height)]],dtype=np.int32)

all_cards = []

cap = cv2.VideoCapture("../2018 Magic World Championship Finals.mp4")

cap.set(cv2.CAP_PROP_POS_FRAMES, 4789)
f=0
while cap.isOpened() and f<2000:
  f+=1
  ret, frame = cap.read()
  image = np.copy(frame)

  pre_proc = preprocess_image(frame)
  cnts_sort, cnt_is_card = find_cards(pre_proc)
  new_cards = []
  for i in range(len(cnts_sort)):
    if cnt_is_card[i] == 1:
      temp = preprocess_card(cnts_sort[i],frame)
      if temp is not None:
        new_cards.append(temp)
      #image = draw_results(image, new_cards[-1])

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
