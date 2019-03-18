import cv2
import numpy as np
import imutils

class Query_card:
    """Structure to store information about query cards in the camera image."""

    def __init__(self):
        self.contour = [] # Contour of card
        self.width, self.height = 0, 0 # Width and height of card
        self.corner_pts = [] # Corner points of card
        self.center = [] # Center point of card
        self.warp = [] # 200x300, flattened, grayed, blurred image

def preprocess_image(image):
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(5,5),0)

  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  clahe_output = clahe.apply(blur)
  return cv2.adaptiveThreshold(clahe_output,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

def preprocess_card(contour, image):
  qCard = Query_card()
  qCard.contour = contour

  peri = cv2.arcLength(contour,True)
  approx = cv2.approxPolyDP(contour,0.12*peri,True)
  pts = np.float32(approx)
  qCard.corner_pts = pts

  x,y,w,h = cv2.boundingRect(contour)
  qCard.width, qCard.height = w,h

  average = np.sum(pts, axis=0)/len(pts)
  cent_x = int(average[0][0])
  cent_y = int(average[0][1])
  qCard.center = [cent_x, cent_y]
 
  qCard.warp = flattener(image, pts, w, h)
  
  return qCard

def flattener(image, pts, w, h):
  temp_rect = np.zeros((4,2),dtype="float32")
  
  s = np.sum(pts, axis = 2)
  
  tl = pts[np.argmin(s)]
  br = pts[np.argmax(s)]
  diff = np.diff(pts, axis = -1)
  tr = pts[np.argmin(diff)]
  bl = pts[np.argmax(diff)]

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

  maxWidth = 200
  maxHeight = 300

  dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
  M = cv2.getPerspectiveTransform(temp_rect,dst)
  warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
  #warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
  
  return warp

def find_cards(thresh_image):

  CARD_MAX_AREA = 100000
  CARD_MIN_AREA = 1000  

  dummy,cnts,hier = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
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
    peri = cv2.arcLength(cnts_sort[i],True)
    approx = cv2.approxPolyDP(cnts_sort[i],0.01*peri,True)
    
    if size < CARD_MAX_AREA and size > CARD_MIN_AREA and len(approx) == 4:
      cnt_is_card[i] = 1

  return cnts_sort, cnt_is_card

def draw_results(image, qCard):
  
  x = qCard.center[0]
  y = qCard.center[1]
  cv2.circle(image,(x,y),5,(255,0,0),-1)

  return image

cap = cv2.VideoCapture("../2018 Magic World Championship Finals.mp4")

cap.set(cv2.CAP_PROP_POS_FRAMES, 10000)
f=0
while cap.isOpened() and f<100:
  f+=1
  ret, frame = cap.read()
  image = np.copy(frame)

  pre_proc = preprocess_image(frame)
  cnts_sort, cnt_is_card = find_cards(pre_proc)
  cards = []
  for i in range(len(cnts_sort)):
    if cnt_is_card[i] == 1:
      cards.append(preprocess_card(cnts_sort[i],frame))
      image = draw_results(image, cards[-1])

  if len(cards) > 0:
    temp_cnts = []
    for i in range(len(cards)):
      temp_cnts.append(cards[i].contour)
    cv2.drawContours(image,temp_cnts, -1, (255,0,0), 2)
  



  cv2.imshow('Card Detector',image)
  key = cv2.waitKey(1) & 0xFF
  if key == ord("q"):
    cam_quit = 1  

cv2.destroyAllWindows()
#videostream.stop()