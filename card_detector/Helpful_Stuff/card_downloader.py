from skimage import io
import cv2
import requests
import urllib
import numpy as np
import json
import time

# query = "https://api.scryfall.com/cards/search?q=date%3Eemn"
# response = requests.get(query)
# if response.status_code == 200:
#     returned = response.json()
#     num_cards = returned["total_cards"]
#     cards = returned["data"]
#     has_more = returned["has_more"]
#     while has_more:
#         has_more = False
#         next_link = returned["next_page"]
#         time.sleep(.1)#wait for server
#         response = requests.get(next_link)
#         if response.status_code == 200:
#             returned = response.json()
#             has_more = returned["has_more"]
#             cards.extend(returned["data"])
#         else:
#             print(next_link)
#
#     print(num_cards == len(cards))
#     f = open("../Cards.json", "w")
#     f.write(json.dumps(cards))
#     f.close()

#download images
f = open("../Cards.json", "r")
cards = f.read()
cards = json.loads(cards)
path = "../images/"
for card in cards:
    img_name = card['name'] + '.jpg'
    if 'image_uris' in card:
        url = card['image_uris']['normal']
    elif 'card_faces' in card:
        for card_face in card['card_faces']:
            img_name = card_face['name'] + '.jpg'
            url = card_face['image_uris']['normal']
            image = io.imread(url)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            fullname = path + img_name
            cv2.imwrite(fullname, image)
            time.sleep(.1)
        continue
    else:
        print(card)
    image = io.imread(url)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fullname = path+img_name
    cv2.imwrite(fullname,image)
    time.sleep(.1)
print('done')
