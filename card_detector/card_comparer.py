import json
import cv2
import time

class card_comparer():
    ###
    # card_dir is the location/name of the json file containing the cards
    # img_dir is the location/name of the directory containing the card images
    ###
    def __init__(self, card_dir, img_dir, cv2_feature_detector):
        f = open(card_dir, "r")
        cards = f.read()
        cards = json.loads(cards)
        f.close()
        self.img_dir = img_dir
        self.fd = cv2_feature_detector
        self.descriptors = []
        for card in cards:
            img = cv2.imread(self.img_dir + card['name'] + '.jpg')
            if img is not None: # make provisions for dfc's
                _, des = self.fd.detectAndCompute(img,None)
                self.descriptors.append((des, card['name']))
            card = None

    def compare(self, img):
        max_matches = 0
        match_name = ''
        _, des = self.fd.detectAndCompute(img, None)
        # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        FLANN_INDEX_LSH = (20, 12, 1)
        #index_params = dict(table_number=12,
        #                    key_size=20,
        #                    multi_probe_level=2)
        #search_params = dict(checks=50)
        #matcher = cv2.FlannBasedMatcher(index_params, search_params)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        for tdes, name in self.descriptors:
            matches = matcher.knnMatch(des, tdes, k=2)
            if len(matches) > max_matches:
                max_matches = len(matches)
                match_name = name
        return match_name


comp = card_comparer("Cards.json", "images/", cv2.BRISK_create())
print('grok')
print(time.perf_counter())
print(comp.compare(cv2.imread('images/Spireside Infiltrator.jpg')))
print(time.perf_counter())