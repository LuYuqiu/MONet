# https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/

from requests import exceptions
import argparse
import requests
import cv2
import os

ap = argparse.ArgumentParser()
# ap.add_argument("-q", "--query", required=True,
#                 help="search query to Bing Image APT for")
# ap.add_argument("-o", "--output", required=True,
#                 help="path to output directory of image")
args = vars(ap.parse_args())  


# args["query"] = "landscape"
# try:
#     os.mkdir("/home/user/Luyuqiu/datasets/landscape")
# except Exception as e:
#     print("[INFO] Path have made.")
#
# args["output"] = r"/home/user/Luyuqiu/datasets/landscape"

# args["query"] = "mountain"
# try:
#     os.mkdir("./datasets/mountain")
# except Exception as e:
#     print("[INFO] Path have made.")
#
# args["output"] = r"./datasets/mountain"

args["query"] = "landscape"
try:
    os.mkdir("/home/user/Luyuqiu/datasets/landscape")
except Exception as e:
    print("[INFO] Path have made.")

args["output"] = r"/home/user/Luyuqiu/datasets/landscape"


MAX_RESULTS = 1000  
GROUP_SIZE = 50   
API_KEY = "1d04ed8991754aa9bf465a90d882ebb6"
URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"


EXCEPTIONS = set([IOError, exceptions.RequestException,
                  exceptions.HTTPError, exceptions.ConnectionError, exceptions.Timeout])


term = args["query"]  
headers = {"Ocp-Apim-Subscription-key": API_KEY}
params = {"q": term, "offset": 0, "count": GROUP_SIZE}

print("[INFO] searching Bing API for '{}'".format(term))
search = requests.get(URL, headers=headers, params=params)
search.raise_for_status()

result = search.json()
estNumResults = min(result["totalEstimatedMatches"], MAX_RESULTS)
print("[INFO] {} total result for '{}'".format(estNumResults, term))

total = 0


for offset in range(0, estNumResults, GROUP_SIZE):
    print("[INFO] Making request for group {}-{} of {}...".format(offset, offset+GROUP_SIZE, estNumResults))
    params["offset"] = offset
    search = requests.get(URL, headers=headers, params=params)
    search.raise_for_status()
    result = search.json()
    print("[INFO] saving images for group {}-{} of {}...".format(offset, offset+GROUP_SIZE, estNumResults))


    for v in result["value"]:
        try:
            print("[INFO] fetching: {}".format(v["contentUrl"]))
            r = requests.get(v["contentUrl"], timeout=30)


            ext = v["contentUrl"][v["contentUrl"].rfind("."):]  
            p = os.path.sep.join([args["output"], "{}{}".format(str(total).zfill(8), ext)])  

   
            f = open(p, "wb")
            f.write(r.content)
            f.close()

        except Exception as e:
            if type(e) in EXCEPTIONS:
                print("[INFO] Skipping: {}".format(v["contentUrl"]))
                continue

        image = cv2.imread(p)
        if image is None:
            print("[INFO] deleting: {}".format(p))
            os.remove(p)
            continue

        total += 1
