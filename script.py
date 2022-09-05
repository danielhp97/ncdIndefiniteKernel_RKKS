import os
import requests
from requests.auth import HTTPBasicAuth
import json
import requests

TOKEN = "285903da1e00f86b1f4b8ff07357be9644100295"  
USERNAME = "danielhp97"
REPO = "danielhp97/ncdIndefiniteKernel_RKKS" 
DAGSHUB_URL = "https://dagshub.com"
MISSING_FILE_NAME = "7128.json"

missing_data = json.load(open(MISSING_FILE_NAME, "r"))
missing = ["/".join(f.split("/")[-2:]) for f in missing_data]
os.chdir(".dvc/cache")
failed = []
for path in missing:
  try:
    requests.put(f"{DAGSHUB_URL}/{REPO}.dvc/{path}", data=open(path, 'rb'), auth=HTTPBasicAuth(USERNAME, TOKEN))
  except Exception as e:
    print(e)
    failed.append(path)

if len(failed) > 0:
  print(f"The following files could not be found locally or could not : \n{failed}")