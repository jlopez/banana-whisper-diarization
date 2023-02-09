# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import base64
import json
import sys
import requests

from io import BytesIO
import base64
#import banana_dev as banana

#Needs test.mp3 file in directory
with open(f'../notebooks/sample01.mp3','rb') as file:
    mp3bytes = BytesIO(file.read())
mp3 = base64.b64encode(mp3bytes.getvalue()).decode("ISO-8859-1")
model_payload = {"mp3BytesString":mp3, "info": True}

res = requests.post("http://localhost:8000/",json=model_payload)

formatted_json = json.dumps(json.loads(res.text), indent=4)
with open(f'../notebooks/transcript.txt', 'w') as file:
    file.write(formatted_json)
print(formatted_json)


#use following to call deployed model on banana, model_payload is same as above
#out = banana.run("apikey","modelkey",model_payload)
