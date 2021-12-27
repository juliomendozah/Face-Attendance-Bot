import cv2
from deepface import DeepFace
import requests
from skimage import io
import time
import numpy as np
import urllib

faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
meraki_api_key = #Enter your Meraki API Key
cam_serial = #Enter your Serial Number of your Meraki Camera
headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Cisco-Meraki-API-Key": meraki_api_key
    }
def get_rtspurl(cam_serial):
    """
    Get RTSP URL from camera
    """


    try:
        r_rtspurl = requests.request('GET', f"https://api.meraki.com/api/v1/devices/{cam_serial}/camera/video/settings", headers=headers)
        r_rtspurl_json = r_rtspurl.json()
        return r_rtspurl_json["rtspUrl"]
    except Exception as e:
        return print(f"Error when getting image URL: {e}")

def snapshot():
    url = "https://api.meraki.com/api/v1/devices/Q2FV-NX7G-MNB2/camera/generateSnapshot"
    payload = '''{ "fullframe": true }'''
    response = requests.request('POST', url, headers=headers, data = payload)
    snapshot_url=(response.json())['url']
    return snapshot_url
def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
	return image
if __name__ == "__main__":
    # Get a reference to webcam #0 (the default one)
    #video_capture = cv2.VideoCapture(get_rtspurl(cam_serial))
    #process_this_frame = True

    while True:
        #ret, frame = video_capture.read()
        time.sleep(10)
        #frame=io.imread(snapshot())
        frame=url_to_image(snapshot())
        result=DeepFace.analyze(frame,actions=['emotion'],enforce_detection=False)

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray,1.1,4)

        for (x,y,w,h) in faces:
          cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        font =cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(frame,result['dominant_emotion'],(50,50),font,3,(0,0,255),2,cv2.LINE_4)

        #cv2.imshow('Video', frame)
        cv2.imshow('Snapshot', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #video_capture.release()
    cv2.destroyAllWindows()
