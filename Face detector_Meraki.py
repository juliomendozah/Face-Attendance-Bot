# venv36\scripts\activate
import face_recognition
import cv2
import numpy as np
import requests
import json
from requests_toolbelt.multipart.encoder import MultipartEncoder
import time
import os

url_webex = "https://webexapis.com/v1/messages"
WEBEX_ACCESS_TOKEN = "YWIyMmExZjgtNTYxYi00Yzk4LTljYmUtODA0MjYzNWQ0N2U4MzkyY2M0NzAtMjU1_PF84_1eb65fdf-9643-417f-9974-ad72cae0e10f"
# roomId="Y2lzY29zcGFyazovL3VzL1JPT00vNGZmMGQwMDAtZWJmYy0xMWViLWI0OWUtOGRhMjlkNWZiNDJi"#Group Space
roomId = "Y2lzY29zcGFyazovL3VzL1JPT00vMWZhMTI1MTAtMjUzZC0xMWVjLTk1YWEtMDlmNDQwMjE3Zjhi"  # Direct Space

meraki_api_key = "0d7a8b4276fb04606fe0659a37e52dbba345e805"
cam_serial = "Q2FV-NX7G-MNB2"

def get_rtspurl(cam_serial):
    """
    Get RTSP URL from camera
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Cisco-Meraki-API-Key": meraki_api_key
    }

    try:
        r_rtspurl = requests.request('GET', f"https://api.meraki.com/api/v1/devices/{cam_serial}/camera/video/settings", headers=headers)
        r_rtspurl_json = r_rtspurl.json()
        return r_rtspurl_json["rtspUrl"]
    except Exception as e:
        return print(f"Error when getting image URL: {e}")

def send_webex_message(msg, photo, use_mask=False):
    if use_mask:
        msg = msg + " 'd just arrived! That person is using a mask."
    else:
        msg = msg + " 'd just arrived! That person isn't using a mask."

    message = MultipartEncoder({'roomId': roomId,
                                "markdown": msg,
                                'files': (photo, open(photo, 'rb'),
                                          'image/png')})

    response_webex = requests.post(url_webex, data=message,
                                   headers={'Authorization': "Bearer " + WEBEX_ACCESS_TOKEN,
                                            'Content-Type': message.content_type})
    print(response_webex.text)


if __name__ == "__main__":
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(get_rtspurl(cam_serial))

    # Load a sample picture and learn how to recognize it.
    photos = os.listdir('Photos')
    known_face_names = [x.replace('.jpg', '').replace('.jpeg', '').replace('.png', '') for x in photos]
    known_face_encodings = []

    for photo in photos:
        temp = face_recognition.load_image_file('Photos/' + photo)
        known_face_encodings.append(face_recognition.face_encodings(temp)[0])

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    visitors = []

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            use_mask = False
            if name != "Unknown":
                if "_m" in name:
                    use_mask = True
                    name = name.replace('_m', '')

                if name not in visitors:
                    visitors.append(name)
                    temp = 'Visitors/' + name + '_' + time.strftime('%H_%M_%S_%m%d%Y') + '.png'
                    cv2.imwrite(temp, frame)
                    send_webex_message(name, temp, use_mask)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()