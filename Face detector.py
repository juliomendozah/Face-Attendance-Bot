# venv36\scripts\activate
#References:
#-https://towardsdatascience.com/the-ultimate-guide-to-emotion-recognition-from-facial-expressions-using-python-64e58d4324ff
#
import face_recognition
import cv2
import numpy as np
import requests
import json
from requests_toolbelt.multipart.encoder import MultipartEncoder
import time
import os
import pandas as pd
from deepface import DeepFace

faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

url_webex = "https://webexapis.com/v1/messages"
WEBEX_ACCESS_TOKEN = # Enter your Webex Bot Access Token
roomId =   # Enter the Room ID of the Monitoring Space
LOCATION ="Lima"
people=pd.read_csv('People.csv',index_col="Name")

def generate_card (person):
    card = {
        "type": "AdaptiveCard",
        "body": [
            {
                "type": "ColumnSet",
                "columns": [
                    {
                        "type": "Column",
                        "items": [
                            {
                                "type": "Image",
                                "style": "Person",
                                "url": people["Photo_URL"][person],
                                "size": "Medium",
                                "height": "50px"
                            }
                        ],
                        "width": "auto"
                    },
                    {
                        "type": "Column",
                        "items": [
                            {
                                "type": "TextBlock",
                                "text": "Welcome",
                                "weight": "Lighter",
                                "color": "Accent"
                            },
                            {
                                "type": "TextBlock",
                                "weight": "Bolder",
                                "text": person,
                                "wrap": True,
                                "color": "Light",
                                "size": "Large",
                                "spacing": "Small"
                            }
                        ],
                        "width": "stretch"
                    }
                ]
            },
            {
                "type": "ColumnSet",
                "columns": [
                    {
                        "type": "Column",
                        "width": 35,
                        "items": [
                            {
                                "type": "TextBlock",
                                "text": "Date:",
                                "color": "Light"
                            },
                            {
                                "type": "TextBlock",
                                "text": "Hour:",
                                "weight": "Lighter",
                                "color": "Light",
                                "spacing": "Small"
                            },
                            {
                                "type": "TextBlock",
                                "text": "Location:",
                                "weight": "Lighter",
                                "color": "Light",
                                "spacing": "Small"
                            }
                        ]
                    },
                    {
                        "type": "Column",
                        "width": 65,
                        "items": [
                            {
                                "type": "TextBlock",
                                "text": time.strftime('%b %d, %Y'),
                                "color": "Light"
                            },
                            {
                                "type": "TextBlock",
                                "text": time.strftime('%H:%M %p'),
                                "color": "Light",
                                "weight": "Lighter",
                                "spacing": "Small"
                            },
                            {
                                "type": "TextBlock",
                                "text": LOCATION,
                                "weight": "Lighter",
                                "color": "Light",
                                "spacing": "Small"
                            }
                        ]
                    }
                ],
                "spacing": "Padding",
                "horizontalAlignment": "Center"
            },
            {
                "type": "TextBlock",
                "text": "We are happy to have you back in the office! Please keep your distance and wear a mask during your stay in the office. Have a nice day!",
                "wrap": True
            }
        ],
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.2"
    }
    return card

def send_webex_message(person, photo, use_mask=False):
    if use_mask:
        msg = person + " 'd just arrived! That person is using a mask."
    else:
        msg = person + " 'd just arrived! That person isn't using a mask."

    message_photo = MultipartEncoder({'roomId': people["RoomID"][person],
                                "markdown": msg,
                                'files': (photo, open(photo, 'rb'),
                                'image/png')})
    message_card = {
        "roomId": roomId,
        "markdown": msg,
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content":generate_card(person) }
                    ]
                }
    response_webex = requests.post(url_webex, data=message_photo,
                                   headers={'Authorization': "Bearer " + WEBEX_ACCESS_TOKEN,
                                            'Content-Type': message_photo.content_type})
    print(response_webex.text)
    response_webex = requests.post(url_webex,  data=json.dumps(message_card),
                                   headers={'Authorization': "Bearer " + WEBEX_ACCESS_TOKEN,
                                            'Content-Type': 'application/json'})
    print(response_webex.text)


if __name__ == "__main__":
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)

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
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)
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
            cv2.putText(frame, result['dominant_emotion'], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1,cv2.LINE_4,False)
            cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1,cv2.LINE_4,False)

            use_mask = False

            if "_m" in name:
                use_mask = True
                name = name.replace('_m', '')
            if "_a" in name:
                name = name.replace('_a', '')

            if name not in visitors:
                if name != "Unknown":
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
