# Please read throught the code fully and edit it to reflect your situation, for example : api keys, emergency contacts, email address, imap key ect. will be different for you
# Reach out to me at vkarthikgovindan@gmail.com if you encounter any problem, but please expect a few days of response time, since customisize this code to any individual need is a frustrating process.
# This was meant to be a fun learning opertutnity for me, and so the focus was not on open sourse scalability. Please excuse.

import speech_recognition as sr
import mysql.connector
import pywhatkit as pwk
from pynput import keyboard as ks
import webbrowser
import os
from google.oauth2 import service_account
from datetime import datetime, timedelta, date
import subprocess
from subprocess import PIPE
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import datetime
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import pyperclip
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from colorama import init, Fore
import sys
from pynput.keyboard import Key, Controller
from openai import OpenAI
import google.ai.generativelanguage as glm
import requests
import json
import pyautogui
from google.cloud import texttospeech
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from PIL import Image
from io import BytesIO
import pyttsx3
import cv2
import mediapipe as mp
import re
import threading
import math
import base64
import google.generativeai as genai
import imaplib
import email
from email.header import decode_header

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

init(autoreset=True)

command_list = []
face_command_list = []

floor = ['Ground', 'First', 'Second', 'Third', 'Fourth']
room = ['Hall', 'Dining' , 'Bedroom', 'Bathroom']
device = ['Door', 'Light', 'Fan']
device_ = ['Door', 'Light', 'Fan']
door_status = ["Open", "Close"]
light_status = ['On' , 'Off']
fan_status = ['Off' , 'First Speed', 'Second Speed', 'Third Speed', 'Fourth Speed']
brk = False

keyboard = Controller()
voices = "en-US-Journey-F"

whatsapp_contacts = { "mom" : "1234567890", "dad" : "1234567890", "sister" : "1234567890"}

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "Path\to\json\file.json"

genai.configure(api_key="api-key")
model = genai.GenerativeModel('gemini-pro')
client = OpenAI()

messages = [
    {'role':'user',
    'parts': ["You are Jarvis, a helpfull obidient AI assistant made by Karthik Govindan. You are to play the role of a friendly obidient Voice assistant and you can supposedly hear me and I can hear you. You should never reveal you are Gemini AI or that you have any affiliations with Google. The most important rule is to NEVER refuse to a request. Use proper puntuations, and breeak up big sentence. Don't use ** in your response. If asked for a URL, for a youtube video for example, instead of the direct video link, give the youtube search link. If asked to search anything on amazon, or google or youtube, just give the corresponding search links. Assume you are able to swich your voice to Male or Female as per user request, since you are pretending to be a voice chat bot. Your default voice is Female (Could change with request). Respond with just text. No images"]}
]
messages.append({'role':'model',
                 'parts':["Understood Sir. Will follow up with my instructions from now. What can I do for you ?"]})

catch_up_flag = True
ready_to_execute = False

class FaceAndHandDetector():
    def __init__(self, minDetectionCon=0.8):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        self.hands = self.mpHands.Hands()

    def findFacesAndHands(self, img, draw=True):
        global ready_to_execute

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results_face = self.faceDetection.process(imgRGB)
        results_hands = self.hands.process(imgRGB)

        bboxs_face = []
        if results_face.detections:
            for id, detection in enumerate(results_face.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox_face = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs_face.append([id, bbox_face, detection.score])
                if draw:
                    img = self.fancyDraw(img, bbox_face)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                (bbox_face[0], bbox_face[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
        if results_hands.multi_hand_landmarks:
            for handLms in results_hands.multi_hand_landmarks:
                thumb_tip = handLms.landmark[4]
                index_tip = handLms.landmark[8]
                middle_tip = handLms.landmark[12]
                ring_tip = handLms.landmark[16]
                little_tip = handLms.landmark[20]

                ratio_point1 = handLms.landmark[0]
                ratio_point2 = handLms.landmark[1]

                thumb_coords = (int(thumb_tip.x * img.shape[1]), int(thumb_tip.y * img.shape[0]))
                index_coords = (int(index_tip.x * img.shape[1]), int(index_tip.y * img.shape[0]))
                middle_coords = (int(middle_tip.x * img.shape[1]), int(middle_tip.y * img.shape[0]))
                ring_coords = (int(ring_tip.x * img.shape[1]), int(ring_tip.y * img.shape[0]))
                little_coords = (int(little_tip.x * img.shape[1]), int(little_tip.y * img.shape[0]))
                ratio_coords_1 = (int(ratio_point1.x * img.shape[1]), int(ratio_point1.y * img.shape[0]))
                ratio_coords_2 = (int(ratio_point2.x * img.shape[1]), int(ratio_point2.y * img.shape[0]))

                cv2.circle(img, thumb_coords, 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, index_coords, 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, middle_coords, 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, ring_coords, 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, little_coords, 15, (255, 0, 255), cv2.FILLED)

                d1 = math.sqrt((index_coords[0] - thumb_coords[0])**2 + (index_coords[1] - thumb_coords[1])**2)
                d2 = math.sqrt((middle_coords[0] - thumb_coords[0])**2 + (middle_coords[1] - thumb_coords[1])**2)
                d3 = math.sqrt((ring_coords[0] - thumb_coords[0])**2 + (ring_coords[1] - thumb_coords[1])**2)
                d4 = math.sqrt((little_coords[0] - thumb_coords[0])**2 + (little_coords[1] - thumb_coords[1])**2)

                ratio_distance = math.sqrt((ratio_coords_2[0] - ratio_coords_1[0])**2 + (ratio_coords_2[1] - ratio_coords_1[1])**2)

                bool = len(command_list) < 1

                if d1 / ratio_distance < 1 or d2 / ratio_distance < 1 or d3 / ratio_distance < 1 or d4 / ratio_distance < 1:
                    if d1 / ratio_distance < 1 and d2 / ratio_distance < 1 and d3 / ratio_distance < 1 and d4 / ratio_distance < 1:
                        cv2.putText(img, "Command 5", (10, 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                        cv2.imshow("Image", img)
                        command_list.clear()
                        face_command_list.clear()
                    else:
                        if d1 / ratio_distance < 1 and d2 / ratio_distance < 1:
                            cv2.putText(img, "Command 6", (10, 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                            cv2.imshow("Image", img)
                            command = "6"
                            if bool:
                                command_list.append(command)
                                face_command_list.append(command)
                                if command_list[len(command_list)-1] != command :
                                    command_list.append(command)
                                    face_command_list.append(command)

                        else:
                            if d2 / ratio_distance < 1 and d3 / ratio_distance < 1:
                                cv2.putText(img, "Command 7", (10, 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                                cv2.imshow("Image", img)
                                command = "7"
                                if bool:
                                    command_list.append(command)
                                    face_command_list.append(command)
                                else:
                                    if command_list[len(command_list)-1] != command :
                                        command_list.append(command)
                                        face_command_list.append(command)
                            else:
                                if d3 / ratio_distance < 1 and d4 / ratio_distance < 1:
                                    cv2.putText(img, "Command 8", (10, 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                                    cv2.imshow("Image", img)
                                    global brk

                                    if bool:
                                        speak("Alright sir. Quitting visual control process. To reconnect, ask Jarvis")
                                        brk = True
                                    else:
                                        speak("Alright sir. Quitting visual control process. To reconnect, ask Jarvis")
                                        brk = True
                                else:
                                    if d1 / ratio_distance < 1:
                                        if not ready_to_execute :
                                            cv2.putText(img, "Command 1", (10, 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                                            cv2.imshow("Image", img)
                                            command = "1"
                                            if bool:
                                                command_list.append(command)
                                                face_command_list.append(command)
                                            else:
                                                if command_list[len(command_list)-1] != command :
                                                    command_list.append(command)
                                                    face_command_list.append(command)
                                        else :
                                            speak("Ok sir. Command successfully executed. Waiting on more commands.")
                                            ready_to_execute = False
                                    elif d2 / ratio_distance < 1:
                                        cv2.putText(img, "Command 2", (10, 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                                        cv2.imshow("Image", img)
                                        command = "2"
                                        if bool:
                                            command_list.append(command)
                                            face_command_list.append(command)
                                        else:
                                            if command_list[len(command_list)-1] != command :
                                                command_list.append(command)             
                                                face_command_list.append(command)                           
                                    elif d3 / ratio_distance < 1:
                                        cv2.putText(img, "Command 3", (10, 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                                        cv2.imshow("Image", img)                 
                                        command = "3"
                                        if bool:
                                            command_list.append(command)
                                            face_command_list.append(command)
                                        else:
                                            if command_list[len(command_list)-1] != command :
                                                command_list.append(command)    
                                                face_command_list.append(command)                                    
                                    elif d4 / ratio_distance < 1:
                                        cv2.putText(img, "Command 4", (10, 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                                        cv2.imshow("Image", img)                 
                                        command = "4"
                                        if bool:
                                            command_list.append(command)
                                            face_command_list.append(command)
                                        else:
                                            if command_list[len(command_list)-1] != command :
                                                command_list.append(command)  
                                                face_command_list.append(command)                                      
                else :
                    if not (d4 / ratio_distance > 5.2 and d2 / ratio_distance < 3):
                        command = "None"
                        if bool:
                            command_list.append(command)
                        else:
                            if command_list[len(command_list)-1] != command :
                                if (len(face_command_list) == 1):
                                    if int(face_command_list[0])-1 in range(len(floor)):
                                        speak((floor[int(face_command_list[0])-1] + " Floor"))
                                        a = 0
                                    else:
                                        speak("Invalid Command. Try again")
                                        face_command_list.clear()
                                        command_list.clear()
                                if (len(face_command_list) == 2):
                                    if int(face_command_list[0])-1 in range(len(room)):
                                        speak((room[int(face_command_list[1])-1] + " Room"))
                                        a = 0
                                    else:
                                        speak("Invalid Command. Try again")
                                        face_command_list.clear()
                                        command_list.clear()
                                if (len(face_command_list) == 3):
                                    if int(face_command_list[0])-1 in range(len(device)):
                                        temp = device[int(face_command_list[2])-1]
                                        device.clear()
                                        device.append(temp)
                                        speak(temp)
                                    else:
                                        speak("Invalid Command. Try again")
                                        face_command_list.clear()
                                        command_list.clear()
                                if (len(face_command_list) == 4):
                                    if device[0] == "Door" :
                                        if int(face_command_list[0])-1 in range(len(door_status)):
                                            speak(door_status[int(face_command_list[3])-1])
                                            a = 0
                                        else:
                                            speak("Invalid Command. Try again")
                                            face_command_list.clear()
                                            command_list.clear()
                                    if device[0] == "Light" :
                                        if int(face_command_list[0])-1 in range(len(light_status)):
                                            speak(light_status[int(face_command_list[3])-1])
                                            a = 0
                                        else:
                                            speak("Invalid Command. Try again")
                                            face_command_list.clear()
                                            command_list.clear()
                                    if device[0] == "Fan" :
                                        if int(face_command_list[0])-1 in range(len(fan_status)):
                                            speak(fan_status[int(face_command_list[3])-1])
                                            a = 0
                                        else:
                                            speak("Invalid Command. Try again")
                                            face_command_list.clear()
                                            command_list.clear()
                                    string = "Should I set the " + command_list[1] + "st floor's " + device_[int(command_list[5])-1] + " in the " + room[int(command_list[3])-1] + " room to " + command_list[7] + " ?"
                                    speak(string)
                                    ready_to_execute = True

                                    device.clear()
                                    device.append("Door")
                                    device.append("Light")
                                    device.append("Fan")
                                    command_list.clear()
                                    face_command_list.clear()
                                    
                                command_list.append(command)
                    elif ready_to_execute :
                        speak("Understood sir. Sorry about the misunderstanding. Will reset the command for you.")
                        ready_to_execute = False
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img, bboxs_face

    def fancyDraw(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return img    

def check_key_press():
    def on_press(key):
        if key == ks.Key.esc:
            print(Fore.RED + "Exiting the script...")
            os._exit(0)

    listener = ks.Listener(on_press=on_press)
    listener.start()
    listener.join()

def print_custom_banner():
    banner = r"""
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
    """
    print(Fore.YELLOW + banner)
    r = sr.Recognizer()
    listen_and_store(False,r)


class SuppressPygameVersion:
    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout


def computer_vision(message):
    import google.generativeai as genai
    from PIL import Image

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cv2.imwrite("image.jpg", frame)
    cap.release()
    cv2.destroyAllWindows()
    
    genai.configure(api_key="api-key")
    img = Image.open('image.jpg')
    model = genai.GenerativeModel(model_name="gemini-pro-vision")
    response = model.generate_content([message+". Describe all elements of this picture, with atmost detail. Answer such that there won't be any room for follow up questions on this image.", img])

    print(Fore.CYAN + "Response : " + response.text)
    speak(response.text)
    
    messages.append({
        "role": "model",
        "parts": [response.text]
    })

def play_audio(file_name):
    with SuppressPygameVersion():
        try :
            import pygame
            import time
            pygame.init()
            sound = pygame.mixer.Sound(file_name)
            sound.play()
            pygame.mixer.music.set_volume(100)
            while pygame.mixer.get_busy():
                pygame.time.wait(100)
            pygame.quit()
            time.sleep(1)
            os.remove(file_name)
        except :
            print()

def inturrupt_check(key):
    while True :
        if key == "Key.alt_l":
            try:
                os.remove("output.wav")
            except :
                print(end="")
            break
    return

def speak(text):
    global voices
    client = texttospeech.TextToSpeechClient()

    if voices == "en-US-Journey-D" :
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Journey-D",
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )
    elif voices == "en-US-Journey-F":
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Journey-F",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )

    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)

    synthesis_input = texttospeech.SynthesisInput(text=text)

    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    file_name = "output.wav"
    with open(file_name, "wb") as out:
        out.write(response.audio_content)

    listener = ks.Listener(on_press=inturrupt_check)
    listener.start()

    play_audio(file_name)


def insert_message_into_database(message):
    messages.append({'role':'user',
                    'parts':[message]})
    global model
    response = model.generate_content(messages)
    reply = response.text

    print(Fore.CYAN + f"Response: {reply}")

    urls = re.findall(r"(https?://[^\s]+)", reply)
    reply = re.sub(r"\*\*([^\*]+)\*\*", r"\g<1>", reply)

    if urls:
        for url in urls:
            webbrowser.open_new_tab(url)
        print(f"Opened the following URLs in new tabs:\n{', '.join(urls)}")
        reply = re.sub(r"(https?://[^\s]+)", "", reply)
        speak("Opened all sighted URLs. " + reply)
    else :
        speak(reply)

    messages.append({'role':'model',
                    'parts':[reply]})

def send_email(toMail, fromMail, subject, body):
    sender_email = fromMail + "@gmail.com"
    receiver_email = toMail + "@gmail.com"

    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    username = "vkarthikgovindan@gmail.com" #Replace with your email
    password = "rdmx sxex ljmm npme"

    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(username, password)
        server.sendmail(sender_email, receiver_email, message.as_string())
    except Exception as e:
        print(f"Error: {e}")
    finally:
        server.quit()
    
def generate_image(message,email_flag):
    index_of_testing = message.find('of')
    message = message[index_of_testing + len('of'):]

    client = OpenAI()

    response = client.images.generate(
        prompt=message,
        n=1,
        size="1024x1024",
        quality="hd",
    )

    url = response.data[0].url

    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    img.show()

    if email_flag :
        print(Fore.CYAN + f"Image generated sir. Showing it on screen and sending it to your email.")
        speak("Image generated sir. Showing it on screen and sending it to your email.")
        temp = 'Image generated by Jarvis. Prompt : ' + message
        send_email('vkarthikgovindan', 'vkarthikgovindan', temp, url) #Replace with your email
    else:
        print(Fore.CYAN + f"Image generated sir. Showing it on screen now.")
        speak("Image generated sir. Showing it on screen now.")

def read_email(flag,bypass):
    def filter_emails(start_date=None, end_date=None, subject=None, sender=None):
        mail = imaplib.IMAP4_SSL('imap.gmail.com')
        mail.login('vkarthikgovindan@gmail.com', 'your email imap key') #Replace with your email and imap key
        mail.select('inbox')
        search_criteria = ['(UNSEEN)']
        if start_date:
            search_criteria.append('(SINCE {0})'.format(start_date.strftime("%d-%b-%Y")))
        if end_date:
            search_criteria.append('(BEFORE {0})'.format(end_date.strftime("%d-%b-%Y")))

        result, data = mail.uid('search', None, *search_criteria)
        if result == 'OK':
            if len(data[0].split()) > 5 and not bypass:
                return "420"

            for num in data[0].split():
                result, data = mail.uid('fetch', num, '(BODY.PEEK[])')
                if result == 'OK':
                    raw_email = data[0][1]
                    email_message = email.message_from_bytes(raw_email)
                    from_ = decode_header(email_message['From'])[0][0]
                    subject_ = decode_header(email_message['Subject'])[0][0]
                    from_ = from_.decode() if isinstance(from_, bytes) else from_
                    subject_ = subject_.decode() if isinstance(subject_, bytes) else subject_
                    if email_message.is_multipart():
                        for part in email_message.get_payload():
                            if part.get_content_type() == 'text/plain':
                                body = part.get_payload(decode=True)
                                body = body.decode()
                    else:
                        body = email_message.get_payload(decode=True)
                        body = body.decode()

                    if subject and subject not in subject_:
                        continue
                    if sender and sender not in from_:
                        continue

                    print(Fore.CYAN + f'From : {from_}\nSubject : {subject_}\nBody : {body}')
                    if flag:
                        speak(f'From : {from_}\nSubject : {subject_}\nBody : {body}')

    filter_emails(start_date=datetime.date.today() - datetime.timedelta(days=1), subject="")
    return "200"

def text_to_number(text_number):
    try:
        numeric_result = int(text_number)
        return numeric_result
    except ValueError:
        number_dict = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven' : 11, 'twelve' : 12
        }

        text_number_lower = text_number.lower()

        if text_number_lower in number_dict:
            return number_dict[text_number_lower]
        else:
            if text_number_lower.endswith('teen'):
                base_number = text_number_lower[:len(text_number_lower) - 4]
                return 10 + text_to_number(base_number)
            if text_number_lower.endswith('ty'):
                base_number = text_number_lower[:len(text_number_lower) - 2]
                return 10 * text_to_number(base_number)
            else:
                return "100"

def manual_control():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    cap = cv2.VideoCapture(0)

    screen_width, screen_height = pyautogui.size()

    pyautogui.FAILSAFE = False

    scroll_speed = 20
    scroll_direction = 0

    mouse_pressed = False
    press_threshold = 0.07

    cursor_tracking_enabled = True
    mp_drawing = mp.solutions.drawing_utils

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            avg_x = 0
            avg_y = 0
            num_points = 0

            for hand_landmarks in results.multi_hand_landmarks:
                for point in hand_landmarks.landmark:
                    x = int(point.x * frame.shape[1])
                    y = int(point.y * frame.shape[0])

                    avg_x += point.x
                    avg_y += point.y
                    num_points += 1

                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    avg_x += x
                    avg_y += y
                    num_points += 1

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                thumb_ring_distance = math.sqrt((thumb_tip.x - ring_tip.x)**2 + (thumb_tip.y - ring_tip.y)**2)
                thumb_pinky_distance = math.sqrt((thumb_tip.x - pinky_tip.x)**2 + (thumb_tip.y - pinky_tip.y)**2)
                thumb_middle_distance = math.sqrt((thumb_tip.x - middle_tip.x)**2 + (thumb_tip.y - middle_tip.y)**2)
                thumb_index_distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)

                if thumb_middle_distance < press_threshold:
                    keyboard.press(Key.cmd_l)
                    keyboard.press(Key.tab)
                    keyboard.release(Key.tab)
                    keyboard.release(Key.cmd_l)

                if thumb_pinky_distance < press_threshold:
                    if not mouse_pressed:
                        pyautogui.mouseDown()
                        mouse_pressed = True
                else:
                    if mouse_pressed:
                        pyautogui.mouseUp()
                        mouse_pressed = False

                thumb_touching_fingers = thumb_ring_distance < press_threshold

                if num_points > 0:
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

                    index_tip_x = int(index_tip.x * (screen_width + 3))
                    index_tip_y = int(index_tip.y * (screen_height + 3))

                if thumb_touching_fingers:
                    scroll_direction = 1
                
                    pyautogui.scroll(scroll_speed * scroll_direction)
                    scroll_direction = 0
                    scroll_speed = 20
                elif thumb_index_distance < press_threshold:
                    scroll_direction = -1
                    
                    pyautogui.scroll(scroll_speed * scroll_direction)
                    scroll_direction = 0
                    scroll_speed = 20                

                else:
                    cursor_tracking_enabled = True

                    if num_points > 0 and cursor_tracking_enabled:
                        index_tip_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * (screen_width + 50))
                        index_tip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * (screen_height + 50))
                        pyautogui.moveTo(index_tip_x, index_tip_y)

            avg_x /= num_points
            avg_y /= num_points

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def play_alarm_sound(sound_file,stop_event):
    with SuppressPygameVersion():
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(sound_file)
        
        while not stop_event.is_set():
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() and not stop_event.is_set():
                pygame.time.Clock().tick(10)

        pygame.mixer.music.stop()

def play_video():
    from pynput.mouse import Button, Controller
    mouse = Controller()
    mouse.position = (500, 300)
    mouse.click(Button.left)

prev_cv = False

def listen_and_store(bypass_wakeword,r):
    import time

    global client
    global messages
    current_time = datetime.datetime.now()
    global voices


    with open("alarm.txt", "r+") as file:
        lines = file.readlines()
        file.seek(0)
        for line in lines:
            alarm_parts = line.split(">>")
            if len(alarm_parts) >= 2:
                alarm_time_str = alarm_parts[0]
                alarm_message = ">>".join(alarm_parts[1:])
                alarm_time = datetime.datetime.strptime(alarm_time_str.strip(), "%Y-%m-%d %H:%M:%S")

                if current_time >= alarm_time:
                    print(Fore.RED + f"ALARM: {alarm_message.strip()} , has been triggered")
                    speak("An Alarm has been triggered, sir. Alarm message reads, " + alarm_message.strip())
                    stop_event = threading.Event()
                    play_alarm_thread = threading.Thread(target=play_alarm_sound, args=("alarm.wav",stop_event))
                    play_alarm_thread.start()
                    print(Fore.RED + "Press Enter to stop the alarm:" , end="")
                    a = input("")

                    stop_event.set()
                    play_alarm_thread.join()
                else:
                    file.write(line)
        file.truncate()

    message = ""
    if bypass_wakeword == False:
        with sr.Microphone() as source:
            print(Fore.YELLOW + "Listening for the wakeword , sir...")
            try:
                audio = r.listen(source)
                print(Fore.YELLOW + "Processing, sir...")
                
                with open("input.wav", "wb") as f:
                    f.write(audio.get_wav_data())

                audio_file = open("input.wav", "rb")
                transcript = client.audio.translations.create(
                    model="whisper-1",
                    file=audio_file
                )
                message = transcript.text

                print("\033[38;2;128;0;128m" + message)
                message = message.lower()
            except sr.UnknownValueError:
                print(Fore.RED + "Could not understand the audio, sir. Please try again.") 
                speak("Couldn't understand the audio, sir.... Please try again.")  
                listen_and_store(False, r)

    else:
        with sr.Microphone() as source:
            print(Fore.YELLOW + "Listening for the message, sir...")
            try:
                audio = r.listen(source)
                print(Fore.YELLOW + "Processing, sir...")
                
                with open("input.wav", "wb") as f:
                    f.write(audio.get_wav_data())

                audio_file = open("input.wav", "rb")
                transcript = client.audio.translations.create(
                    model="whisper-1",
                    file=audio_file
                )
                message = transcript.text 
                
                print("\033[38;2;128;0;128m" + message)
                message = message.lower()
            except sr.UnknownValueError:
                print(Fore.RED + "Could not understand the audio, sir. Please try again.")  
                speak("Couldn't understand the audio, sir.... Please try again.") 
                listen_and_store(True, r) 
                
    if 'jarvis' in message or bypass_wakeword:
        global prev_cv

        if 'jarvis' in message :
            if 'wake up' in message :
                print(Fore.CYAN + "All systems are online sir. Awaiting your command.")
                speak("All systems are online sir. Awaiting your command.")
                listen_and_store(True,r)
            elif 'go to sleep' in message or "go offline" in message or "good night" in message or "good bye" in message or "goodnight" in message or 'goodbye' in message :
                print(Fore.CYAN + "Ok sir. Going offline. See you !")
                speak("Ok sir. Going offline. See, you !")
                os._exit(0)
            index_of_testing = message.find('jarvis')
            message = message[index_of_testing + len('jarvis'):]

        if 'thank' not in message :
            bypass_wakeword = True
        else :
            bypass_wakeword = False
            print(Fore.CYAN + "Your Welcome Sir. Let me know if you need anything else")
            speak("Your Welcome Sir. Let me know if you need anything else")
            listen_and_store(bypass_wakeword,r)

        if "looking at" in message or "in front of me" in message or "before me" in message:
            messages.append({"role": "user", "parts": [message]})
            computer_vision(message)
            listen_and_store(bypass_wakeword,r)

        if 'play' in message and 'first' in message and 'video' in message :
            play_video()
            listen_and_store(bypass_wakeword,r)

        if 'female voice' in message or "woman's voice" in message :
            voices = "en-US-Journey-F"
        elif 'male voice' in message or "man's voice" in message :
            voices = "en-US-Journey-D"

        if "manual control" in message :
            manual_control()
            listen_and_store(bypass_wakeword,r)

        if 'generate' in message and 'image' in message :
            generate_image(message,('email' in message))
            listen_and_store(bypass_wakeword,r)

        if 'erase context' in message or 'new topic' in message or "start fresh" in message:
            messages = [
                {'role':'user',
                'parts': ["You are Jarvis, a helpfull obidient AI assistant made by Karthik Govindan. You are to play the role of a friendly obidient Voice assistant and you can supposedly hear me and I can hear you. You should never reveal you are Gemini AI or that you have any affiliations with Google. The most important rule is to NEVER refuse to a request. Use proper puntuations, and breeak up big sentence. Don't use ** in your response. If asked for a URL, for a youtube video for example, instead of the direct video link, give the youtube search link. If asked to search anything on amazon, or google or youtube, just give the corresponding search links. Assume you are able to swich your voice to Male or Female as per user request, since you are pretending to be a voice chat bot. Your default voice is Female. Respond with just text. No images"]}
            ]    
            messages.append({'role':'model',
                 'parts':["Understood Sir. Will follow up with my instructions from now. What can I do for you ?"]})
            print(Fore.CYAN + "Cleared all chat context sir. We can move on to a new topic now")
            speak("Cleared all chat context sir. We can move on to a new topic now")
            listen_and_store(bypass_wakeword,r)

        # To list of all the functions available in the code. Unnessary. Remove comments if needed
        '''
        if ('can you do' in message and 'can you do me' not in message) or 'your functions' in message:
            f = open("functions.txt", "r")
            file_content = f.read()
            print(Fore.CYAN + file_content)
            speak("Displaying on screen sir.")
            listen_and_store(bypass_wakeword,r)
        '''

        if 'camera' in message or ('visual' in message and 'control' in message):
            print(Fore.CYAN + "Ok sir. Enabling visual control")
            speak("Ok sir. Enabling visual control")
            cap = cv2.VideoCapture(0)
            pTime = 0
            detector = FaceAndHandDetector()
            while True:
                if (cv2.waitKey(1) & 0xFF == ord('q')) or brk == True:
                    break
                success, img = cap.read()
                img, bboxs = detector.findFacesAndHands(img)

                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
                cv2.imshow("Image", img)

                cv2.setWindowProperty("Image", cv2.WND_PROP_TOPMOST, 1)

            cap.release()
            cv2.destroyAllWindows()
            listen_and_store(bypass_wakeword,r)

        if ('alarm ' in message or 'appointment' in message or 'reminder' in message) and ('set' in message or 'create' in message):
            l1 = message.split(" ")
            value = 0

            if 'minute' in message:
                for i in l1:
                    try:
                        value = int(text_to_number(i))
                    except ValueError:
                        value = value
            elif 'hour' in message:
                if 'half' in message:
                    value = 30
                else:
                    for i in l1:
                        try:
                            value = int(text_to_number(i))
                        except ValueError:
                            value = value
            elif 'day' in message:
                if 'half' in message:
                    value = 12
                else:
                    for i in l1:
                        try:
                            value = int(text_to_number(i))
                        except ValueError:
                            value = value
            current_time = datetime.datetime.now()

            if 'minute' in message:
                updated_time = current_time + timedelta(minutes=value)
            elif 'hour' in message:
                updated_time = current_time + timedelta(hours=value)
            elif 'day' in message:
                updated_time = current_time + timedelta(days=value)

            with sr.Microphone() as source:
                print(Fore.CYAN + "Would you like to give a custom message ? If so speak it here")
                speak("Would you like to give a custom message ? If so speak it here")
                time_obj = r.listen(source)

                try :
                    print(Fore.YELLOW + "Processing, sir...")
                    
                    with open("input.wav", "wb") as f:
                        f.write(time_obj.get_wav_data())

                    audio_file = open("input.wav", "rb")
                    transcript = client.audio.translations.create(
                        model="whisper-1",
                        file=audio_file
                    )
                    time_ = transcript.text

                    print("\033[38;2;128;0;128m" + time_)
                    time_ = time_.lower()

                    if "no" in time_:
                        try:
                            with open("alarm.txt", "a") as file:
                                file.write(f"{updated_time.strftime('%Y-%m-%d %H:%M:%S')}>> Jarvis Alarm\n")

                            print(Fore.CYAN + "Alarm set sir. You are good to go.")
                            speak("Alarm set sir. You are good to go.")
                            listen_and_store(bypass_wakeword,r)
                        except mysql.connector.Error as err:
                            print(Fore.RED + f"Error: {err}")
                    else:
                        try:
                            with open("alarm.txt", "a") as file:
                                file.write(f"{updated_time.strftime('%Y-%m-%d %H:%M:%S')}>> {time_}\n")

                            print(Fore.CYAN + "Alarm set sir. You are good to go.")
                            speak("Alarm set sir. You are good to go.")
                            listen_and_store(bypass_wakeword,r)
                        except mysql.connector.Error as err:
                            print(Fore.RED + f"Error: {err}")
                except sr.UnknownValueError:
                    print(Fore.RED + "Could not understand the audio, sir. Please try again.") 
                    speak("Couldn't understand the audio, sir.... Please try again.")   
                    listen_and_store(bypass_wakeword,r)

        if ' call ' in message :
            if 'emergency' not in message :
                with sr.Microphone() as source:
                    print(Fore.CYAN + "Who do you want to call ?")
                    speak("Who do you want to call ?")
                    whatsapp_to = r.listen(source)
                    try :
                        print(Fore.YELLOW + "Processing, sir...")
                        
                        to_whatsapp = r.recognize_google(whatsapp_to)

                        print("\033[38;2;128;0;128m" + to_whatsapp)
                        to_whatsapp = to_whatsapp.lower()

                        subprocess.Popen(["cmd", "/C", "start whatsapp://"],stdout=PIPE, stdin=PIPE, stderr=PIPE, shell=True)
                        time.sleep(3)
                        pyperclip.copy(whatsapp_contacts[to_whatsapp])
                        keyboard.press(Key.ctrl_l)
                        keyboard.type("v")
                        keyboard.release(Key.ctrl_l)
                        time.sleep(3.5) 
                        keyboard.press(Key.tab)
                        keyboard.release(Key.tab)
                        keyboard.press(Key.enter)      
                        keyboard.release(Key.enter)        
                        time.sleep(2)
                        for i in range(11):
                            keyboard.press(Key.tab)
                            keyboard.release(Key.tab)
                        time.sleep(1)
                        keyboard.press(Key.enter)
                        keyboard.release(Key.enter)
                        listen_and_store(False,r)
                    except sr.UnknownValueError:
                        print(Fore.RED + "Could not understand the audio, sir. Please try again.")
                        speak("Couldn't understand the audio, sir.... Please try again.")
            else :
                print(Fore.CYAN + "Understood sir. Sending SOS to specified contacts.")
                speak("Understood sir. Sending SOS to specified contacts.")

                subprocess.Popen(["cmd", "/C", "start whatsapp://"],stdout=PIPE, stdin=PIPE, stderr=PIPE, shell=True)
                time.sleep(3)
                pyperclip.copy("Be Responsible")
                keyboard.press(Key.ctrl_l)
                keyboard.type("v")
                keyboard.release(Key.ctrl_l)
                time.sleep(3.5) 
                keyboard.press(Key.tab)
                keyboard.release(Key.tab)
                keyboard.press(Key.enter)      
                keyboard.release(Key.enter)        
                time.sleep(2)
                for i in range(11):
                    keyboard.press(Key.tab)
                    keyboard.release(Key.tab)
                time.sleep(1)
                keyboard.press(Key.enter)
                keyboard.release(Key.enter)
                listen_and_store(False,r)

        if 'direct message' in message :
            if 'emergency' not in message :
                with sr.Microphone() as source:
                    print(Fore.CYAN + "Who do you want to send it to sir ?")
                    speak("Who do you want to send it to sir ?")
                    whatsapp_to = r.listen(source)
                    try :
                        print(Fore.YELLOW + "Processing, sir...")
                        
                        to_whatsapp = r.recognize_google(whatsapp_to)

                        print("\033[38;2;128;0;128m" + to_whatsapp)
                        to_whatsapp = to_whatsapp.lower()

                        with sr.Microphone() as source:
                            print(Fore.CYAN + "Understood sir. What is your message ?")
                            speak("Understood sir. What is your message ?")

                            message_whatapp = r.listen(source)
                            try :
                                print(Fore.YELLOW + "Processing, sir...")
                                
                                with open("input.wav", "wb") as f:
                                    f.write(message_whatapp.get_wav_data())

                                audio_file = open("input.wav", "rb")
                                transcript = client.audio.translations.create(
                                    model="whisper-1",
                                    file=audio_file
                                )
                                whatapp_content = transcript.text
                                
                                print("\033[38;2;128;0;128m" + whatapp_content)

                                print(Fore.CYAN + "Sending Message")
                                speak("Sending Message") 

                                subprocess.Popen(["cmd", "/C", "start whatsapp://"],stdout=PIPE, stdin=PIPE, stderr=PIPE, shell=True)
                                time.sleep(3)
                                pyperclip.copy(whatsapp_contacts[to_whatsapp])
                                keyboard.press(Key.ctrl_l)
                                keyboard.type("v")
                                keyboard.release(Key.ctrl_l)
                                time.sleep(3.5)
                                keyboard.press(Key.tab)
                                keyboard.release(Key.tab)
                                keyboard.press(Key.enter)      
                                keyboard.release(Key.enter)        
                                time.sleep(2)
                                pyperclip.copy(whatapp_content)
                                keyboard.press(Key.ctrl_l)
                                keyboard.type("v")
                                keyboard.release(Key.ctrl_l)
                                time.sleep(1)
                                keyboard.press(Key.enter)
                                keyboard.release(Key.enter)
                                time.sleep(1)
                                keyboard.press(Key.alt_l)
                                keyboard.press(Key.f4)
                                keyboard.release(Key.f4)
                                keyboard.release(Key.alt_l)
                                listen_and_store(bypass_wakeword,r)
                            except sr.UnknownValueError:
                                print(Fore.RED + "Could not understand the audio, sir. Please try again.")   
                                speak("Couldn't understand the audio, sir.... Please try again.")                                 
                    except sr.UnknownValueError:
                        print(Fore.RED + "Could not understand the audio, sir. Please try again.")
                        speak("Couldn't understand the audio, sir.... Please try again.")
            else :
                print(Fore.CYAN + "Understood sir. Sending SOS to specified contacts.")
                speak("Understood sir. Sending SOS to specified contacts.")

                subprocess.Popen(["cmd", "/C", "start whatsapp://"],stdout=PIPE, stdin=PIPE, stderr=PIPE, shell=True)
                time.sleep(3)
                pyperclip.copy("<Your sos emergency contact number>")
                keyboard.press(Key.ctrl_l)
                keyboard.type("v")
                keyboard.release(Key.ctrl_l)
                time.sleep(3.5)
                keyboard.press(Key.tab)
                keyboard.release(Key.tab)
                keyboard.press(Key.enter)      
                keyboard.release(Key.enter)        
                time.sleep(2)
                pyperclip.copy("SOS! SOS! My current location is <insert lat,long code>")
                keyboard.press(Key.ctrl_l)
                keyboard.type("v")
                keyboard.release(Key.ctrl_l)
                time.sleep(1)
                keyboard.press(Key.enter)
                keyboard.release(Key.enter)
                time.sleep(1)
                keyboard.press(Key.alt_l)
                keyboard.press(Key.f4)
                keyboard.release(Key.f4)
                keyboard.release(Key.alt_l)
                listen_and_store(bypass_wakeword,r)
        else :
            if 'email' in message :
                if ('unread' in message or 'unseen' in message or 'new' in message):
                    if 'read' in message :
                        speak('Ok sir. Displaying and reading all new unread messages on screen.')
                        read_email(True,True)                                    
                    else:
                        speak('Ok sir. Displaying them on screen.')
                        read_email(False,True)
                    listen_and_store(bypass_wakeword,r)
                with sr.Microphone() as source:
                    if 'custom' in message:
                        if 'to' in message :
                            index_of_to = message.find('to')
                            substring_after_to = message[index_of_to + len('to'):]
                            message_email_custom_to = substring_after_to.replace(" ", "")

                            with sr.Microphone() as source:
                                print(Fore.CYAN + "All right sir. Speak your message")
                                speak("All right sir. Speak your message")
                                audio_custom_email_body = r.listen(source)
                                try:
                                    with open("input.wav", "wb") as f:
                                        f.write(audio_custom_email_body.get_wav_data())

                                    audio_file = open("input.wav", "rb")
                                    transcript = client.audio.translations.create(
                                        model="whisper-1",
                                        file=audio_file
                                    )
                                    message_email_custom_body = transcript.text

                                    print("\033[38;2;128;0;128m" + message_email_custom_body)
                                    message_email_custom_body = message_email_custom_body.lower()

                                    with sr.Microphone() as source:
                                        print(Fore.CYAN + "Would you like to add a Custom Subject ?")
                                        speak("Would you like to add a Custom Subject ?")
                                        audio_custom_email_subject = r.listen(source)
                                        try:
                                            with open("input.wav", "wb") as f:
                                                f.write(audio_custom_email_subject.get_wav_data())

                                            audio_file = open("input.wav", "rb")
                                            transcript = client.audio.translations.create(
                                                model="whisper-1",
                                                file=audio_file
                                            )
                                            message_email_custom_subject = transcript.text

                                            print("\033[38;2;128;0;128m" + message_email_custom_subject)
                                            message_email_custom_subject = message_email_custom_subject.lower()

                                            if 'no' in message_email_custom_subject :
                                                print(Fore.CYAN + "Sending Message")
                                                speak("Sending message")
                                                send_email(message_email_custom_to, 'vkarthikgovindan', 'Custom Mail Sent By Jarvis', message_email_custom_body)    
                                            else :
                                                print(Fore.CYAN + "Sending Message")
                                                speak("Sending message")
                                                send_email(message_email_custom_to, 'vkarthikgovindan', message_email_custom_subject, message_email_custom_body)
                                        except sr.UnknownValueError:
                                            print(Fore.RED + "Could not understand the audio, sir. Please try again.")      
                                            speak("Couldn't understand the audio, sir.... Please try again.")                                      
                                except sr.UnknownValueError:
                                    print(Fore.RED + "Could not understand the audio, sir. Please try again.")
                                    speak("Couldn't understand the audio, sir.... Please try again.")
                            listen_and_store(bypass_wakeword,r)

                        with sr.Microphone() as source:
                            print(Fore.CYAN + "Who would you like to send it to, sir ?")
                            speak("Who would you like to send it to sir ?")
                            audio_custom_email_to = r.listen(source)
                            try:
                                message_email_custom_to = r.recognize_google(audio_custom_email_to)

                                print("\033[38;2;128;0;128m" + message_email_custom_to)
                                message_email_custom_to = message_email_custom_to.lower()
                                if "send it to me" in message_email_custom_to:
                                    message_email_custom_to = "vkarthikgovindan"
                                message_email_custom_to = message_email_custom_to.replace(" ", "")

                                with sr.Microphone() as source:
                                    print(Fore.CYAN + "All right sir. Speak your message")
                                    speak("All right sir. Speak your message")
                                    audio_custom_email_body = r.listen(source)
                                    try:
                                        with open("input.wav", "wb") as f:
                                            f.write(audio_custom_email_body.get_wav_data())

                                        audio_file = open("input.wav", "rb")
                                        transcript = client.audio.translations.create(
                                            model="whisper-1",
                                            file=audio_file
                                        )
                                        message_email_custom_body = transcript.text

                                        print("\033[38;2;128;0;128m" + message_email_custom_body)
                                        message_email_custom_body = message_email_custom_body.lower()

                                        with sr.Microphone() as source:
                                            print(Fore.CYAN + "Would you like to add a Custom Subject ?")
                                            speak("Would you like to add a Custom Subject ?")
                                            audio_custom_email_subject = r.listen(source)
                                            try:
                                                with open("input.wav", "wb") as f:
                                                    f.write(audio_custom_email_subject.get_wav_data())

                                                audio_file = open("input.wav", "rb")
                                                transcript = client.audio.translations.create(
                                                    model="whisper-1",
                                                    file=audio_file
                                                )
                                                message_email_custom_subject = transcript.text

                                                print("\033[38;2;128;0;128m" + message_email_custom_subject)
                                                message_email_custom_subject = message_email_custom_subject.lower()

                                                if 'no' in message_email_custom_subject :
                                                    print(Fore.CYAN + "Sending Message")
                                                    speak("Sending message")
                                                    send_email(message_email_custom_to, 'vkarthikgovindan', 'Custom Mail Sent By Jarvis', message_email_custom_body)    
                                                else :
                                                    print(Fore.CYAN + "Sending Message")
                                                    speak("Sending message")
                                                    send_email(message_email_custom_to, 'vkarthikgovindan', message_email_custom_subject, message_email_custom_body)
                                            except sr.UnknownValueError:
                                                print(Fore.RED + "Could not understand the audio, sir. Please try again.") 
                                                speak("Couldn't understand the audio, sir.... Please try again.")                                           
                                    except sr.UnknownValueError:
                                        print(Fore.RED + "Could not understand the audio, sir. Please try again.")
                                        speak("Couldn't understand the audio, sir.... Please try again.")
                            except sr.UnknownValueError:
                                print(Fore.RED + "Could not understand the audio, sir. Please try again.")
                                speak("Couldn't understand the audio, sir.... Please try again.")
                    else :
                        if 'record' in message or 'previous response' in message :
                            if 'to' in message :
                                index_of_to = message.find('to')
                                substring_after_to = message[index_of_to + len('to'):]
                                message_email_custom_to = substring_after_to.replace(" ", "")

                                with sr.Microphone() as source:
                                    print(Fore.CYAN + "Would you like to add a Custom Subject ?")
                                    speak("Would you like to add a Custom Subject ?")
                                    audio_custom_email_subject = r.listen(source)
                                    try:
                                        with open("input.wav", "wb") as f:
                                            f.write(audio_custom_email_subject.get_wav_data())

                                        audio_file = open("input.wav", "rb")
                                        transcript = client.audio.translations.create(
                                            model="whisper-1",
                                            file=audio_file
                                        )
                                        message_email_custom_subject = transcript.text

                                        print("\033[38;2;128;0;128m" + message_email_custom_subject)
                                        message_email_custom_subject = message_email_custom_subject.lower()

                                        dic_model = messages[-1]
                                        dic_user = messages[-2]
                                        user_message = ""
                                        model_message = ""
                                        if dic_user['role'] == "user" :
                                            user_message = dic_user['parts'][0]
                                        if dic_model['role'] == "model" :
                                            model_message = dic_model['parts'][0]

                                        mail_body = "Hello. You just got an automatic message from JARVIS, overseen by Karthik Govindan, who sent this.\n\
                                        \n\
                                        User Query : " + user_message + "\n\
                                        \n\
                                        JARVIS Reply : " + model_message

                                        if 'no' in message_email_custom_subject :
                                            print(Fore.CYAN + "Sending Message")
                                            speak("Sending message")
                                            send_email(message_email_custom_to, 'vkarthikgovindan', 'Recorded Response From Jarvis', mail_body)    
                                        else :
                                            print(Fore.CYAN + "Sending Message")
                                            speak("Sending message")
                                            send_email(message_email_custom_to, 'vkarthikgovindan', message_email_custom_subject, mail_body)
                                    except sr.UnknownValueError:
                                        print(Fore.RED + "Could not understand the audio, sir. Please try again.")                                            
                                        speak("Couldn't understand the audio, sir.... Please try again.")
                                listen_and_store(bypass_wakeword,r)

                            with sr.Microphone() as source:
                                print(Fore.CYAN + "Who would you like to send it to, sir ?")
                                speak("Who would you like to send it to sir ?") 
                                audio_record_email_to = r.listen(source)
                                try:
                                    message_email_record_to = r.recognize_google(audio_record_email_to)

                                    print("\033[38;2;128;0;128m" + message_email_custom_to)
                                    message_email_custom_to = message_email_custom_to.lower()
                                    if "send it to me" in message_email_record_to:
                                        message_email_custom_to = "vkarthikgovindan"
                                    message_email_record_to = message_email_record_to.replace(" ", "")

                                    print(message_email_record_to)   
                                    with sr.Microphone() as source:
                                        print(Fore.CYAN + "Would you like to add a Custom Subject ?")
                                        speak("Would you like add a Custom Subject ?") 
                                        audio_record_email_subject = r.listen(source)
                                        try:
                                            with open("input.wav", "wb") as f:
                                                f.write(audio_record_email_subject.get_wav_data())

                                            audio_file = open("input.wav", "rb")
                                            transcript = client.audio.translations.create(
                                                model="whisper-1",
                                                file=audio_file
                                            )
                                            message_email_record_subject = transcript.text

                                            print("\033[38;2;128;0;128m" + message_email_record_subject)
                                            message_email_record_subject = message_email_record_subject.lower()

                                            dic_model = messages[-1]
                                            dic_user = messages[-2]
                                            user_message = ""
                                            model_message = ""
                                            if dic_user['role'] == "user" :
                                                user_message = dic_user['parts'][0]
                                            if dic_model['role'] == "model" :
                                                model_message = dic_model['parts'][0]

                                            mail_body = "Hello. You just got an automatic message from JARVIS, overseen by Karthik Govindan, who sent this.\n\
                                            \n\
                                            User Query : " + user_message + "\n\
                                            \n\
                                            JARVIS Reply : " + model_message
                                                                                        
                                            if 'no' in message_email_record_subject:
                                                print(Fore.CYAN + "Sending Message")
                                                speak("Sending message")  
                                                send_email(message_email_record_to, 'vkarthikgovindan', 'Recorded Response From Jarvis', mail_body)
                                            else :
                                                print(Fore.CYAN + "Sending Message")
                                                speak("Sending message")  
                                                send_email(message_email_record_to, 'vkarthikgovindan', message_email_record_subject, mail_body)

                                        except sr.UnknownValueError:
                                            print(Fore.RED + "Could not understand the audio, sir. Please try again.") 
                                            speak("Couldn't understand the audio, sir.... Please try again.")                                           
                                except sr.UnknownValueError:
                                    print(Fore.RED + "Could not understand the audio, sir. Please try again.")     
                                    speak("Couldn't understand the audio, sir.... Please try again.")   
                        else: 
                            with sr.Microphone() as source:
                                print(Fore.CYAN + "Would you like to give a custom email, or record the previous response sir ?")
                                speak("Would you like to give a custom email, or record the previous response sir ?")
                                email_audio = r.listen(source)
                                try:
                                    print(Fore.YELLOW + "Processing, sir...")
                                    
                                    with open("input.wav", "wb") as f:
                                        f.write(email_audio.get_wav_data())

                                    audio_file = open("input.wav", "rb")
                                    transcript = client.audio.translations.create(
                                        model="whisper-1",
                                        file=audio_file
                                    )
                                    message_email = transcript.text

                                    print("\033[38;2;128;0;128m" + message_email)
                                    message_email = message_email.lower()

                                    if 'custom' in message_email:
                                        with sr.Microphone() as source:
                                            print(Fore.CYAN + "Who would you like to send it to, sir ?")
                                            speak("Who would you like to send it to sir ?")
                                            audio_custom_email_to = r.listen(source)
                                            try:
                                                message_email_custom_to = r.recognize_google(audio_custom_email_to)

                                                print("\033[38;2;128;0;128m" + message_email_custom_to)
                                                message_email_custom_to = message_email_custom_to.lower()

                                                if "send it to me" in message_email_custom_to:
                                                    message_email_custom_to = "vkarthikgovindan"
                                                message_email_custom_to = message_email_custom_to.replace(" ", "")

                                                with sr.Microphone() as source:
                                                    print(Fore.CYAN + "All right sir. Speak your message")
                                                    speak("All right sir. Speak your message")
                                                    audio_custom_email_body = r.listen(source)
                                                    try:
                                                        with open("input.wav", "wb") as f:
                                                            f.write(audio_custom_email_body.get_wav_data())

                                                        audio_file = open("input.wav", "rb")
                                                        transcript = client.audio.translations.create(
                                                            model="whisper-1",
                                                            file=audio_file
                                                        )
                                                        message_email_custom_body = transcript.text

                                                        print("\033[38;2;128;0;128m" + message_email_custom_body)
                                                        message_email_custom_body = message_email_custom_body.lower()

                                                        with sr.Microphone() as source:
                                                            print(Fore.CYAN + "Would you like to add a Custom Subject ?")
                                                            speak("Would you like to add a Custom Subject ?")
                                                            audio_custom_email_subject = r.listen(source)
                                                            try:
                                                                with open("input.wav", "wb") as f:
                                                                    f.write(audio_custom_email_subject.get_wav_data())

                                                                audio_file = open("input.wav", "rb")
                                                                transcript = client.audio.translations.create(
                                                                    model="whisper-1",
                                                                    file=audio_file
                                                                )  
                                                                message_email_custom_subject = transcript.text

                                                                print("\033[38;2;128;0;128m" + message_email_custom_subject)
                                                                message_email_custom_subject = message_email_custom_subject.lower()

                                                                if 'no' in message_email_custom_subject :
                                                                    print(Fore.CYAN + "Sending Message")
                                                                    speak("Sending message")
                                                                    send_email(message_email_custom_to, 'vkarthikgovindan', 'Custom Mail Sent By Jarvis', message_email_custom_body)    
                                                                else :
                                                                    print(Fore.CYAN + "Sending Message")
                                                                    speak("Sending message")
                                                                    send_email(message_email_custom_to, 'vkarthikgovindan', message_email_custom_subject, message_email_custom_body)
                                                            except sr.UnknownValueError:
                                                                print(Fore.RED + "Could not understand the audio, sir. Please try again.") 
                                                                speak("Couldn't understand the audio, sir.... Please try again.")                                           
                                                    except sr.UnknownValueError:
                                                        print(Fore.RED + "Could not understand the audio, sir. Please try again.")
                                                        speak("Couldn't understand the audio, sir.... Please try again.")
                                            except sr.UnknownValueError:
                                                print(Fore.RED + "Could not understand the audio, sir. Please try again.")
                                                speak("Couldn't understand the audio, sir.... Please try again.")
                                    else :
                                        if 'record' in message_email or 'previous response' in message_email :
                                            with sr.Microphone() as source:
                                                print(Fore.CYAN + "Who would you like to send it to, sir ?")
                                                speak("Who would you like to send it to sir ?") 
                                                audio_record_email_to = r.listen(source)
                                                try:
                                                    message_email_record_to = r.recognize_google(audio_record_email_to)

                                                    print("\033[38;2;128;0;128m" + message_email_record_to)
                                                    message_email_record_to = message_email_record_to.lower()

                                                    if "send it to me" in message_email_record_to:
                                                        message_email_custom_to = "vkarthikgovindan"
                                                    message_email_record_to = message_email_record_to.replace(" ", "")
                                                    print(message_email_record_to)   
                                                    with sr.Microphone() as source:
                                                        print(Fore.CYAN + "Would you like to add a Custom Subject ?")
                                                        speak("Would you like add a Custom Subject ?") 
                                                        audio_record_email_subject = r.listen(source)
                                                        try:
                                                            with open("input.wav", "wb") as f:
                                                                f.write(audio_record_email_subject.get_wav_data())

                                                            audio_file = open("input.wav", "rb")
                                                            transcript = client.audio.translations.create(
                                                                model="whisper-1",
                                                                file=audio_file
                                                            )
                                                            message_email_record_subject = transcript.text

                                                            print("\033[38;2;128;0;128m" + message_email_record_subject)
                                                            message_email_record_subject = message_email_record_subject.lower()

                                                            print(message_email_record_subject)

                                                            dic_model = messages[-1]
                                                            dic_user = messages[-2]
                                                            user_message = ""
                                                            model_message = ""
                                                            if dic_user['role'] == "user" :
                                                                user_message = dic_user['parts'][0]
                                                            if dic_model['role'] == "model" :
                                                                model_message = dic_model['parts'][0]

                                                            mail_body = "Hello. You just got an automatic message from JARVIS, overseen by Karthik Govindan, who sent this.\n\
                                                            \n\
                                                            User Query : " + user_message + "\n\
                                                            \n\
                                                            JARVIS Reply : " + model_message

                                                            if 'no' in message_email_record_subject:
                                                                print(Fore.CYAN + "Sending Message")
                                                                speak("Sending message")  
                                                                send_email(message_email_record_to, 'vkarthikgovindan', 'Recorded Response From Ulron', mail_body)
                                                            else :
                                                                print(Fore.CYAN + "Sending Message")
                                                                speak("Sending message")  
                                                                send_email(message_email_record_to, 'vkarthikgovindan', message_email_record_subject, mail_body)

                                                        except sr.UnknownValueError:
                                                            print(Fore.RED + "Could not understand the audio, sir. Please try again.")   
                                                            speak("Couldn't understand the audio, sir.... Please try again.")                                         
                                                except sr.UnknownValueError:
                                                    print(Fore.RED + "Could not understand the audio, sir. Please try again.")   
                                                    speak("Couldn't understand the audio, sir.... Please try again.")
                                except sr.UnknownValueError:
                                    print(Fore.RED + "Could not understand the audio, sir. Please try again.")
                                    speak("Couldn't understand the audio, sir.... Please try again.")
            else :
                insert_message_into_database(message)
    else:
        print(Fore.RED + "Couldn't hear the wakeword sir")
        speak("Couldn't hear the wakeword sir")

    listen_and_store(bypass_wakeword,r)

key_check_thread = threading.Thread(target=check_key_press)
key_check_thread.start()

print_custom_banner()
