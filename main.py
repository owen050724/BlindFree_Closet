import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import time, os
from urllib.request import urlopen
from bs4 import BeautifulSoup
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from colorthief import ColorThief
import colorsys
from PIL import Image, ImageOps

main_link = 'C:\\Users\\owen0\\Desktop\\haekaton\\'
dic1={'T-shirt':'티셔츠','dress_shirts':'셔츠','sweater':'스웨터','turtle_neck':'폴라티','hood':'후드','jeans':'청바지','sweatpants':'츄리닝','slacks':'슬랙스'}
def set_name(name):
    global color_link
    global ori_link
    global save_link
    color_link = main_link + f'img\\color_{name}.jpg'
    ori_link = main_link + f'img\\{name}.jpg'
    save_link = main_link + f'data\\{name}'

#음성 인식(STT)
def listen(recognizer, audio):
    try:
        text = recognizer.recognize_google(audio, language='ko')
        print('[사용자] '+text)
        answer(text)
    except sr.UnknownValueError:
        print('인식 실패') # 음성 인식 실패 시
    except sr.RequestError:
        print('요청 실패 : {0}'.format(e)) #API Key 오류, 또는 네트워크 오류

#대답
def answer(input_text):
    answer_text = ' '
    if '안녕' in input_text:
        answer_text = '안녕하세요? 반갑습니다.'
    elif '날씨' in input_text:
        answer_text1, atext = weather()
        answer_text = answer_text1+atext
    elif '옷 정보' in input_text:
        img_save('test')
        answer_text1 = clothes_info()
        set_name(str(answer_text1))
        image = cv2.imread(main_link+'test.jpg')
        cv2.imwrite(ori_link, image)        
        clr = color(ori_link)
        answer_text = f'이 옷은 {dic1[answer_text1]}입니다. \n 이 옷의 색깔은 {clr}입니다.'
    elif '옷 추천' in input_text:
        img_save('test')
        answer_text1 = clothes_info()
        set_name(str(answer_text1))
        image = cv2.imread(main_link+'test.jpg')
        cv2.imwrite(ori_link, image)        
        clr = color(ori_link)
        if answer_text1 in ['T-shirt','dress_shirts','sweater', 'turtle_neck','hood']:
            upco = clr
            
            if upco == '흰색':
                answer_text2='블랙,베이지,연청색,진청색의 하의를 추천합니다.'
            elif upco == '빨간색' :
                answer_text2='와인색의 하의를 추천합니다.'
            elif upco == '노란색':
                answer_text2='청색,베이지색,와인색의 하의를 추천합니다.'
            elif upco == '주황색':
                answer_text2='진청색의 하의를 추천합니다.'
            elif upco == '초록색':
                answer_text2='진청색의 하의를 추천합니다.'
            elif upco == '파란색':
                answer_text2='베이지색, 와인색, 검은색의 하의를 추천합니다.'
            elif upco == '검정색':
                answer_text2='진청색, 베이지, 검은 색의 하의를 추천합니다.'
            elif upco == '회색':
                answer_text2='검은색의 하의를 추천합니다.'
            else :
                answer_text2='리스트에 없는 옷입니다.'             
            answer_text = f'{clr}의 {answer_text1}는'+answer_text2
        else:
            doco = clr

            if doco == '파란색' :
                answer_text2='흰 색의 상의를 추천합니다.'
            elif doco == '회색' :
                answer_text2='흰 색의 상의를 추천합니다.'
            elif doco == '초록색' :
                answer_text2='흰 색의 상의를 추천합니다.'
            elif doco == '검정색' :
                answer_text2='검은색, 하얀색, 회색의 상의를 추천합니다.'
            else:
                answer_text2='리스트에 없는 옷입니다.'
            answer_text = f'{clr}의 {answer_text1}는'+answer_text2
        
    
    elif '고마워' in input_text:
        answer_text = '별 말씀을요.'
    elif '종료' in input_text:
        answer_text = '다음에 또 만나요.'
        speak(answer_text)
        stop_listening(wait_for_stop=False)
        return 0
    elif '등록' in input_text:
        regit()
    else:
        answer_text = '다시 한 번 말씀해주시겠어요?'
    
    speak(answer_text)

def regit():
    f = open(main_link+'database.txt', 'r')
    lines = f.readlines()
    count = len(lines)+1
    f.close
    img_save('test')
    answer_text1 = clothes_info()
    set_name(str(answer_text1))
    image = cv2.imread(main_link+'test.jpg')
    cv2.imwrite(save_link+str(count)+'.jpg', image)        
    clr = color(save_link+str(count)+'.jpg')
    f = open(main_link+'database.txt', 'a')
    f.write(f"{count} {str(answer_text1)} {clr}")
    f.close
    
#읽기(TTS)
def speak(text):
    print('[인공지능] '+text)
    file_name = 'voice.mp3'
    tts = gTTS(text=text, lang='ko')
    tts.save(file_name)
    playsound(file_name)
    if os.path.exists(file_name):
        os.remove(file_name)
        
def weather():
    url="https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query=%EB%82%B4%EC%9D%BC+%EB%B6%80%EC%82%B0+%EB%82%A0%EC%94%A8"
    html=urlopen(url).read()
    url0="https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query=%EB%B6%80%EC%82%B0%20%ED%95%B4%EC%9A%B4%EB%8C%80%EA%B5%AC%20%EB%82%A0%EC%94%A8"
    html0=urlopen(url0).read()

    soup=BeautifulSoup(html,'html.parser')
    soup0=BeautifulSoup(html0,'html.parser')

    temp=soup.find(class_="temperature_info")
    now=soup0.find(class_='temperature_text')

    temp1=temp.text[5:]
    now1=now.text[0:6]+"는"+now.text[6:]

    answer_text1 = now1+"입니다."
    
    ntem=float(now1[-6:-2])
    if ntem>=23:
        atext='가벼운 옷차림을 추천드립니다. 반팔이나 민소매가 좋아요.'
    elif ntem<23 and ntem>=17:
        atext='가벼운 옷차림을 추천드립니다.소매가 긴 옷이 좋아요.'
    elif ntem<17 and ntem>=10:
        atext='따뜻한 옷차림을 추천드립니다. 너무 두꺼운 옷보다는 얇은 옷을 여러겹 껴입으세요'
    elif ntem<10:
        atext='따뜻한 옷차림을 추천드립니다. 매우 추우니 야상 혹은 패딩을 입는 것이 좋아요.'
    
    return answer_text1,atext

def clothes_info():
    np.set_printoptions(suppress=True)

    model = load_model("C:\\Users\\owen0\\Desktop\\haekaton\\realuse\\keras_model.h5", compile=False)
    class_names = open("C:\\Users\\owen0\\Desktop\\haekaton\\realuse\\labels.txt", "r", encoding="UTF8").readlines()

    while True:
        image = cv2.imread("C:\\Users\\owen0\\Desktop\\haekaton\\test.jpg")
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image/127.5)-1

    
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        
        print("Class: ", class_name[2:], end="")
        print("Confidence Score: ", str(np.round(confidence_score*100))[:-2], "%")
        
        clothes_info_result = class_name[2:-1]
        return clothes_info_result
    cv2.destroyAllWindows()
    
def img_save(name):
    cap= cv2.VideoCapture(1)
    if cap.isOpened():
    
        while True:
            ret,frame = cap.read()
        
            if ret:
                cv2.imshow('camera', frame)

                if cv2.waitKey(1):
                    cv2.imwrite(main_link+f"{name}.jpg", frame)
                    break
            else:
                print('no frame')
                break
    else:
        print('no camera!')
    cap.release()
    cv2.destroyAllWindows()

def pal(link):
    ct = ColorThief(link)
    dominant_color = ct.get_color(quality=1)
    plt.imshow([[dominant_color]])
    plt.savefig(color_link)
    image = cv2.imread(color_link)
    resized_image = cv2.resize(image, (1280 ,960))
    resized_image_1 = resized_image[500:800, 400:750]
    cv2.imwrite(color_link, resized_image_1)

def color(link):
    pal(link)
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model(main_link + "keras_Model.h5", compile=False)

    #   Load the labels
    class_names = open(main_link + "labels.txt", "r",encoding="UTF8").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(color_link).convert("RGB")
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    #image.save('C:\\Users\\bj894\\OneDrive\\Desktop\\yee\\img\\croppedImage.PNG')
    #croppedImage = image.crop((50,50,50,50))
    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    return class_name[2:-1]



r = sr.Recognizer()
m = sr.Microphone()

speak('무엇을 도와드릴까요?')
stop_listening = r.listen_in_background(m, listen)
# stop_listening(wait_for_stop=False) 더 이상 듣지 않음

while True:
    time.sleep(0.1)