from os import makedirs
import cv2
import numpy as np
import socket
from os.path import isdir
from os import listdir
from os.path import isfile, join
from scipy.misc import face

# TCP_IP = '10.10.20.37'
# TCP_PORT = 9010
# sock = socket.socket()
# sock.connect((TCP_IP, TCP_PORT))

face_dirs='/home/iot/study/python/opencv/Facial-Recognition/faces/'

face_classifier = cv2.CascadeClassifier('/home/iot/study/python/opencv/haarcascade_frontalface_default.xml') #얼굴 인식용 xml파일


def face_extractor(img): #전체 사진에서 얼굴 부위만 잘라 리턴
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #흑백처리

    faces = face_classifier.detectMultiScale(gray,1.3,5) #얼굴찾기

    if faces is():
        return None     #찾은 얼굴이 없으면 None으로 리턴
    
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]   #얼굴들이 있으면해당 얼굴 크기만큼 cropped_face에 잘라 넣기 
   
    return cropped_face     #cropped_face 리턴

def take_picture(name):
    
    if not isdir(face_dirs+name): #해당 유저 폴더 없다면 생성
        makedirs(face_dirs+name)
    
    cap = cv2.VideoCapture(0) #카메라 실행

    count = 0 #저장할 이미지 카운트 변수

    while True:
        ret, frame = cap.read()     #카메라로 부터 사진 1장 얻기 

        if face_extractor(frame) is not None:     #얼굴 감지 하여 얼굴만 가져오기
            count+=1
            face = cv2.resize(face_extractor(frame),(200,200))         #얼굴 이미지 크기를 200x200으로 조정 

            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)    #조정된 이미지를 흑백으로 변환 

            file_name_path = face_dirs+name+'/user'+str(count)+'.jpg' # faces/얼굴 이름/userxx.jpg 로 저장      
            cv2.imwrite(file_name_path,face)

            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)         #화면에 얼굴과 현재 저장 개수 표시 
            cv2.imshow('Face Cropper',face)
        else:
            print("Face not Found")
            pass

        if cv2.waitKey(1)==13 or count==100: # 얼굴 사진 100장을 다 얻었거나 enter키 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Colleting Samples Complete!!!')
    

def study_picture(name): #사용자 얼굴 학습
    data_path = face_dirs + name + '/'

    face_pics = [f for f in listdir(data_path) if isfile(join(data_path,f))]     #파일만 리스트로 만듬
    
    Training_Data, Labels = [], [] #데이터와 매칭될 라벨 변수 

    for i, files in enumerate(face_pics): #파일 개수 만큼 루프 
        image_path = data_path + face_pics[i] 
        image_array = np.fromfile(image_path, np.uint8) # 컴퓨터가 읽을수 있게 넘파이로 변환 
        images = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)     #이미지 불러오기 
        if images is None:     #이미지 파일이 아니거나 못 읽어 왔다면 무시
            continue
        Training_Data.append(np.asarray(images, dtype=np.uint8))     #Training_Data 리스트에 이미지를 바이트 배열로 추가
        Labels.append(i)     #Labels 리스트엔 카운트 번호 추가 

    if len(Labels) == 0: #훈련할 데이터가 없다면 종료.
        print("There is no data to train.")
        exit()
    
    
    Labels = np.asarray(Labels, dtype=np.int32) #Labels를 32비트 정수로 변환 

    model = cv2.face.LBPHFaceRecognizer_create() #모델 생성 

    model.train(np.asarray(Training_Data), np.asarray(Labels)) #학습 시작 

    print("Model Training Complete!!!!!")

    return model #학습 모델 리턴


def study_pictures(): # 여러 사용자 학습

    data_path = face_dirs   #faces 폴더의 하위 폴더를 학습

    model_dirs = [f for f in listdir(data_path) if isdir(join(data_path,f))]    # 폴더만 색출

    models = {} #학습 모델 저장할 딕셔너리

    for model in model_dirs:     # 각 폴더에 있는 얼굴들 학습
        print('model :' + model)
       
        result = study_picture(model) # 학습 시작

        if result is None:         # 학습이 안되었다면 패스!
            continue

        print('model2 :' + model)
        models[model] = result         # 학습되었으면 저장

    return models         # 학습된 모델 딕셔너리 리턴


def face_detector(img, size = 0.5): #얼굴 검출
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)
        if faces is():
            return img,[]
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))
        return img,roi   #검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달


def check_picture(models):    #얼굴 인식

    cap = cv2.VideoCapture(0)     #카메라 열기 
    
    while True:
        ret, frame = cap.read()         #카메라로 부터 사진 한장 읽기 

        image, face = face_detector(frame)        # 얼굴 검출 시도 
        try:            
            min_score = 999       #가장 낮은 점수로 예측된 사람의 점수
            min_score_name = ""   #가장 높은 점수로 예측된 사람의 이름
            
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)             #검출된 사진을 흑백으로 변환 

            for key, model in models.items():             #위에서 학습한 모델로 예측시도
                result = model.predict(face)                
                if min_score > result[1]:
                    min_score = result[1]
                    min_score_name = key
                             
            if min_score < 500:             #min_score 신뢰도이고 0에 가까울수록 자신과 같다는 뜻이다.
                confidence = int(100*(1-(min_score)/300))
                display_string = str(confidence)+'% Confidence it is ' + min_score_name                 # 유사도 화면에 표시 
            cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)

            if confidence > 75:             #75 보다 크면 동일 인물로 간주해 UnLocked! 
                cv2.putText(image, "Unlocked : " + min_score_name, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', image)
            else:         #75 이하면 타인.. Locked!!! 
                cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Cropper', image)
        except:             #얼굴 검출 안됨 
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Cropper', image)
            pass
        if cv2.waitKey(1)==13:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # sock.send('ㅁㅁㅁㅁ'.encode('utf-8'))
    # while True:
    #     data = sock.recv(1024).decode()
    #     print('signal : ' + data)
    #     split_data = data.split('/')
    #     if split_data[0] == "!signup": #회원가입
    #         take_picture(split_data[1]) #사진 저장할 이름 넣어서 호출
    #         models=study_pictures()
    #     elif split_data[0] == "!login": #로그인
    #         check_picture(models)
    #     else:
    #         print('else로 빠짐')
    #         print(data)
    take_picture('Bang')
    models=study_pictures()
    check_picture(models)
