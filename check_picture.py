import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# 여기서부터는 study.py와 동일 
data_path = '/home/iot/study/python/opencv/Facial-Recognition/faces'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
Training_Data, Labels = [], []
for i, files in enumerate(onlyfiles): #파일 개수 만큼 루프 
    image_path = data_path + '/'+files 
    image_array = np.fromfile(image_path, np.uint8) # 컴퓨터가 읽을수 있게 넘파이로 변환 
    images = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)     #이미지 불러오기 
    if images is None:     #이미지 파일이 아니거나 못 읽어 왔다면 무시
        continue
    Training_Data.append(np.asarray(images, dtype=np.uint8))     #Training_Data 리스트에 이미지를 바이트 배열로 추가
    Labels.append(i)     #Labels 리스트엔 카운트 번호 추가 
if len(Labels) == 0:
    print("There is no data to train.")
    exit()

Labels = np.asarray(Labels, dtype=np.int32)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model Training Complete!!!!!")
# 여기까지 study.py와 동일 

# 여긴 take.py와 거의 동일 
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return img,[]
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))
    return img,roi   #검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달
# 여기까지 take.py와 거의 동일 


cap = cv2.VideoCapture(0) #카메라 열기 

while True:
    ret, frame = cap.read()     #카메라로 부터 사진 한장 읽기 

    image, face = face_detector(frame)     # 얼굴 검출 시도 
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)         #검출된 사진을 흑백으로 변환 

        result = model.predict(face)         #위에서 학습한 모델로 예측시도 

        if result[1] < 500:         #result[1]은 신뢰도이고 0에 가까울수록 자신과 같다는 뜻이다. 
            
            confidence = int(100*(1-(result[1])/300))
                 
            display_string = str(confidence)+'% Confidence it is user'   # 유사도 화면에 표시 
        cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
        
      
        if confidence > 75:   #75 보다 크면 동일 인물로 간주해 UnLocked! 
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', image)
        else:  #75 이하면 타인.. Locked!!!    
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)
    except:  #얼굴 검출 안됨      
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass
    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()