import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

data_path = '/home/iot/study/python/opencv/Facial-Recognition/faces'

onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))] #faces폴더에 있는 파일 리스트 얻기 

Training_Data, Labels = [], [] #데이터와 매칭될 라벨 변수 

for i, files in enumerate(onlyfiles): #파일 개수 만큼 루프 
    image_path = data_path + '/'+files 
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


