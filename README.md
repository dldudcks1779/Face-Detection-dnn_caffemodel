<div>
  <p align="center">
    <img width="500" src="result_face_video.gif"> 
  </p>
</div>

## DNN(Deep Neural Networks)를 이용한 Face Detection
* #### caffemodel 파일 : Object Detection을 위해 사전 훈련된 모델인 MobileNet SSD caffemodel을 사용(약 20개의 객체 인식)
* #### prototxt 파일 : 모델의 레이어 구성 및 속성 정의

---
### 실행 환경
* #### Ubuntu
* #### OpenCV Version : 3.x.x
  * ##### 설치 : https://blog.naver.com/dldudcks1779/222020005648
* #### imutils
  * ##### 설치 : sudo pip3 install imutils
* #### face_recognition
  * ##### 설치 : sudo pip3 install face_recognition
  
---
## 이미지 얼굴 인식
* #### 이미지를 저장하지 않을 경우
  * sudo python3 face_recognition_image.py --input 이미지 경로
    * 예) sudo python3 face_recognition_image.py --input ./face_image.jpg
* #### 이미지를 저장할 경우
  * sudo python3 face_recognition_image.py --input 이미지 경로 --output 저장할 이미지 경로
    * 예) sudo python3 face_recognition_image.py --input ./face_image.jpg --output ./result_face_image.jpg

<div>
  <p align="center">
    <img width="300" src="face_image.jpg"> 
    <img width="300" src="result_face_image.jpg">
  </p>
</div>

---
## 비디오 얼굴 인식 - 웹캠 또는 동영상(webcam or video)
* #### 비디오를 저장하지 않을 경우
  * webcam : sudo python3 face_recognition_video.py
    * 예) sudo python3 face_recognition_video.py
  * video : sudo python3 face_recognition_video.py --input 비디오 경로
    * 예) sudo python3 face_recognition_video.py --input ./face_video.avi
* #### 비디오를 저장할 경우
  * webcam : sudo python3 face_recognition_video.py --output 저장할 비디오 경로
    * 예) sudo python3 face_recognition_video.py --output ./result_face_video.avi
  * video : sudo python3 face_recognition_video.py --input 비디오 경로 --output 저장할 비디오 경로
    * 예) sudo python3 face_recognition_video.py --input ./face_video.avi --output ./result_face_video.avi

---
