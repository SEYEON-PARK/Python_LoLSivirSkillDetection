# import the opencv library
import cv2
import torch # 토치 추가
import numpy as np
import time

# define a video capture object

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best1.pt')  # local model
filepath = './one.webm'
video = cv2.VideoCapture(filepath) #'' 사이에 사용할 비디오 파일의 경로 및 이름을 넣어주도록 함

if not video.isOpened():
    print("Could not Open :", filepath)
    exit(0)
    
#불러온 비디오 파일의 정보 출력
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer=cv2.VideoWriter('./result.mp4', fourcc, fps, (width, height))

blue_color=(255,0,0)
other_color=(0, 0, 255)
use=False # e 스킬 사용했는지 확인!
sucess=False # e 스킬 성공했는지 확인!
f=fps # 60 프레임 보겠다!
count_good=0 # 성공 횟수!
count_bad=0 # 실패 횟수!
count_frame=0 # 3번 동안 성공 유지했는지 보겠다!
frame_index=0

print("length :", length)
print("width :", width)
print("height :", height)
print("fps :", fps)
print(type(None))

while(video.isOpened() and frame_index<length):
  frame_index+=1
  print("start time : ", time.time())
    # Capture the video frame
	# by frame
  ret, frame = video.read()
  if(type(frame)==type(None)):
    continue
	
  results=model(frame) # 만든 모델로 frame 분석!
  print(results.pandas().xyxy[0].to_json(orient="records"))
  print(results.pandas().xyxy[0].xmin)
  print(len(results.pandas().xyxy[0].xmin))
  for i in range(len(results.pandas().xyxy[0].xmin)):
    if(results.pandas().xyxy[0].name[i]=="0" and results.pandas().xyxy[0].confidence[i]>=0.7): # 스킬 발동했다면
      frame=cv2.rectangle(frame, (int(results.pandas().xyxy[0].xmin[i]), int(results.pandas().xyxy[0].ymin[i])), (int(results.pandas().xyxy[0].xmax[i]), int(results.pandas().xyxy[0].ymax[i])), blue_color, 3)
      cv2.putText(frame, "use "+str(results.pandas().xyxy[0].confidence[i]), (int(results.pandas().xyxy[0].xmin[i]), int(results.pandas().xyxy[0].ymin[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
      if(use==False):
          print("시비르 E 발동 시간 : ", frame_index/fps)
          use=True # use를 True로 변경!
      
    if(use==True): # 만약, use가 True라면
      if(f>0): # f(프레임수)가 0보다 크다면
        if(results.pandas().xyxy[0].name[i]=="1" and results.pandas().xyxy[0].confidence[i]>=0.7): # 만약, 성공했다면
          count_frame+=1 # 성공 프레임 1 증가하기
          if(count_frame>=3): # 성공 프레임이 3개라면
            frame=cv2.rectangle(frame, (int(results.pandas().xyxy[0].xmin[i]), int(results.pandas().xyxy[0].ymin[i])), (int(results.pandas().xyxy[0].xmax[i]), int(results.pandas().xyxy[0].ymax[i])), other_color, 3)
            cv2.putText(frame, "success "+str(results.pandas().xyxy[0].confidence[i]), (int(results.pandas().xyxy[0].xmin[i]), int(results.pandas().xyxy[0].ymin[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            count_good+=1 # 성공 횟수 1 증가시키기
            f=fps # 프레임 세는 것도 초기화하기
            use=False # use를 False로 바꾸기
            print("시비르 E 성공 시간 : ", frame_index/fps)
            #cv2.imshow('frame.jpg', frame)
            #cv2.waitKey(0)
            ##writer.write(frame)
            
        f-=1 # f를 1 감소시키기기
      else:
        print("시비르 E 실패 시간 : ", frame_index/fps)
        f=fps
        count_bad+=1
        use=False
  writer.write(frame)
  print("end time : ", time.time())

# After the loop release the cap object
video.release()
writer.release()

print("length :", length)
print("width :", width)
print("height :", height)
print("fps :", fps)
print("성공 : ", count_good)
print("실패 : ", count_bad)
print("성공률 : ", count_good/(count_good+count_bad))
print("게임 시간 : ", length/fps)


# Destroy all the windows
cv2.destroyAllWindows()


"""

# import the opencv library
import cv2
import torch # 토치 추가
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best1.pt')  # local model
filepath = './one.mp4'
video = cv2.VideoCapture(filepath) #'' 사이에 사용할 비디오 파일의 경로 및 이름을 넣어주도록 함

if not video.isOpened():
    print("Could not Open :", filepath)
    exit(0)
    
# define a video capture object

blue_color=(255,0,0)
other_color=(0, 0, 255)

while(video.isOpened()):
    
    # Capture the video frame
	# by frame
	ret, frame = video.read()
	
	
	results=model(frame) # 만든 모델로 frame 분석!
	print(results.pandas().xyxy[0].to_json(orient="records"))
	print(results.pandas().xyxy[0].xmin)
	print(len(results.pandas().xyxy[0].xmin))
	for i in range(len(results.pandas().xyxy[0].xmin)):
		if(results.pandas().xyxy[0].name[i]==0 and results.pandas().xyxy[0].confidence[i]>=0.7):
			frame=cv2.rectangle(frame, (int(results.pandas().xyxy[0].xmin[i]), int(results.pandas().xyxy[0].ymin[i])), (int(results.pandas().xyxy[0].xmax[i]), int(results.pandas().xyxy[0].ymax[i])), blue_color, 3)
			cv2.putText(frame, "발동"+str(results.pandas().xyxy[0].confidence[i]), (int(results.pandas().xyxy[0].xmin[i]), int(results.pandas().xyxy[0].ymin[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
			cv2.imshow('frame.jpeg', frame)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		if(results.pandas().xyxy[0].name[i]==1 and results.pandas().xyxy[0].confidence[i]>=0.7):
			frame=cv2.rectangle(frame, (int(results.pandas().xyxy[0].xmin[i]), int(results.pandas().xyxy[0].ymin[i])), (int(results.pandas().xyxy[0].xmax[i]), int(results.pandas().xyxy[0].ymax[i])), other_color, 3)
			cv2.putText(frame, "성공"+str(results.pandas().xyxy[0].confidence[i]), (int(results.pandas().xyxy[0].xmin[i]), int(results.pandas().xyxy[0].ymin[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
			cv2.imshow('frame.jpeg', frame)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
	
	# the 'q' button is set as the
	# quitting button you may use any
	# desired button of your choice
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# After the loop release the cap object
video.release()
# Destroy all the windows
cv2.destroyAllWindows()
"""

'''
# import the opencv library
import cv2
import torch # 토치 추가
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # local model

# define a video capture object
vid = cv2.VideoCapture(0)

blue_color=(255,0,0)

while(True):
	img=np.zeros((1080, 1920, 3), np.uint8) # 빈 캔버스 만들기
	img=cv2.rectangle(img, (10, 10), (100, 100), blue_color, 3) # 제일 왼쪽 위에 꺼, 오른쪽 아래꺼
	
    
    # Capture the video frame
	# by frame
	ret, frame = vid.read()
	
	
	results=model(frame) # 만든 모델로 frame 분석!
	print(results.pandas().xyxy[0].to_json(orient="records"))
	print(results.pandas().xyxy[0].xmin)
	print(len(results.pandas().xyxy[0].xmin))
	for i in range(len(results.pandas().xyxy[0].xmin)):
		frame=cv2.rectangle(frame, (int(results.pandas().xyxy[0].xmin[i]), int(results.pandas().xyxy[0].ymin[i])), (int(results.pandas().xyxy[0].xmax[i]), int(results.pandas().xyxy[0].ymax[i])), blue_color, 3)
		cv2.putText(frame, results.pandas().xyxy[0].name[i]+str(results.pandas().xyxy[0].confidence[i]), (int(results.pandas().xyxy[0].xmin[i]), int(results.pandas().xyxy[0].ymin[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the resulting frame
	cv2.imshow('frame', frame)
	
	# the 'q' button is set as the
	# quitting button you may use any
	# desired button of your choice
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
'''