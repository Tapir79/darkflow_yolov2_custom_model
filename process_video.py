import cv2
from darkflow.net.build import TFNet
import numpy as np 
import time 

# Custom options
# weights are in ckpt folder. as PROFILE file
# if last file is called 'tiny-yolo-voc-1c-2700' then 'load': 2700
# option = {
#     'model': 'cfg/tiny-yolo-voc-1c.cfg',
#     'load': 2700,
#     'threshold': 0.15,
#     'gpu': 0.9
# }

# original YOLOv2 options
option = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.15,
    'gpu': 0.9
}

# initialize the model
tfnet = TFNet(option)
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]

#capture = cv2.VideoCapture('IMG_0975.mp4')
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while(capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    results = tfnet.return_predict(frame)
    
    # video is playing
    if ret:
        for color, result in zip(colors, results):
            
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            confidence = result['confidence']
            topx = result['topleft']['x']
            topy = result['topleft']['y']
            conf = (topx, (topy+50))
            label = result['label']
            frame = cv2.rectangle(frame, tl, br, color, 7)   
            frame = cv2.putText(frame, str(confidence), conf, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
            frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 2)
        print(results)
        cv2.imshow('frame', frame)
        #print('FPS {:.1f}'.format(1/(time.time() -stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
    else:
        capture.release()
        cv2.destroyAllWindows()