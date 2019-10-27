# Variant 1 - car goes up

import cv2 as cv
import time
import numpy as np
import tensorflow as tf
import pyrebase

camera_no = "0"
stream_link = "http://vss2live.dot.ga.gov:80/lo/gdot-cam-150.stream/playlist.m3u8"

config = {
  "apiKey": "AIzaSyB_JDKzdASym_BESMe_mmXLtzL8K5glj1M",
  "authDomain": "hackgt-traffic.firebaseapp.com",
  "databaseURL": "https://hackgt-traffic.firebaseio.com/",
  "storageBucket": "hackgt-traffic.appspot.com"
}
firebase = pyrebase.initialize_app(config)

db = firebase.database()

with tf.gfile.FastGFile('/Users/nut/Downloads/frozen_inference_graph.pb', 'rb') as f:

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:

    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    cap = cv.VideoCapture(stream_link)

    if not cap.isOpened():
        print("Video stream not captured")
        exit()

    num = 1

    threshold = 0.1
    outer_bbox = [0,0,0,0]
    newcar = True
    car_frames = 0

    speeds = []
    speed = 0
    avg_speed = 0

    while True:
        print("reading %d" % num)
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        line_y = int(height * 2 / 3)

        #if time.localtime()[3] > 6 or time.localtime()[3] <= 22:
        rows = frame.shape[0]
        cols = frame.shape[1]
        inp = cv.resize(frame, (300, 300))
        inp = inp[:, :, [2, 1, 0]]

        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
            sess.graph.get_tensor_by_name('detection_scores:0'),
            sess.graph.get_tensor_by_name('detection_boxes:0'),
            sess.graph.get_tensor_by_name('detection_classes:0')],
            feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        num_detections = int(out[0][0])
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > 0.3:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                # remove boxes that are too big (definitely not cars)
                if (right - x) < (width / 3) and (bottom - y) < (line_y * 2 / 3):
                    cv.rectangle(frame, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=1)
                    if newcar:
                        if ((y < line_y + threshold * (bottom - y)) and (y > line_y - threshold * (bottom - y))):
                            outer_bbox = [(y - threshold * (bottom - y)), (x - threshold * (right - x)),(bottom + threshold * (bottom - y)),(right + threshold * (right - x))]
                            cv.rectangle(frame, (int(outer_bbox[1]),int(outer_bbox[0])), (int(outer_bbox[3]), int(outer_bbox[2])), (0, 120, 255), 2)
                            newcar = False
                    if not newcar:
                        if (x > outer_bbox[1] and x < outer_bbox[3] or right > outer_bbox[1] and right < outer_bbox[3]) and (y > outer_bbox[0] and y < outer_bbox[2] or bottom > outer_bbox[0] and bottom < outer_bbox[2]):
                            car_frames += 1
                            #print(car_frames)
                            outer_bbox = [(y - threshold * (bottom - y)), (x - threshold * (right - x)),(bottom + threshold * (bottom - y)),(right + threshold * (right - x))]
                            cv.rectangle(frame, (int(outer_bbox[1]),int(outer_bbox[0])), (int(outer_bbox[3]), int(outer_bbox[2])), (0, 80, 255), 2)
                            if (bottom < line_y):
                                newcar = True
                                #print(car_frames)
                                speed = 4.5 / (car_frames * 0.02) * 3.6
                                print("Apparent speed is " + str(speed))
                                car_frames = 0
        
        if len(speeds) < 50:
            speeds.append(speed)
        else:
            if avg_speed == 0 and speed < 200 and speed != 0:
                speeds.pop(0)
                speeds.append(speed)
                speed = 0
            avg_speed = sum(speeds) / len(speeds)
            print("Calculated average speed is " + str(avg_speed))
            db.child("cameras").child(camera_no)
            db.set(avg_speed)

        cv.line(frame, (0, line_y), (width, line_y), (255, 0, 0), 1)
        cv.imshow("frame", frame)

        num += 1
        if cv.waitKey(20) == 27:
            break

    cap.release()
    cv.destroyAllWindows()