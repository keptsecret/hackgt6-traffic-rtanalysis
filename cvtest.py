import cv2 as cv
import time
import numpy as np
import tensorflow as tf

#with tf.gfile.FastGFile('/Users/nut/Downloads/frozen_inference_graph.pb', 'rb') as f:
with tf.gfile.FastGFile('/Users/nut/Downloads/faster_rcnn_inception_v2_coco_2018_01_28.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:

    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    cap = cv.VideoCapture("http://vss2live.dot.ga.gov:80/lo/gdot-cam-017.stream/playlist.m3u8")
    #cap = cv.VideoCapture("http://vss2live.dot.ga.gov:80/lo/gdot-cam-014.stream/playlist.m3u8")
    #cap = cv.VideoCapture("http://vss1live.dot.ga.gov:80/lo/atl-cam-931.stream/playlist.m3u8")

    if not cap.isOpened():
        print("Video stream not captured")
        exit()

    num = 1


    while True:
        print("reading %d" % num)
        print(time.localtime()[3])
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        line_x = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        line_y = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) * 3 / 4)

        if time.localtime()[3] > 6 or time.localtime()[3] <= 17:
            rows = frame.shape[0]
            cols = frame.shape[1]
            inp = cv.resize(frame, (300, 300))
            inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

            out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                sess.graph.get_tensor_by_name('detection_scores:0'),
                sess.graph.get_tensor_by_name('detection_boxes:0'),
                sess.graph.get_tensor_by_name('detection_classes:0')],
                feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

            # Visualize detected bounding boxes.
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
                    if (right - x) < (line_x / 3) and (bottom - y) < (line_y * 2 / 3):
                        cv.rectangle(frame, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=1)

            #result = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            cv.line(frame, (0, line_y), (line_x, line_y), (255, 0, 0), 1)
            cv.imshow("frame", frame)
        if time.localtime()[3] <= 6 or time.localtime()[3] > 17:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(frame, np.array([170,70,50]), np.array([180,255,255]))
            cv.line(frame, (0, y), (x, y), (255, 0, 0), 1)
            result = cv.bitwise_and(frame, frame, mask = mask)
            cv.imshow('frame', result)

        num += 1
        cv.waitKey(30)
        if cv.waitKey(20) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()