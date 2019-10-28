# HackGT 6 - TrafficWatcher project
This is a project I worked on for HackGT 6 where traffic livestreams were analyzed real time with _opencv_ and _tensorflow_.
What is done in this script is the vehicles from the traffic livestreams are tracked and the average speed is calculated for the time on that stretch of road.

This python script requires the _opencv_ and _tensorflow_ modules for python, as well as _pyrebase_ for sending data to Firebase.

---
The script accesses Georgia Department of Transportation surveillance camera [livestream feeds](http://www.511ga.org/#traffic_speeds_layer&a_con_ctl&cam_ctl&msg_ctl&xpln_ctl&zoom=9&lat=3998196.19706&lon=-9394143.80966). Then the feed is opened in _opencv_ and the frames are analyzed for cars, which are then tracked. The speed can then be calculated, along with the average speed of 50 cars, eliminating anomalies.
_Tensorflow_ is used to detect car models using a pretrained model based on the [Coco image dataset](http://cocodataset.org/#home).

The information can then be shown on an app, the code for which can be found on my [project partner's page](https://github.com/kevinquayle/traffic-watcher).
