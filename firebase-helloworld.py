import pyrebase

config = {
  "apiKey": "AIzaSyB_JDKzdASym_BESMe_mmXLtzL8K5glj1M",
  "authDomain": "hackgt-traffic.firebaseapp.com",
  "databaseURL": "https://hackgt-traffic.firebaseio.com/",
  "storageBucket": "hackgt-traffic.appspot.com"
}

firebase = pyrebase.initialize_app(config)

db = firebase.database()
db.child("cameras").child("2")
data = 123
db.set(data)
