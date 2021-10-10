import face_recognition
import argparse
import pickle
import cv2

ap=argparse.ArgumentParser()
ap.add_argument("-e","--encodings", required=True)
ap.add_argument("-i","--image",required=True)
ap.add_argument("-d","--detection-method",type=str,default="cnn")
args=vars(ap.parse_args())

print("loading encodings")
data=pickle.loads(open(args["encodings"],"rb").read())

image=cv2.imread(args["image"])
image=cv2.resize(image,(500,500),cv2.INTER_NEAREST)
rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

print("recognising faces")

boxes=face_recognition.face_locations(rgb,model="hog")
encodings=face_recognition.face_encodings(rgb,boxes)

names=[]

for encoding in encodings:
    matches=face_recognition.compare_faces(data["encodings"],encoding)
    name="unknown"

    if True in matches:
        matchedIdxs=[i for (i,b) in enumerate(matches) if b]
        counts={}

        for i in matchedIdxs:
            name=data["names"][i]
            counts[name]=counts.get(name,0)+1

        name=max(counts, key=counts.get)

    names.append(name)

for ((top,right,bottom,left),name) in zip(boxes,names):
    cv2.rectangle(image,(left,top),(right,bottom),(0,255,0),2)
    y = top - 15
    cv2.putText(image,name,(left,y),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)

cv2.imshow("Image",image)
cv2.waitKey(0)
