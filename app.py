from flask import Flask,render_template,Response,request,flash
import sqlite3
import cv2
import face_recognition
import pickle

app=Flask(__name__)
app.config['SECRET_KEY'] = "This is my super key"

data=pickle.loads(open("encodings.pickle","rb").read())

rollno=""
predrollno=""
det=[]

def generate_frames():
    camera=cv2.VideoCapture(0)
    global rollno,predrollno
    while True:
        while True:
            success,frame=camera.read()
            if not success:
                break
            else:
                rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                # rgb=cv2.resize(rgb,(100,100),interpolation=cv2.INTER_NEAREST)
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
                        predrollno=name
                    names.append(name)

                for ((top,right,bottom,left),name) in zip(boxes,names):
                    cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)
                    y = top - 15
                    cv2.putText(frame,name,(left,y),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)

                ret,buffer=cv2.imencode('.jpg',frame)
                frame=buffer.tobytes()
                yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')                      





@app.route('/', methods=['POST','GET'])
def login():
    if request.method=='POST':
        result = request.form
        conn = sqlite3.connect("student.db")
        c = conn.cursor()
        global det
        c.execute("SELECT * FROM STUDENT_DETAILS WHERE roll=?",[result['roll']])
        det = c.fetchall()
        if len(det)!=0:
            if det[0][4]==result['pwd']:
                global rollno
                rollno = det[0][0]
                return render_template('index.html', user = det[0])
            else :
                flash("Wrong Password")
        else :
            flash("User Does not exit")
    return render_template('login.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/temp')
def temp():
    return render_template('index.html',user=det[0])
    
@app.route('/status')
def succ():
    global predrollno,rollno
    if predrollno==rollno:
        return render_template('success.htmL',user=rollno)
    return render_template('fail.html',user=rollno)

# Invalid URL
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

# Internal Server Error
@app.errorhandler(500)
def page_not_found(e):
    return render_template("500.html"), 500


if __name__=="__main__":
    app.run(debug=True)

