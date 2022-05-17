import datetime
import cv2
import face_recognition as fr
import os

def DbEncodings(images):
    encList = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        enc = fr.face_encodings(image)[0]
        encList.append(enc)
    return encList

def Attendance(name):
    with open('Attendance_Register.csv', 'r+') as f:
        DataList = f.readlines()
        names = []
        for data in DataList:
            ent = data.split(',')
            names.append(ent[0])
        if name not in names:
            curr = datetime.now()
            dt = curr.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dt}')

pathlib = 'ImagesAttendance'
images = []
Names = []
myList = os.listdir(pathlib)
print(myList)
for cl in myList:
    currImg = cv2.imread(f'{pathlib}/{cl}')
    images.append(currImg)
    Names.append(os.path.splitext(cl)[0])
print(Names)

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    image = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    facesInFrame = fr.face_locations(image)
    encodesInFrame = fr.face_encodings(image, facesInFrame)

for encodeFace, faceLoc in zip(encodesInFrame, facesInFrame):
    matchList = fr.compare_faces(encodeKnown, encodeFace)
    faceDist = fr.face_distance(encodeKnown, encFace)
    match = np.argmin(faceDist)
    if matchList[match]:
        name = Names[match].upper()
        Attendance(name)

encodeKnown = DbEncodings(images)
print('Encoding Complete')
