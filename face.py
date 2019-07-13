import face_recognition
import cv2
input_video = cv2.VideoCapture("a1.mp4")
length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc('M','P','E','G')

output_video = cv2.VideoWriter('output.avi', fourcc, 25.07, (1280, 720))


tony_image = face_recognition.load_image_file("tony.jpg")
tony_face_encoding = face_recognition.face_encodings(tony_image)[0]

thor_image = face_recognition.load_image_file("thor.jpg")
thor_face_encoding = face_recognition.face_encodings(thor_image)[0]


cap_image = face_recognition.load_image_file("steve.jpg")
cap_face_encoding = face_recognition.face_encodings(cap_image)[0]

known_faces = [
    tony_face_encoding,
    thor_face_encoding,
    cap_face_encoding
]
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
  
    ret, frame = input_video.read()
    frame_number += 1

    
    if not ret:
        break

    
    rgb_frame = frame[:, :, ::-1]

    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
     
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.60)

     
        name = None
        if match[0]:
            name = "Tony Stark"
        elif match[1]:
            name = "Thor"
        elif match[2]:
            name = "Cap America"

        face_names.append(name)

   
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

     
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

   
    print("Writing frame {} / {}".format(frame_number, length))
    output_video.write(frame)


input_video.release()
cv2.destroyAllWindows()