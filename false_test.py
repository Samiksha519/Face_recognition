import cv2
import face_recognition


Person2 = face_recognition.load_image_file('nupur1.jpeg')
Person2 = cv2.cvtColor(Person2, cv2.COLOR_BGR2RGB)
Test_person2 = face_recognition.load_image_file('vicky2.jpg')
Test_person2 = cv2.cvtColor(Test_person2, cv2.COLOR_BGR2RGB)

face_image = face_recognition.face_locations(Person2)[0]
print(face_image)
encode = face_recognition.face_encodings(Person2)[0]
print(encode)
cv2.rectangle(Person2, (face_image[3], face_image[0]), (face_image[1], face_image[2]), (255, 0, 255), 3)

Test_image = face_recognition.face_locations(Test_person2)[0]
encodeTestFace = face_recognition.face_encodings(Test_person2)[0]
cv2.rectangle(Test_person2, (Test_image[3], Test_image[0]), (Test_image[1], Test_image[2]), (0, 255, 0), 3)

res = face_recognition.compare_faces([encode], encodeTestFace)
face_Distance = face_recognition.face_distance([encode], encodeTestFace)
print(res, face_Distance)
cv2.putText(Test_person2, f'{res} {round(face_Distance[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Nupur Maheshwari', Person2)
if res:
    cv2.imshow('Image identified as Nupur', Test_person2)
else:
    cv2.imshow('Test image does not match', Test_person2)


cv2.waitKey(0)
cv2.destroyAllWindows()