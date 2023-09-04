import cv2 as cv
import face_recognition

# Tải ảnh
imgHCM = face_recognition.load_image_file('img/hcm.jpg')
imgHCM = cv.cvtColor(imgHCM, cv.COLOR_BGR2RGB)

imgHCM_check = face_recognition.load_image_file('img/hcmCheck.jpg')
imgHCM_check = cv.cvtColor(imgHCM_check, cv.COLOR_BGR2RGB)

imgVNG_check = face_recognition.load_image_file('img/vngCheck.jpg')
imgVNG_check = cv.cvtColor(imgVNG_check, cv.COLOR_BGR2RGB)

imgDN_check = face_recognition.load_image_file('img/trumpCheck.jpg')
imgDN_check = cv.cvtColor(imgDN_check, cv.COLOR_BGR2RGB)

# Xác định vị trí khuôn mặt
HCMLoc = face_recognition.face_locations(imgHCM)[0]
HCMCheckLoc = face_recognition.face_locations(imgHCM_check)[0]
VNGCheckLoc = face_recognition.face_locations(imgVNG_check)[0]
DNCheckLoc = face_recognition.face_locations(imgDN_check)[0]

# Mã hóa
encodeHCM = face_recognition.face_encodings(imgHCM)[0]
encodeHCM_check = face_recognition.face_encodings(imgHCM_check)[0]
encodeVNG_check = face_recognition.face_encodings(imgVNG_check)[0]
encodeDN_check = face_recognition.face_encodings(imgDN_check)[0]

# Vẽ khung bao bọc khuôn mặt
cv.rectangle(imgHCM, (HCMLoc[3], HCMLoc[0]), (HCMLoc[1], HCMLoc[2]), (255, 0, 255), 2)
cv.rectangle(imgHCM_check, (HCMCheckLoc[3], HCMCheckLoc[0]), (HCMCheckLoc[1], HCMCheckLoc[2]), (255, 0, 255), 2)
cv.rectangle(imgVNG_check, (VNGCheckLoc[3], VNGCheckLoc[0]), (VNGCheckLoc[1], VNGCheckLoc[2]), (255, 0, 255), 2)
cv.rectangle(imgDN_check, (DNCheckLoc[3], DNCheckLoc[0]), (DNCheckLoc[1], DNCheckLoc[2]), (255, 0, 255), 2)

# So sánh, true khi distance < tolerance
result1 = face_recognition.compare_faces([encodeHCM], encodeHCM_check, tolerance=0.5)
distance1 = face_recognition.face_distance([encodeHCM], encodeHCM_check)
cv.putText(imgHCM_check, f"{result1[0]}, Similar: {round((1-distance1[0])*100,2)},%", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

result2 = face_recognition.compare_faces([encodeHCM], encodeVNG_check, tolerance=0.5)
distance2 = face_recognition.face_distance([encodeHCM], encodeVNG_check)
cv.putText(imgVNG_check, f"{result2[0]}, Similar: {round((1-distance2[0])*100,2)},%", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

result3 = face_recognition.compare_faces([encodeHCM], encodeDN_check, tolerance=0.5)
distance3 = face_recognition.face_distance([encodeHCM], encodeDN_check)
cv.putText(imgDN_check, f"{result3[0]}, Similar: {round((1-distance3[0])*100,2)},%", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Hiển thị ảnh
# cv.imshow('Screen1', imgHCM)
cv.imshow('Screen2', imgHCM_check)
cv.imshow('Screen3', imgVNG_check)
cv.imshow('Screen4', imgDN_check)
cv.waitKey(0)
cv.destroyAllWindows()
