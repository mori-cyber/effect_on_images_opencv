import cv2
face_detector = cv2.CascadeClassifier('Nariz.xml')
video_capture = cv2.VideoCapture('mm.mp4')
sticker = cv2.imread('sticker_.png',cv2.IMREAD_UNCHANGED)

print(sticker.shape)

sticker_image = sticker[:,:,0:3]
sticker_image_gray = cv2.cvtColor(sticker_image,cv2.COLOR_BGR2GRAY)
sticker_mask = sticker[:,:,3]
# sticker_mask =sticker_mask/255


while True:
    ret,frame = video_capture.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(frame_gray,1.3)
    for i,face in enumerate(faces):
        x,y,w,h = face
        image_face= frame_gray[y:y+h,x:x+w]

        sticker_image_gray_resized = cv2.resize(sticker_image_gray,(w,h))
        sticker_mask_gray_resized = cv2.resize(sticker_mask,(w,h))
        sticker_image_gray_resized = sticker_image_gray_resized.astype(float)/255
        sticker_mask_gray_resized = sticker_mask_gray_resized.astype(float) / 255
        image_face = image_face.astype(float)/255
        # frame_gray[y:y+h,x:x+w] = sticker_image_gray_resized
        forground = cv2.multiply(sticker_image_gray_resized,sticker_mask_gray_resized)
        background = cv2.multiply(image_face,1-sticker_mask_gray_resized)
        result = cv2.add(forground,background)
        result*=255
        frame_gray[y:y+h,x:x+w]=result
    cv2.imshow('output',frame_gray)
    cv2.waitKey(1)




