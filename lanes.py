import cv2
import numpy as np
def reign_of_interest(img,vertices):
    mask = np.zeros_like(img)
    chn_co = img.shape[2]
    match_mask_color = (255,)*chn_co
    print(vertices)
    cv2.fillPoly(mask,vertices,match_mask_color)
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image

def roi(img):

    low_black = np.array([0,0,124])
    high_black = np.array([185,37,255])
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,low_black,high_black)
    r2 = cv2.bitwise_and(img,img,mask=mask)
    contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)>0:
        mx = max(contours, key = cv2.contourArea)
        mask = np.zeros(img.shape[:2],np.uint8)
        cv2.fillPoly(mask,[mx],(255,255,255))
    result = cv2.bitwise_and(img,img,mask = mask)
    
    return result
    
    
def draw_lines(img,lines):
    Nimg = np.copy(img)
    line_img = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_img,(x1,y1),(x2,y2),(0,255,0),thickness=5)
    Nimg = cv2.addWeighted(img,0.8,line_img,0.6,0.0)
    return Nimg
def work(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    height = image.shape[0]
    width = image.shape[1]
    
    roi_v = [(0,height),(0.78*width,0),(width,0),(width,0.15*height),(0.85*width,height)]
    
    crp_img = roi(image)
    
    kernel = np.ones((3, 3), np.float32) / 9    
    smoothed = cv2.filter2D(crp_img, -1, kernel)
     
    gray_image = cv2.cvtColor(smoothed,cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image,27,142)
    
    
    lines = cv2.HoughLinesP(canny_image,rho=1,theta=np.pi/180,
    threshold=60,lines=np.array([]),minLineLength=100,maxLineGap=150)
    return lines
def run(path):
    cap = cv2.VideoCapture(path)
    img=[]
    i=0
    fps = cap.get(cv2.CAP_PROP_FPS)
    f_w = int(cap.get(3)) 
    f_h = int(cap.get(4)) 
    
    size = (f_w, f_h) 
    while i<fps:
        ret,frame = cap.read()
        img.append(frame)
        i+=1
    cap.release()
    cap = cv2.VideoCapture(path)
    newF = np.asarray(img)
    neWF = np.median(newF,axis=0).astype(np.uint8)

    ls = work(neWF)
    # output = cv2.VideoWriter('output.avi',  
    #                          cv2.VideoWriter_fourcc(*'MJPG'), 
    #                          fps, size) 
    while cap.isOpened():
        ret,frame = cap.read()
        if not ret or (cv2.waitKey(1) & 0xFF == ord('q')):
            break
        if len(ls)>0:
            frame = draw_lines(frame,ls)
        cv2.imshow('frame',frame)
        #output.write(frame)
        
    cap.release()
    #output.release()
    cv2.destroyAllWindows()
run('video.mp4')