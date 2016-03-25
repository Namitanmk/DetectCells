import cv
import cv2
import numpy as np

FILENAME = "3.tif"
HIGH_PARAM = 20
LOW_PARAM = 20

def auto_canny(image, sigma=0.33):

	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

def segment(img):
    
    new_img = np.zeros(img.shape, dtype=np.float64)
    new_img = np.copy(img)
    new_img = np.float64(new_img) #So that we can exeed 255
    pos = new_img > (np.mean(img) + HIGH_PARAM) 
    new_img[pos] += 100
    others = new_img < (np.mean(img) + LOW_PARAM) 
    new_img[new_img > 255] = 255
    new_img[others] -= 100
    new_img[new_img < 0] = 0
    return new_img

def threshold(img):
    
    img = np.uint8(img)
    img /= 255
    img *= 255
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 2)
    return img

def count_cells(img, oimg):
    
    contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    
    cv2.drawContours(img, contours, -1, (0, 255, 255), 8)
    print "The number of cells are", len(contours)
    
    f = open('eratio.csv', 'w')
    f.write('x,y,ratio\n')
    for cnt in contours:
        if cnt.shape[0] >= 5: #To fit ellipse
            area = cv2.contourArea(cnt)
            ellipse = cv2.fitEllipse(cnt)
            M = cv2.moments(cnt)
            denom = M['m00']
            if denom == 0:
                denom = 1
            cx = int(M['m10']/denom)
            cy = int(M['m01']/denom)
            ratio = ellipse[1][0] * 1.0 /ellipse[1][1]
            f.write(str(cx) + "," + str(cy) + "," + str(ratio) + "\n")
            if area > 150:
                cv2.ellipse(oimg,ellipse,(0,255,0),5)
    f.close()
    return oimg
    
def process_image(img):
	
    oimg = np.copy(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    img = v
    cv2.imshow("Gray2.tif", img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#    img = auto_canny(img)
    img = clahe.apply(img)
    cv2.imshow('filtered', img)
    img = segment(img)
    cv2.imshow('segment', img)
    img = threshold(img)
    cv2.imshow('bw_img', img)
    print img 
    img = count_cells(img, oimg)
    cv2.imshow('ellipse_img', oimg)
    cv2.imwrite('ellipse_img.tif', oimg)    
    k = cv2.waitKey(0)
    
    cv2.destroyAllWindows()

def main():
    
    img = cv2.imread(FILENAME)
    process_image(img) 

main()
