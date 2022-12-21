import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2



target = r'C:\Users\w_alm\OneDrive\Desktop\Lauretta\Test Data\fumo'
data = []
for images in os.listdir(target):
    if images.endswith(".jpg"):
        data.append(images)
       

i = 0
arr_blue = []
arr_red = []
data_name = []

for i in range(len(data)):
    path = target + '\\' + data[i]

    read_img = cv2.imread(path)


    lower_blue = np.array([200,0,0])
    upper_blue = np.array([255,20,20])
    mask_blue =cv2.inRange(read_img, lower_blue, upper_blue)

    lower_red = np.array([0,0,200])
    upper_red = np.array([20,20,255])
    mask_red =cv2.inRange(read_img, lower_red, upper_red)
    
# Just checking if the masking of the images was proper
#cv2.imshow('mask_red', mask_red)
#cv2.imshow('mask_blue', mask_blue)
#cv2.imshow('original',read_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

    output_blue = read_img.copy()
    output_red = read_img.copy()

    circles_blue = cv2.HoughCircles(mask_blue,cv2.HOUGH_GRADIENT,1,20,param1=20,param2=6.5,
                               minRadius=0,maxRadius=5)
    circles_red = cv2.HoughCircles(mask_red,cv2.HOUGH_GRADIENT,1,20,param1=20,param2=6.5,
                               minRadius=0,maxRadius=5)
    index = 0
    if circles_blue is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles_blue = np.round(circles_blue[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles_blue:
            # Putting a small rectangle on the color dot
            cv2.rectangle(output_blue, (x - 5, y - 5), (x + 5, y + 5), (255, 0, 255), -1)
            
            index = index + 1
            
        arr_blue.append(index)   
        print ('No. of circles blue detected = {}'.format(index))
    
    index = 0
    if circles_red is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles_red = np.round(circles_red[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles_red:
            # Putting a small rectangle on the color dot
            cv2.rectangle(output_red, (x - 4, y - 4), (x + 4, y + 4), (255, 0, 255), -1)

            index = index + 1
            
        arr_red.append(index)
        print ('No. of circles red detected = {}'.format(index))
    
    print('Image Number: ', i)
    data_name.append(data[i])

    
data_list=list(zip(arr_blue, arr_red, data_name))
df = pd.DataFrame(data_list)
print('Showing index',df) # Index 0 is row (blue) index 1 columns (red) and 2 is image file name


# Ordering the Data Frame first to only show first row then columns in ascending order. PS I did not want to loop lol
testing = df[df[0] == 1]
t = testing.sort_values(by=1, ascending = True)

t = t.reset_index(drop=True)
print(t)

W = cv2.imread(target + '\\' + t[2][0])
X = cv2.imread(target + '\\' + t[2][1])
Y = cv2.imread(target + '\\' + t[2][2])
im_h1 = cv2.hconcat([W, X])
im_h2 = cv2.hconcat([im_h1, Y])

testing = df[df[0] == 2]
t = testing.sort_values(by=1, ascending = True)

t = t.reset_index(drop=True)

W = cv2.imread(target + '\\' + t[2][0])
X = cv2.imread(target + '\\' + t[2][1])
Y = cv2.imread(target + '\\' + t[2][2])
im_h3 = cv2.hconcat([W, X])
im_h4 = cv2.hconcat([im_h3, Y])

testing = df[df[0] == 3]
t = testing.sort_values(by=1, ascending = True)

t = t.reset_index(drop=True)

W = cv2.imread(target + '\\' + t[2][0])
X = cv2.imread(target + '\\' + t[2][1])
Y = cv2.imread(target + '\\' + t[2][2])
im_h5 = cv2.hconcat([W, X])
im_h6 = cv2.hconcat([im_h5, Y])




im_v1 = cv2.vconcat([im_h2,im_h4])
Fumo_attack = cv2.vconcat([im_v1, im_h6])


cv2.imshow('MASTER FUMO',Fumo_attack)
cv2.waitKey(0)
cv2.destroyAllWindows()

# sources that help solve the problem
# https://stackoverflow.com/questions/44439555/count-colored-dots-in-image

