"""
Script used to prepare the URFD dataset: 

1- It takes the downloaded RGB images of the camera 0, divided into two
   folders: 'Falls' and 'ADLs', where all the images of a video (comprised in a
   folder) are inside one of those folders.

2- It creates another folder for the new dataset with the 'Falls' and
   'NotFalls' folders. All the ADL videos are moved to the 'NotFalls' folder.
   The images within the original 'Falls' folder are divided in three stages:
   (i) the pre-fall ADL images (they go to the new 'NotFalls' folder),
   (ii) the fall itself (goes to the new 'Falls' folder) and
   (iii) the post-fall ADL images (to 'NotFalls').
   
All the images are resized to size (W,H) - both W and H are variables of the
script.

The script should generate all the necessary folders.
"""

import cv2
import os
import csv
import sys
import numpy as np

# Path where the images are stored
basepath="E:/Accident_detection"

data_folder = basepath + '/URFD_images/'
# Path to save the images
output_path = basepath + '/URFD_images/'
# Label files
falls_labels = basepath + '/accident-cam0-falls.csv'
notfalls_labels = basepath + '/urfall-cam0-adls.csv'

W, H = 224, 224 # shape of new images (resize is applied)

if not os.path.exists(output_path):
    print("make")
    os.makedirs(output_path + 'Accident')
    os.makedirs(output_path + 'Not Accident')
    
# =====================================================================
# READ LABELS AND STORE THEM
# =====================================================================

labels = {'Accident': dict(), 'Not Accident': dict()}

# For falls videos: read the CSV where frame-level labels are given
with open(falls_labels, 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    event_type = 'Accident'
    for row in spamreader:
        elems = row[0].split(',')  # read a line in the csv 
        if not elems[0] in labels[event_type]:
            labels[event_type][elems[0]] = []
        if int(elems[2]) == 0 or int(elems[2]) == -1:
            labels[event_type][elems[0]].append(0)
        elif int(elems[2]) == 1:
            labels[event_type][elems[0]].append(1)

# For ADL videos: read the CSV where frame-level labels are given
with open(notfalls_labels, 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    event_type ='Not Accident'
    for row in spamreader:
        elems = row[0].split(',')  # read a line in the csv 
        if not elems[0] in labels[event_type]:
            labels[event_type][elems[0]] = []
        if int(elems[2]) == 0 or int(elems[2]) == -1:
            labels[event_type][elems[0]].append(0)
        elif int(elems[2]) == 1:
            labels[event_type][elems[0]].append(1)
            
print('Label files processed')
            
# =====================================================================
# PROCESS THE DATASET
# =====================================================================

# Get all folders: each one contains the set of images of the video
folders = [f for f in os.listdir(data_folder)
             if os.path.isdir(os.path.join(data_folder, f))]


for folder in folders:
    print('{} videos =============='.format(folder))
    events = [f for f in os.listdir(data_folder + folder) 
                if os.path.isdir(os.path.join(data_folder + folder, f))]
    events.sort() 
    for nb_event, event, in enumerate(events):
        # Create the appropriate folder
        if folder == 'ADLs':
            event_id = event[:6]
            new_folder = output_path + 'NotFalls/notfall_{}'.format(event_id)
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)     
        elif folder == 'Falls':
            event_id = event[:7]
            # "No falls" come before and after the fall, so the respective
            # folders must be created
            new_folder = output_path + 'Falls/fall_{}'.format(event_id)
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)    
#        print(event_id)      
        folder_created = False
        path_to_images = data_folder + folder + '/' + event + '/'
        
        # Load all the images of the video
        images = [f for f in os.listdir(path_to_images) 
                    if os.path.isfile(os.path.join(path_to_images, f))]
        images.sort()
        fall_detected = False # whether a fall has been detected in the video
        for nb_image, image in enumerate(images):
            x = cv2.imread(path_to_images + image)
            
            # If the image is part of an ADL video no fall need to be
            # considered
            if folder == 'ADLs':
                # Save the image
                save_path = (output_path +
                    'NotFalls/notfall_{}'.format(event_id) + 
                    '/frame{:04}.jpg'.format(nb_image))
                cv2.imwrite(save_path,
                            cv2.resize(x, (W,H)),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                
                
            elif folder == 'Falls':
                event_type = 'falls'
                if labels[event_type][event_id][nb_image] == 0: # ADL
                    if fall_detected:
                        # Create another folder for an ADL event,
                        # i.e. the post-fall ADL event
                        new_folder = (output_path +
                                    'NotFalls/notfall_{}_post'.format(
                                    event_id))
                        if not os.path.exists(new_folder):
                            os.makedirs(new_folder) 
                        
                        save_path = (output_path +
                                    'NotFalls/notfall_{}_post'.format(
                                    event_id) +
                                    '/frame{:04}.jpg'.format(nb_image))
                    else:
                        new_folder = (output_path +
                                    'NotFalls/notfall_{}_pre'.format(event_id))
                        if not os.path.exists(new_folder):
                            os.makedirs(new_folder) 
                        save_path = (output_path +
                                    'NotFalls/notfall_{}_pre'.format(
                                    event_id) +
                                    '/frame{:04}.jpg'.format(nb_image))
                    cv2.imwrite(save_path,
                                cv2.resize(x, (224,224)),
                                [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    
                elif labels[event_type][event_id][nb_image] == 1: # actual fall
                    save_path = (output_path + 
                                'Falls/fall_{}'.format(event_id) +
                                '/frame{:04}.jpg'.format(nb_image))
                    cv2.imwrite(save_path,
                                cv2.resize(x, (224,224)),
                                [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    # If fall is detected in a video set the variable to True
                    # used to discern between pre- and post-fall ADL events
                    fall_detected = True
 
print('End of the process, all the images stored within the {} folder'.format(output_path))


####################################################################################################


def create_video_with_frame_num():
    import numpy as np
    import pandas as pd
    np.random.seed(47)
    import random
    random.seed(47)
    import cv2 as cv
    import os
    import h5py
    import math
    from pathlib import Path
    
    
    # video_path=r"C:\alka\FULL_FDD_Data\RGB\fall-01-cam0-rgb"
    video_path="E://Accident_detection//URFD_images//MVI_1049.avi"

    save_path="E://Accident_detection//URFD_images"
    
    delay = 0
    
    ''' Add frame number to a video '''
    if (not os.path.isfile(video_path)):
        print("{} is not a valid file".format(video_path))

    video = cv.VideoCapture(video_path);
    if (not video.isOpened()):
        print("{} cannot be opened".format(video_path))

    video_name, video_ext = os.path.splitext(os.path.basename(video_path))
    
    if (save_path[-1] != "/"):
        save_path += "/"
        save=True
    try:
        Path(save_path).mkdir(parents = True, exist_ok = True)
    except :
        print("Cannot create directory {}".format(save_path))

    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv.CAP_PROP_FPS)
    out = cv.VideoWriter(
        "{}{}_frames{}".format(save_path, video_name, video_ext),
        cv.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))

    frame_num = 1
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if (delay > 0):
            delay -= 1

        frame = cv.putText(
            frame, "Frame: {}".format(frame_num), (5, 20),
            fontFace = cv.FONT_HERSHEY_SIMPLEX, fontScale = 0.5,
            color = (0, 255, 0), lineType = cv.LINE_AA
        )
        if (save):
            out.write(frame)
        else:
            cv.imshow("Clip", frame)
            if cv.waitKey(30) == 27:
                break

        frame_num += 1

    video.release()
    out.release()
#####################################################For creating Frame to Video   
# import cv2
# import os
# from pathlib import Path


# def Create_Video(Foldername,VideoN):
#     for i in range(10,31,1):
#         # Foldername="fall-01-cam0-rgb"
#         VideoN="fall-" + str(i)  #"4"
#         # try:
#         dirfile=[]
#         basepath = Path(r"C:\alka\FULL_FDD_Data\RGB\fall-" + str(i) +"-cam0-rgb") #Path(Foldername)     #'data4/')
#         Foldername=basepath
#         files_in_basepath = basepath.iterdir()
            
#         for item in files_in_basepath:
#             if item.is_file():
#                 x=item.name
#                 x=x.split(".")[0]
#                 dirfile.append(x[-3:])
                
        
#         Ipath= str(Foldername) + "\\fall-" + str(i) +"-cam0-rgb-"
#         lsorted = sorted(dirfile,key=lambda x: int(os.path.splitext(x)[0]))
#         FSorted=[]
#         for i in lsorted:
#             FSorted.append(Ipath + i + '.png')
            
        
#         img_array = []
        
        
#         for filename in FSorted:
#             img = cv2.imread(filename)
#             height, width, layers = img.shape
#             size = (width,height)
#             img_array.append(img)
         
         
#         out = cv2.VideoWriter(VideoN + ".mp4",cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

#         for i in range(len(img_array)):
#             out.write(img_array[i])
#         out.release()
        
#         # return VideoN + ".mp4"
        
#         # except:pass
                

