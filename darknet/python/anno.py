import os
video_dir = '/video/mls/'
videos = os.listdir(video_dir)
print(videos)
frame_gap = 15
start_index = 800000
frame_per_video = 100
for v in videos:
    command = 'python truncated_get_xml_and_background_6cls_.py cfg/coco.data cfg/yolov3-nob-608.cfg /opt/darknet/weights/header_int_aaa_yolov3.weights ' + str(start_index) + ' ' + str(frame_gap) + ' ' + video_dir + v + ' ' +  str(frame_per_video)
    print(command)    
    os.system(command)
    start_index = start_index + frame_per_video
    
