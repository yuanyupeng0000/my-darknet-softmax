from ctypes import *
import math
import random
import sys
cv2_root = "/usr/local/lib-bk/python2.7/dist-packages/cv2"
sys.path.insert(0,cv2_root)
cv2_lib_avcodec = "/usr/lib/x86_64-linux-gnu/bkavcodec"
sys.path.insert(0,cv2_lib_avcodec)
import cv2

if(len(sys.argv) < 7):
    print("please input data_file cfg_file weights_file save_start_index save_fram_gap and video_file from cmdline by order!")
    exit(1)
video_file = sys.argv[6]
data_file = sys.argv[1]
cfg_file = sys.argv[2]
weights_file = sys.argv[3]
image_index_start = int(sys.argv[4])
gap = int(sys.argv[5])
image_index = image_index_start
def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]
    
def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res
#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("../libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

temp_index = 0
if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    net = load_net(cfg_file, weights_file, 0)
    meta = load_meta(data_file)
    
    cap = cv2.VideoCapture(video_file)

    #image_index = 0
    #gap = 0

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    # Read until video is completed
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if int(image_index%gap) == 0:
                
                #frame = frame[180:, 120:1720, :] #27.11.64.148_01_20180920120811979.mp4;
                                                  
                '''
                img1 = frame[:180, :, :] #----------------
                img2 = frame[:370, 1000:1350, :]
                dst1 = cv2.blur(img1 , (16 ,32))
                dst2 = cv2.blur(img2 , (16, 32))
                frame[:180,:,:] = dst1
                frame[:370, 1000:1350, :] = dst2 #27.11.64.149_01_20180920183555808.mp4
                                                 
                '''

                '''
                img1 = frame[:180, :, :] #----------------
                img2 = frame[:320, 1000:1350, :]
                dst1 = cv2.blur(img1 , (16 ,32))
                dst2 = cv2.blur(img2 , (16, 32))
                frame[:180,:,:] = dst1
                frame[:320, 1000:1350, :] = dst2 #27.11.64.149_01_20180920175502673.mp4
                                                 #27.17.64.107_01_20180919183341690.mp4;

                '''

                '''
                benchmark = int(image_index - image_index_start)/gap
                print(benchmark)
                if benchmark < 1000:
                    img1 = frame[:380, :, :] #----------------
                    #img2 = frame[:450, 500:780, :]
                    dst1 = cv2.blur(img1 , (16 ,32))
                    #dst2 = cv2.blur(img2 , (16, 32))
                    frame[:380,:,:] = dst1
                    #frame[:450, 500:780, :] = dst2 #27.11.64.149_01_20180920175502673.mp4
                                                
                else:
                    img1 = frame[:380, :, :] #----------------
                    img2 = frame[:450, 500:780, :]
                    dst1 = cv2.blur(img1 , (16 ,32))
                    dst2 = cv2.blur(img2 , (16, 32))
                    frame[:380,:,:] = dst1
                    frame[:450, 500:780, :] = dst2 #27.17.64.107_01_20180919183341690.mp4;

                '''
                '''
                img1 = frame[:380, :, :] #----------------
                img2 = frame[:450, 500:780, :]
                dst1 = cv2.blur(img1 , (16 ,32))
                dst2 = cv2.blur(img2 , (16, 32))
                frame[:380,:,:] = dst1
                frame[:450, 500:780, :] = dst2 #27.17.64.107_01_2018091918564462.mp4;
                '''
                '''
                img1 = frame[:360, :, :]
                dst1 = cv2.blur(img1 , (16 ,32))
                frame[:360,:,:] = dst1
                #27.17.64.107_01_20180920115125352.mp4;
                '''
                '''
                img1 = frame[:300, :, :]
                dst1 = cv2.blur(img1 , (16 ,32))
                frame[:300,:,:] = dst1 #whsj
                img1 = frame[:200, :, :]
                dst1 = cv2.blur(img1 , (16 ,32))
                frame[:200,:,:] = dst1
                '''
                
                '''
                img1 = frame[:340, :, :]
                dst1 = cv2.blur(img1 , (16 ,32))
                img2 = frame[:450, 780:1250, :]
                dst2 = cv2.blur(img2, (16,32))
                frame[:340,:,:] = dst1 #27.17.64.107_01_2018092017
                frame[:450, 780:1250, :]  = dst2
                '''

                '''
                img1 = frame[1150:, 520:2400, :]
                frame = img1 #27.14.32.8_01_20180921181130405.mp4
                '''

                '''
                temp_index = temp_index +1
                if(temp_index%2 == 1):
                    if(temp_index < 125):
                        continue
                    img1 = frame[600:1000, 1100:2000, :]
                    frame = img1 #27.17.32.4_01_20180921174942256.mp4
                else:
                    img2 = frame[1000:, 1400:3400, :]
                    frame = img2 #27.17.32.4_01_20180921174942256.mp4
                ''' 
                '''            
                img1 = frame[1120:, 400:2731, :]
                frame = img1 #27.11.32.3_01_20180921181804985.mp4
                '''
                            
                '''
                img1 = frame[300:, 200:-300, :]
                frame = img1 #27.11.32.16_01_20180921183741410_1.mp4
                '''
                
                temp_index = temp_index +1
                if(temp_index%3 == 1):
                    img1 = frame[700:, :2048, :]
                    frame = img1 #
                elif(temp_index%3 == 2):
                    img2 = frame[700:, 1024:3072, :]
                    frame = img2 #guangzhou_high_sky
                else:
                    img3 = frame[700:, 2048:, :]
                    frame = img3 #guangzhou_high_sky
                

                cv2.imwrite("image/" + str(image_index_start + (image_index-image_index_start)/gap).zfill(11) + ".jpg", frame)
                fname = str(image_index_start + (image_index-image_index_start)/gap).zfill(11)
                origimg_path = "image/" + fname + ".jpg"
                r = detect(net, meta, origimg_path)

                origimg = cv2.imread(origimg_path)
                height = origimg.shape[0]
                width = origimg.shape[1]
                depth = origimg.shape[2]

                xml_str = "<annotation>\n\t\
                <folder>image</folder>\n\t\
                <filename>" + origimg_path.split('/')[1] + "</filename>\n\t\
                " + "<path>" + origimg_path + "</path>\n\t\
                <source>\n\t\t\
                <database>Unknown</database>\n\t\
                </source>\n\t\
                <size>\n\t\t\
                <width>" + str(width) + "</width>\n\t\t\
                <height>" + str(height) + "</height>\n\t\t\
                <depth>" + str(depth) + "</depth>\n\t\
                </size>\n\t\
                <segmented>0</segmented>"
                #print(xml_str)

                for i in range(len(r)):
                    cls = r[i][0]
                    if(cls not in ['bus','car','truck']):
                        continue
                    score = r[i][1]
                    bbox = r[i][2]

                    x1 = max(0, (bbox[0] - bbox[2]/2))
                    y1 = max(0, (bbox[1] - bbox[3]/2))
                    x2 = min(width, (bbox[0] + bbox[2]/2))
                    y2 = min(height, (bbox[1] + bbox[3]/2))      

                    p1 = (int(x1), int(y1))
                    p2 = (int(x2), int(y2))

                    obj_str = "\n\t\
                    <object>\n\t\t\
                    <name>" + cls + "</name>\n\t\t\
                    <pose>Unspecified</pose>\n\t\t\
                    <truncated>0</truncated>\n\t\t\
                    <difficult>0</difficult>\n\t\t\
                    <bndbox>\n\t\t\t\
                    <xmin>" + str(int(x1)) + "</xmin>\n\t\t\t\
                    <ymin>" + str(int(y1)) + "</ymin>\n\t\t\t\
                    <xmax>" + str(int(x2)) + "</xmax>\n\t\t\t\
                    <ymax>" + str(int(y2)) + "</ymax>\n\t\t\
                    </bndbox>\n\t\
                    </object>"

                    xml_str += obj_str

                    cv2.rectangle(origimg, p1, p2, (0,255,0))
                    p3 = (max(p1[0], 15), max(p1[1], 15))
                    title = "%s:%.2f" % (cls, score)
                    cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
                xml_str += "\n</annotation>"
                fileObject = open("xml/" + fname + '.xml', 'w')  
                fileObject.write(xml_str)   
                fileObject.close()
                print("Created : " + str(image_index_start + (image_index-image_index_start)/gap))
                if((int(image_index) - int(image_index_start))/gap == 150):
                    print('Got 150 samples')
                    break;
                #print(xml_str)
                #cv2.imshow("yolov3", origimg)
                # Press Q on keyboard to  exit
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
            image_index = image_index + 1
