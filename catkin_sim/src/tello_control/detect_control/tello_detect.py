# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
#TODO: 注意剪裁图片，提高准确率
#TODO: 检测不出圆，可以增加直接对黄棕和蓝、yellow的判断，防止漏盘
#TODO:可以根据距离，预计圆的大小，防止误判

class TelloDetect():
    def __init__(self):
        def _r(h,s,v):
            return np.array([int(np.around(h*179)),int(np.around(s*255)),int(np.around(v*255))])
        self.hsv_dict={
            'yellow':[_r(0.100,0.4467,0.3333),_r(0.1998,1.0,1.0)],
            'black':[_r(0,0,0),_r(1,1,0.12+0.05)],  #黑的下限很低了？
            'white':[_r(0,0,0.3267),_r(1,0.217,1)],
            'blue':[_r(0.45,0.47,0.12),_r(0.7333,1,1)],
            'brown':[_r(0,0.4435,0.14),_r(0.115, 1, 1)],
            'brown_2':[_r(1-0.115,0.75,0.14),_r(1,1,1)]

        }
        self.result_list=['b','f','v']

    def detect(self, img, x=0, y=0, w=0, h=0,width=960,height=720,show=False,min_radius_k=10,max_radius_k=3):
        cv2.convertScaleAbs(img, img, alpha=1.0, beta=0)
        while width>1200:
            width=np.around(width/2)
            height=np.around(height/2)
            img=cv2.resize(img,(width,height))

        #img=img[height/8:height*2/3, :width*3/4]
        if w != 0 and h != 0:
            img=img[y:y+h, x:x+w]
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        #gray_img=cv2.GaussianBlur(gray_img,(5,5),1)
        gray_img = cv2.medianBlur(gray_img,3)#中值模糊
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        circles = cv2.HoughCircles(gray_img,             #vec:x,y,r
                                    cv2.HOUGH_GRADIENT,
                                    1,                  #放缩
                                    width/2,                 #两个圆心之间的最小间距                      
                                    param1=15,         #边缘检测阈值
                                    param2=90,          #近似圆的阈值
                                    minRadius=height/min_radius_k,
                                    maxRadius=height/max_radius_k)
        if type(circles)==type(None) or len(circles)==0:
            return None
        #print("    finish HoughCircles ")
        circles = np.uint16(np.around(circles))
        if show==True :
            cv_image=img.copy()
            for i in circles[0, :]: #
                cv2.circle(cv_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(cv_image, (i[0], i[1]), 2, (0, 0, 255), 3)
            cv2.imshow("view",cv_image)
            cv2.waitKey(1000)


        max_r=0
        x=0#width
        y=0#height
        for vec in circles[0, :]:
            if vec[2]>max_r:
                max_r=int(vec[2])
                x=int(vec[0])
                y=int(vec[1])
        #circles mask
        #反掩膜
        #mask=np.zeros(img.shape[:2],dtype='uint8') 
        #mask=cv2.circle(mask,(x, y), max_r,255,-1)#透明
        #img_mask = cv2.circle(hsv_img, (x, y), max_r, (60, 147, 204),-11)#覆盖为绿色
        #img_mask=cv2.bitwise_and(hsv_img,hsv_img,mask=mask) #得到掩膜后的处理
        #同时缩减
        y_r=max(y-max_r,0)
        y_r_=min(y+max_r,height)
        x_r=max(x-max_r,0)
        x_r_=min(x+max_r,width)
        img_mask=hsv_img[y_r:y_r_,x_r:x_r_,:]
        #print("circles:",y_r,y_r_,x_r,x_r_)
        res_num=np.array([0,0,0])#b,f,v
        color_num=np.array([0,0,0])

        b_mask_1 = cv2.inRange(img_mask, self.hsv_dict['brown'][0], self.hsv_dict['brown'][1])
        b_mask_2 = cv2.inRange(img_mask, self.hsv_dict['brown_2'][0], self.hsv_dict['brown_2'][1])
        b_mask=cv2.bitwise_or(b_mask_1, b_mask_2)
        num_brown=len(np.where(b_mask!=0)[0])


        f_mask_1 = cv2.inRange(img_mask, self.hsv_dict['black'][0], self.hsv_dict['black'][1])
        f_mask_2 = cv2.inRange(img_mask, self.hsv_dict['white'][0], self.hsv_dict['white'][1])
        num_w=len(np.where(f_mask_2!=0)[0])
        num_black=len(np.where(f_mask_1!=0)[0])



        v_mask_1 = cv2.inRange(img_mask, self.hsv_dict['yellow'][0], self.hsv_dict['yellow'][1])
        v_mask_2 = cv2.inRange(img_mask, self.hsv_dict['blue'][0], self.hsv_dict['blue'][1])
        num_y=len(np.where(v_mask_1!=0)[0])
        num_b=len(np.where(v_mask_2!=0)[0])

            
        color_num[0]=num_brown
        res_num[0]=num_brown


        if num_w*5>num_black and num_black*5>num_w:
            res_num[1]=num_w+num_black
        color_num[1]=num_w+num_black  

        #认为确实可能出现yellow和blue比例极端，但不可能出现白色比y和b都多得多
        #if num_y*5>num_b and num_b*5>num_y:
        #    res_num[2]=num_y+num_b+num_w
        res_num[2]=num_y+num_b+num_w
        if num_w>num_y*5 and num_w>num_b*5:
            res_num[2]=num_y+num_b
        color_num[2]=num_y+num_b

        result_idx= np.argmax(res_num) 
        #print("result_idx_temp",result_idx) 
        circle_num=int(np.pi*max_r**2)
        all_color_num=color_num.sum()

        print('num_brown',num_brown,'num_y',num_y,'num_b',num_b,'num_w',num_w,'num_black',num_black,"circle_num",circle_num,"all_color_num",all_color_num)

        #后处理
        #黑色和白色太多，排球判断不出来 排球的关键色有蓝色和黄色
        if (result_idx==0 and num_y>1.0/3* num_brown) or (result_idx==1 and num_y>1.0/3* num_black):
            result_idx=2 
            print("Get yellow and must be volleyball")
        if (result_idx==0 and num_b>1.0/3* num_brown) or (result_idx==1 and num_b>1.0/3* num_black):
            result_idx=2
            print("Get blue and must be volleyball")

        #防止检测区域太大，或者几条规则下来后很小的颜色成立主流
        if res_num[result_idx]<all_color_num/4 or res_num[result_idx]<circle_num/10:
           # all_num=all_num-color_num[np.where(color_num!=res_num[result_idx])].sum()
           # if res_num[result_idx]<all_num/4:
            print("the color %s nums too few"%result_idx)
            result_idx= None






        if show==True  :
            cv_image=img.copy()
            for i in circles[0, :]: #
                cv2.circle(cv_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(cv_image, (i[0], i[1]), 2, (0, 0, 255), 3)
            cv_image=cv_image[y_r:y_r_,x_r:x_r_,:]
            #image=cv2.bitwise_and(cv_image,cv_image,mask=v_mask_2*255)
            image=cv2.add(cv_image, np.zeros(np.shape(cv_image), dtype=np.uint8), mask=f_mask_1*255)
            cv2.imshow("view",image)
            cv2.waitKey(2000)  
        
        


        return result_idx


def main_old():
    td=TelloDetect()
    label_list=[]
    lines=open('data/val.txt', 'r').readlines()    
    for line in lines:
        s=str(line).rstrip()
        #img=cv2.imread(s)
        f_label=open('data/labels/'+s.split('.')[0][-6:]+'.txt','r')
        label_list.append(int(f_label.readline().split(" ")[0]))
        f_label.close()
    lines=open('data/val.txt', 'r').readlines()

    cv2.namedWindow("view",0)
    cv2.resizeWindow("view", 960, 720)
    for i, line in enumerate(lines):
        s=str(line).rstrip()
        img=cv2.imread(s)
        print(" ")
        print('------start to det----------')
        t1=time.time()
        res=td.detect(img,width=img.shape[1],height=img.shape[0],show=True)
        dis_t=time.time()-t1
        print('get det',dis_t,'s')
        Check=False
        if (res==None and label_list[i]==3) or  res==label_list[i]:
            Check=True
        if Check==False:
            print("%s, answer:%s ,detect:%s"%(s,label_list[i],res))

if __name__ == '__main__':
    td=TelloDetect()
    label_list=[]
    lines=open('data/val.txt', 'r').readlines()    
    for line in lines:
        s=str(line).rstrip()
        #img=cv2.imread(s)
        f_label=open('data/labels/'+s.split('.')[0][-6:]+'.txt','r')
        label_list.append(int(f_label.readline().split(" ")[0]))
        f_label.close()
    lines=open('data/val.txt', 'r').readlines()

    cv2.namedWindow("view",0)
    cv2.resizeWindow("view", 960, 720)
    for i, line in enumerate(lines):
        s=str(line).rstrip()
        img=cv2.imread(s)
        print(" ")
        print('------start to det----------')
        t1=time.time()
        res=td.detect(img,0,0,960,720/5*4,width=img.shape[0],height=img.shape[1],show=True,min_radius_k=20,max_radius_k=4)
        #res=td.detect(img,width=img.shape[1],height=img.shape[0],show=True,min_radius_k=20,max_radius_k=4)

        dis_t=time.time()-t1
        print('get det',dis_t,'s')
        print("%s, answer:%s ,detect:%s"%(s,label_list[i],res))


