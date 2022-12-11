# -*- coding: utf-8 -*-
import cv2
from control_pkg import sl4p
import time
import random
from control_pkg import tello_center
import threading

import rospy
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class JudgeClient(tello_center.Service):

    def __init__(self):
        tello_center.Service.__init__(self)
        self.logger = sl4p.Sl4p('judge_client')
        self.time_start=0
        self.send_targets_thread_=None
        self.targets_list=['e' for i in range(5)]

        #rosnode
        rospy.init_node('tello_state', anonymous=True,disable_signals=True)
        self.is_ready_ = False  # 无人机是否已准备完毕
        self.is_takeoff_command_received_ = False  # 是否收到起飞准许信号

        self.readyPub_ = rospy.Publisher('/ready', Bool, queue_size=100)
        self.seenfirePub_ = rospy.Publisher('/seenfire', Bool, queue_size=100)
        self.targetresultPub_ = rospy.Publisher('/target_result', String, queue_size=100)
        self.donePub_ = rospy.Publisher('/done', Bool, queue_size=100)

        self.takeoffSub_ = rospy.Subscriber('/takeoff', Bool, self.takeoffCallback)
        self.spin_thread_ = threading.Thread(target=rospy.spin)
        self.spin_thread_.daemon = True
        self.spin_thread_.start()

        self.state_pub = rospy.Publisher('tello_state',String, queue_size=3)
        self.img_pub = rospy.Publisher('tello_image', Image, queue_size=5)


    # 接收上位机发送的准许起飞信号
    def takeoffCallback(self, msg):
        if msg.data and self.is_ready_ and not self.is_takeoff_command_received_:
            self.is_takeoff_command_received_ = True
            #print('已接收到准许起飞信号。')
        pass

    def takeoff(self):
        #TODO：发送准备完成，监听指令，获取许可
        self.time_start=time.time()

        self.is_ready_ = True
        ready_msg = Bool()
        ready_msg.data = 1
        self.readyPub_.publish(ready_msg)
        while not self.is_takeoff_command_received_:
            time.sleep(0.1)
        self.logger.info('takeoff')
        return

    def seen_fire(self):
        #TODO:发布成功
        self.logger.info('seen_fire')
        
        seenfire_msg = Bool()
        seenfire_msg.data = 1
        self.seenfirePub_.publish(seenfire_msg)

        time_span=time.time()-self.time_start
        self.logger.info('-----------seen fire span: %s   ------------'%time_span)
        return

    def send_task_done(self):
        #发送降落指令，确保无人机已经降落
        done_msg = Bool()
        done_msg.data = 1
        self.donePub_.publish(done_msg)
        
        self.logger.info(' land done!!!!!!')
        time_span=time.time()-self.time_start
        self.logger.info('-----------time_span: %s   ------------'%time_span)
        return

    def send_targets(self,targets):
        tar_list=['b','f','v']
        for i in range(5):
            if targets[i] is None:
                continue
            self.targets_list[i]=tar_list[targets[i]]
        print(self.targets_list)
        self.send_targets_thread_ = threading.Thread(target=self.send_targets_task)
        self.send_targets_thread_.daemon = True
        self.send_targets_thread_.start()
        self.logger.info(' send targets')
        #发送detect结果  本地显示
        return 
    def send_targets_task(self):
        while True:
            for i in range(5):
                targetresult_msg = String()
                targetresult_msg.data=str(i+1)+self.targets_list[i]
                self.targetresultPub_.publish(targetresult_msg)
                time.sleep(0.1)
        pass

    def PubImgAndState(self,frame):
        tello_state='mid:-1;x:100;y:100;z:-170;mpry:1,180,1;pitch:0;roll:0;yaw:-19;'
        self.state_pub.publish(tello_state)

        if frame is None or frame.size == 0:
            return False
        #img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img=frame
        try:
            img_msg = CvBridge().cv2_to_imgmsg(img, 'bgr8')
            img_msg.header.frame_id = rospy.get_namespace()
        except CvBridgeError as err:
            rospy.logerr('fgrab: cv bridge failed - %s' % str(err))
            return False
        self.img_pub.publish(img_msg)
        
        pass
