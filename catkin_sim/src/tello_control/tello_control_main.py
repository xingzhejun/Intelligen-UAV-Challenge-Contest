#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import time
import threading
import random
import numpy as np

import rospy
from std_msgs.msg import String
# if you can not find cv2 in your python, you can try this. usually happen when you use conda.
import control_pkg.tello_base  as tello_base
from  control_pkg.tello_center  import Service
from  judge_pkg import judge_client
from utils_pkg.drone_util import *
import  detect_pkg.tello_detect as tello_detect
from detect_pkg.redball_detecter import *
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import cv2

class Stage:
    def __init__(self, name, func_do=None, func_into=None, func_leave=None, args=None):
        self.name = name
        self.args = args
        self.func_do = func_do
        self.func_into = func_into
        self.func_leave = func_leave

    def on_into_stage(self):
        if self.func_into is not None:
            self.func_into()

    def do(self):
        if self.func_do is not None:
            if self.args is not None:
                self.func_do(self.args)
            else:
                self.func_do()

    def on_leave_stage(self):
        if self.func_leave is not None:
            self.func_leave()



class MainControl(Service):

    def __init__(self):
        Service.__init__(self)
        self.logger = sl4p.Sl4p("main_control")
        self.backend =tello_base.TelloBackend()  
        self.detector = tello_detect.TelloDetect()  
        self.judge = judge_client.JudgeClient()
        self.stage = None

        self.stage_wait_for_start = Stage('wait_for_start', func_do=self.wait_for_start)
        self.stage_find_fire = Stage('find_fire', func_do=self.search_fire)
        self.search_min = False  #第一轮搜完，先搜右边
        self.search_fire_move=False#出界了，先move一下
        self.search_fire_try=2 #1 为左，表示fire在左边
        self.stage_go_to_step2_start_pos = Stage('step2', func_do=self.step2)
        self.stage_land = Stage('land', func_do=self.land)
        self.stage_test = Stage('debug', func_do=self.debug)

        self.results=[None for i in range(5)]
        self.ps=[
            [1-2.5-0.225, -1.5+0.5, 1.7-0.25-0.1],
            [1.5+0.5-2.5,1.5+0.25, 1-0.25-0.1],
            [0, 0, 1.275-0.25-0.1],
            [2.5-1-0.5, -1.5-0.25, 1.875-0.25-0.1],
            [-1.0-0.25+2.5,-0.5+1.5,1.5-0.25-0.1]
        ]


        self.start()


        

    def jump_stage(self, stage, args=None):
        #self.logger.info("jump stage: %s" % str(stage.name))
        if self.stage is not None:
            self.stage.on_leave_stage()
        self.stage = stage
        self.stage.args = args
        self.stage.on_into_stage()

    def wait_for_start(self):
        if self.backend.drone.has_takeoff:
            self.jump_stage(self.stage_find_fire)
            #self.jump_stage(self.stage_land)
            return
        if not self.backend.drone.has_takeoff:
            self.backend.drone.mon()
            self.judge.takeoff()
            res=self.backend.drone.takeoff()#time out =7
            #goto(self.backend, 0.1-2.5, 0, 1.8, self.flag, tol=0.30)  
            #look_at(self.backend, 0,10000, 0, self.flag)
            if res=='FALSE':
                self.backend.drone.go(0,-0.20, 0.85,speed=70)
            else :
                self.backend.drone.go(0,-0.20, 0.85,speed=70)
            #self.backend.drone.move_backward(50)
            #self.backend.drone.go(0, 0, 0.75)    #TODO:for a test 

            image, state = self.backend.drone.get_image_and_state()  
            print(state.mid,state.x,state.y,state.z,state.pitch,state.yaw,state.roll)

    def land(self):
        if self.backend.drone.has_takeoff:
            
            self.backend.drone.land()
            self.judge.send_task_done()
        time.sleep(0.01)


    def search_fire(self):
        image, state = self.backend.drone.get_image_and_state()  
        #shape (720,960)
        '''
        if state.mid != -1 and False:
            if abs(state.mpry[1]) > 8 and not 150 < state.y + 50 < 250:
                if state.mpry[0] > 0:
                    self.backend.drone.rotate_ccw(state.mpry[0])
                else:
                    self.backend.drone.rotate_cw(-state.mpry[0])
                return
            if state.x < 30:
                self.backend.drone.move_forward(0.2)
                return
        '''
        x, y, w, h =self.detect_fire(skip=5)
        if x is not None:
            view = 45/180.0*np.pi
            det = 10/180.0*np.pi
            _y, _dis_y, _det_y = solve_system(det, (np.pi - view)/2, view, 720, y, y+h, 10)#x x+h

            pix_size = 1.0/w*10  # 单位像素对应的实际长度,球的直径=10cm
            la_x = 480
            la_y = 360#这是中心？168 360 才对吧
            cx = int(x + w/2)#中心
            cy = int(y + h/2)

            rh = _y(la_y) - _y(cy)  # (360 - cy) * pix_size  #场地的z轴方向上，球心的高度 #两种方法均可
            ry = (cx - la_x)*pix_size  # (cx - 480) * pix_size  #场地的y轴方向上：到球中心的距离，y轴的坐标，
            #无人接斜着飞过去也行，就不应该考虑斜着带来的距离
            #ry += (180 - (state.x if state.mid != -1 else 70))*np.tan(state.mpry[1]/180.0*np.pi)
            ry=-ry
            
            print("drone to fire z-axis: ",rh," and easy method:",(360 - cy) * pix_size)
            print("drone to fire y-axis (no yaw): ",ry)
            
            
            mz = 0
            my = 0
            if rh > -10: #要在球心的上方穿越
                mz = max(min(abs(rh + 20), 40), 22)/100.0
                #up
            elif rh < -30: 
                mz = -max(min(abs(rh + 20), 40), 22)/100.0
                #down

            if abs(ry) > 10:#y为正则向左
                dis = max(min(abs(ry), 40), 22)/100.0
                if ry < 0:
                    my = -dis
                    #self.backend.drone.move_left(dis)
                else:
                    my = dis
                    #self.backend.drone.move_right(dis)

            if mz == 0 and my == 0:
                # 位置调整已经比较准确，rush
                if self.search_fire_move==True:
                    #self.backend.drone.move_up(0.5)
                    print(" seen red_fire and state.mid == -1")
                    #self.backend.drone.move_right(0.2)
                    self.backend.drone.move_forward(0.6)
                    self.search_fire_move=False
                else:
                #估算到墙的距离
                    d_x=100 - (state.x+250) if state.mid != -1 else (-10)
                    dis = min(195, ((100 - ((state.x+250) if state.mid != -1 else (-10))) + 75))/100.0
                    self.backend.drone.go(dis, ry / 100.0, (rh + 20-d_x/45.0*16) / 100.0,speed=80)
                    #print("now try to direct rush and go ",dis,ry / 100.0,(rh + 20) / 100.0)
                
                    # clamp_abs(round(ry), 20, 40) / 100.0, clamp_abs(round(rh + 20), 20, 40) / 100.0)
                    self.jump_stage(self.stage_go_to_step2_start_pos)
                    #time.sleep(2)#TODO

                    #self.jump_stage(self.stage_land)
                    self.judge.seen_fire()
            else:
                #print(" not get accurate pose and go ",0,my,mz)
                if my>0.35:
                    self.search_fire_try=1
                elif my<-0.35:
                    self.search_fire_try=2

                image, state = self.backend.drone.get_image_and_state()
                if state.mid != -1 and state.z>=220 and (state.x+250)>=55:
                    self.backend.drone.go(-0.3, 0, -0.7)#防止飞太高看到墙后面
                else :
                    self.backend.drone.go(0, my, mz)
        else:
            
            if state.mid == -1:
                #v1 = vec3(state.x, state.y, state.z)/100.0 - vec3(0.15-2.5,-0.3, 1.65)#0.6,0.9,1.8
                #v2 = vec3(state.x, state.y, state.z)/100.0 - vec3(0.15-2.5, 1.0, 1.65)#
                #l1 = np.linalg.norm(v1)
                #l2 = np.linalg.norm(v2)
                #if self.search_min and l1 < l2:
                #self.backend.drone.move_up(0.5)
                print(" no red_fire and state.mid == -1")
                if self.search_fire_try==1:
                    self.backend.drone.move_right(0.25)
                elif self.search_fire_try==2 :
                    self.backend.drone.move_left(0.25)
                #self.backend.drone.move_forward(0.7)
            else:

                if self.search_min:
                    goto(self.backend, 0.08-2.5, 0.02, 1.7, self.flag,tol=0.4)
                    look_at(self.backend, 10000, 0, 0, self.flag,tol=4)
                    
                else:
                    goto(self.backend, 0.08-2.5, 0.72+0.35, 1.7, self.flag,tol=0.4)
                    look_at(self.backend, 10000, 0, 0, self.flag,tol=4)
                print("no fire and try to go another place")
                self.search_min = not self.search_min



            # elif state.x > 70:
            #    self.backend.drone.move_backward(max(state.x > 70, 25)/100.0)
            # elif abs(state.z - 180) > 15:
            #     if state.z > 180:
            #         self.backend.drone.move_down(max(min(state.z - 180, 50), 22)/100.0)
            #     else:
            #         self.backend.drone.move_up(max(min(180 - state.z, 50), 22)/100.0)
            # else:
            #     self.backend.drone.move_right(0.5)
            #     if 160 < state.y + 50 < 240:
            #        self.backend.drone.move_right(0.8)

    def step2(self):

        #goto(self.backend, -0.5-0.8, -0.25-0.5, self.ps[2][2]+0.55, self.flag,speed=100)
        #image, state = self.backend.drone.get_image_and_state() 
        '''
        goto(self.backend,
            self.ps[1][0]-0.4,     
            self.ps[2][2]-1,
            self.ps[2][2]+0.65, 
            self.flag,
            tol=0.5,
            max_x=200.0,
            max_y=200.0,
            max_z=60,
            speed=100)
        
        goto(self.backend,
            self.ps[2][0],
            self.ps[2][2]-1-0.75,
            self.ps[2][2]+0.2, 
            self.flag,
            tol=0.35,
            speed=100)
        '''
        goto(self.backend,
            self.ps[2][0],
            self.ps[2][2]-1-0.75-0.16,
            self.ps[2][2]+0.65, 
            self.flag,
            tol=0.35,
            max_x=215.0,
            max_y=215.0,
            max_z=60,
            speed=100)
        image, state = self.backend.drone.get_image_and_state()

        self.backend.drone.go(0,0,self.ps[2][2]+0.05-state.z/100.0,speed=100)

        look_at(self.backend,  self.ps[2][0],self.ps[2][1],self.ps[2][2], self.flag)
        self.detect_object(3)


        '''
        image, state = self.backend.drone.get_image_and_state()
        self.backend.drone.go(0,0,self.ps[0][2]+0.25+0.2-state.z/100.0)
        '''
        image, state = self.backend.drone.get_image_and_state()
        self.backend.drone.go(0,0.3,self.ps[0][2]+0.25-state.z/100.0,speed=100)
        '''
        goto(self.backend,
            self.ps[2][0]+0.1, 
            self.ps[2][2]-1-0.5, 
            self.ps[0][2]+0.25, 
            self.flag,
            speed=60)
        '''

        look_at(self.backend,  self.ps[0][0],self.ps[0][1],self.ps[0][2], self.flag)
        image, state = self.backend.drone.get_image_and_state()
        if state.z/100.0>=self.ps[0][2]+0.3:
            self.detect_object(1,min_radius_k=20,max_radius_k=4)
        else:
            self.detect_object(1,hint=(0,0,960,720/5*4),min_radius_k=20,max_radius_k=4)

        '''
        goto(self.backend,
            self.ps[3][0]+0.4, 
            self.ps[3][1]+0.75, 
            self.ps[3][2]+0.1, 
            self.flag,
            tol=0.30,
            speed=60)
        '''
        goto(self.backend,
            self.ps[3][0]+0.3, 
            self.ps[3][1]+0.88, 
            self.ps[3][2]+0.32, 
            self.flag,
            max_x=200.0,
            max_y=200.0,
            tol=0.4,
            speed=100)

        look_at(self.backend, self.ps[3][0],self.ps[3][1],self.ps[3][2], self.flag)
        self.detect_object(4)

        goto(self.backend,
            self.ps[4][0]+1.2, 
            self.ps[4][1], 
            self.ps[4][2],
            self.flag, 
            max_x=200.0,
            max_y=200.0,
            tol=0.35,
            speed=100) 
        
        look_at(self.backend, self.ps[4][0],self.ps[4][1],self.ps[4][2], self.flag)
        self.detect_object(5)

        goto(self.backend,
            2.5-0.3, 
            0-0.2, 
            self.ps[4][2]-0.4, 
            self.flag,
            tol=0.8,
            max_x=200.0,
            max_y=200.0,
            max_z=90,
            speed=100)


        self.GetFinalTargets()
        self.judge.send_targets(self.results)
        self.jump_stage(self.stage_land)

    def detect_fire(self, count=10,skip=1,sleep_duration=0.1):
        preimg = None
        frame = 0
        while self.flag() and count > 0:
            img, state = self.backend.drone.get_image_and_state()
            if img is preimg:
                time.sleep(sleep_duration)
                continue
            frame += 1
            if frame > skip:
                frame = 0
                count -= 1
                preimg = img
                x, y, w, h = find_red_ball(img)
                if x is not None:
                    break
        return x, y, w, h

    def detect_object(self, idx, hint=None, count=10, skip=1, sleep_duration=0.05,min_radius_k=10,max_radius_k=3):
        '''
        跳过skip帧后开始检测
        共检测count帧
        如果检测速度太快，则在下一帧前sleep
        hint决定检测图片的区域
        '''
        self.logger.info("----------Start to detect %s" % (idx))
        if hint is None:
            hint = (0, 0, 0, 0)
        preimg = None
        frame = 0
        result=None
        while self.flag() and count > 0:
            img, state = self.backend.drone.get_image_and_state()
            if img is preimg:
                time.sleep(sleep_duration)
                continue
            frame += 1
            if frame > skip:
                frame = 0
                count -= 1
                preimg = img
                result = self.detector.detect(img, hint[0], hint[1], hint[2], hint[3],min_radius_k=min_radius_k,max_radius_k=max_radius_k)
                if result !=None:
                    break
        if result !=None:
            self.logger.info("-----------at %d detect %s" % (idx, str(result)))
            self.results[idx-1]=result
        else:
            self.logger.info("-----------at %d detect None" % idx)

    def debug(self):
        self.counts=1
        image, state = self.backend.drone.get_image_and_state()
        if state.mid == -2:
            self.backend.drone.mon()
        if self.counts%10==0:
            self.backend.drone.mon()
            self.counts=1
        #find_red_ball(image)
        #print(state.mid,state.x,state.y,state.z,state.pitch,state.yaw,state.roll)
        self.detect_object(0)
        self.counts+=1
        time.sleep(0.1)


    def loop(self):
        if self.stage is not None:
            self.stage.do()

    def loop_thread(self):
        #tello_center.wait_until_proxy_available(self.backend)
        self.backend.drone.wait_for_image_and_state()
        self.jump_stage(self.stage_wait_for_start)
        #self.jump_stage(self.stage_test)
        while True:
            self.loop()
            time.sleep(0.01)
    def exit_thread(self):
        while True:
            line = str(raw_input())
            if 'exit' in line:
                self.run_flag=False
                while True:
                    self.backend.drone.land()
                    time.sleep(1)
                    #return

    def imshow_thread(self):
        cv2.namedWindow("obs",0)
        cv2.resizeWindow("obs", 960, 720)
        while self.flag()==True:
            try:
                img, state = self.backend.drone.get_image_and_state()
                #cv2.imshow("obs",img)
                #cv2.waitKey(80)#20Hz
                self.judge.PubImgAndState(img)
                time.sleep(0.08)
            except:
                time.sleep(1)


    def start(self):
        self.loop_thread_ = threading.Thread(target=self.loop_thread)
        self.loop_thread_.daemon = True
        self.loop_thread_.start()

        self.imshow_thread_ = threading.Thread(target=self.imshow_thread)
        self.imshow_thread_.daemon = True
        self.imshow_thread_.start()

    def GetFinalTargets(self):
        flags=[False,False,False]
        flags_num=[0,0,0]
        for i in range(5):
            for j in range(3):
                if self.results[i]==j:
                    flags[j]=True
                    flags_num[j]+=1
        #冲突情况
        #信任优先级：
        team=[4,3,0,2,1]
        collision_flag=False
        for num in flags_num:
            if num>=2:
                collision_flag=True
                break
        #查出1/2/3个颜色，其中撞车了
        if collision_flag==True:
            for i in range(5):
                for j in range(i):
                    if self.results[team[i]]!=None and self.results[team[i]]==self.results[team[j]]:#与优先级高的撞了
                        for k in [2,1,0]:
                            if flags[k]==False:
                                self.results[team[i]]=k
                                flags_num[self.results[team[j]]]-=1
                                flags_num[k]+=1
                                flags[k]=True
                                break
                            if k==0:#三个颜色全满，只能为None
                                self.results[team[i]]=None
                                flags_num[team[i]]-=1

        #只查出0/1种颜色情况
        sum=0
        for i in range(3):
            sum+=flags_num[i]
        if sum<=1:
            for team_idx in team[::-1]:
                if self.results[team_idx]==None:
                    for k in range(3):
                        if flags[k]==False:
                            self.results[team_idx]=k
                            flags[k]=True
                            flags_num[k]+=1
                            break
                

        for i in range(3):
            if flags[i]==False:
                self.results[1]=i
















if __name__ == '__main__':
    mc=MainControl()
    mc.exit_thread()

    

