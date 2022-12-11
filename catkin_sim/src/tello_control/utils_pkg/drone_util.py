
# -*- coding: utf-8 -*-
from control_pkg import tello_base, tello_data
from utils_pkg.world import *
import control_pkg.sl4p as sl4p
import numpy as np
import typing
import time

go_logger = sl4p.Sl4p('__go__')


def clamp_abs(v, minv, maxv):
    vv = max(min(abs(v), maxv), minv)
    return vv if v > 0 else (0 if v == 0 else -vv)


def find_most_possible_object(collect):
    poss = []
    for ss in collect:
        if collect[ss]['count'] < 6 or collect[ss]['max_conf'] < 0.80 or collect[ss]['object_conf'] < 0.4:
            continue
        else:
            poss.append(collect[ss])
    if len(poss) == 0:
        return None
    if len(poss) == 1:
        return poss[0]
    if len(poss) > 1:
        max_obj = None
        for s in poss:
            if max_obj is None or s['object_conf'] > max_obj['object_conf']:
                max_obj = s
        return max_obj


def go_abs(drone, state, dis):
    """
    预处理了yaw的移动，输入x, y, z基于原始坐标系就可以了
    """
    if state.mid == -1:
        raise BaseException("mid == -1")
    hpr = state.get_hpr()
    dis = np.matmul(hpr2matrix(hpr), dis)
    drone.go(dis[1], dis[0], dis[2])


def clamp_ang(ang):
    return (ang + 180) % 360 - 180


def look_at(backend, x, y, z, flag,tol=1):
    '''
    don't care z
    '''
    while True:
        if not flag():
            return
        state = backend.drone.get_state()
        if state.mid == -1:
            backend.drone.move_up(0.3)
        else:
            break
    state = backend.drone.get_state()
    dis = vec3(x, y, z) * 100 - vec3(state.x, state.y, state.z)
    ang = int(np.arctan2(dis[1], dis[0]) / np.pi * 180)
    dis_ang = state.mpry[1] - ang
    dis_ang = clamp_ang(dis_ang)
    if not flag():
        return
    if dis_ang < -tol:
        backend.drone.rotate_ccw(-dis_ang)
    elif dis_ang > tol:
        backend.drone.rotate_cw(dis_ang)
    else:
        return


def goto_new(backend, x, y, z, flag, itridx=0, tol=0.45,speed=60,max_x=130.0,max_y=130.0,max_z=45.0,no_check_flag=False):
    if itridx >= 5:
        return
    while True:
        if not flag():
            return
        state = backend.drone.get_state()
        if state.mid == -1:
            backend.drone.move_up(0.3)
        else:
            break
    if not flag():
        return
    state = backend.drone.get_state()
    prestate = state
    dis = vec3(x, y, z)*100 - vec3(state.x, state.y, state.z)
    ang = state.mpry[1] / 180.0 * np.pi
    dis[0], dis[1] = dis[0] * np.cos(ang) + dis[1] * np.sin(ang), dis[0] * np.sin(ang) - dis[1] * np.cos(ang)
    if abs(dis[0]) > 11 or abs(dis[1]) > 11 or abs(dis[2]) > 11:
        if abs(dis[0]) > 11:
            s = -1 if dis[0] < 0 else 1
            dis[0] = s * min(max_x, max(20.0, abs(dis[0])))#130
        if abs(dis[1]) > 11:
            s = -1 if dis[1] < 0 else 1
            dis[1] = s * min(max_y, max(20.0, abs(dis[1])))#130
        if abs(dis[2]) > 11:
            s = -1 if dis[2] < 0 else 1
            dis[2] = s * min(max_z, max(20.0, abs(dis[2])))#45

        dis = dis / 100.0
        backend.drone.go(dis[0], -dis[1], dis[2],speed=speed)
        if no_check_flag==True:
            return
        state = backend.drone.get_state()
        if state.mid == -1:
            return
        if np.linalg.norm(vec3(prestate.x, prestate.y, prestate.z) - vec3(state.x, state.y, state.z)) < 3:
            go_logger.info('maybe mid error,ready to land')
            #backend.drone.land()
            time.sleep(2)
            return 
            goto_new(backend, x, y, z, flag, itridx=itridx, tol=tol,speed=speed,max_x=max_x,max_y=max_y,max_z=max_z,no_check_flag=no_check_flag)
            return
        dis = vec3(x, y, z)*100 - vec3(state.x, state.y, state.z)
        dis = dis / 100.0
        l = float(np.linalg.norm(dis))
        go_logger.info('dis %s' % l)
        if l > tol:
            goto_new(backend, x, y, z, flag, itridx=itridx + 1,speed=speed)


def curve_to(backend,x1, y1, z1,x2, y2, z2,flag,speed=60):
    while True:
        if not flag():
            return
        state = backend.drone.get_state()
        if state.mid == -1:
            backend.drone.move_up(0.3)
        else:
            break
    if not flag():
        return
    state = backend.drone.get_state()
    dis1 = vec3(x1, y1, z1)*100 - vec3(state.x, state.y, state.z)
    ang = state.mpry[1] / 180.0 * np.pi
    dis1[0], dis1[1] = dis1[0] * np.cos(ang) + dis1[1] * np.sin(ang), dis1[0] * np.sin(ang) - dis1[1] * np.cos(ang)
    dis1 = dis1 / 100.0

    dis2 = vec3(x2, y2, z2)*100 - vec3(state.x, state.y, state.z)
    ang = state.mpry[1] / 180.0 * np.pi
    dis2[0], dis2[1] = dis2[0] * np.cos(ang) + dis2[1] * np.sin(ang), dis2[0] * np.sin(ang) - dis2[1] * np.cos(ang)
    dis2 = dis2 / 100.0
    
    backend.drone.curve(dis1[0], -dis1[1], dis1[2],dis2[0], -dis2[1], dis2[2],speed)



def goto_old(backend, x, y, z, flag, itridx=0, tol=0.45):
    if itridx >= 5:
        return
    while True:
        if not flag():
            return
        state = backend.drone.get_state()
        if state.mid == -1:
            backend.drone.move_up(0.3)
        else:
            break
    if not flag():
        return
    state = backend.drone.get_state()
    dis = vec3(x, y, z)*100 - vec3(state.x, state.y, state.z)
    dis_f = min(150.0, np.linalg.norm(vec3(dis[0], dis[1], 0)))/100.0
    dis_u = dis[2]/100.0
    if dis_f <= 0.25 and abs(dis_u) <= 0.25:
        return

    look_at(backend, x, y, z, flag)
    state = backend.drone.get_state()
    dis = vec3(x, y, z) * 100 - vec3(state.x, state.y, state.z)
    dis_f = max(0.0, min(150.0, np.linalg.norm(vec3(dis[0], dis[1], 0)) - 10)) / 100.0
    dis_u = dis[2] / 100.0
    if not flag():
        return
    if dis_f > 0.15 or abs(dis_u) > 0.11:
        if abs(dis_u) > 0.11:
            sign = -1 if dis_u < 0 else 1
            dis_u = sign * max(0.2, abs(dis_u))
        backend.drone.go(dis_f, 0, dis_u)
        state = backend.drone.get_state()
        if state.mid != -1:
            dis = vec3(x, y, z)*100 - vec3(state.x, state.y, state.z)
            dis[2] *= 1.7
            dis = np.linalg.norm(dis/100.0)
            go_logger.info("dis %.2f" % dis)
            if dis > tol:
                goto_old(backend, x, y, z, flag, itridx=itridx + 1, tol=tol)


goto = goto_new


if __name__ == '__main__':
    print(clamp_ang(100))
    print(clamp_ang(400))
    print(clamp_ang(180))
    print(clamp_ang(-30))
    print(clamp_ang(-190))

