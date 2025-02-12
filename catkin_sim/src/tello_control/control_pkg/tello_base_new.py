# -*- coding: utf-8 -*-
import cv2
import socket
import threading
import time
import numpy as np
import libh264decoder
from control_pkg.stats import Stats
import control_pkg.sl4p as sl4p
from  control_pkg.tello_center import Service
from  utils_pkg.fps import FpsRecoder
from control_pkg.tello_data import TelloData


class TimeoutException(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)



class MyTello:
    """Wrapper class to interact with the Tello drone."""

    KEY_STATE_FPS = 'state_fps'
    KEY_VIDEO_FPS = 'video_fps'

    def __init__(self, local_ip, local_port, imperial=False, command_timeout=.3, tello_ip='192.168.10.1',
                 tello_port=8889):
        """
        Binds to the local IP/port and puts the Tello into command mode.

        :param local_ip (str): Local IP address to bind.
        :param local_port (int): Local port to bind.
        :param imperial (bool): If True, speed is MPH and distance is feet.
                             If False, speed is KPH and distance is meters.
        :param command_timeout (int|float): Number of seconds to wait for a response to a command.
        :param tello_ip (str): Tello IP.
        :param tello_port (int): Tello port.
        """
        self.logger = sl4p.Sl4p("tello_base", "1;33")
        self.do_print_info = True
        self.filter = None
        self.request_lock = threading.Lock()
        self.response_handler_lock = threading.Lock()
        self.response_handler = None

        self.abort_flag = False
        self.decoder = libh264decoder.H264Decoder()
        self.command_timeout = command_timeout
        self.imperial = imperial
        self.response = None
        self.frame = None  # numpy array BGR -- current camera output frame
        self.is_freeze = False  # freeze current camera output
        self.last_frame = None

        self.log = []
        self.MAX_TIME_OUT = 10.0

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # socket for sending cmd
        self.socket_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # socket for receiving video stream
        self.socket_state = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # state socket
        self.tello_ip = tello_ip
        self.tello_address = (tello_ip, tello_port)
        self.local_video_port = 11111  # port for receiving video stream
        self.last_height = 0
        self.socket.bind((local_ip, local_port))

        # thread for receiving cmd ack
        self.receive_thread = threading.Thread(target=self._receive_thread)
        self.receive_thread.daemon = True

        self.receive_thread.start()

        # to receive video -- send cmd: command, streamon
        self.socket.sendto(b'command', self.tello_address)
        self.logger.info('into command mode')
        self.socket.sendto(b'streamon', self.tello_address)
        self.logger.info('open video stream')

        # thread for receiving video
        self.socket_video.bind((local_ip, self.local_video_port))
        self.receive_video_thread = threading.Thread(target=self._receive_video_thread)
        self.receive_video_thread.daemon = True
        self.receive_video_thread.start()

        # state receive
        self.results = None
        self.socket_state.bind((local_ip, 8890))
        self.receive_state_thread = threading.Thread(target=self._receive_state_thread)
        self.receive_state_thread.daemon = True
        self.receive_state_thread.start()

        self.stop = False
        self.latest_safe_state = None

        self.state_lock = threading.Lock()
        self.video_lock = threading.Lock()
        self.state = None
        self.image = None
        self.has_takeoff = False

    def print_info(self, msg):
        if self.do_print_info:
            if self.filter is not None and not self.filter(msg):
                return
            self.logger.info(msg)

    def __del__(self):
        """Closes the local socket."""
        self.land()
        self.socket.close()
        self.socket_state.close()
        self.socket_video.close()

    def get_image_and_state(self):
        self.video_lock.acquire()
        self.state_lock.acquire()
        try:
            return self.image, self.state  # type: (np.ndarray, TelloData)
        finally:
            self.state_lock.release()
            self.video_lock.release()

    def get_state(self):
        self.state_lock.acquire()
        try:
            return self.state
        finally:
            self.state_lock.release()

    def wait_for_image_and_state(self, timeout=60):
        start = time.time()
        while True:
            img, s = self.get_image_and_state()
            if img is not None and s is not None:
                self.logger.info('image and state got')
                return
            if time.time() - start > timeout:
                self.logger.error("timeout: wait_for_image_and_state")
                raise TimeoutException(msg="timeout: wait_for_image_and_state")
            time.sleep(0.1)

    def _receive_thread(self):
        """Listen to responses from the Tello.

        Runs as a thread, sets self.response to whatever the Tello last returned.

        """

        while True:
            try:
                response, ip = self.socket.recvfrom(3000)
                self.response = response.decode()
                if len(self.log) != 0:
                    self._on_response(self.response)
                    self.log[-1].add_response(self.response)
                    # self.print_info(self.response)
            except socket.error as exc:
                self.print_info("Caught exception socket.error : %s"%exc)

    def _on_response(self, response):
        self.response_handler_lock.acquire()
        try:
            self.print_info("OnResponse: " + response)
            if self.response_handler is not None:
                self.response_handler(response)
        finally:
            self.response_handler_lock.release()

    def _set_response_handler(self, response_handler):
        self.response_handler_lock.acquire()
        try:
            self.response_handler = response_handler
        finally:
            self.response_handler_lock.release()

    def _receive_video_thread(self):
        '''
        video_fps = FpsRecoder(MyTello.KEY_VIDEO_FPS)
        cap = cv2.VideoCapture("udp://0.0.0.0:11111")
        frame_to_skip = 120  # 跳过前120帧
        while True:
            _, img = cap.read()

            if img is not None:
                video_fps.on_loop()
                if frame_to_skip > 0:
                    frame_to_skip -= 1
                else:
                    self.image = img
                    # cv2.imshow('raw', img)
                    # cv2.waitKey(1)
                    #self.video_lock.acquire()
                    #try:
                    #    self.image = img
                    #finally:
                    #    self.video_lock.release()
            else:
                time.sleep(0.001)
        '''
        packet_data = ""
        while True:
            try:
                res_string, ip = self.socket_video.recvfrom(2048)
                packet_data += res_string
                # end of frame
                if len(res_string) != 1460:
                    for frame in self._h264_decode(packet_data):
                        #self.image = frame
                        self.image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    packet_data = ""

            except socket.error as exc:
                print ("Caught exception socket.error : %s" % exc)
    def _h264_decode(self, packet_data):
        """
        decode raw h264 format data from Tello
        
        :param packet_data: raw h264 data array
       
        :return: a list of decoded frame
        """
        res_frame_list = []
        frames = self.decoder.decode(packet_data)
        for framedata in frames:
            (frame, w, h, ls) = framedata
            if frame is not None:
                # print 'frame size %i bytes, w %i, h %i, linesize %i' % (len(frame), w, h, ls)

                frame = np.fromstring(frame, dtype=np.ubyte, count=len(frame), sep='')
                frame = (frame.reshape((h, ls / 3, 3)))
                frame = frame[:, :w, :]
                res_frame_list.append(frame)

        return res_frame_list

    def _receive_state_thread(self):
        state_fps = FpsRecoder(MyTello.KEY_STATE_FPS)
        while True:
            try:
                state, ip = self.socket_state.recvfrom(1024)
                state = state.decode()
                out = state.replace(';', ';\n')
                self.results = out.split()
                if not (self.results == 'ok'):
                    state_fps.on_loop()
                    s = TelloData("".join(self.results[0:8]) + self.results[15])
                    self.state_lock.acquire()
                    try:
                        self.state = s
                        if s.mid != -1:
                            self.latest_safe_state = s
                    finally:
                        self.state_lock.release()
                # self.print_info(self.response)
            except BaseException as exc:
                self.logger.error("Caught exception : %s"%exc)

    def send_command(self, command, timeout=None):
        """
        Send a command to the Tello and wait for a response.

        :param command: Command to send.
        :return (str): Response from Tello.

        """
        if timeout is None:
            timeout = self.MAX_TIME_OUT

        self.print_info("command: %s"%str(command))
        if self.stop:
            time.sleep(0.1)
            return ""
        self.request_lock.acquire()
        try:
            self.log.append(Stats(command, len(self.log)))
            # self.abort_flag = False
            # timer = threading.Timer(self.command_timeout, self.set_abort_flag)

            self.socket.sendto(command.encode('utf-8'), self.tello_address)
            start = time.time()
            while not self.log[-1].got_response():
                now = time.time()
                diff = now - start
                if diff > timeout:
                    self.logger.error("timeout: %s"%command)
                    break
                    # raise TimeoutException("[tello] command timeout: " + command)

            self.print_info("Done!!! sent command: %s to %s"%(command, self.tello_ip))
            return self.log[-1].got_response()
        finally:
            time.sleep(0.3)
            self.request_lock.release()

    def set_abort_flag(self):
        """
        Sets self.abort_flag to True.

        Used by the timer in Tello.send_command() to indicate to that a response

        timeout has occurred.

        """

        self.abort_flag = True

    def takeoff(self):
        """
        Initiates take-off.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """
        self.has_takeoff = True
        return self.send_command('takeoff', timeout=7.0)

    def set_speed(self, speed):
        """
        Sets speed.

        This method expects KPH or MPH. The Tello API expects speeds from
        1 to 100 centimeters/second.

        Metric: .1 to 3.6 KPH
        Imperial: .1 to 2.2 MPH

        Args:
            speed (int|float): Speed.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """

        speed = float(speed)

        if self.imperial is True:
            speed = int(round(speed*44.704))
        else:
            speed = int(round(speed*27.7778))

        return self.send_command('speed %s'%speed)

    def rotate_cw(self, degrees):
        """
        Rotates clockwise.

        Args:
            degrees (int): Degrees to rotate, 1 to 360.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """

        return self.send_command('cw %s'%degrees)

    def rotate_ccw(self, degrees):
        """
        Rotates counter-clockwise.

        Args:
            degrees (int): Degrees to rotate, 1 to 360.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """
        return self.send_command('ccw %s'%degrees)

    def flip(self, direction):
        """
        Flips.

        Args:
            direction (str): Direction to flip, 'l', 'r', 'f', 'b'.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """

        return self.send_command('flip %s'%direction)

    def get_response(self):
        """
        Returns response of tello.

        Returns:
            int: response of tello.

        """
        response = self.response
        return response

    def get_height(self):
        """Returns height(dm) of tello.

        Returns:
            int: Height(dm) of tello.

        """
        height = self.send_command('height?')
        height = str(height)
        height = filter(str.isdigit, height)
        try:
            height = int(height)
            self.last_height = height
        except:
            height = self.last_height
            pass
        return height

    def get_battery(self):
        """Returns percent battery life remaining.

        Returns:
            int: Percent battery life remaining.

        """

        battery = self.send_command('battery?')

        try:
            battery = int(battery)
        except:
            pass

        return battery

    def get_flight_time(self):
        """Returns the number of seconds elapsed during flight.

        Returns:
            int: Seconds elapsed during flight.

        """

        flight_time = self.send_command('time?')

        try:
            flight_time = int(flight_time)
        except:
            pass

        return flight_time

    def get_speed(self):
        """Returns the current speed.

        Returns:
            int: Current speed in KPH or MPH.

        """

        speed = self.send_command('speed?')

        try:
            speed = float(speed)

            if self.imperial is True:
                speed = round((speed/44.704), 1)
            else:
                speed = round((speed/27.7778), 1)
        except:
            pass

        return speed

    def land(self):
        """Initiates landing.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """
        self.has_takeoff = False
        return self.send_command('land')

    def go(self, x, y, z, speed=60):
        """
        x+ 前
        y+ 左
        z+ 上
        @return:
        """
        x = int(x * 100.0)
        y = int(y * 100.0)
        z = int(z * 100.0)
        return self.send_command('go %d %d %d %d' % (x, y, z, speed))

    def curve(self,x1, y1, z1,x2, y2, z2,speed=60):
        """
        x+ 前
        y+ 左
        z+ 上
        @return:
        """
        x1 = int(x1 * 100.0)
        y1 = int(y1 * 100.0)
        z1 = int(z1 * 100.0)
        x2 = int(x2 * 100.0)
        y2 = int(y2 * 100.0)
        z2 = int(z2 * 100.0)
        return self.send_command('curve %d %d %d %d %d %d %d' % (x1, y1, z1,x2, y2, z2, speed))

    def move(self, direction, distance):
        """Moves in a direction for a distance.

        This method expects meters or feet. The Tello API expects distances
        from 20 to 500 centimeters.

        Metric: .02 to 5 meters
        Imperial: .7 to 16.4 feet

        Args:
            direction (str): Direction to move, 'forward', 'back', 'right' or 'left'.
            distance (int|float): Distance to move.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """

        distance = float(distance)

        if self.imperial is True:
            distance = int(round(distance*30.48))
        else:
            distance = int(round(distance*100))

        return self.send_command('%s %s'%(direction, distance))

    def move_backward(self, distance):
        """Moves backward for a distance.

        See comments for Tello.move().

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """

        return self.move('back', distance)

    def move_down(self, distance):
        """Moves down for a distance.

        See comments for Tello.move().

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """

        return self.move('down', distance)

    def move_forward(self, distance):
        """Moves forward for a distance.

        See comments for Tello.move().

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """
        return self.move('forward', distance)

    def move_left(self, distance):
        """Moves left for a distance.

        See comments for Tello.move().

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """
        return self.move('left', distance)

    def move_right(self, distance):
        """Moves right for a distance.

        See comments for Tello.move().

        """
        return self.move('right', distance)

    def move_up(self, distance):
        """Moves up for a distance.

        See comments for Tello.move().

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """

        return self.move('up', distance)

    def mon(self):
        return self.send_command('mon')

class TelloBackend(Service):
    def __init__(self):
        Service.__init__(self)
        self.drone= MyTello('', 8888)
        self.drone.stop = False
        self.drone.do_print_info = False  #TODO
        self.drone.logger.info("wait for image and state")
        self.drone.wait_for_image_and_state()

        #self.drone.land() 系统退出前请先降落
