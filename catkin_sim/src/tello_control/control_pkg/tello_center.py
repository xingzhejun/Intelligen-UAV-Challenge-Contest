# -*- coding: utf-8 -*-
class Service:
    def __init__(self):
        self.started = False
        self.run_flag = True

    def flag(self):
        return self.run_flag

    def available(self):
        return True

    def on_register(self):
        pass

    def call_start(self):
        self.start()
        self.started = True

    def start(self):
        pass

    def on_request_exit(self):
        self.run_flag = False