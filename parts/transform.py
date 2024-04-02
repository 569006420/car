# -*- coding: utf-8 -*-

import time

class Lambda:

    def __init__(self, f):

        self.f = f

    def run(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def shutdown(self):
        return




class PIDController:


    def __init__(self, p=0, i=0, d=0, debug=False):

        # initialize gains
        self.Kp = p
        self.Ki = i
        self.Kd = d

        # The value the controller is trying to get the system to achieve.
        self.target = 0

        # initialize delta t variables
        self.prev_tm = time.time()
        self.prev_feedback = 0
        self.error = None

        # initialize the output
        self.alpha = 0

        # debug flag (set to True for console output)
        self.debug = debug

    def run(self, target_value, feedback):
        curr_tm = time.time()

        self.target = target_value
        error = self.error = self.target - feedback

        # Calculate time differential.
        dt = curr_tm - self.prev_tm

        # Initialize output variable.
        curr_alpha = 0

        # Add proportional component.
        curr_alpha += self.Kp * error

        # Add integral component.
        curr_alpha += self.Ki * (error * dt)

        # Add differential component (avoiding divide-by-zero).
        if dt > 0:
            curr_alpha += self.Kd * ((feedback - self.prev_feedback) / float(dt))

        # Maintain memory for next loop.
        self.prev_tm = curr_tm
        self.prev_feedback = feedback

        # Update the output
        self.alpha = curr_alpha

        if (self.debug):
            print('PID target value:', round(target_value, 4))
            print('PID feedback value:', round(feedback, 4))
            print('PID output:', round(curr_alpha, 4))

        return curr_alpha
