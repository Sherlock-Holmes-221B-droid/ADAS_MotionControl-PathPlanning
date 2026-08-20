class PIDSpeedController:

    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral   = 0.0
        self.prev_error = 0.0

    def compute_control(self, target_speed, current_speed, dt):

        error              =  (target_speed - current_speed)
        self.integral      += (error * dt)
        derivative         =  (error - self.prev_error)/dt
        accel              =  (self.kp * error + self.ki * self.integral + self.kd * derivative)

        self.prev_error    = error

        return accel