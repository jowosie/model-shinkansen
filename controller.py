class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp                # Proportional Gain
        self.Ki = Ki                # Integral Gain
        self.Kd = Kd                # Derivative Gain
        self.setpoint = setpoint    # Target Value

        # Internal Variables
        self.previous_error = 0.0
        self.integral = 0.0

    def calculate(self, current_value, dt):
        error = self.setpoint - current_value

        # Proportional Term
        p_term = self.Kp * error

        # Integral Term
        self.integral += error * dt
        i_term = self.Ki * self.integral

        # Derivative Term
        derivative = (error - self.previous_error) / dt
        d_term = self.Kd * derivative

        # Update previous error for next iteration
        self.previous_error = error

        # Total output
        return p_term + i_term + d_term