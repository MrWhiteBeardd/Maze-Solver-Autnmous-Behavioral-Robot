from controller import Robot
import cv2
import numpy as np

# Create the Robot instance and initialize devices
robot = Robot()
timestep = int(robot.getBasicTimeStep())
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Initialize distance sensors
sensors = []
sensor_names = ['ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7']
for name in sensor_names:
    sensor = robot.getDevice(name)
    sensor.enable(timestep)
    sensors.append(sensor)

# Initialize the onboard camera
camera = robot.getDevice('camera')
camera.enable(timestep)

# Define behavior tree nodes
class Node:
    def run(self):
        pass

class Selector(Node):
    def __init__(self, children):
        self.children = children

    def run(self):
        for child in self.children:
            if child.run():
                return True
        return False

class Sequence(Node):
    def __init__(self, children):
        self.children = children

    def run(self):
        for child in self.children:
            if not child.run():
                return False
        return True

class CheckObstacle(Node):
    def __init__(self, sensors, threshold):
        self.sensors = sensors
        self.threshold = threshold

    def run(self):
        for sensor in self.sensors:
            if sensor.getValue() > self.threshold:
                return True
        return False

class MoveForward(Node):
    def run(self):
        left_motor.setVelocity(2.0)  # Reduced speed
        right_motor.setVelocity(2.0)  # Reduced speed
        return True

class Turn(Node):
    def __init__(self, direction):
        self.direction = direction

    def run(self):
        if self.direction == 'left':
            left_motor.setVelocity(1.0)  # Reduced speed
            right_motor.setVelocity(-1.0)  # Reduced speed
        elif self.direction == 'right':
            left_motor.setVelocity(-1.0)  # Reduced speed
            right_motor.setVelocity(1.0)  # Reduced speed
        return True

class DetectBall(Node):
    def __init__(self, camera, colors):
        self.camera = camera
        self.colors = colors
        self.current_color = 0
        self.color_names = ['Red', 'Green', 'Blue', 'Black']
        self.detected_sequence = []
        self.color_detected = False

    def run(self):
        # If a ball of the current color has been detected, move on to the next color
        if self.color_detected:
            self.current_color = (self.current_color + 1) % len(self.colors)
            self.color_detected = False

        # Capture an image from the robot's camera
        camera_data = self.camera.getImageArray()
        image = np.array(camera_data, dtype=np.uint8)

        # Convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Define the color range for the current color
        lower_color = np.array(self.colors[self.current_color][0])
        upper_color = np.array(self.colors[self.current_color][1])

        # Create a mask for the current color
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If a contour is found, print a message on the console
        if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > 50:  # Reduced contour size
            print(f'Detected {self.color_names[self.current_color]} ball')
            self.detected_sequence.append(self.color_names[self.current_color])
            print(f'Sequence of detected balls: {self.detected_sequence}')
            self.color_detected = True
            return True

        return False

# Define the color ranges for the balls
colors = [
    [[0, 100, 100], [10, 255, 255]],  # Red
    [[60, 50, 50], [80, 255, 255]],  # Green
    [[110, 50, 50], [130, 255, 255]],  # Blue
    [[0, 0, 0], [180, 255, 30]]  # Black
]

# Construct the behavior tree
obstacle_threshold = 80
root = Selector([
    Sequence([
        CheckObstacle(sensors[0:3], obstacle_threshold),
        Turn('right')
    ]),
    Sequence([
        CheckObstacle(sensors[5:8], obstacle_threshold),
        Turn('left')
    ]),
    DetectBall(camera, colors),
    MoveForward()
])

# Main loop
while robot.step(timestep) != -1:
    root.run()
