
from tdmclient import ClientAsync
import time
from image_detection import *
from evolutionary import *
import numpy as np
import cv2

from copy import deepcopy
import random

seeker_program = """
var reset_delay = 100
call prox.comm.enable(1)

prox.comm.tx = 1

onevent prox.comm
    if prox.comm.rx != 0 then
        timer.period[1] = reset_delay
    end

onevent timer1
    prox.comm.rx = 0
    timer.period[1] = 0
"""

old_seeker_program = """
var send_interval = 200  # time in milliseconds
var reset_delay = 100 # reset message time interval
var signal_flag = 0

timer.period[0] = send_interval
call prox.comm.enable(1)

onevent timer0
    prox.comm.tx = 1
    
onevent prox.comm
    if prox.comm.rx != 0 then
        signal_flag = prox.comm.rx
        timer.period[1] = reset_delay
    end

onevent timer1
    if prox.comm.rx != 0 then
        prox.comm.rx = 0
        signal_flag = 0
        timer.period[1] = 0
    end
"""

camera = cv2.VideoCapture(0)

class SeekerController:
    def __init__(self):
        self.is_tagged = False
        self.in_grey_area = False
        self.last_message_time = None
        self.speeds = (None, None)
        self.reload_grey = False
        self.cur_gen = None

        self.all_weights = []

        MAX_MOTOR_SPEED_FORWARD = 450
        #MAX_MOTOR_SPEED_BACKWORD = -600
        MAX_MOTOR_SPEED_BACKWORD = 0


        # Set the LED lights on the robot
        def led_state(node, color):
            node.v.leds.top = color
            node.v.leds.bottom.left = color
            node.v.leds.bottom.right = color

        # Running the Thymio robot
        def run_motor(node, left, right):
            print(left, right)
            node.v.motor.left.target = left  
            node.v.motor.right.target = right   
        

        # Detect which area the robot is in
        def area_detection(reflected_values, node):
            #Detect black lines
            if reflected_values[0] < 200 or reflected_values[1] < 200:
                led_state(node, [32, 0, 0]) # Turn red
                return 100, -100

            #Detects grey area
            elif (reflected_values[0] > 900) and (reflected_values[1] > 900):
                led_state(node, [32, 8, 0]) # Turn orange
                self.in_grey_area = True
                return None

            else:
                led_state(node, [32,0, 0]) # Turn red
                self.in_grey_area = False
                return None

        def run(weight, node):
            
            MAX_AREA = 480*640
            
            while True:
                
                prox_values = node.v.prox.horizontal
                
                if (sum(prox_values) > 20000): #or self.is_tagged:
                    camera.release()
                    cv2.destroyAllWindows()
                    break
                
                hsv, image = take_picture(camera)
                
                if image is not None:
                    
                    blue_area, blue_direction = get_image(hsv, image, np.array([110, 50, 50]), np.array([130, 255, 255]))
                    green_area, green_direction = get_image(hsv, image, np.array([35, 50, 50]), np.array([85, 255, 255]))
                    blue_area = blue_area / MAX_AREA
                    green_area = green_area / MAX_AREA
                    
                    model = NN(5)
                    
                    input_weights = torch.tensor(weight, dtype=torch.float32).view(2, 5)

                    with torch.no_grad():
                        model.fc.weight = nn.Parameter(input_weights)

                    if blue_direction is None:
                        blue_direction = 0
                    if green_direction is None:
                        green_direction = 0
                    
                    blue_direction = blue_direction / 5
                    green_direction = green_direction / 5

                    input_nodes = [blue_direction, blue_area, green_direction, green_area, -1]

                    x = torch.tensor(input_nodes, dtype=torch.float32).unsqueeze(0)
                    
                    # Forward pass through the model
                    output = model(x)

                    # Use the output for further processing (e.g., control the robot)

                    left_motor_speed = output[0][0].item() * (MAX_MOTOR_SPEED_FORWARD - MAX_MOTOR_SPEED_BACKWORD) + MAX_MOTOR_SPEED_BACKWORD
                    right_motor_speed = output[0][1].item() * (MAX_MOTOR_SPEED_FORWARD - MAX_MOTOR_SPEED_BACKWORD) + MAX_MOTOR_SPEED_BACKWORD
                    
                    reflected_values = node.v.prox.ground.reflected
                    detected_speeds = area_detection(reflected_values, node)
                    
                    self.speeds = detected_speeds if detected_speeds is not None else (left_motor_speed, right_motor_speed)
                    run_motor(node, int(self.speeds[0]), int(self.speeds[1]))
                    
                    node.flush()

        with ClientAsync() as client:

            async def prog():
                """
                Asynchronous function controlling the Thymio.
                """

                # Lock the node representing the Thymio to ensure exclusive access.
                with await client.lock() as node:
                    # Compile and send the program to the Thymio.
                    error = await node.compile(seeker_program)
                    if error is not None:
                        print(f"Compilation error: {error['error_msg']}")
                    else:
                        error = await node.run()
                        if error is not None:
                            print(f"Error {error['error_code']}")

                    # Wait for the robot's proximity sensors to be ready.
                    await node.wait_for_variables({"prox.horizontal"})
                    print("Thymio started successfully!")
                    

                    self.speeds = node.v.motor.left.target, node.v.motor.right.target
                    
                    # Set time interval on 200ms
                    node.v.timer.period[0] = 10000
                    # Set initial LED state
                    led_state(node, [32, 0, 0])
                    node.flush() #send the initial state to Thymio

                    weights = [0.6563,0.8201,3.8740,1.1422,0.0,0.6669,2.0687,5.5944,1.0300,0.9585]
                    #weights = [0.6900386347151003, 0.773884953257336, 3.608458471193268, 1.1529785254089457, 0.0, 0.8958776042164931, 2.7195329805705195, 5.210423864706167, 1.050714740775008, 0.8399548479269693]
                    run(weights, node)
                    
                    message = node.v.prox.comm.rx

                    """
                    Get the value of the message received from the other Thymio
                    the value is 0 if no message has been received and 
                    gets set to a new value when a message is received" 
                    """
                    # if the robot is blocked or it has been tagged, the program should terminate
                    prox_values = node.v.prox.horizontal
                    

                    node.flush()  # Send the set commands to the robot.

                    await client.sleep(0.3)  # Pause for 0.3 seconds before the next iteration.

                    # Once out of the loop, stop the robot and set the top LED to red.
                    print("Thymio stopped successfully!")
                    node.v.motor.left.target = 0
                    node.v.motor.right.target = 0
                    node.v.leds.top = [32, 0, 0]
                    node.flush()

            # Run the asynchronous function to control the Thymio.
            client.run_async_program(prog)

if __name__ == "__main__":

    # Instantiate the AvoiderController class, which initializes and starts the robot's behavior for the avoider.
    SeekerController()
