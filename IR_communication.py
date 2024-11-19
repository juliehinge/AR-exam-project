

# TODO correct color scheme

from tdmclient import ClientAsync
import time

seeker_program = """
var send_interval = 200  # time in milliseconds
timer.period[0] = send_interval

leds.top = [32, 0, 0]
leds.bottom.left = [32, 0, 0]
leds.bottom.right = [32, 0, 0]

call prox.comm.enable(1)
onevent timer0
    prox.comm.tx = 1
"""

avoider_program = """
var send_interval = 200  # time in milliseconds
timer.period[0] = send_interval
call prox.comm.enable(1)
leds.top = [0, 32, 0]
timer.period[0] = send_interval

onevent timer0
    prox.comm.tx = 2
    
onevent prox.comm
    if prox.comm.rx == 1 then
        leds.top = [32, 0, 32]
        leds.bottom.left = [32, 0, 32]
        leds.bottom.right = [32, 0, 32]
    end
    
"""



class AvoiderController:
    def __init__(self):
        self.is_tagged = False
        self.in_grey_area = False
        self.last_message_time = None

        # Set the LED lights on the robot
        def led_state(node, color):
            node.v.leds.top = color
            node.v.leds.bottom.left = color
            node.v.leds.bottom.right = color

        # Running the Thymio robot
        def run_motor(node, left, right):
            node.v.motor.left.target = left  
            node.v.motor.right.target = right   
        
        # Obstacle avoidance bevahior
        def behaviorOA(prox_values, reflected_values):
            """
            Obstacle avoidance behavior function.
            Given the proximity sensor values, it determines the Thymio's motion.
            """

            # 5 is in the back left
            # 0 is the front left
            # 6 is in the back right
            # Reflected value 0 is left buttom

            # If an object is detected in front
            if prox_values[2] > 1500:
                return -100, -100
            # If an object is detected on the left
            elif prox_values[0] > 1000:
                return -100, 100
            # If an object is detected on the right
            elif prox_values[4] > 1000:
                return 100, -100
            # If no object is detected, move forward
            else:
                return 100, 100

        # Detect which area the robot is in
        def area_detection(reflected_values, node):
            #Detect black lines
            if reflected_values[0] < 200 or reflected_values[1] < 200:
                return -100, -100

            #Detects grey area
            elif (reflected_values[0] > 200 and reflected_values[0] < 500) or (reflected_values[1] > 200 and reflected_values[1] < 500):
                led_state(node, [0, 32, 0])
                self.in_grey_area = True
                return 0, 0
            else:
                return 100, 100



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

                    # Enable the proximity communication 
                    node.v.prox.comm.enable = 1
                    # Set time interval on 200ms
                    node.v.timer.period[0] = 200
                    # Set initial LED state
                    led_state(node, [0, 0, 32])
                    node.flush() #send the initial state to Thymio
                    
                    # Transmits the message "2" when robot is not in the grey area and it has been
                    # away from the grey area for a minimum of 5 sec.
                    async def on_timer0():
                        if (not self.in_grey_area) and (self.last_message_time is None or time.time() - self.last_message_time >= 5):
                            node.v.prox.comm.tx = 2
                            node.flush()
                            self.last_message_time = None #reset last message time

                    # When Thymio receives a message "1" it is tagged, stops and turns purple
                    async def on_prox_comm():
                        message = node.v.prox.comm.rx
                        print(f"message from Thymio: {message}")
                        if message == 1:
                            led_state(node, [32, 0, 32])    # turns purple
                            run_motor(node, 0, 0)           # stop motor
                            node.flush()
                            self.is_tagged = True      # robot is tagged and should terminate
                        elif (message == 2) and self.in_grey_area:
                            led_state(node, [0, 0, 32])    # turns purple
                            run_motor(node, 100, 100)           # stop motor
                            node.flush()
                            self.last_message_time = time.time()

                    while True:
                        # get the values of the proximity sensors
                        prox_values = node.v.prox.horizontal
                        reflected_values = node.v.prox.ground.reflected
                        
                        await on_timer0()
                        await on_prox_comm()
                        """
                        Get the value of the message received from the other Thymio
                        the value is 0 if no message has been received and 
                        gets set to a new value when a message is received" 
                        """
                        # if the robot is blocked or it has been tagged, the program should terminate
                        if (sum(prox_values) > 20000) or self.is_tagged:
                            break

                        #speeds = behaviorOA(prox_values, reflected_values)
                        
                        speeds = area_detection(reflected_values, node)
                        run_motor(node, speeds[1], speeds[0])

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
    AvoiderController()

