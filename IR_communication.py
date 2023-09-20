

# TODO correct color scheme

from tdmclient import ClientAsync

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


        def avoid_black(reflected_values, node):

            if reflected_values[0] < 200 or reflected_values[1] < 200:
                return -100, -100

            elif (reflected_values[0] > 200 and reflected_values[0] < 500) or (reflected_values[1] > 200 and reflected_values[1] < 500):
                node.v.leds.top = [0, 0, 32]
                node.v.leds.bottom.left = [0, 0, 32]
                node.v.leds.bottom.right = [0, 0, 32]

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

                    while True:
                        # get the values of the proximity sensors
                        prox_values = node.v.prox.horizontal
                        reflected_values = node.v.prox.ground.reflected
                        

                        """
                        Get the value of the message received from the other Thymio
                        the value is 0 if no message has been received and 
                        gets set to a new value when a message is received" 
                        """

                        message = node.v.prox.comm.rx
                        print(f"message from Thymio: {message}")

                        if sum(prox_values) > 20000:
                            break

                       # speeds = behaviorOA(prox_values, reflected_values)
                        
                        speeds = avoid_black(reflected_values, node)

                        node.v.motor.left.target = speeds[1]
                        node.v.motor.right.target = speeds[0]

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

