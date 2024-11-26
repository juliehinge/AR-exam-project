

# TODO correct color scheme

from tdmclient import ClientAsync
import time
from image_detection import *
from evolutionary import *
import numpy as np 

from copy import deepcopy
import random


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
        self.speeds = None, None
        self.reload_grey = False
        self.cur_gen = None

        self.all_weights = []

        MAX_MOTOR_SPEED_FORWARD = 700
        MAX_MOTOR_SPEED_BACKWORD = -600


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
                led_state(node, [0, 0, 32]) # Turn blue
                return -100, -100

            #Detects grey area
            elif (reflected_values[0] > 200 and reflected_values[0] < 500) or (reflected_values[1] > 200 and reflected_values[1] < 500):
                led_state(node, [0, 32, 0]) # Turn green
                self.in_grey_area = True
                return 0, 0
            else:
                led_state(node, [0, 0, 32]) # Turn blue
                self.in_grey_area = False
                # 700 is the max speed
                # 600 is the max speed for going backwords
                return 100, 100

        def run(attributes):
            weights = attributes[0]

            for weight in weights:
                total_fitness = 0
                for _ in range(200):
                    hsv, image = take_picture()
                    if image is not None:    
                        red_area, red_direction = get_image(hsv, image, np.array([0, 120, 70]), np.array([10, 255, 255]))
                        green_area, green_direction = get_image(hsv, image, np.array([0, 100, 0]), np.array([50, 255, 50]))
                        model = NN()
                        input_weights = torch.tensor(weight, dtype=torch.float32).view(2, 7)  # Reshape to (2, 6)

                        with torch.no_grad():
                            model.fc.weight = nn.Parameter(input_weights)
                        
                        input_nodes = [red_direction, red_area, green_direction, green_area, self.in_grey_area, self.reload_grey, -1]
                        print(input_nodes)
                        x = torch.tensor(input_nodes, dtype=torch.float32).unsqueeze(0)
                        
                        # Forward pass through the model
                        output = model(x)

                        # Use the output for further processing (e.g., control the robot)
                        left_motor_speed = output[0][0].item() * (MAX_MOTOR_SPEED_FORWARD - MAX_MOTOR_SPEED_BACKWORD) + MAX_MOTOR_SPEED_BACKWORD
                        right_motor_speed = output[0][1].item() * (MAX_MOTOR_SPEED_FORWARD - MAX_MOTOR_SPEED_BACKWORD) + MAX_MOTOR_SPEED_BACKWORD
                        self.speeds(left_motor_speed, right_motor_speed)

                        total_fitness += fitness_function(self.speeds, self.in_grey_area, self.reload_grey, red_area, green_area)

                self.all_weights[weight][1] = total_fitness
                

        def generate_weights():
            
            all_weights = []
            for _ in range(10):
                weights = [random.randint(-5, 5) for _ in range(2*(7))]
                all_weights.append((weights, None))

            return all_weights



        def elitism_selection(all_weights, n):
            ''' Selects the top n robots based on their fitness (second value in tuple) '''

            # Sort by the second value of each tuple (fitness value), which is x[1]
            sorted_robots = sorted(all_weights, key=lambda x: x[1], reverse=True)
            
            # Select the top n robots
            top_n = sorted_robots[:n]
            
            return top_n



        def roulette_wheel_selection(all_weights):
            ''' Selects 2 robots from the population using roulette wheel selection. '''

            # Calculate total fitness (ignoring negative values by taking max with 0)
            total_fitness = sum(max(x[1], 0) for x in all_weights)
            
            if total_fitness == 0:
                # Avoid division by zero if all fitness values are zero
                selection_probs = [1 / len(all_weights)] * len(all_weights)
            else:
                # Calculate selection probabilities based on normalized fitness values
                selection_probs = [max(x[1], 0) / total_fitness for x in all_weights]
            
            # Use np.random.choice to select two robots with replacement
            selected_indices = np.random.choice(len(all_weights), size=2, p=selection_probs, replace=True)
            selected_robots = [all_weights[i] for i in selected_indices]
            
            # Return the two selected robots
            return selected_robots[0], selected_robots[1]



        def crossover(all_weights, n_crossovers):
            ''' Crossover function to generate offspring from the selected robots '''
            
            crossover_rate = 0.8
            offspring = []
            
            # Work with a shuffled copy to avoid modifying the original list
            shuffled_weights = all_weights[:]
            random.shuffle(shuffled_weights)
            
            for _ in range(n_crossovers):
                # Select two robots using roulette wheel selection
                robot1, robot2 = roulette_wheel_selection(shuffled_weights)
                
                if random.random() < crossover_rate:
                    # Create deep copies of robots to avoid modifying originals
                    child1 = deepcopy(robot1)
                    child2 = deepcopy(robot2)
                    
                    if random.random() < 0.5:
                        # Perform crossover
                        midpoint = len(robot1[0]) // 2  # Assume weights are in robot1[0]
                        child1[0][:midpoint] = robot1[0][:midpoint]
                        child1[0][midpoint:] = robot2[0][midpoint:]
                        child2[0][:midpoint] = robot2[0][:midpoint]
                        child2[0][midpoint:] = robot1[0][midpoint:]
                    else:
                            
                        # Perform crossover by copying weights directly
                        child1[0] = robot2[0][:]  # Copy weights from robot2 to child1
                        child2[0] = robot1[0][:]  # Copy weights from robot1 to child2
                        
                    # Append the offspring
                    offspring.append(child1)
                    offspring.append(child2)
                else:
                    # No crossover, append deep copies of the parents
                    offspring.append(deepcopy(robot1))
                    offspring.append(deepcopy(robot2))
            
            return offspring




        def mutation(attribute):
            ''' Mutates the attributes with a probability of mutation_rate '''
            mutation_rate = 0.9
            weights = attribute[0]

            mutated_weights = []
            for weight in weights:
                if random.random() < mutation_rate:
                    weight = weight + random.uniform(-0.1,0.1) * weight
                    mutated_weights.append(weight)
                else:
                    mutated_weights.append(weight)
                
            return (mutated_weights, attribute[1])



        def average_fitness(all_weights):
            ''' Returns the average fitness of the population '''

            total_fitness = 0
            for weights in all_weights:
                total_fitness += weights[1]

            return total_fitness / len(all_weights)



        def population_converged(all_weights, threshold):
            ''' Checks if the population has converged by checking the standard deviation of the fitness values '''

            fitness_values = [fitness[1] for fitness in all_weights]
            std_dev = np.std(fitness_values)
            return std_dev < threshold


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


                    # Enable the proximity communication 
                   # node.v.prox.comm.enable = 1
                   # node.v.prox.comm.tx = 1

                    
                    # Set time interval on 200ms
                    node.v.timer.period[0] = 200
                    # Set initial LED state
                    led_state(node, [0, 0, 32])
                    node.flush() #send the initial state to Thymio
                    
                    # Transmits the message "2" when robot is not in the grey area and it has been
                    # away from the grey area for a minimum of 5 sec.
                    async def on_timer0():
                        if (not self.in_grey_area) and (self.last_message_time is None or time.time() - self.last_message_time >= 5):
                            self.reload_grey = False
                            node.v.prox.comm.tx = 2
                            node.flush()
                            self.last_message_time = None #reset last message time

                    # When Thymio receives a message "1" it is tagged, stops and turns purple
                    async def on_prox_comm():
                        message = node.v.prox.comm.rx
                       # print(f"message from Thymio: {message}")
                        if message == 1:
                            led_state(node, [32, 0, 32])    # turns purple
                            run_motor(node, 0, 0)           # stop motor
                            node.flush()
                            self.is_tagged = True      # robot is tagged and should terminate
                        elif (message == 2) and self.in_grey_area:
                            self.reload_grey = True
                            led_state(node, [0, 0, 32])    # turns purple
                            run_motor(node, 100, 100)           # stop motor
                            node.flush()
                            self.last_message_time = time.time()


                    self.all_weights = generate_weights()
                    run(self.all_weights)
                    gen_n = 0
                    MAX_GENERATIONS = 100
                    

                    initial_average_fitness = average_fitness(self.all_weights)
                    convergence_threshold = 1

                    while not population_converged(self.all_weights, convergence_threshold) or gen_n < MAX_GENERATIONS:

                        gen_n += 1
                        elite = elitism_selection(self.all_weights, 2)
                        new_gen = crossover(self.all_weights, 4) + elite
                        mutated = [mutation(weight) if weight not in elite else weight for weight in new_gen]
                        
                        self.all_weights = mutated

                        run(self.all_weights)



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

