

# TODO correct color scheme
# TODO correct the reflection vales
# TODO Grey area has to ignore a 1 message

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

        def run(attributes, node):
            
            MAX_AREA = 480*640
            
            for i,weight in enumerate(attributes):
                prox_values = node.v.prox.horizontal
                
                if (sum(prox_values) > 20000): #or self.is_tagged:
                    camera.release()
                    cv2.destroyAllWindows()
                    break
                
                weight = weight[0]
                total_fitness = 0
                
                for _ in range(20):
                    
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

                        total_fitness += fitness_function_seeker(self.speeds, blue_area, green_area) if detected_speeds is None else 0
                        
                tmp = (self.all_weights[i][0], total_fitness)
                self.all_weights[i] = tmp

        def generate_weights():
            
            all_weights = []
            for _ in range(10):
                weights = [random.randint(0, 5) for _ in range(2*(5))]
                #weights = [random.uniform(0, 0.75) for _ in range(2*5)]
                all_weights.append((weights, 0))

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

                    # Convert to lists to allow modifications
                    child1 = list(child1)
                    child1[0] = list(child1[0])
                    child2 = list(child2)
                    child2[0] = list(child2[0])

                    if random.random() < 0.5:
                        # Perform crossover at midpoint
                        midpoint = len(robot1[0]) // 2  
                        child1[0][:midpoint] = robot1[0][:midpoint]
                        child1[0][midpoint:] = robot2[0][midpoint:]
                        child2[0][:midpoint] = robot2[0][:midpoint]
                        child2[0][midpoint:] = robot1[0][midpoint:]
                    else:
                        # Perform crossover by copying weights directly
                        child1[0] = robot2[0][:]
                        child2[0] = robot1[0][:]

                    # Convert back to tuples 
                    child1[0] = tuple(child1[0])
                    child1 = tuple(child1)
                    child2[0] = tuple(child2[0])
                    child2 = tuple(child2)

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
                    
                    # Set time interval on 200ms
                    node.v.timer.period[0] = 10000
                    # Set initial LED state
                    led_state(node, [32, 0, 0])
                    node.flush() #send the initial state to Thymio

                    self.all_weights = generate_weights()
                    run(self.all_weights, node)


                    gen_n = 0
                    MAX_GENERATIONS = 100
                    
                    initial_average_fitness = average_fitness(self.all_weights)
                    convergence_threshold = 5

                    while not population_converged(self.all_weights, convergence_threshold) or gen_n < MAX_GENERATIONS:

                        gen_n += 1
                        elite = elitism_selection(self.all_weights, 2)
                        new_gen = crossover(self.all_weights, 4) + elite
                        mutated = [mutation(weight) if weight not in elite else weight for weight in new_gen]
                        
                        self.all_weights = mutated


                        run(self.all_weights, node)
                        
                        
                        message = node.v.prox.comm.rx

                        """
                        Get the value of the message received from the other Thymio
                        the value is 0 if no message has been received and 
                        gets set to a new value when a message is received" 
                        """
                        # if the robot is blocked or it has been tagged, the program should terminate
                        prox_values = node.v.prox.horizontal
                        if (sum(prox_values) > 20000): #or self.is_tagged:
                            camera.release()
                            cv2.destroyAllWindows()
                            with open("final_weights", "w") as file:
                                for weight, fitness in self.all_weights:
                                    file.write(f"Weights: {weight} Fitness: {fitness}\n")
                            break

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
