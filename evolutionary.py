import math
import random
import numpy as np
import torch
import torch.nn as nn

# TODO mutation for weights (maybe)




class NN(nn.Module):
    # Input: direction(red), area, direction(greem), in(grey) (boolean), -1
    def __init__(self, input_size=5):
        super(NN, self).__init__()
        # Output: left_motor_speed, right_motor_speed
        self.fc = nn.Linear(input_size, 2)

    def forward(self, x):
        x = self.fc(x)
        x = torch.relu(x)
        return x


def elitism_selection(robots, n):
    ''' Selects the top n robots based on their fitness '''
    sorted_robots = sorted(robots, key=lambda x: x.fitness, reverse=True)
    top_n = sorted_robots[:n]
    return top_n


def roulette_wheel_selection(robots):
    ''' Selects n robots from the population using roulette wheel selection. '''

    total_fitness = sum(max(robot.fitness, 0) for robot in robots)
    if total_fitness == 0:
        # Avoid division by zero if all fitness values are zero
        selection_probs = [1 / len(robots)] * len(robots)
    else:
        selection_probs = [max(robot.fitness, 0) /
                           total_fitness for robot in robots]

    selected_robots = np.random.choice(
        robots, size=2, p=selection_probs, replace=True)
    l = selected_robots.tolist()
    return l[0], l[1]


def truncation_selection(robots, n):
    ''' Selects n robots from the population using truncation selection
        selects the top n robots based on their fitness
    '''

    sorted_robots = sorted(robots, key=lambda x: x.fitness, reverse=True)
    top_n = sorted_robots[:n]

    return top_n


def crossover(robots, n_crossovers):
    ''' Crossover function to generate offspring from the selected robots '''

    crossover_rate = 0.8
    offspring = []
    random.shuffle(robots)
    for _ in range(n_crossovers):
        robot1, robot2 = roulette_wheel_selection(robots)
        if random.random() < crossover_rate:
            # copy the attributes of the robots
            child1 = DifferentialDriveRobot(random.randint(0, width), random.randint(0, height), random.uniform(
                0, 2 * math.pi), robot1.base_speed, robot1.wall_threshold, robot1.doorway_threshold, robot1.turn_speed, robot1.weights)
            child2 = DifferentialDriveRobot(random.randint(0, width), random.randint(0, height), random.uniform(
                0, 2 * math.pi), robot2.base_speed, robot2.wall_threshold, robot2.doorway_threshold, robot2.turn_speed, robot2.weights)

            crossover_n = random.randint(0, len(attributes))
            crossover_set = set()
            while len(crossover_set) < crossover_n:
                crossover_set.add(random.choice(attributes))

            for attr in crossover_set:
                if attr == 'weights':
                    if random.random() < 0.5:
                        # Crossover weights by breaking in half
                        midpoint = len(robot1.weights) // 2
                        child1.weights[:midpoint] = robot1.weights[:midpoint]
                        child1.weights[midpoint:] = robot2.weights[midpoint:]
                        child2.weights[:midpoint] = robot2.weights[:midpoint]
                        child2.weights[midpoint:] = robot1.weights[midpoint:]
                    else:
                        # Swap weights without breaking in half
                        child1.weights = robot2.weights[:]
                        child2.weights = robot1.weights[:]
                else:
                    setattr(child1, attr, getattr(robot2, attr))
                    setattr(child2, attr, getattr(robot1, attr))

            offspring.append(child1)
            offspring.append(child2)
        else:
            offspring.append(robot1)
            offspring.append(robot2)

    return offspring


def mutation(robot):
    ''' Mutates the robot's attributes with a probability of mutation_rate '''
    mutation_rate = 0.9
    for attr in attributes:
        if random.random() < mutation_rate:
            current_value = getattr(robot, attr)
            if isinstance(current_value, list):
                # Handle mutation for list attributes (e.g., weights)
                mutated_value = [
                    value + random.uniform(-0.1, 0.1) * value for value in current_value]
                setattr(robot, attr, mutated_value)
            else:
                # Handle mutation for scalar attributes
                mutation = random.uniform(-0.1, 0.1) * current_value
                setattr(robot, attr, current_value + mutation)
    return robot


def generate_random_robots(n):
    ''' Generates n robots with random attributes to a list '''
    robots = []
    for _ in range(n):
        x = random.randint(0, width)
        y = random.randint(0, height)
        theta = random.uniform(0, 2 * math.pi)
        # Reduced range for more reasonable speeds
        base_speed = random.randint(MIN_MOTOR_SPEED, MAX_MOTOR_SPEED)
        wall_threshold = random.randint(50, 200)
        doorway_threshold = random.randint(50, 200)
        turn_speed = random.randint(50, 300)
        weights = [random.randint(-5, 5) for _ in range(2*(NUM_BEAMS + 1))]
        robot = DifferentialDriveRobot(
            x, y, theta, base_speed, wall_threshold, doorway_threshold, turn_speed, weights)
        robots.append(robot)

    return robots


def fitness_function(weights, motor_speeds, in_grey_area, reload_grey, red_area, green_area):
    ''' Normalized fitness function for the robot '''
    
    # TODO test if imbalance needs to be included

    left_motor, right_motor = motor_speeds

    # Define maximum values for normalization
    MAX_MOTOR_SPEED_FORWARD = 700
    MAX_MOTOR_SPEED_BACKWORD = -600
    MIN_SPEED = 0
    # Width times height of the image
    MAX_AREA = 480*640
    # Cap fitness to a maximum value to avoid large spikes
    MAX_FITNESS = 1.0


    # Calculate normalized speed (in range [0, 1])
    speed = abs((left_motor + right_motor) / 2)
    normalized_speed = (speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)

    # Calculate normalized imbalance (delta_v)
   # delta_v = abs(left_motor - right_motor) / max(left_motor + right_motor, 1)
   # normalized_imbalance = math.sqrt(delta_v)  # Smoother scaling, already between 0 and 1

    # Normalize area
    normalized_red_area = red_area / MAX_AREA
    normalized_green_area = green_area / MAX_AREA

    if in_grey_area:
        # Minimize motor speeds
        fitness = MAX_FITNESS - normalized_speed
    elif reload_grey:
        # Minimize red area, and maximize speed
        fitness = (1 - normalized_red_area) * normalized_speed
    else:
        if normalized_green_area != 0 and normalized_red_area != 0:
            fitness = (1 - normalized_red_area) * normalized_speed
        else:    
            fitness = (1 - normalized_red_area) * normalized_green_area
  
    # Include imbalance
    #fitness = fitness - normalized_imbalance * 0.1  

    return min(fitness, MAX_FITNESS)


def test_generation(robots, gen_n):
    ''' Makes the whole generation run and calculates the fitness of each robot '''

    last_time = pygame.time.get_ticks()

    for robot in robots:
        total_fitness = 0
        for _ in range(200):
            time_step = (pygame.time.get_ticks() - last_time) / 1000
            last_time = pygame.time.get_ticks()
            robot_pose = robot.predict(time_step)
            lidar_scans, _intersect_points = lidar.generate_scans(
                robot_pose, env.get_environment())

            model = RobotNN()
            weights = torch.tensor(robot.weights, dtype=torch.float32).view(2, 61)  # Reshape to (2, 61)

            with torch.no_grad():
                model.fc.weight = nn.Parameter(weights)

            normalized_lidar_scans = [scan / MAX_LIDAR_BEAM_DISTANCE for scan in lidar_scans]
            x = torch.tensor(normalized_lidar_scans, dtype=torch.float32).unsqueeze(0)

            # Add an additional input node with a value of -1
            additional_input = torch.tensor([[-1.0]], dtype=torch.float32)
            x = torch.cat((x, additional_input), dim=1)

            # Forward pass through the model
            output = model(x)

            # Use the output for further processing (e.g., control the robot)
            left_motor_speed = output[0][0].item() * (MAX_MOTOR_SPEED - MIN_MOTOR_SPEED) + MIN_MOTOR_SPEED
            right_motor_speed = output[0][1].item() * (MAX_MOTOR_SPEED - MIN_MOTOR_SPEED) + MIN_MOTOR_SPEED
            robot.set_motor_speeds(left_motor_speed, right_motor_speed)

            total_fitness += fitness_function(robot, lidar_scans)
        

        robot.fitness = total_fitness

    print()
    best = max(robots, key=lambda robot: robot.fitness)
    best.x = width / 2
    best.y = height / 2
    best.theta = 2.6

    #visualization(best)
    print(f"Average fitness of gen {gen_n}: {int(average_fitness(robots))}")


def average_fitness(robots):
    ''' Returns the average fitness of the population '''

    total_fitness = 0
    for robot in robots:
        total_fitness += robot.fitness

    return total_fitness / len(robots)


def population_converged(robots, threshold):
    ''' Checks if the population has converged by checking the standard deviation of the fitness values '''

    fitness_values = [robot.fitness for robot in robots]
    std_dev = np.std(fitness_values)
    return std_dev < threshold



def main():
    ''' Main function to run the genetic algorithm '''

    n = 10  # number of robots in each generation
    population = generate_random_robots(n)
    gen_n = 0
    test_generation(population, gen_n)

    initial_average_fitness = average_fitness(population)
    convergence_threshold = 2

    MAX_GENERATIONS = 100
    while not population_converged(population, convergence_threshold) or gen_n < MAX_GENERATIONS:

        gen_n += 1
        elite = elitism_selection(population, 2)
        new_gen = crossover(population, 4) + elite
        mutated = [
            mutation(robot) if robot not in elite else robot for robot in new_gen]
        test_generation(mutated, gen_n)
        population = mutated

    # print(max(population, key=lambda robot: robot.fitness).fitness)

    pygame.quit()


if __name__ == "__main__":
    main()