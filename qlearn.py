import numpy as np
import random
from test import get_direction

#drive = Drive.ThymioController()
state_size = 5
action_size = 5
#Initialize q-table values to 0
Q = np.zeros((state_size, action_size))
print(Q)
FORWARD = 0
BACKWRDS = 1
LEFT = 2
RIGHT = 3
SPEED_STRAIGHT = 100
            
async def perform_action(client, action):
    with await client.lock() as node:
        if action == FORWARD:
            node.v.motor.left.target = SPEED_STRAIGHT
            node.v.motor.right.target = SPEED_STRAIGHT
        elif action == RIGHT:
            node.v.motor.left.target = -SPEED_STRAIGHT
            node.v.motor.right.target = SPEED_STRAIGHT
        elif action == LEFT:
            node.v.motor.left.target = SPEED_STRAIGHT
            node.v.motor.right.target = -SPEED_STRAIGHT
        elif action == BACKWRDS:
            node.v.motor.left.target = -SPEED_STRAIGHT
            node.v.motor.right.target = -SPEED_STRAIGHT
        node.flush()

        await client.sleep(.1) # May not be required


epsilon = 1
if random.uniform(0,1) < epsilon:
    dir = random.uniform(0,5) 
    perform_action(FORWARD)
else:
    max_index_flat = np.argmax(Q)
    dir = np.unravel_index(max_index_flat, Q.shape)
    