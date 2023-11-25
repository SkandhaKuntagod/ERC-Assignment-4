import cv2
import mediapipe as mp
import numpy as np
import time


# Initialize OpenCV Webcam dimensions
VIDEO_X, VIDEO_Y = 640, 480

# Initialize webcam

cap=cv2.VideoCapture(0)

# Initialize hand tracking

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,             # Only 1 hand's index finger taken in consideration
                       model_complexity=1,
                       min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize paddle and puck positions

# Stores paddle positions (random) as a tuple in function 'update_paddle_position()'
paddle = np.random.randint(0, [VIDEO_X - 75, VIDEO_Y - 18], size=2)
# Storing coordinates of the puck as a numpy array [X, Y] (initially at the midpoint)
puck = np.array([VIDEO_X/2, VIDEO_Y/2]).astype(int)


# Initial velocity
initial_puck_velocity = np.array([10, 10])
puck_velocity = initial_puck_velocity.copy()

# Initialize necessary variables
extra_pixel = 0

# Load target image and resize it to 30,30

target_image = cv2.resize(cv2.imread('target.png'), (30, 30))
# Converting the target image from RGB to RGBA
target_image = cv2.cvtColor(target_image, cv2.COLOR_RGB2RGBA)

# Initialize 5 target positions randomly(remember assignment 2!!)
 
target_hit = np.zeros((5,), dtype=np.int16)
target_positions = np.random.randint(0, [VIDEO_X - 30, VIDEO_Y - 30], size=(5, 2))
# Leaving a boundary of 30 pixels at the bottom and right side to accomodate the whole target within the frame

# Initialize score

score = 0

# Initialize timer variables
start_time = time.time()
game_duration = 30  # 1/2 minute in seconds
previous_time = time.time()

# Function to check if the puck is within a 5% acceptance region of a target
def is_within_acceptance(puck, target, acceptance_percent=5):
  #complete the function
  #7
    return (
        0#make changes here!! #8
    )

# Function to update paddle positions
def update_paddle_position(results):
    # Extracting HandLandmark number assigned to index finger tip
    index_finger = mp_hands.HandLandmark.INDEX_FINGER_TIP     # For index finger tip, it is 8
    
    # If there is a hand on the screen
    if results.multi_hand_landmarks:
        # Extracting the details of the hand
        hand = results.multi_hand_landmarks[0]
        
        # Updating paddle's postions
        global paddle
        paddle = tuple((int(hand.landmark[index_finger].x * VIDEO_X), int(hand.landmark[index_finger].y * VIDEO_Y)))

    return 

def update_puck_position(time_difference, extra_pixel):
    # Calculate the distance covered in due time
    distance_covered = puck_velocity.copy() * time_difference + extra_pixel
    #np.array(((current_time - previous_time) * puck_velocity*10))
    # Update puck position
    if int(distance_covered[0]) != 0:
        global puck
        puck += distance_covered.astype(int)

    return distance_covered[0] - int(distance_covered[0])

while True:
    # Calculate remaining time and elapsed time in minutes and seconds   
    # previous_time = current_time
    current_time = time.time()
    elapsed_time = round(current_time - start_time, 2)
    
    # Calculating remaining time
    remaining_time = game_duration - elapsed_time
    
    # Read a frame from the webcam
    success, frame=cap.read()
    # Safety-check
    if not success:
        print('Ignoring empty camera frame')
        break

    # Flip the frame horizontally for a later selfie-view display
    image = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the frame with mediapipe hands
    results = hands.process(image)

    # Update paddle position based on index finger tip
    update_paddle_position(results)

    # Update puck position based on its velocity
    #extra_pixel = update_puck_position(current_time - previous_time, extra_pixel)

    # Check for collisions with the walls
    # collision_walls()

    # Check for collisions with the paddle
    # collision_paddle()

    # Check for collisions with the targets(use is_within_acceptance)    
    #18
            # Increase the player's score
            # Remove the hit target
            # Increase puck velocity after each hit by 2(you will have to increase both x and y velocities

    # Draw paddle, puck, and targets on the frame and overlay target image on frame
    
    # Overlaying the target images on the frame (one by one)
    for index, target_position in enumerate(target_positions):
        
        # Skipping the target if it is already hit
        if target_hit[index] == 1:
            continue
        
        # Extracting the 30x30 Region of Interest from the frame
        target_roi = image[target_position[1]: (target_position[1] + target_image.shape[0]), #image[Y, X]
                          target_position[0]: (target_position[0] + target_image.shape[1])]
        
        # Opacity of the target_image scaled down to 0 to 1
        alpha = target_image[:, :, 3] / 255.0
        beta = 1.0 - alpha
        
        # Updating the RGB array for each pixel (overlaying the target image onto frame)
        for colour_code in range(0, 3):
            target_roi[:, :, colour_code] = (alpha * target_image[:, :, colour_code] +
                                  beta * target_roi[:, :, colour_code])
            
    # Overlaying puck on the frame
    # Extract a square around the centre of the puck of side 25 pixels
    puck_roi = image[(puck[1] - 12):(puck[1] + 13),
                     (puck[0] - 12):(puck[0] + 13)]
    
    # Iterating over each pixel
        # Each row (from top to bottom)
    for y in range(0, 25):
        # Each column (from L to R)
        for x in range(0, 25):
            ### Checking if the pixel lies in the circle of radius of the puck's radius
            # Colouring inner circle Red (radius = 8)
            if ((y-12)**2 + (x-12)**2) <= 64:
                puck_roi[y, x] = np.array([255, 0, 0])

            # Colouring outer circle Orange (radius = 12)
            elif ((y-12)**2 + (x-12)**2) <= 144:
                puck_roi[y, x] = np.array([255, 165, 0])

    # Overlaying paddle on the frame
    # Extract a rectangle around the centre of the paddle (75x18)
    paddle_roi = image[(paddle[1] - 9):(paddle[1] + 9),
                       (paddle[0] - 37):(paddle[0] + 38)]
    
    paddle_roi[:, :] = np.array([0, 255, 0])

    # Display the player's score on the frame (Top Right)
    # X: 640x0.9 = 576
    # Y: 480x0.1 = 48
    cv2.putText(image, f'Score: {score}', (576, 48), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Display the remaining time on the frame (Top Mid)
    cv2.putText(image, f'{round(remaining_time, 2)} secs', (320, 48), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # cv2.putText(image, text, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Check if all targets are hit or time is up
    if all in target_hit == 1:
        print('GG!')
        # Print stuff on screen!
        break
    
    if elapsed_time >= 10:
        # Do something
        image[:, :] = [0, 0, 0]
        cv2.putText(image, 'YOU LOSE!\n Your Score', (320, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3, cv2.LINE_AA)
        cv2.imshow('Virtual Air Hockey', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        time.sleep(5)
        print('You lost!')
        break

    # Display the resulting frame
    cv2.imshow('Virtual Air Hockey', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Exit the game when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
