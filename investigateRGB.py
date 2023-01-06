from Env import Env
from RandomAgent import RandomAgent
import matplotlib.pyplot as plt
import random


def getUniqueColourPixels(observation_action):
    rgb_colors = {}
    pixels = observation_action[0]
    for row in pixels:
        for pixel in row:
            # Add color to dictionary. Cannot use array as key so convert to tuple.
            rgb_colors[pixel[0], pixel[1], pixel[2]] = pixel
    return rgb_colors.values()


def getEpisodeColourSpace(episodeParameter):
    unique_rgb_colors = []
    for observation_action in episodeParameter:
        # Get unique rgb pixels from state observation.
        rgb_colors = getUniqueColourPixels(observation_action)
        for color in rgb_colors:
            color_tuple = color[0], color[1], color[2]
            if color_tuple not in unique_rgb_colors:
                unique_rgb_colors.append(color_tuple)
    return unique_rgb_colors


def plotObservationImage(state_action):
    plt.imshow(state_action[0], aspect='auto')
    plt.draw()


def plotRgb3dColorSpace(rgb_colors, title, ax):
    x = []
    y = []
    z = []
    cs = []
    for color in rgb_colors:
        x.append(color[0])
        y.append(color[1])
        z.append(color[2])
        c = (color[0] / 255, color[1] / 255, color[2] / 255)
        cs.append(c)
        ax.text(color[0], color[1], color[2], color, color=c)
    ax.scatter(x, y, z, c=cs)
    ax.set_title(title)
    ax.set_xlabel('R axis')
    ax.set_ylabel('G axis')
    ax.set_zlabel('B axis')


def investigateRgbObservations(episodeParameter):
    # Get first and last state_action pair.
    first_state_action = episodeParameter[0]
    last_state_action = episodeParameter[len(episodeParameter) - 1]

    print()
    print("RGB Observation dimensions: ", first_state_action[0].shape)
    print("RGB Observation type: ", type(first_state_action[0]))
    print()
    # Get all unique rgb colours seen in episode observations.
    episode_rgb_colors = getEpisodeColourSpace(episodeParameter)
    print("Numbers of unique colours in episode observations: ", len(episode_rgb_colors))
    print()

    # Initialise figure with 2 sub-plots and show image of first and last observations.
    fig1, axes = plt.subplots(1, 2)
    axes[0].imshow(first_state_action[0], aspect='auto')
    axes[0].set_title("First Observation")
    axes[1].imshow(last_state_action[0], aspect='auto')
    axes[1].set_title("Last Observation")

    fig2 = plt.figure()
    # Set up the axes for the first 3d colour space plot.
    ax = fig2.add_subplot(1, 2, 1, projection='3d')
    # Get unique rgb colours seen in first observation and plot 3d colour space.
    plotRgb3dColorSpace(getUniqueColourPixels(first_state_action), "First Observation Colour Space", ax)
    # Get unique rgb colours seen in last observation and plot 3d colour space.
    ax = fig2.add_subplot(1, 2, 2, projection='3d')
    # Set up the axes for the last 3d colour space plot.
    plotRgb3dColorSpace(getUniqueColourPixels(last_state_action), "Last Observation Colour Space", ax)

    # Plot 3d colour space of unique rgb colors seen in the whole episode.  
    fig3 = plt.figure()
    ax = plt.axes(projection='3d')
    plotRgb3dColorSpace(episode_rgb_colors, "Episode Observation Colour Space", ax)

    # Show all figures made. 
    plt.show()


def printNumberOfObjectsDetected(objects):
    print()
    for obs in ['player', 'pole', 'tree']:
        print(obs, " objects detected: ", len(objects.get(obs, None)))
    print()


def plot2Images(observation, ob_detect):
    # Initialise figure with 2 sub-plots and show image of first and last observations.
    fig1, axes = plt.subplots(1, 2)
    axes[0].imshow(observation, aspect='auto')
    axes[0].set_title("Observation Image")
    axes[1].imshow(ob_detect, aspect='auto')
    axes[1].set_title("Object Detection")


def pixelDictTo2dRgbArray(pixel_dict):
    object2dImage = [[[255, 255, 255] for j in range(160)] for i in range(250)]
    for object_type in pixel_dict:
        obs = pixel_dict.get(object_type)
        for ob_pixels in obs:
            for (row, col) in ob_pixels:
                pixel = ob_pixels.get((row, col))
                object2dImage[row][col] = [pixel[0], pixel[1], pixel[2]]
    return object2dImage


def testObjectDetection(episode):
    for state in episode:
        objects = skiing.identifyObjects(state[0])
        printNumberOfObjectsDetected(objects)
        plot2Images(state[0], pixelDictTo2dRgbArray(objects))
        plt.show()


# Initialise skiing environment.
skiing = Env('Skiing-v4')
# Initialise agent using environment.
agent = RandomAgent(skiing)
# Generate episode using agent.
episode, reward = agent.generateEpisode()
# investigateRgbObservations(episode)
testObjectDetection(episode)
