import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(n_games, avg_turns):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('avg number of turns')
    plt.plot()
    plt.plot(n_games, avg_turns)
    ax = plt.gca()
    plt.show(block=False)
    plt.pause(.1)