import matplotlib.pyplot as plt
import matplotlib.animation as animation
from queue import Queue
from threading import Thread
import time

class LiveLossPlot:
    def __init__(self):
        self.losses = []
        self.queue = Queue()
        self.running = True

        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training Loss")
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=10, blit=False)

        self.thread = Thread(target=self.process_queue)
        self.thread.start()

    def update_plot(self, frame):
        self.line.set_data(range(len(self.losses)), self.losses)
        self.ax.relim()
        self.ax.autoscale_view()
        return self.line,

    def process_queue(self):
        while self.running:
            while not self.queue.empty():
                loss = self.queue.get()
                self.losses.append(loss)
            time.sleep(0.1)

    def add_loss(self, loss):
        self.queue.put(loss)

    def show(self):
        plt.show()
        self.running = False
        self.thread.join()

# Example usage:
if __name__ == '__main__':
    import random
    
    plotter = LiveLossPlot()

    def mock_training():
        for i in range(200):
            loss = 2.5 / (i + 1) + random.random() * 0.1
            plotter.add_loss(loss)
            time.sleep(0.1)

    training_thread = Thread(target=mock_training)
    training_thread.start()

    plotter.show()
    training_thread.join()
