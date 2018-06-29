import matplotlib.pyplot as plt
import time
from sys import argv
def redneck_tensorboard(run_id, xlim=None, fig=None):
    try:
        if fig == None:
            fi=plt.figure()
            ax = fi.add_subplot(311)
            ax.set_title('Training Loss')
            bx = fi.add_subplot(313)
            bx.set_title('Validation Loss')
            cx = fi.add_subplot(311)
            dx = fi.add_subplot(313)
            fig = (fi, ax, bx, cx, dx)
        with open(str(run_id)+'.log') as f:
            contents = f.readlines()
        new = [contents[i:i+5] for i  in range(0, len(contents), 5)][:-1]
        val_loss = [float(i[3].split(' ')[2]) for i in new]
        train_loss = [float(i[2].split(' ')[2]) for i in new]
        fig[1].set_title('Training Loss: {}'.format(train_loss[-1]))
        fig[2].set_title('Validation Loss: {}'.format(val_loss[-1]))
        fig[3].axhline(y=13)
        fig[4].axhline(y=20)
        if xlim == None:
            fig[1].plot(train_loss)
            fig[2].plot(val_loss)
        elif type(xlim) != type(5):
            raise TypeError("expected xlim to be of type integer or None, got '{}'".format(xlim))
        else:
            fig[1].plot(train_loss[:xlim])
            fig[2].plot(val_loss[:xlim])
        plt.pause(0.05)
        plt.draw()
    except KeyboardInterrupt:
        quit()
    return fig
try:
    fig = redneck_tensorboard(argv[1])
    while True:
        print('going to display...')
        time.sleep(3)
        fig = redneck_tensorboard(argv[1], fig=fig)
        print('displayed.')
except KeyboardInterrupt:
    quit()
