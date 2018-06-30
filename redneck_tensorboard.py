import time
from sys import argv
import visdom
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('-r','--run_id', help='Run Id of your makeshift log.', required=True)
args = parser.parse_args()
vis = visdom.Visdom()
def redneck_tensorboard(run_id, windows, xlim=None):
    try:
        win = [0,0,0]
        with open(str(run_id)+'.log') as f:
            contents = f.readlines()
        new = [contents[i:i+5] for i  in range(0, len(contents), 5)][:-1]
        val_loss = [float(i[3].split(' ')[2]) for i in new]
        train_loss = [float(i[2].split(' ')[2]) for i in new]
        val_acc = [float(i[3].split(' ')[4]) for i in new]
        train_acc = [float(i[2].split(' ')[4]) for i in new]
        title1 = ('Training Loss: {}'.format(train_loss[-1]))
        title2 = ('Validation Loss: {}'.format(val_loss[-1]))
        title3 = ('Training Accuracy: {}'.format(train_acc[-1]))
        title4 = ('Validation Accuracy: {}'.format(val_acc[-1]))
        if xlim == None:
            pass
        elif type(xlim) != type(5):
            raise TypeError("expected xlim to be of type integer or None, got '{}'".format(xlim))
        else:
            train_loss = train_loss[:xlim]
            val_loss = val_loss[:xlim]
        win[0] = vis.text('<h1>'+new[-1][0]+'</h1>', win=windows[0])
        win[1] = vis.line(Y=np.array(train_loss), X=np.array(range(len(train_loss))), name=title1, win=windows[1], update='replace', opts=dict(fillarea=False,showlegend=False,width=800,height=800,xlabel='Epochs',ylabel='Loss',title=title2,marginleft=30, marginright=30,marginbottom=80,margintop=30,))
        win[2] = vis.line(Y=np.array(val_loss), X=np.array(range(len(val_loss))), name=title2, win=windows[2], update='replace', opts=dict(fillarea=False,showlegend=False,width=800,height=800,xlabel='Epochs',ylabel='Loss',title=title1,marginleft=30, marginright=30,marginbottom=80,margintop=30,))
        win[3] = vis.line(Y=np.array(train_acc), X=np.array(range(len(train_acc))), name=title3, win=windows[3], update='replace', opts=dict(fillarea=False,showlegend=False,width=800,height=800,xlabel='Epochs',ylabel='Accuracy',title=title3,marginleft=30, marginright=30,marginbottom=80,margintop=30,))
        win[4] = vis.line(Y=np.array(val_acc), X=np.array(range(len(val_acc))), name=title4, win=windows[4], update='replace', opts=dict(fillarea=False,showlegend=False,width=800,height=800,xlabel='Epochs',ylabel='Accuracy',title=title4,marginleft=30, marginright=30,marginbottom=80,margintop=30,))
    except KeyboardInterrupt:
        quit()
    return win
try:
    r_id = args.run_id
    fig = [vis.text('Initializing Training'),vis.line(np.array([0])),vis.line(np.array([0])),vis.line(np.array([0])),vis.line(np.array([0]))]
    while True:
        time.sleep(3)
        fig = redneck_tensorboard(r_id, fig)
except KeyboardInterrupt:
    quit()
