from pylsl import StreamInlet, resolve_stream, resolve_streams
import numpy as np


def prepare_data(time):
    a = resolve_streams()
    inlet = StreamInlet(a[0])
    print(a[0].name())
    data=[]
    while len(data)<=time:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        sample, timestamp = inlet.pull_sample()
        data.append(sample)
    return data