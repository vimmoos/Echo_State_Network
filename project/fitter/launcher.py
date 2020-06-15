from  project.esn.tasks import test_0
from celery  import group as g
import celery as cel
from project.fitter.ifitter import ADispatcher,Combiner
from dataclasses import dataclass

@dataclass
class My_dispatcher(ADispatcher):
    algortihm_task: cel.Task =  test_0



'''

leaking_rates = [x/10 for x in range(10)]

r = test_0.delay()

r.get()

batch =  g(test_0.s(leaking_rate= x)  for x in leaking_rates)

r_batch = batch()

r_batch.get()
'''
import plotly.express as px
import pandas as pd

from project.esn.fitter.afilter import DemoFitter

fitter = DemoFitter({"leaking_rate":0,"seed":40},trace=[],limit=20)

dispatcher = My_dispatcher()

combiner = Combiner(fitter = fitter , dispatcher = dispatcher)

~combiner


def trace_to_df(trace):
    return pd.DataFrame(
        trace,
        columns=['x','y'])

df = trace_to_df(fitter.trace)

def plot_trace_df(df):
    fig = px.scatter(df, x='x', y='y')
    return fig

fig = plot_trace_df(df)

fig.show()
