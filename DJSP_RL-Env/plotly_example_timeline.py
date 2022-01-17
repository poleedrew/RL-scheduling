import plotly.express as px
import pandas as pd
import numpy as np

df = pd.DataFrame([
    dict(Task="Job A", Start='2009-01-01', Finish='2009-02-28', Completion_pct=50),
    dict(Task="Job B", Start='2009-03-05', Finish='2009-04-15', Completion_pct=25),
    dict(Task="Job C", Start='2009-02-20', Finish='2009-05-30', Completion_pct=75)
])
# df = pd.DataFrame([
#     dict(Task="Job A", Start=100, Finish=200, Completion_pct=50),
#     dict(Task="Job B", Start=500, Finish=1000, Completion_pct=25),
#     dict(Task="Job C", Start=700, Finish=1000, Completion_pct=75)
# ])

fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Completion_pct")
fig.update_yaxes(autorange="reversed")
# fig.update_xaxes(tickmode='linear', tickvals=np.linspace(0, 1000, num=10))
# fig.update_xaxes(tickformat='%d', tickformatstops=[dict(dtickrange=[0, 1000])])
# fig.update_xaxes(tickformat='%d', tickvals=np.linspace(0, 1000, num=11))
# fig.update_layout(xaxis=dict(title='time', tickmode='linear', tickvals=np.linspace(0, 1000, num=11), tickformat='%d'))
fig.update_layout(xaxis=dict(title='time', tickformat='%d'))
fig.write_html('plotly_example_timeline.html')