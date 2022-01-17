import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd

# df = pd.DataFrame([
#     dict(Task="Job A", Start='2009-01-01', Finish='2009-02-28'),
#     dict(Task="Job B", Start='2009-03-05', Finish='2009-04-15'),
#     dict(Task="Job C", Start='2009-02-20', Finish='2009-05-30')
# ])
df = pd.DataFrame([
    dict(Task="Job A", Start=100, Finish=200),
    dict(Task="Job B", Start=500, Finish=1000),
    dict(Task="Job C", Start=700, Finish=1000),
])

fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task")
fig.layout.xaxis.type = 'linear'
df['delta'] = df['Finish'] - df['Start']
fig.data[0].x = df.delta.tolist()
# fig = ff.create_gantt(df)
# fig.update_layout(xaxis_type='linear')
fig.write_html('plotly_example_1.html')