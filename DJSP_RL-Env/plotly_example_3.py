import plotly.express as px
import pandas as pd

df = pd.DataFrame([
   dict(Task = "Job A", Start = 1, Finish = 4),
   dict(Task = "Job B", Start = 2, Finish = 6),
   dict(Task = "Job C", Start = 3, Finish = 10)
])
df['delta'] = df['Finish'] - df['Start']

fig = px.timeline(df, x_start = "Start", x_end = "Finish", y = "Task")
fig.update_yaxes(autorange = "reversed")

fig.layout.xaxis.type = 'linear'
fig.data[0].x = df.delta.tolist()
# f = fig.full_figure_for_development(warn = False)
fig.write_html('plotly_example_3.html')