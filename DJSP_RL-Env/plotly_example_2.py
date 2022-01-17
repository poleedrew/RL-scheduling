import plotly.express as px
df = px.data.tips()
print(df)
fig = px.bar(df, x="total_bill", y="day", orientation='h')
fig.show()