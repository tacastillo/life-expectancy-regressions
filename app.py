import pickle
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

from dash.dependencies import Input, Output

df = pd.read_csv('./data/led.csv')
df = df[df['Year'] == 2015]
df = pd.get_dummies(df, columns=['Status'], drop_first=True)

imports = pickle.load(open('./model.pickle', 'rb'))

model = imports['model']
imputer = imports['imputer']
features = imports['features']
target = imports['target']
mae = imports['mean_absolute_error']

countries = df['Country'].apply(lambda country: {
    "label": country,
    "value": country
})

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

equation = f"Age = {model.intercept_:.3f}"
for coefficient, feature in zip(model.coef_, features):
    equation = equation + f" + {coefficient:.3f}_{feature}_"

app.layout = html.Div([
    dcc.Dropdown(
        id='country-input',
        options=countries,
        value='Albania'
    ),
    html.H5([
        dcc.Markdown(
            children=equation
        )
    ]),
    html.Div(id='model-output'),
    html.H5(children="Mean Absolute Error:"),
    html.P(children=f"{mae:.1f} years")
])

@app.callback(
    Output('model-output', 'children'),
    Input('country-input', 'value')
)
def update_prediction(input_value):
    country = df[df['Country'] == input_value].iloc[0]
    input_data = imputer.transform([country[features]])
    predicted_age = model.predict(input_data)[0]

    equation = f"{model.intercept_:.3f}"
    for coefficient, feature in zip(model.coef_, features):
        equation = equation + f" + {coefficient:.3f}(_{country[feature]:.3f}_)"

    return html.Div([
        html.H5(children="Predicted:"),
        dcc.Markdown(children=f"{predicted_age:.1f} years = {equation}"),
        html.H5(children='Actual:'),
        html.P(children=f"{country[target]} years")
    ])


if __name__ == '__main__':
    app.run_server(debug=True)