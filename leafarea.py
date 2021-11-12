import pandas as pd #(versio0.24.2)
import dash         #(version 1.0.0)
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State
import dash_bootstrap_components as dbc
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy import stats

import plotly       #(version 4.4.1)
import plotly.express as px
import plotly.io as pio
pio.templates.default = "none"
#PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"
df = pd.read_csv("Raw Tree Data.csv")
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )

app.layout = dbc.Container([
                dbc.Row(
                    [
                        dbc.Col(html.H4("Dashboard by Jatin Mahour", className="ml-2",style={"font-family":"Times New Roman",'color':'#3a3733'})),
                        dbc.Col(
                        dbc.Button(
            "Code",
            href="https://github.com/jatinmahour/Land-use-Classification/blob/main/ag.py",
            #download="my_data.txt",
            external_link=True,
            color="dark",
            style={"font-family":"Times New Roman","position": "fixed", "top": 10, "right": 20, "width": 70}
        )),
                    ],
                    align="center",
                ),
        html.H1("Leaf Area Estimate",style={"font-family":"Times New Roman",'marginTop': 20,'color':'#3a3733','padding':20,'backgroundColor':'#3f92ad'}),
        html.Hr(),
        html.H5("Leaves are the exchange surfaces of plants, absorbing light, taking in carbon dioxide, "
                "and releasing oxygen and water vapour. Leaf area measurement is vital for studying plant responses "
                "to the environment and for optimizing yields in a changing climate. The total leaf area of a tree is "
                "a difficult thing to measure, it could be a destructive method to pull all the leaves off a tree,"
                " measure them individually, then add the areas together, or a non-destructive digital imaging method,"
                " which too is a difficult process. Therefore, I have carried out this statistical analysis to estimate"
                " the leaf area of a tree from its dbh(Diameter at breast height) and tree height. ",style={"font-family":"Times New Roman",'color':'#3a3733'} ,className="ml-2"),
        #html.H5("Sample:",style={"font-family":"Times New Roman",'color':'#3a3733'} ,className="ml-2"),
         html.H5( "Over a period of 14 years, scientists with the U.S. Forest Service Pacific Southwest Research Station"
                " recorded data from a consistent set of measurements on over 14,000 trees, with 171 distinct species in 17 U.S. cities."
                " The online database is available at http://dx.doi.org/10.2737/RDS-2016-0005. I have used a subset of this database"
                " for my study, containing the most common tree species found in the USA. "
                ,style={"font-family":"Times New Roman",'color':'#3a3733'} ,className="ml-2")
        ,
    html.Br(),
        html.Div([
            dbc.Tabs([
                   dbc.Tab(label="Red Maple", tab_id="RM"),
                   dbc.Tab(label="Sweet Gum", tab_id="SG"),
                   dbc.Tab(label="Honey Locust", tab_id="HL"),
                   dbc.Tab(label="Camphor", tab_id="CP"),
                   dbc.Tab(label="Ginkgo", tab_id="GK"),
            ],id="tabs"),

        ]),
html.Div(id="content"),
html.Div(id="content1"),
html.Div(id="content2"),
html.Div(id="content3"),
html.Div(id="content4")

])



#-------------------------------- Code for Red Maple Tree species----------------------------------------------#

df1=df[df['CommonName'].isin(['Red maple'])]
df1 = df1[['DBH (cm)', 'TreeHt (m)', 'Leaf (m2)']]
df1['Leaf (m2)'], _ = stats.boxcox(df1['Leaf (m2)'])
df1['DBH (cm)'], _ = stats.boxcox(df1['DBH (cm)'])
df1['TreeHt (m)'], _ = stats.boxcox(df1['TreeHt (m)'])
y = df1['Leaf (m2)']
X = df1[['DBH (cm)','TreeHt (m)']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('std_scalar', StandardScaler())
])

X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train,y_train)

table_header = [
    html.Thead(html.Tr([html.Th("Model"), html.Th("Linear Regression")]))
]

row1 = html.Tr([html.Td("Intercept"), html.Td("8.407")])
row2 = html.Tr([html.Td("Coefficient(DBH)"), html.Td("2.419856")])
row3 = html.Tr([html.Td("Coefficient(TreeHt)"), html.Td("1.903088")])
row4 = html.Tr([html.Td("MAE"), html.Td("1.394086")])
row5 = html.Tr([html.Td("MSE"), html.Td("3.072183")])
row6 = html.Tr([html.Td("RMSE"), html.Td("1.752764")])
row7 = html.Tr([html.Td("R2 Square"), html.Td("0.861066")])
row8 = html.Tr([html.Td("Cross Validation"), html.Td("0.7669")])

table_body = [html.Tbody([row1, row2, row3, row4,row5,row6,row7,row8])]

@app.callback(Output('content', 'children'),
              Input('tabs', 'active_tab'))


def render_tab_content(active_tab):
        if active_tab == "RM":
            return html.Div([
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        html.H6(
                            "Acer rubrum, the red maple, also known as swamp, water or soft maple, is one of the most common and widespread deciduous trees of eastern and central North America. The U.S. Forest service recognizes it as the most abundant native tree in eastern North America.[4] The red maple ranges from southeastern Manitoba around the Lake of the Woods on the border with Ontario and Minnesota, east to Newfoundland, south to Florida, and southwest to East Texas. Many of its features, especially its leaves, are quite variable in form. At maturity, it often attains a height of around 30 m (100 ft). Its flowers, petioles, twigs and seeds are all red to varying degrees. Among these features, however, it is best known for its brilliant deep scarlet foliage in autumn.",style={"font-family":"Times New Roman",'marginTop': 20,'color':'#3a3733','padding':20} ,className="ml-2")
                        ,dcc.Link('Wikipedia', href='https://en.wikipedia.org/wiki/Acer_rubrum'),
                    ]),
                    dbc.Col(
                        html.Div(
                            html.Img(src=app.get_asset_url('RM.jpg'), height="300px", width="300px",style={'marginTop': 20,'padding':20} ,className="ml-2")
                        )
                    ),
                    dbc.Col([
                html.Div([
                    dbc.Button(
                        "Model Parameters",
                        id="simple-toast-toggle",
                        color="dark",
                        n_clicks=0,
                        className="mb-3", style={ "position": "static","top": 350, "left": 95, "width": 150}
                    ),
                    dbc.Toast(
                        [
                            dbc.Table(table_header + table_body, bordered=True, dark=True, hover=True, responsive=True,
                            striped=True,)
                        ],
                        id="simple-toast",
                        header="Red Maple  ",
                        icon="dark",
                        is_open=False,
                        dismissable=True,
                        style={"position": "static", "top": 200, "right": 55, "width": 300}
                    )
                    ]),
                html.Div([
                html.H5('DBH (cm)'),
                html.Br(),
                dcc.Input(value='20', type='text', id='g1'),
                ]),
                html.Br(),
                html.Div([
                html.H5('Tree Height (m)'),
                html.Br(),
                dcc.Input(value='10', type='text', id='g2'),
                ]),
                html.Br(),
                html.Div([
                    html.H4(id="prediction_result1")
                ])
                ],style={"font-family":"Times New Roman",'marginTop': 20,'color':'#3a3733','padding':20} ,className="ml-2")

                ])
                ])

@app.callback(
    Output("simple-toast", "is_open"),
    [Input("simple-toast-toggle", "n_clicks")],
)
def open_toast(n):
    if n:
        return True
    return False

@app.callback(Output(component_id="prediction_result1",component_property="children"),
[Input("g1","value"), Input("g2","value")])

def update_prediction(g1, g2):
    input_X = np.array([g1,
                        g2]).reshape(1, -1)
    pred = lin_reg.predict(input_X)
    return "Leaf Area (m2): {}".format((pred))

@app.callback(
    Output("positioned-toast", "is_open"),
    [Input("positioned-toast-toggle", "n_clicks")],
)
def open_toast(n):
    if n:
        return True
    return False

#---------------------------SweetGum-----------------------------------------#

df2=df[df['ScientificName'].isin(['Liquidambar styraciflua'])]
df2 = df2[['DBH (cm)', 'TreeHt (m)', 'Leaf (m2)']]
df2['Leaf (m2)'], _ = stats.boxcox(df2['Leaf (m2)'])
df2['DBH (cm)'], _ = stats.boxcox(df2['DBH (cm)'])
df2['TreeHt (m)'], _ = stats.boxcox(df2['TreeHt (m)'])
y1 = df2['Leaf (m2)']
X1 = df2[['DBH (cm)','TreeHt (m)']]

from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=101)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('std_scalar', StandardScaler())
])

X1_train = pipeline.fit_transform(X1_train)
X1_test = pipeline.transform(X1_test)


from sklearn.linear_model import LinearRegression

lin_reg1 = LinearRegression(normalize=True)
lin_reg1.fit(X1_train,y1_train)

table_header1 = [
    html.Thead(html.Tr([html.Th("Model"), html.Th("Linear Regression")]))
]

row1 = html.Tr([html.Td("Intercept"), html.Td("12.2622")])
row2 = html.Tr([html.Td("Coefficient(DBH)"), html.Td("3.689972")])
row3 = html.Tr([html.Td("Coefficient(TreeHt)"), html.Td("1.708784")])
row4 = html.Tr([html.Td("MAE"), html.Td("2.164048")])
row5 = html.Tr([html.Td("MSE"), html.Td("9.018037")])
row6 = html.Tr([html.Td("RMSE"), html.Td("3.003005")])
row7 = html.Tr([html.Td("R2 Square"), html.Td("0.775473	")])
row8 = html.Tr([html.Td("Cross Validation"), html.Td("0.756115")])

table_body1 = [html.Tbody([row1, row2, row3, row4,row5,row6,row7,row8])]

@app.callback(Output('content1', 'children'),
              Input('tabs', 'active_tab'))


def render_tab_content(active_tab):
        if active_tab == "SG":
            return html.Div([
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        html.H6(
                            "American sweetgum (Liquidambar styraciflua), also known as American storax,[2] hazel pine,[3]"
                            " bilsted,[4] redgum,[2] satin-walnut,[2] star-leaved gum,[4] alligatorwood,[2] or simply sweetgum,"
                            "[2][5] is a deciduous tree in the genus Liquidambar native to warm temperate areas of eastern North"
                            " America and tropical montane regions of Mexico and Central America. Sweet gum is one of the main "
                            "valuable forest trees in the southeastern United States, and is a popular ornamental tree in temperate climates."
                            " It is recognizable by the combination of its five-pointed star-shaped leaves (similar to maple leaves)"
                            " and its hard, spiked fruits. It is currently classified in the plant family Altingiaceae, but was formerly"
                            " considered a member of the Hamamelidaceae.[6]",style={"font-family":"Times New Roman",'marginTop': 20,'color':'#3a3733','padding':20} ,className="ml-2")
                        ,dcc.Link('Wikipedia', href='https://en.wikipedia.org/wiki/Liquidambar_styraciflua'),
                    ]),
                    dbc.Col(
                        html.Div(
                            html.Img(src=app.get_asset_url('ss.jpg'), height="300px", width="300px",style={'marginTop': 20,'padding':20} ,className="ml-2")
                        )
                    ),
                    dbc.Col([
                html.Div([
                    dbc.Button(
                        "Model Parameters",
                        id="simple-toast-toggle1",
                        color="dark",
                        n_clicks=0,
                        className="mb-3", style={ "position": "static","top": 350, "left": 95, "width": 150}
                    ),
                    dbc.Toast(
                        [
                            dbc.Table(table_header1 + table_body1, bordered=True, dark=True, hover=True, responsive=True,
                            striped=True,)
                        ],
                        id="simple-toast1",
                        header="Sweet Gum",
                        icon="dark",
                        is_open=False,
                        dismissable=True,
                        style={"position": "static", "top": 200, "right": 55, "width": 300}
                    )
                    ]),
                html.Div([
                html.H5('DBH (cm)'),
                html.Br(),
                dcc.Input(value='20', type='text', id='g11'),
                ]),
                html.Br(),
                html.Div([
                html.H5('Tree Height (m)'),
                html.Br(),
                dcc.Input(value='10', type='text', id='g21'),
                ]),
                html.Br(),
                html.Div([
                    html.H4(id="prediction_result11")
                ])
                ],style={"font-family":"Times New Roman",'marginTop': 20,'color':'#3a3733','padding':20} ,className="ml-2")

                ])
                ])

@app.callback(
    Output("simple-toast1", "is_open"),
    [Input("simple-toast-toggle1", "n_clicks")],
)
def open_toast(n):
    if n:
        return True
    return False

@app.callback(Output(component_id="prediction_result11",component_property="children"),
[Input("g11","value"), Input("g21","value")])

def update_prediction(g11, g21):
    input_X1 = np.array([g11,
                        g21]).reshape(1, -1)
    pred1 = lin_reg1.predict(input_X1)
    return "Leaf Area (m2): {}".format((pred1))

@app.callback(
    Output("positioned-toast1", "is_open"),
    [Input("positioned-toast-toggle1", "n_clicks")],
)
def open_toast(n):
    if n:
        return True
    return False
#----------------------------HoneyLocust-----------------------------------#

df3=df[df['ScientificName'].isin(['Gleditsia triacanthos'])]
df3 = df3[['DBH (cm)', 'TreeHt (m)', 'Leaf (m2)']]
df3['Leaf (m2)'], _ = stats.boxcox(df3['Leaf (m2)'])
df3['DBH (cm)'], _ = stats.boxcox(df3['DBH (cm)'])
df3['TreeHt (m)'], _ = stats.boxcox(df3['TreeHt (m)'])
y2 = df3['Leaf (m2)']
X2 = df3[['DBH (cm)','TreeHt (m)']]

from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=101)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('std_scalar', StandardScaler())
])

X2_train = pipeline.fit_transform(X2_train)
X2_test = pipeline.transform(X2_test)


from sklearn.linear_model import LinearRegression

lin_reg2 = LinearRegression(normalize=True)
lin_reg2.fit(X2_train,y2_train)

table_header2 = [
    html.Thead(html.Tr([html.Th("Model"), html.Th("Linear Regression")]))
]

row1 = html.Tr([html.Td("Intercept"), html.Td("8.6767")])
row2 = html.Tr([html.Td("Coefficient(DBH)"), html.Td("2.369135")])
row3 = html.Tr([html.Td("Coefficient(TreeHt)"), html.Td("2.107601")])
row4 = html.Tr([html.Td("MAE"), html.Td("1.479103")])
row5 = html.Tr([html.Td("MSE"), html.Td("3.503522")])
row6 = html.Tr([html.Td("RMSE"), html.Td("1.87177")])
row7 = html.Tr([html.Td("R2 Square"), html.Td("0.855078	")])
row8 = html.Tr([html.Td("Cross Validation"), html.Td("0.639469")])

table_body2 = [html.Tbody([row1, row2, row3, row4,row5,row6,row7,row8])]

@app.callback(Output('content2', 'children'),
              Input('tabs', 'active_tab'))


def render_tab_content(active_tab):
        if active_tab == "HL":
            return html.Div([
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        html.H6(
                            "The honey locust (Gleditsia triacanthos), also known as the thorny locust or thorny honeylocust, is a deciduous tree in the family Fabaceae, native to central North America where it is mostly found in the moist soil of river valleys.[2] Honey locust is highly adaptable to different environments, has been introduced worldwide, and is an aggressive invasive species.[2]",style={"font-family":"Times New Roman",'marginTop': 20,'color':'#3a3733','padding':20} ,className="ml-2")
                        ,dcc.Link('Wikipedia', href='https://en.wikipedia.org/wiki/Honey_locust'),
                    ]),
                    dbc.Col(
                        html.Div(
                            html.Img(src=app.get_asset_url('hl.jpg'), height="300px", width="300px",style={'marginTop': 20,'padding':20} ,className="ml-2")
                        )
                    ),
                    dbc.Col([
                html.Div([
                    dbc.Button(
                        "Model Parameters",
                        id="simple-toast-toggle1",
                        color="dark",
                        n_clicks=0,
                        className="mb-3", style={ "position": "static","top": 350, "left": 95, "width": 150}
                    ),
                    dbc.Toast(
                        [
                            dbc.Table(table_header2 + table_body2, bordered=True, dark=True, hover=True, responsive=True,
                            striped=True,)
                        ],
                        id="simple-toast1",
                        header="Honey Locust",
                        icon="dark",
                        is_open=False,
                        dismissable=True,
                        style={"position": "static", "top": 200, "right": 55, "width": 300}
                    )
                    ]),
                html.Div([
                html.H5('DBH (cm)'),
                html.Br(),
                dcc.Input(value='20', type='text', id='g12'),
                ]),
                html.Br(),
                html.Div([
                html.H5('Tree Height (m)'),
                html.Br(),
                dcc.Input(value='10', type='text', id='g22'),
                ]),
                html.Br(),
                html.Div([
                    html.H4(id="prediction_result_HL")
                ])
                ],style={"font-family":"Times New Roman",'marginTop': 20,'color':'#3a3733','padding':20} ,className="ml-2")

                ])
                ])

@app.callback(
    Output("simple-toast2", "is_open"),
    [Input("simple-toast-toggle2", "n_clicks")],
)
def open_toast(n):
    if n:
        return True
    return False

@app.callback(Output(component_id="prediction_result_HL",component_property="children"),
[Input("g12","value"), Input("g22","value")])

def update_prediction(g12, g22):
    input_X2 = np.array([g12,
                        g22]).reshape(1, -1)
    pred2 = lin_reg2.predict(input_X2)
    return "Leaf Area (m2): {}".format((pred2))

@app.callback(
    Output("positioned-toast2", "is_open"),
    [Input("positioned-toast-toggle2", "n_clicks")],
)
def open_toast(n):
    if n:
        return True
    return False

#-----------------------------Camphor Tree---------------------------------------------#

df4=df[df['ScientificName'].isin(['Cinnamomum camphora'])]
df4 = df4[['DBH (cm)', 'TreeHt (m)', 'Leaf (m2)']]
df4['Leaf (m2)'], _ = stats.boxcox(df4['Leaf (m2)'])
df4['DBH (cm)'], _ = stats.boxcox(df4['DBH (cm)'])
df4['TreeHt (m)'], _ = stats.boxcox(df4['TreeHt (m)'])
y3 = df4['Leaf (m2)']
X3 = df4[['DBH (cm)','TreeHt (m)']]

from sklearn.model_selection import train_test_split
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.3, random_state=101)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('std_scalar', StandardScaler())
])

X3_train = pipeline.fit_transform(X3_train)
X3_test = pipeline.transform(X3_test)


from sklearn.linear_model import LinearRegression

lin_reg3 = LinearRegression(normalize=True)
lin_reg3.fit(X3_train,y3_train)

table_header3 = [
    html.Thead(html.Tr([html.Th("Model"), html.Th("Linear Regression")]))
]

row1 = html.Tr([html.Td("Intercept"), html.Td("12.1945")])
row2 = html.Tr([html.Td("Coefficient(DBH)"), html.Td("2.584311")])
row3 = html.Tr([html.Td("Coefficient(TreeHt)"), html.Td("2.321831")])
row4 = html.Tr([html.Td("MAE"), html.Td("1.640523")])
row5 = html.Tr([html.Td("MSE"), html.Td("4.713431")])
row6 = html.Tr([html.Td("RMSE"), html.Td("2.171044")])
row7 = html.Tr([html.Td("R2 Square"), html.Td("0.788066	")])
row8 = html.Tr([html.Td("Cross Validation"), html.Td("0.669806")])

table_body3 = [html.Tbody([row1, row2, row3, row4,row5,row6,row7,row8])]

@app.callback(Output('content3', 'children'),
              Input('tabs', 'active_tab'))


def render_tab_content(active_tab):
        if active_tab == "CP":
            return html.Div([
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        html.H6(
                            "Cinnamomum camphora is a species of evergreen tree that is commonly known "
                            "under the names camphor tree, camphorwood or camphor laurel.[1]The leaves have a glossy, waxy appearance and smell of camphor when crushed."
                            " In spring, it produces bright green foliage with masses of small white flowers. It produces clusters of black, "
                            "berry-like fruit around 1 cm (0.39 in) in diameter. Its pale bark is very rough and fissured vertically.",style={"font-family":"Times New Roman",'marginTop': 20,'color':'#3a3733','padding':20} ,className="ml-2")
                        ,dcc.Link('Wikipedia', href='https://en.wikipedia.org/wiki/Cinnamomum_camphora'),
                    ]),
                    dbc.Col(
                        html.Div(
                            html.Img(src=app.get_asset_url('CP.jpg'), height="300px", width="300px",style={'marginTop': 20,'padding':20} ,className="ml-2")
                        )
                    ),
                    dbc.Col([
                html.Div([
                    dbc.Button(
                        "Model Parameters",
                        id="simple-toast-toggle1",
                        color="dark",
                        n_clicks=0,
                        className="mb-3", style={ "position": "static","top": 350, "left": 95, "width": 150}
                    ),
                    dbc.Toast(
                        [
                            dbc.Table(table_header3 + table_body3, bordered=True, dark=True, hover=True, responsive=True,
                            striped=True,)
                        ],
                        id="simple-toast1",
                        header="Camphor",
                        icon="dark",
                        is_open=False,
                        dismissable=True,
                        style={"position": "static", "top": 200, "right": 55, "width": 300}
                    )
                    ]),
                html.Div([
                html.H5('DBH (cm)'),
                html.Br(),
                dcc.Input(value='20', type='text', id='g13'),
                ]),
                html.Br(),
                html.Div([
                html.H5('Tree Height (m)'),
                html.Br(),
                dcc.Input(value='10', type='text', id='g23'),
                ]),
                html.Br(),
                html.Div([
                    html.H4(id="prediction_result_CP")
                ])
                ],style={"font-family":"Times New Roman",'marginTop': 20,'color':'#3a3733','padding':20} ,className="ml-2")

                ])
                ])

@app.callback(
    Output("simple-toast3", "is_open"),
    [Input("simple-toast-toggle3", "n_clicks")],
)
def open_toast(n):
    if n:
        return True
    return False

@app.callback(Output(component_id="prediction_result_CP",component_property="children"),
[Input("g13","value"), Input("g23","value")])

def update_prediction(g13, g23):
    input_X3 = np.array([g13,
                        g23]).reshape(1, -1)
    pred3 = lin_reg3.predict(input_X3)
    return "Leaf Area (m2): {}".format((pred3))

@app.callback(
    Output("positioned-toast3", "is_open"),
    [Input("positioned-toast-toggle3", "n_clicks")],
)
def open_toast(n):
    if n:
        return True
    return False
#-----------------------------------ginkgo--------------------------------------------------#

df5=df[df['ScientificName'].isin(['Ginkgo biloba'])]
df5 = df5[['DBH (cm)', 'TreeHt (m)', 'Leaf (m2)']]
df5['Leaf (m2)'], _ = stats.boxcox(df5['Leaf (m2)'])
df5['DBH (cm)'], _ = stats.boxcox(df5['DBH (cm)'])
df5['TreeHt (m)'], _ = stats.boxcox(df5['TreeHt (m)'])
y4 = df5['Leaf (m2)']
X4 = df5[['DBH (cm)','TreeHt (m)']]

from sklearn.model_selection import train_test_split
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.3, random_state=101)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('std_scalar', StandardScaler())
])

X4_train = pipeline.fit_transform(X4_train)
X4_test = pipeline.transform(X4_test)


from sklearn.linear_model import LinearRegression

lin_reg4 = LinearRegression(normalize=True)
lin_reg4.fit(X4_train,y4_train)

table_header4 = [
    html.Thead(html.Tr([html.Th("Model"), html.Th("Linear Regression")]))
]

row1 = html.Tr([html.Td("Intercept"), html.Td("6.8200")])
row2 = html.Tr([html.Td("Coefficient(DBH)"), html.Td("2.959519")])
row3 = html.Tr([html.Td("Coefficient(TreeHt)"), html.Td("0.825341")])
row4 = html.Tr([html.Td("MAE"), html.Td("1.733883")])
row5 = html.Tr([html.Td("MSE"), html.Td("4.168125")])
row6 = html.Tr([html.Td("RMSE"), html.Td("2.041599")])
row7 = html.Tr([html.Td("R2 Square"), html.Td("0.804941	")])
row8 = html.Tr([html.Td("Cross Validation"), html.Td("0.379647")])

table_body4 = [html.Tbody([row1, row2, row3, row4,row5,row6,row7,row8])]

@app.callback(Output('content4', 'children'),
              Input('tabs', 'active_tab'))


def render_tab_content(active_tab):
        if active_tab == "GK":
            return html.Div([
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        html.H6(
                            "Ginkgo biloba, commonly known as ginkgo or gingko (/ˈɡɪŋkoʊ, ˈɡɪŋkɡoʊ/ GINK-oh, -⁠goh)[4][5]"
                            " also known as the maidenhair tree,[6] is a species of tree native to China. It is the last living"
                            " species in the order Ginkgoales, which first appeared over 290 million years ago. Fossils very similar"
                            " to the living species, belonging to the genus Ginkgo, extend back to the Middle Jurassic "
                            "approximately 170 million years ago.[2] The tree was cultivated early in human history and"
                            " remains commonly planted.Gingko leaf extract is commonly used as a dietary supplement, but "
                            "there is no scientific evidence that it supports human health or is effective against any disease.[7][8]",style={"font-family":"Times New Roman",'marginTop': 20,'color':'#3a3733','padding':20} ,className="ml-2")
                        ,dcc.Link('Wikipedia', href='https://en.wikipedia.org/wiki/Ginkgo_biloba'),
                    ]),
                    dbc.Col(
                        html.Div(
                            html.Img(src=app.get_asset_url('GK.jpg'), height="300px", width="300px",style={'marginTop': 20,'padding':20} ,className="ml-2")
                        )
                    ),
                    dbc.Col([
                html.Div([
                    dbc.Button(
                        "Model Parameters",
                        id="simple-toast-toggle1",
                        color="dark",
                        n_clicks=0,
                        className="mb-3", style={ "position": "static","top": 350, "left": 95, "width": 150}
                    ),
                    dbc.Toast(
                        [
                            dbc.Table(table_header4 + table_body4, bordered=True, dark=True, hover=True, responsive=True,
                            striped=True,)
                        ],
                        id="simple-toast1",
                        header="Ginkgo",
                        icon="dark",
                        is_open=False,
                        dismissable=True,
                        style={"position": "static", "top": 200, "right": 55, "width": 300}
                    )
                    ]),
                html.Div([
                html.H5('DBH (cm)'),
                html.Br(),
                dcc.Input(value='20', type='text', id='g14'),
                ]),
                html.Br(),
                html.Div([
                html.H5('Tree Height (m)'),
                html.Br(),
                dcc.Input(value='10', type='text', id='g24'),
                ]),
                html.Br(),
                html.Div([
                    html.H4(id="prediction_result_GK")
                ])
                ],style={"font-family":"Times New Roman",'marginTop': 20,'color':'#3a3733','padding':20} ,className="ml-2")

                ])
                ])

@app.callback(
    Output("simple-toast4", "is_open"),
    [Input("simple-toast-toggle4", "n_clicks")],
)
def open_toast(n):
    if n:
        return True
    return False

@app.callback(Output(component_id="prediction_result_GK",component_property="children"),
[Input("g14","value"), Input("g24","value")])

def update_prediction(g14, g24):
    input_X4 = np.array([g14,
                        g24]).reshape(1, -1)
    pred4 = lin_reg4.predict(input_X4)
    return "Leaf Area (m2): {}".format((pred4))

@app.callback(
    Output("positioned-toast4", "is_open"),
    [Input("positioned-toast-toggle4", "n_clicks")],
)
def open_toast(n):
    if n:
        return True
    return False

if __name__=='__main__':
    app.run_server(debug=True, port=8000)
