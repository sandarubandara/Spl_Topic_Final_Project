import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go

# Load the full dataset with predictions
decoded_X_full = pd.read_csv('predicted_crop_yields_full_new.csv')

# Load historical data
historical_data = pd.read_csv('final_crop_analysis.csv')

# Ensure 'Year' is numeric for proper sorting
historical_data['Year'] = pd.to_numeric(historical_data['Year'], errors='coerce')
#decoded_X_full['Year'] = pd.to_numeric(decoded_X_full['Year'], errors='coerce')

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Crop Yield Prediction Dashboard", style={'text-align': 'center'}),
    
    html.Div([
        html.Label("Select Country:"),
        dcc.Dropdown(
            id='country-dropdown',
            options=[{'label': country, 'value': country} for country in sorted(decoded_X_full['Country'].unique())],
            value=decoded_X_full['Country'].unique()[0]
        )
    ], style={'margin-bottom': '20px'}),
    
    html.Div([
        html.Label("Select Region:"),
        dcc.Dropdown(
            id='region-dropdown',
            options=[],
            value=None,
            placeholder="Select a region"
        )
    ], style={'margin-bottom': '20px'}),
    
    html.Div([
        html.Label("Select Crop Type (Optional):"),
        dcc.Dropdown(
            id='crop-dropdown',
            options=[],
            value=None,
            placeholder="Select a crop type (optional)"
        )
    ], style={'margin-bottom': '20px'}),
    
    html.Div([
        html.H3("Historical Data and Prediction:"),
        dcc.Graph(id='line-chart')
    ])
])

@app.callback(
    Output('region-dropdown', 'options'),
    [Input('country-dropdown', 'value')]
)
def update_region_dropdown(selected_country):
    filtered_regions = decoded_X_full[decoded_X_full['Country'] == selected_country]['Region'].unique()
    return [{'label': region, 'value': region} for region in sorted(filtered_regions)]

@app.callback(
    Output('crop-dropdown', 'options'),
    [Input('country-dropdown', 'value'),
     Input('region-dropdown', 'value')]
)
def update_crop_dropdown(selected_country, selected_region):
    filtered_crops = decoded_X_full[
        (decoded_X_full['Country'] == selected_country) & 
        (decoded_X_full['Region'] == selected_region)
    ]['Crop_Type'].unique()
    return [{'label': crop, 'value': crop} for crop in sorted(filtered_crops)]

@app.callback(
    Output('line-chart', 'figure'),
    [Input('country-dropdown', 'value'),
     Input('region-dropdown', 'value'),
     Input('crop-dropdown', 'value')]
)

def update_line_chart(selected_country, selected_region, selected_crop):
    historical_filtered = historical_data[
        (historical_data['Country'] == selected_country) & 
        (historical_data['Region'] == selected_region)
    ]
    
    prediction_filtered = decoded_X_full[
        (decoded_X_full['Country'] == selected_country) & 
        (decoded_X_full['Region'] == selected_region)
    ]
    
    fig = go.Figure()
    
    if selected_crop:
        historical_filtered = historical_filtered[historical_filtered['Crop_Type'] == selected_crop]
        prediction_filtered = prediction_filtered[prediction_filtered['Crop_Type'] == selected_crop]
        
        historical_filtered['Year'] = pd.to_numeric(historical_filtered['Year'], errors='coerce')
        historical_filtered = historical_filtered.sort_values(by='Year')
        
        fig.add_trace(go.Scatter(
            x=historical_filtered['Year'],
            y=historical_filtered['Crop_Yield_Mt_Per_Ha'],
            mode='lines+markers',
            name=f'Historical Data ({selected_crop})'
        ))
        
        if not prediction_filtered.empty:
            max_year = prediction_filtered['Year'].max()
            predicted_yield = prediction_filtered[prediction_filtered['Year'] == max_year]['Predicted_Crop_Yield_Mt_Per_Ha'].iloc[0]
                
           # predicted_yield = prediction_filtered.iloc[0]['Predicted_Crop_Yield_Mt_Per_Ha']
            fig.add_trace(go.Scatter(
                x=[2020],
                y=[predicted_yield],
                mode='markers',
                marker=dict(size=10, color='red'),
                name=f'Predicted Value ({selected_crop})'
            ))
    else:
        for crop in historical_filtered['Crop_Type'].unique():
            crop_historical = historical_filtered[historical_filtered['Crop_Type'] == crop]
            crop_prediction = prediction_filtered[prediction_filtered['Crop_Type'] == crop]
            
            crop_historical['Year'] = pd.to_numeric(crop_historical['Year'], errors='coerce')
            crop_historical = crop_historical.sort_values(by='Year')
            
            fig.add_trace(go.Scatter(
                x=crop_historical['Year'],
                y=crop_historical['Crop_Yield_Mt_Per_Ha'],
                mode='lines+markers',
                name=f'Historical Data ({crop})'
            ))
            
            if not crop_prediction.empty:
                predicted_yield = crop_prediction.iloc[0]['Predicted_Crop_Yield_Mt_Per_Ha']
                fig.add_trace(go.Scatter(
                    x=[2020],
                    y=[predicted_yield],
                    mode='markers',
                    marker=dict(size=10, color='red'),
                    name=f'Predicted Value ({crop})'
                ))
    
    fig.update_layout(
        title="Crop Yield Over Time",
        xaxis_title="Year",
        yaxis_title="Crop Yield (MT/HA)",
        legend_title="Legend",
        template="plotly_white"
    )
    
    return fig

import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(historical_data['Crop_Yield_Mt_Per_Ha'], kde=True)
plt.title("Before Transformation")
#plt.show()

historical_data['transformed_yield'] = historical_data['Crop_Yield_Mt_Per_Ha'] ** 2 

from scipy.stats import boxcox

# Make sure there are no 0 or negative values (Box-Cox needs strictly positive data)
data = historical_data['Crop_Yield_Mt_Per_Ha']
data_positive = data - data.min() + 1

""" transformed, fitted_lambda = boxcox(data_positive)
historical_data['transformed_yield'] = transformed
print("Lambda used:", fitted_lambda)

sns.histplot(historical_data['transformed_yield'], kde=True)
plt.title("After Transformation")
plt.show() """
    
if __name__ == '__main__':
    app.run(debug=True) 
