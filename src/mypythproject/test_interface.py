import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from datetime import datetime
from tickers import csv_to_ticker_dict
import pandas as pd
from pybacktestchain.data_module import get_stock_data


# Your modules
from pybacktestchain.broker import Backtest, StopLoss
from pybacktestchain.blockchain import load_blockchain
from optimization_techniques import  MaxSharpe, MinVariance, MaxReturn
from pybacktestchain.data_module import FirstTwoMoments


###############################################################################
# Dictionary mapping Symbol -> Security
###############################################################################
sp_dict = csv_to_ticker_dict()
universe_options = [
    {"label": security, "value": symbol}
    for symbol, security in sp_dict.items()
]
###############################################################################
# Create Dash App
###############################################################################
app = dash.Dash(__name__)

app.layout = html.Div(style={'fontFamily': 'Arial', 'margin': '30px'}, children=[

    html.H1("Backtest", style={'marginBottom': '20px'}),

    # Date Pickers
    html.Div([
        html.Label("Initial Date:"),
        dcc.DatePickerSingle(
            id='initial-date',
            date='2019-01-01',
            display_format='YYYY-MM-DD'
        )
    ], style={'marginBottom': '20px'}),

    html.Div([
        html.Label("Final Date:"),
        dcc.DatePickerSingle(
            id='final-date',
            date='2020-01-01',
            display_format='YYYY-MM-DD'
        )
    ], style={'marginBottom': '20px'}),

    # Information Class
    html.Div([
        html.Label("Information Class:"),
        dcc.Dropdown(
            id='information-class',
            options=[
                {'label': 'FirstTwoMoments (default)', 'value': 'first_two_moments'},
                {'label': 'MaxReturn', 'value': 'max_return'},
                {'label': 'MinVariance', 'value': 'min_variance'},
                {'label': 'MaxSharpe', 'value': 'max_sharpe'},
            ],
            value='first_two_moments',
            clearable=False
        )
    ], style={'width': '300px', 'marginBottom': '20px'}),

    # Universe Dropdown (multi-select)
    html.Div([
        html.Label("Select S&P500 Securities:"),
        dcc.Dropdown(
            id='universe-dropdown',
            options=universe_options,
            value=[],    
            multi=True,
            placeholder='Pick one or more securities'
        )
    ], style={'width': '400px', 'marginBottom': '20px'}),

    # Name Blockchain
    html.Div([
        html.Label("Name Blockchain:"),
        dcc.Input(
            id='name-blockchain',
            type='text',
            value='backtest',
            style={'width': '200px'}
        )
    ], style={'marginBottom': '20px'}),

    # Verbose Checkbox
    html.Div([
        dcc.Checklist(
            id='verbose',
            options=[{'label': 'Enable verbose logging?', 'value': 'verbose'}],
            value=[],
            style={'marginTop': '10px'}
        )
    ]),

    # Run Backtest Button
    html.Button(
        "Run Backtest",
        id='run-button',
        n_clicks=0,
        style={'marginTop': '20px', 'marginBottom': '20px'}
    ),

    # Output Container
    html.Div(id='output-container', style={'whiteSpace': 'pre-wrap'})

])

###############################################################################
# Callback
###############################################################################
@app.callback(
    Output('output-container', 'children'),
    Input('run-button', 'n_clicks'),
    State('initial-date', 'date'),
    State('final-date', 'date'),
    State('information-class', 'value'),
    State('universe-dropdown', 'value'),  
    State('name-blockchain', 'value'),
    State('verbose', 'value')
)
def run_backtest(
    n_clicks,
    init_date_str,
    final_date_str,
    information_class_str,
    selected_symbols,
    name_blockchain,
    verbose_opts
):
    if n_clicks == 0:
        return "Pick some options, select securities, then click 'Run Backtest' above."

    # Convert date strings to datetime
    try:
        init_date = datetime.strptime(init_date_str[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        init_date = datetime(2019, 1, 1)

    try:
        final_date = datetime.strptime(final_date_str[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        final_date = datetime(2020, 1, 1)

    # Map info class
    if information_class_str == 'first_two_moments':
        information_class = FirstTwoMoments
    elif information_class_str == 'max_return':
        information_class = MaxReturn
    elif information_class_str == 'min_variance':
        information_class = MinVariance
    elif information_class_str == 'max_sharpe':
        information_class = MaxSharpe
    else:
        information_class = FirstTwoMoments  # fallback

    risk_model = StopLoss  # default
    verbose = ('verbose' in verbose_opts) if verbose_opts else False

    # Instantiate Backtest
    backtest = Backtest(
        initial_date=init_date,
        final_date=final_date,
        information_class=information_class,
        risk_model=risk_model,
        name_blockchain=name_blockchain,
        verbose=verbose
    )

    # selected_symbols is a list of the "value" items from the dropdown
    # (i.e., the TICKER symbols)
    backtest.universe = selected_symbols or []

    # Run it (dummy)
    backtest.run_backtest()

    # Load transaction log
    transaction_log = backtest.broker.get_transaction_log()

    # Load & display the blockchain (dummy)
    block_chain = load_blockchain(name_blockchain)
    blockchain_str = str(block_chain)
    blockchain_valid = block_chain.is_valid()

    # Format output
    return (
        f"Backtest Parameters:\n"
        f"--------------------\n"
        f"Initial Date: {init_date.strftime('%Y-%m-%d')}\n"
        f"Final Date: {final_date.strftime('%Y-%m-%d')}\n"
        f"Information Class: {information_class_str}\n"
        f"Name Blockchain: {name_blockchain}\n"
        f"Verbose: {verbose}\n\n"
        f"Selected Symbols: {selected_symbols}\n\n"
        f"Blockchain Data:\n"
        f"----------------\n"
        f"{blockchain_str}\n\n"
        f"Is the blockchain valid? {blockchain_valid}"
    )

###############################################################################
# Main
###############################################################################
if __name__ == '__main__':
    app.run_server(debug=True)