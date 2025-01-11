import pandas as pd

def csv_to_ticker_dict(csv_file_path="src/mypythproject/sp500_companies.csv", print_dict=False):
    df = pd.read_csv(csv_file_path, delimiter=';')
    ticker_dict = dict(zip(df['Symbol'], df['Security']))

    if print_dict:
        print(ticker_dict)  # Only prints if you explicitly set print_dict=True

    return ticker_dict