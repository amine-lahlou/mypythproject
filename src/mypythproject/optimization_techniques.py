import yfinance as yf
import pandas as pd 
from sec_cik_mapper import StockMapper
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging 
from scipy.optimize import minimize
import numpy as np
from pybacktestchain.data_module import Information

@dataclass
class MaxSharpe(Information):
    def compute_portfolio(self, t: datetime, information_set: dict):
        try:
            mu = information_set['expected_return']        
            Sigma = information_set['covariance_matrix']    
            n = len(mu)

            # Sharpe objective
            def sharpe_ratio_neg(x):
                eps = 1e-8 # Add small epsilon to avoid division by zero
                numerator = x.dot(mu)
                denominator = np.sqrt(x.dot(Sigma).dot(x)) + eps
                return - numerator / denominator 

            # constraints:
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = [(0.0, 1.0)] * n

            # initial guess, equal weights
            x0 = np.ones(n) / n

            res = minimize(sharpe_ratio_neg,
                           x0,
                           constraints=constraints,
                           bounds=bounds,
                           method='SLSQP')

            # prepare dictionary
            portfolio = {k: None for k in information_set['companies']}

            # if converged update
            if res.success:
                for i, company in enumerate(information_set['companies']):
                    portfolio[company] = res.x[i]
            else:
                raise Exception("Optimization did not converge")

            return portfolio

        except Exception as e:
            # if something goes wrong return an equal weigh portfolio but let the user know
            logging.warning("Error computing portfolio, returning equal weight portfolio")
            logging.warning(e)
            return {k: 1/len(information_set['companies']) 
                    for k in information_set['companies']}
        
    
    def compute_information(self, t: datetime):
        # get the data module
        data = self.slice_data(t)
        # the information set will be a dictionary with data
        information_set = {}
        # sort data by ticker and date
        data = data.sort_values(by=[self.company_column, self.time_column])

        # compute returns
        data['return'] = data.groupby(self.company_column)[self.adj_close_column].pct_change()

        # expected return by company
        information_set['expected_return'] = data.groupby(self.company_column)['return'].mean().to_numpy()

        # covariance matrix

        # 1. pivot the data
        data_pivot = data.pivot(index=self.time_column, 
                                columns=self.company_column, 
                                values=self.adj_close_column)
        # drop missing values
        data_pivot = data_pivot.dropna(axis=0)
        # 2. compute the covariance matrix
        covariance_matrix = data_pivot.cov().to_numpy()

        # add to information set
        information_set['covariance_matrix'] = covariance_matrix
        information_set['companies'] = data_pivot.columns.to_numpy()

        return information_set

@dataclass
class MinVariance(Information):
    def compute_portfolio(self, t: datetime, information_set: dict):
        try:
            mu = information_set['expected_return']        
            Sigma = information_set['covariance_matrix']    
            n = len(mu)

            # Min Variance objective
            def variance_obj(x):
                return x.dot(Sigma).dot(x)

            # constraints:
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = [(0.0, 1.0)] * n

            # initial guess, equal weights
            x0 = np.ones(n) / n

            res = minimize(variance_obj,
                           x0,
                           constraints=constraints,
                           bounds=bounds,
                           method='SLSQP')

            # prepare dictionary
            portfolio = {k: None for k in information_set['companies']}

            # if converged update
            if res.success:
                for i, company in enumerate(information_set['companies']):
                    portfolio[company] = res.x[i]
            else:
                raise Exception("Optimization did not converge")

            return portfolio

        except Exception as e:
            # if something goes wrong return an equal weight portfolio but let the user know
            logging.warning("Error computing portfolio, returning equal weight portfolio")
            logging.warning(e)
            return {k: 1/len(information_set['companies']) 
                    for k in information_set['companies']}
        
    
    def compute_information(self, t: datetime):
        # get the data module
        data = self.slice_data(t)
        # the information set will be a dictionary with data
        information_set = {}
        # sort data by ticker and date
        data = data.sort_values(by=[self.company_column, self.time_column])

        # compute returns
        data['return'] = data.groupby(self.company_column)[self.adj_close_column].pct_change()

        # expected return by company
        information_set['expected_return'] = data.groupby(self.company_column)['return'].mean().to_numpy()

        # covariance matrix

        # 1. pivot the data
        data_pivot = data.pivot(index=self.time_column, 
                                columns=self.company_column, 
                                values=self.adj_close_column)
        # drop missing values
        data_pivot = data_pivot.dropna(axis=0)
        # 2. compute the covariance matrix
        covariance_matrix = data_pivot.cov().to_numpy()

        # add to information set
        information_set['covariance_matrix'] = covariance_matrix
        information_set['companies'] = data_pivot.columns.to_numpy()

        return information_set
    
@dataclass
class MaxReturn(Information):
    def compute_portfolio(self, t: datetime, information_set: dict):
        try:
            mu = information_set['expected_return']        
            Sigma = information_set['covariance_matrix']    
            n = len(mu)

            # Max Return objective
            def neg_return(x):
                return -x.dot(mu)

            # constraints:
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = [(0.0, 1.0)] * n

            # initial guess, equal weights
            x0 = np.ones(n) / n

            res = minimize(neg_return,
                           x0,
                           constraints=constraints,
                           bounds=bounds,
                           method='SLSQP')

            # prepare dictionary
            portfolio = {k: None for k in information_set['companies']}

            # if converged update
            if res.success:
                for i, company in enumerate(information_set['companies']):
                    portfolio[company] = res.x[i]
            else:
                raise Exception("Optimization did not converge")

            return portfolio

        except Exception as e:
            # if something goes wrong return an equal weight portfolio but let the user know
            logging.warning("Error computing portfolio, returning equal weight portfolio")
            logging.warning(e)
            return {k: 1/len(information_set['companies']) 
                    for k in information_set['companies']}
        
    
    def compute_information(self, t: datetime):
        # get the data module
        data = self.slice_data(t)
        # the information set will be a dictionary with data
        information_set = {}
        # sort data by ticker and date
        data = data.sort_values(by=[self.company_column, self.time_column])

        # compute returns
        data['return'] = data.groupby(self.company_column)[self.adj_close_column].pct_change()

        # expected return by company
        information_set['expected_return'] = data.groupby(self.company_column)['return'].mean().to_numpy()

        # covariance matrix

        # 1. pivot the data
        data_pivot = data.pivot(index=self.time_column, 
                                columns=self.company_column, 
                                values=self.adj_close_column)
        # drop missing values
        data_pivot = data_pivot.dropna(axis=0)
        # 2. compute the covariance matrix
        covariance_matrix = data_pivot.cov().to_numpy()

        # add to information set
        information_set['covariance_matrix'] = covariance_matrix
        information_set['companies'] = data_pivot.columns.to_numpy()

        return information_set