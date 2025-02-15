from pybacktestchain.broker import Backtest, StopLoss
from pybacktestchain.blockchain import load_blockchain
from datetime import datetime
from optimization_techniques import MaxSharpe
from optimization_techniques import MinVariance
from optimization_techniques import MaxReturn
from pybacktestchain.data_module import FirstTwoMoments

# Set verbosity for logging
verbose = False  # Set to True to enable logging, or False to suppress it

backtest = Backtest(
    initial_date=datetime(2019, 1, 1),
    final_date=datetime(2020, 1, 1),
    information_class=FirstTwoMoments,
    risk_model=StopLoss,
    name_blockchain='backtest',
    verbose=verbose
)
backtest.universe = ['BAC','JPM','GS']
backtest.run_backtest()
block_chain = load_blockchain('backtest')

print(str(block_chain))
# check if the blockchain is valid
print(block_chain.is_valid())
