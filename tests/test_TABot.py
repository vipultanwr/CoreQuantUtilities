
import unittest
import pandas as pd
import numpy as np

from CoreQuantUtilities.ta_strategies.TABot import getTACombinedSignals

class TestTABot(unittest.TestCase):
    def setUp(self):
        # Create a sample dataframe for testing
        self.sample_data = pd.DataFrame({
            'open': np.random.rand(100) * 100,
            'high': np.random.rand(100) * 100,
            'low': np.random.rand(100) * 100,
            'close': np.random.rand(100) * 100,
            'volume': np.random.rand(100) * 1000,
        })
        self.sample_data.index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100))

    def test_getTACombinedSignals(self):
        signals = getTACombinedSignals(self.sample_data, returnall=True)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertFalse(signals.empty)

        for col in signals.columns:
            for val in signals[col].dropna().values:
                if isinstance(val, np.ndarray):
                    for v in val.flatten():
                        self.assertIn(v, [-1, 0, 1])
                else:
                    self.assertIn(val, [-1, 0, 1])



if __name__ == '__main__':
    unittest.main()
