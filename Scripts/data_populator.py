"""
Data Populator Module 
--------------------------------------------

- Robust numeric detection
- Yield guaranteed float
- Rounded numeric precision
- Protected identifier columns
"""

import pandas as pd
import numpy as np


class RangeDataPopulator:
    def __init__(self, n_samples: int = 100, decimal_places: int = 2):
        self.n_samples = n_samples
        self.decimal_places = decimal_places

        # Protected categorical columns
        self.protected_columns = ["Crop Type", "Crop Species"]

    # ---------------------------------------------------------
    # Robust numeric / range detection
    # ---------------------------------------------------------
    def _extract_range_or_value(self, value):

        if pd.isna(value):
            return value

        value = str(value).strip().replace("–", "-").replace("—", "-")

        # Try range split first
        if "-" in value:
            parts = value.split("-")
            if len(parts) == 2:
                try:
                    low = float(parts[0].strip())
                    high = float(parts[1].strip())
                    return (min(low, high), max(low, high))
                except ValueError:
                    pass  # Not numeric range

        # Try single float conversion
        try:
            return float(value)
        except ValueError:
            return value  # Categorical

    # ---------------------------------------------------------
    # Populate a single row
    # ---------------------------------------------------------
    def _populate_row(self, row: pd.Series) -> pd.DataFrame:

        populated_data = {}

        for col in row.index:
            value = row[col]

            # Keep identifiers untouched
            if col in self.protected_columns:
                populated_data[col] = [value] * self.n_samples
                continue

            parsed = self._extract_range_or_value(value)

            # Range
            if isinstance(parsed, tuple):
                low, high = parsed
                samples = np.linspace(low, high, self.n_samples)
                np.random.shuffle(samples)
                samples = np.round(samples, self.decimal_places)
                populated_data[col] = samples

            # Single numeric
            elif isinstance(parsed, float):
                rounded = round(parsed, self.decimal_places)
                populated_data[col] = [rounded] * self.n_samples

            # Categorical
            else:
                populated_data[col] = [value] * self.n_samples

        return pd.DataFrame(populated_data)

    # ---------------------------------------------------------
    # Populate full dataframe
    # ---------------------------------------------------------
    def populate(self, df: pd.DataFrame) -> pd.DataFrame:

        all_rows = []

        for _, row in df.iterrows():
            all_rows.append(self._populate_row(row))

        populated_df = pd.concat(all_rows, ignore_index=True)

        # Final numeric enforcement (safe)
        for col in populated_df.columns:
            if col not in self.protected_columns:
                populated_df[col] = pd.to_numeric(populated_df[col], errors="coerce")

        return populated_df
