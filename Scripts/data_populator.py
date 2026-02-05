import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional, List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RangeDataPopulator:
    """Class for generating synthetic data from range-based agricultural data"""
    
    def __init__(self, num_samples_per_row: int = 100):
        """
        Initialize the data populator.
        
        Parameters:
        -----------
        num_samples_per_row : int
            Number of synthetic samples to generate for each original row
        """
        self.num_samples_per_row = num_samples_per_row
        self.original_df = None
        self.synthetic_df = None
        
    @staticmethod
    def extract_range_or_value(val: Union[str, float, int]) -> Union[Tuple[float, float], float, None]:
        """
        Extract range or single value from a cell.
        
        Parameters:
        -----------
        val : str, float, or int
            The value from the dataframe cell
        
        Returns:
        --------
        Union[Tuple[float, float], float, None]
            - Tuple(min, max) for ranges
            - float for single values
            - None for non-numeric or empty values
        """
        if pd.isna(val):
            return None
        
        val_str = str(val).strip()
        
        # Handle ranges with different dash types
        if any(sep in val_str for sep in ['-', '–', '—']):
            for sep in ['–', '—']:
                val_str = val_str.replace(sep, '-')
            
            parts = val_str.split('-')
            if len(parts) == 2:
                try:
                    part1 = float(parts[0].strip())
                    part2 = float(parts[1].strip())
                    
                    # Check if it's actually a single value
                    if abs(part1 - part2) < 1e-10:  # Account for floating point errors
                        return part1
                    
                    # Ensure min, max order
                    min_val = min(part1, part2)
                    max_val = max(part1, part2)
                    return (min_val, max_val)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Could not parse range '{val_str}': {e}")
                    return None
        
        # Try single numeric value
        try:
            return float(val_str)
        except ValueError:
            logger.debug(f"Could not parse as number: '{val_str}'")
            return None
    
    @staticmethod
    def generate_sample(value_info: Union[Tuple[float, float], float, str], 
                        original_value: Union[str, float, int]) -> Union[float, str]:
        """
        Generate a single sample based on value type.
        
        Parameters:
        -----------
        value_info : Union[Tuple, float, str]
            Extracted range or value information
        original_value : Union[str, float, int]
            Original value from dataframe
        
        Returns:
        --------
        Union[float, str]
            Generated sample value
        """
        if isinstance(value_info, tuple):  # Range
            min_val, max_val = value_info
            return np.random.uniform(min_val, max_val)
        elif isinstance(value_info, (int, float)):  # Single number
            return value_info
        else:  # Keep original (for categorical/text)
            return original_value
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file
        
        Returns:
        --------
        pd.DataFrame
            Loaded dataframe
        """
        logger.info(f"Loading data from {filepath}")
        self.original_df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(self.original_df)} rows with {len(self.original_df.columns)} columns")
        return self.original_df
    
    def generate_synthetic_data(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate synthetic data from the loaded dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame, optional
            DataFrame to process. If None, uses self.original_df
        
        Returns:
        --------
        pd.DataFrame
            Generated synthetic data
        """
        if df is None:
            df = self.original_df
            if df is None:
                raise ValueError("No data loaded. Call load_data() first or provide df parameter.")
        
        logger.info(f"Generating {self.num_samples_per_row} samples per row...")
        
        synthetic_data = []
        
        for idx, row in df.iterrows():
            if idx % 10 == 0:  # Log progress every 10 rows
                logger.info(f"Processing row {idx + 1}/{len(df)}")
            
            # Generate samples for this row
            row_samples = []
            for sample_num in range(self.num_samples_per_row):
                sample_row = {}
                
                for col in df.columns:
                    val = row[col]
                    value_info = self.extract_range_or_value(val)
                    
                    # Generate sample value
                    sample_row[col] = self.generate_sample(value_info, val)
                
                row_samples.append(sample_row)
            
            synthetic_data.extend(row_samples)
        
        self.synthetic_df = pd.DataFrame(synthetic_data)
        logger.info(f"Generated {len(self.synthetic_df)} synthetic rows")
        
        return self.synthetic_df
    
    def save_synthetic_data(self, filepath: str):
        """
        Save synthetic data to CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the synthetic data
        """
        if self.synthetic_df is None:
            raise ValueError("No synthetic data generated. Call generate_synthetic_data() first.")
        
        logger.info(f"Saving synthetic data to {filepath}")
        self.synthetic_df.to_csv(filepath, index=False)
        logger.info("Save completed successfully")
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the generation process.
        
        Returns:
        --------
        Dict
            Dictionary containing statistics
        """
        if self.original_df is None or self.synthetic_df is None:
            return {}
        
        stats = {
            'original_rows': len(self.original_df),
            'synthetic_rows': len(self.synthetic_df),
            'samples_per_row': self.num_samples_per_row,
            'columns': list(self.original_df.columns),
            'generation_ratio': len(self.synthetic_df) / len(self.original_df)
        }
        
        return stats


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic data from range-based agricultural data')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('output_file', help='Path to output CSV file')
    parser.add_argument('--samples', type=int, default=100, 
                       help='Number of samples per row (default: 100)')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create populator
    populator = RangeDataPopulator(num_samples_per_row=args.samples)
    
    # Load data
    populator.load_data(args.input_file)
    
    # Generate synthetic data
    synthetic_df = populator.generate_synthetic_data()
    
    # Save results
    populator.save_synthetic_data(args.output_file)
    
    # Print statistics
    stats = populator.get_statistics()
    print("\n" + "="*50)
    print("GENERATION STATISTICS")
    print("="*50)
    print(f"Original rows: {stats['original_rows']}")
    print(f"Synthetic rows: {stats['synthetic_rows']}")
    print(f"Samples per original row: {stats['samples_per_row']}")
    print(f"Generation ratio: {stats['generation_ratio']:.1f}x")
    print(f"Output saved to: {args.output_file}")


if __name__ == "__main__":
    main()