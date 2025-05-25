import pandas as pd
import re
import time

class KCCPreprocessor:
    def __init__(self, input_path: str, output_path: str = "kcc_preprocessed_chunks.csv"):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.input_path)
        print(f"Data loaded. Shape: {self.df.shape}")

    @staticmethod
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s.,?-]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def apply_cleaning(self):
        self.df['QueryText_clean'] = self.df['QueryText'].apply(self.clean_text)
        self.df['KccAns_clean'] = self.df['KccAns'].apply(self.clean_text)

    def filter_empty(self):
        self.df = self.df[(self.df['QueryText_clean'] != "") & (self.df['KccAns_clean'] != "")]
        print(f"Filtered non-empty queries and answers. Shape: {self.df.shape}")

    def normalize_metadata(self):
        meta_cols = ['StateName', 'Crop', 'DistrictName', 'Category', 'QueryType', 'Season', 'Sector']
        for col in meta_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip().str.title()

        for col in ['Year', 'Month', 'Day']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(int)

    def combine_chunks(self):
        self.df['chunk'] = self.df['QueryText_clean'] + " " + self.df['KccAns_clean']

    def save_data(self):
        output_columns = ['chunk', 'StateName', 'Crop', 'DistrictName',
                          'Year', 'Month', 'Day', 'Category', 'QueryType', 'Season', 'Sector']
        self.df[output_columns].to_csv(self.output_path, index=False)
        print(f"Preprocessed data saved to '{self.output_path}'.")

    def run(self):
        self.load_data()
        self.apply_cleaning()
        self.filter_empty()
        self.normalize_metadata()
        self.combine_chunks()
        self.save_data()
        