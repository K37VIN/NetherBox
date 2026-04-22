import pandas as pd

from src.logger import logger
from src.utils.main_utils import load_dataset


class DataLoader:
    """
    Single entry point for loading training data.
    Extend this class to add new data sources without touching pipeline logic.
    """

    def from_file(self, path: str) -> pd.DataFrame:
        """Load data from a local file path (CSV, Excel, JSON, Parquet)."""
        logger.info(f"Loading data from file: {path}")
        return load_dataset(path)

    def from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pass a DataFrame directly (useful for testing)."""
        logger.info(f"Using in-memory DataFrame: {df.shape}")
        return df.copy()

    def from_mongodb(self, collection_name: str, query: dict = None) -> pd.DataFrame:
        """Load data from a MongoDB collection."""
        from src.configuration.mongo_db_connection import MongoDBClient
        client = MongoDBClient()
        col = client.get_collection(collection_name)
        records = list(col.find(query or {}, {"_id": 0}))
        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df)} records from MongoDB collection '{collection_name}'")
        return df
