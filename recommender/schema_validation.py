"""
Schema validation using Pandera.
Ensures MovieLens dataset matches expected structure before training.
"""

import pandera.pandas as pa
from pandera import Column, DataFrameSchema, Check

schema = DataFrameSchema({
    "user_id": Column(int, Check.ge(1)),
    "movie_id": Column(int, Check.ge(1)),
    "rating": Column(float, Check.between(0, 5), coerce=True),
    "timestamp": Column(int, Check.ge(0))
})

def validate_dataset(df):
    """Validate dataset against schema."""
    try:
        schema.validate(df, lazy=True)
        print(" Schema validation passed.")
        return True
    except pa.errors.SchemaErrors as e:
        print(" Schema validation failed.")
        print(e.failure_cases)
        return False
