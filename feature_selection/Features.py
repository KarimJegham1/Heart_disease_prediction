from feast import Field,Entity,ValueType,FeatureStore,FeatureView
from feast import FileSource
from feast.types import Float32, Int32
from datetime import datetime
def Feature_selection(df):
    patient = Entity(name="patient", join_keys=["patient_id"])
    cp = Field(name="cp", dtype=Float32)
    thalach = Field(name="thalach", dtype=Int32)
    ca = Field(name="ca", dtype=Int32)
    thal = Field(name="thal", dtype=Int32)
    df['timestamp'] = datetime.now()

    df.to_parquet("heart_disease.parquet")
    # Point File Source to the saved file
    data_source = FileSource(
        path="heart_disease.parquet",
        event_timestamp_column="timestamp",
        created_timestamp_column="created",
    )

    # Create a Feature View of the features
    heart_disease_fv = FeatureView(
        name="heart_disease",
        entities=[patient],
        schema=[cp, thalach, ca, thal],
        source=data_source,
    )
    # Create a store of the data and apply the features
    store = FeatureStore(repo_path="C:/Users/TOPINFORMATIQUE/anaconda3/Lib/site-packages/feast/templates/local/feature_repo")
    store.apply([patient, heart_disease_fv])