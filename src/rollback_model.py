from mlflow.tracking import MlflowClient

REGISTERED_NAME = "quickfoods-delivery-predictor"

def main():
    client = MlflowClient()
    
    # Rollback to version 1
    client.set_registered_model_alias(
        REGISTERED_NAME,
        "champion",
        "1"
    )
    
    print("✅ Rolled back to version 1")

if __name__ == "__main__":
    main()