import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from openai import OpenAI


def run_fine_tuning(
    client: OpenAI,
    training_data_path: str,
    model_id: str,
    hyperparameters: Dict[str, Any],
    log_dir: str = "fine_tuning_logs",
    csv_path: str = "fine_tuning_experiments.csv",
) -> Tuple[Any, Any]:
    """
    Run a fine-tuning job using the provided OpenAI client and log all relevant information.

    Args:
        client (OpenAI): Preconfigured OpenAI client instance.
        training_data_path (str): Path to the training data file.
        model_id (str): Base model identifier.
        hyperparameters (Dict[str, Any]): Hyperparameters for the fine-tuning job.
        log_dir (str, optional): Directory to save fine-tuning logs.
        csv_path (str, optional): Path to a CSV file for logging experiments.

    Returns:
        Tuple[Any, Any]: The fine-tuning job object and the fine-tuned model ID.
    """
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Load training data and compute statistics
    with open(training_data_path, "r") as f:
        training_data = [json.loads(line) for line in f]

    data_stats = {
        "n_samples": len(training_data),
        "n_messages": sum(len(sample["messages"]) for sample in training_data),
        "has_system_message": any(
            "system" in msg["role"]
            for sample in training_data
            for msg in sample["messages"]
        ),
        "training_data_path": str(Path(training_data_path).resolve()),
        "training_data_size_bytes": os.path.getsize(training_data_path),
    }

    # Upload training file
    print("Uploading training file...")
    with open(training_data_path, "rb") as file:
        training_file = client.files.create(file=file, purpose="fine-tune")

    # Create fine-tuning job
    print(f"Starting fine-tuning job with model {model_id}...")
    job = client.fine_tuning.jobs.create(
        model=model_id, training_file=training_file.id, hyperparameters=hyperparameters
    )

    # Wait for job completion by polling status
    print("Waiting for job completion...")
    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        print(f"Status: {job.status}")

        if job.status in ["succeeded", "failed", "cancelled"]:
            break

        time.sleep(60)

    # Prepare log entry with job details and training data statistics
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "job_id": job.id,
        "base_model": job.model,
        "fine_tuned_model": job.fine_tuned_model,
        "training_file": job.training_file,
        "validation_file": job.validation_file,
        "status": job.status,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
        "trained_tokens": job.trained_tokens,
        "hyperparameters": hyperparameters,
        "training_data_stats": data_stats,
    }

    # Save detailed log as a JSON file
    json_filename = (
        f"fine_tuning_job_{job.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    json_filepath = Path(log_dir) / json_filename
    with open(json_filepath, "w") as f:
        json.dump(log_entry, f, indent=2)
    print(f"Detailed log saved to: {json_filepath}")

    # Prepare experiment data for CSV logging
    experiment_data = {
        "date": [datetime.now()],
        "job_id": [job.id],
        "base_model": [job.model],
        "model_id": [job.fine_tuned_model],
        "training_file": [job.training_file],
        "training_data_path": [data_stats["training_data_path"]],
        "n_samples": [data_stats["n_samples"]],
        "n_messages": [data_stats["n_messages"]],
        "has_system_message": [data_stats["has_system_message"]],
        "training_data_size_bytes": [data_stats["training_data_size_bytes"]],
        "n_epochs": [hyperparameters.get("n_epochs")],
        "batch_size": [hyperparameters.get("batch_size")],
        "learning_rate_multiplier": [hyperparameters.get("learning_rate_multiplier")],
        "trained_tokens": [job.trained_tokens],
        "status": [job.status],
    }

    df = pd.DataFrame(experiment_data)
    csv_file = Path(csv_path)
    if csv_file.exists():
        df.to_csv(csv_file, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_file, index=False)
    print(f"Experiment logged to: {csv_file}")

    # Final status output
    if job.status == "succeeded":
        print(
            f"Fine-tuning completed successfully! New model ID: {job.fine_tuned_model}"
        )
    else:
        print(f"Fine-tuning ended with status: {job.status}")

    return job, job.fine_tuned_model
