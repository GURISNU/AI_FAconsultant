# upload_data.py
import os
import pandas as pd
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def load_csv(file_path):
    """Load CSV file if it exists, otherwise return an empty DataFrame."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"File not found: {file_path}. Skipping...")
        return pd.DataFrame()


def prepare_documents(stats_data, stats_z_data, stats_data_p, stats_z_data_p, fa_contract_data, data_type):
    """Prepare documents from the given datasets."""
    documents = []

    # Player stats data
    for _, row in stats_data.iterrows():
        row_dict = row.to_dict()
        
        row_dict.pop("Rank", None)
        row_dict.pop("Name", None)
        row_dict.pop("Year", None)
        row_dict.pop("Position", None)
        
        metadata = {
            "player_name": row.get("Name", ""),
            "year": row.get("Year", ""),
            "position": row.get("Position", ""),
        }
        content = f"Stats: {row_dict}"
        documents.append(Document(page_content=content, metadata=metadata))

    # Z-normalized stats data
    for _, row in stats_z_data.iterrows():
        row_dict = row.to_dict()
        
        row_dict.pop("Rank", None)
        row_dict.pop("Name", None)
        row_dict.pop("Year", None)
        row_dict.pop("Position", None)

        
        metadata = {
            "player_name": row.get("Name", ""),
            "year": row.get("Year", ""),
            "position": row.get("Position", ""),
            "data_type": data_type,
        }
        content = f"Stats: {row_dict}"
        documents.append(Document(page_content=content, metadata=metadata))

    # Pitcher stats data
    for _, row in stats_data_p.iterrows():
        row_dict = row.to_dict()
        
        row_dict.pop("Rank", None)
        row_dict.pop("Name", None)
        row_dict.pop("Year", None)
        
        metadata = {
            "player_name": row.get("Name", ""),
            "year": row.get("Year", ""),
            "position": row.get("Position", ""),
        }
        content = f"Stats: {row_dict}"
        documents.append(Document(page_content=content, metadata=metadata))

    # Z-normalized Pitcher stats data
    for _, row in stats_z_data_p.iterrows():
        row_dict = row.to_dict()
        
        row_dict.pop("Rank", None)
        row_dict.pop("Name", None)
        row_dict.pop("Year", None)
        
        metadata = {
            "player_name": row.get("Name", ""),
            "year": row.get("Year", ""),
            "position": row.get("Position", ""),
            "data_type": data_type,
        }
        content = f"Stats: {row_dict}"
        documents.append(Document(page_content=content, metadata=metadata))
    # FA contract data
    for _, row in fa_contract_data.iterrows():
        metadata = {
            "player_name": row.get("name", ""),
            "position": row.get("position", ""),
            "contract_year": row.get("year", ""),
            "contract_type": "FA",
        }
        content = (
            f"Contract Period: {row.get('contract_period', 'N/A')}, "
            f"Total Value: {row.get('total_amount', 'N/A')}, "
            f"Contract Option: {row.get('options', 'N/A')}, "
            f"Name: {row.get('name', '')}"
        )
        documents.append(Document(page_content=content, metadata=metadata))

    return documents


def upload_to_chromadb(persist_directory, batch_size=5000):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # File paths
    file_paths = {
        "batter_stats": os.path.join(current_dir, "batter_data.csv"),
        "batter_stats_z": os.path.join(current_dir, "batter_data_norm.csv"),
        "fa_contracts": os.path.join(current_dir, "fa_contract_data.csv"),
        # Placeholder for future files
        "pitcher_stats": os.path.join(current_dir, "pitcher_data.csv"),
        "pitcher_stats_z": os.path.join(current_dir, "pitcher_data_norm.csv"),
    }

    # Load data
    batter_stats = load_csv(file_paths["batter_stats"])
    batter_stats_z = load_csv(file_paths["batter_stats_z"])
    fa_contracts = load_csv(file_paths["fa_contracts"])

    # Placeholder: Future pitcher data
    pitcher_stats = load_csv(file_paths["pitcher_stats"])
    pitcher_stats_z = load_csv(file_paths["pitcher_stats_z"])

    # Prepare documents
    documents = prepare_documents(batter_stats, batter_stats_z, pitcher_stats, pitcher_stats_z, fa_contracts, data_type="z_type")

    # Initialize ChromaDB
    embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    db = Chroma(persist_directory=persist_directory, embedding_function=embed_model)

    # Function to add documents in batches
    def add_documents_in_batches(db, documents, batch_size=5000):
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            db.add_documents(batch)
            print(f"Added batch {i // batch_size + 1} / {len(documents) // batch_size + 1}")

    # Upload documents
    add_documents_in_batches(db, documents, batch_size=batch_size)
    print("Documents added successfully.")

persist_directory = "./chroma_data"
upload_to_chromadb(persist_directory=persist_directory)