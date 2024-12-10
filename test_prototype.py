# test_process_data.py
import os
from process_data import load_chromadb, initialize_llm, predict_fa_contract

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Main test execution
if __name__ == "__main__":
    # Set up ChromaDB and LLM
    persist_directory = "./chroma_data"  # Update this if your ChromaDB directory is different
    db = load_chromadb(persist_directory)
    llm = initialize_llm()
    # messages = [
    #     SystemMessage(content="You are AI chatbot who anser the questions about baseball"),
    #     HumanMessage(content="안녕? 너 야구 좋아하니?"),
    #     AIMessage(content="I really love baseball! How can I help you?")
    # ]
    # Test parameters
    player_name = "이정후"
    year = 2022

    # Run prediction    
    print(f"Testing FA contract prediction for {player_name} in {year}...")
    result = predict_fa_contract(player_name, year, db, llm)

    # Output result
    if result:
        print("\nPredicted FA Contract:\n")
        print(result)
    else:
        print(f"Prediction failed for {player_name}.")
