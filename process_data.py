# process_data.py
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# Initialize ChromaDB
def load_chromadb(persist_directory):
    embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    return Chroma(persist_directory=persist_directory, embedding_function=embed_model)


# Initialize LLM
def initialize_llm():
    return ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model_name="gpt-4o"
    )


# Extract stats from content
def extract_stats_from_content(content: str):
    stats = {}
    lines = content.split(", ")
    for line in lines:
        if ":" not in line:
            continue

        key, *values = line.split(":")
        value = ":".join(values).strip()

        if key.strip() == "Stats":
            try:
                stats_dict = eval(value.strip())
                stats.update(stats_dict)
            except (SyntaxError, ValueError):
                continue
        else:
            try:
                stats[key.strip()] = float(value)
            except ValueError:
                continue

    return stats


# Normalize stats
def normalize_stats(stats):
    normalized_stats = {}
    for key, value in stats.items():
        clean_key = key.strip("'").strip('"')
        if clean_key != "Year":
            normalized_stats[clean_key] = value
    return normalized_stats


# Fetch player stats from ChromaDB
def fetch_player_stats(db, player_name: str, end_year: int, years: int = 4):
    start_year = end_year - years + 1
    query = f"{player_name} 선수의 {start_year}년부터 {end_year}년까지의 z_type 성적을 가져와 주세요."
    results = db.similarity_search(query, k=years * 10, filter={"player_name": player_name})
    # for result in results:
    #     print(result.metadata, result.page_content)
    filtered_results_standardized = [
        doc for doc in results
        if doc.metadata.get("player_name") == player_name
        and start_year <= doc.metadata.get("year", 0) <= end_year
        and doc.metadata.get("data_type") == "z_type"
    ]

    final_results = []
    for doc in filtered_results_standardized:
        year = doc.metadata.get("year", 0)
        position = doc.metadata.get("position", "")
        stats = extract_stats_from_content(doc.page_content)
        normalized_stats = normalize_stats(stats)
        
        final_results.append({
            "year": year,
            "stats": normalized_stats,
            "position": position,
        })
        
        
        
    if not final_results:
        raise ValueError(f"선수 '{player_name}'의 데이터가 없습니다.")

    return final_results

def fetch_actual_stats(db, player_name: str, years: list):
    """
    Fetch up to 2 stats for a given player and list of years, using only non-z_type data.
    """
    actual_stats = []
    for year in years:
        filters = {
            "$and": [
                {"player_name": {"$eq": player_name}},
                {"year": {"$eq": year}}
            ]
        }
        # 최대 2개의 데이터 가져오기
        results = db.similarity_search(query="Retrieve stats", k=2, filter=filters)

        # z_type 제외 및 유효 데이터 선택
        for result in results:
            if result.metadata.get("data_type") != "z_type":  # z_type이 아닌 데이터만 선택
                stats = extract_stats_from_content(result.page_content)
                actual_stats.append({
                    "year": year,
                    "stats": stats,
                    "metadata": result.metadata
                })
                break  # 첫 번째 유효 데이터를 찾으면 종료

    if not actual_stats:
        print(f"No non-z_type stats found for {player_name} in years: {years}")
    return actual_stats

# Aggregate normalized stats
def aggregate_normalized_stats(stats):
    aggregated_stats = {}
    for stat in stats:
        for key, value in stat["stats"].items():
            aggregated_stats[key] = aggregated_stats.get(key, 0) + value

    num_years = len(stats)
    aggregated_stats = {key: round(value / num_years, 3) for key, value in aggregated_stats.items()}
    return aggregated_stats


# Calculate similarity between players
def calculate_similarity(player_stats: dict, target_stats: dict, query_keys: list, weighting: dict) -> float:
    """
    두 선수의 유사도를 계산합니다.
    - 높을수록 좋은 지표는 현재 값이 클수록 유사도를 증가.
    - 낮을수록 좋은 지표는 현재 값이 작을수록 유사도를 증가.
    """
    similarity_score = 0.0
    total_weight = 0.0

    for key in query_keys:
        if key in player_stats and key in target_stats:
            player_value = player_stats[key]
            target_value = target_stats[key]

            # 높을수록 좋은 지표 처리 (승리 등)
            if key in ["W", "SV", "HLD", "OPS_z", "wRC+_z", "WAR_z"]:
                similarity = max(0, 1 - abs(player_value - target_value) / max(target_value, 1))
            
            # 낮을수록 좋은 지표 처리 (ERA, WHIP, FIP 등)
            elif key in ["ERA_z", "WHIP_z", "FIP_z", "ERA", "WHIP", "FIP"]:
                similarity = max(0, 1 - abs(player_value - target_value) / max(player_value, 1))
            
            # 가중치 적용
            weight = weighting.get(key, 1.0)
            similarity_score += similarity * weight
            total_weight += weight

    if total_weight == 0:
        return 0.0

    return similarity_score / total_weight

# Find most similar players
def find_most_similar_players(db, player_name, year, k=10, similarity_threshold=0.6):
    base_stats = fetch_player_stats(db, player_name, year)
    print(f"Base stats: {base_stats}")
    base_aggregated_stats = aggregate_normalized_stats(base_stats)
    print(f"aggregated base stat: {base_aggregated_stats}")
    
    position = base_stats[0].get("position", "")
    is_pitcher = position == "P"
    
    if is_pitcher:
        sp_status = base_stats[0]["stats"].get("Binary_SP")
        if sp_status is None:
            raise ValueError(f"{player_name} 선수의 선발/구원 구분이 없습니다.")
        
        if sp_status == 1:
            weighting = {
                "ERA_z": 1.0,
                "WHIP_z": 1.0,
                "FIP_z": 1.0,
                "승_z": 0.5,
            }
        else:
            weighting = {
                "ERA_z": 1.0,
                "WHIP_z": 1.0,
                "FIP_z": 1.0,
                "홀드_z": 0.5,
                "세이브_z": 0.5,
            }
        query_keys = list(weighting.keys())
        query = " AND ".join(
            [f"{key}가 {base_aggregated_stats[key]:.3f}보다 크고," for key in query_keys if key in base_aggregated_stats]
        )
        query += "z_type 성적을 가져와 주세요"
    else:
        weighting = {"WAR_z": 1.0, "OPS_z": 1.0, "wRC+_z": 1.0}
        query_keys = ["WAR_z", "OPS_z", "wRC+_z"]

        query = " AND ".join(
            [f"{key}가 {base_aggregated_stats[key]:.3f}와 거의 동일하고," for key in query_keys if key in base_aggregated_stats]
        )
        query += "z_type 성적을 가져와 주세요"
    
    if is_pitcher:
        db_results = db.similarity_search(query, k=100, filter={"position": "P"})
    else:
        db_results = db.similarity_search(query, k=100)
    print("\nsimilarity search result:")
    for idx, result in enumerate(db_results, start=1):
        print(f"{idx}. Metadata: {result.metadata}, Page Content: {result.page_content[:200]}...")  # Print only the first 200 characters of content

    candidates = []
    for result in db_results:
        similar_player_name = result.metadata.get("player_name")
        result_year = result.metadata.get("year")
        result_position = result.metadata.get("position")

        if is_pitcher and result.metadata.get("position") != position:
            continue
        
        try:
            player_stats = fetch_player_stats(db, similar_player_name, result_year, 4)
            aggregated_stats = aggregate_normalized_stats(player_stats)
            analysis_years = sorted([player_stats[0]["year"], player_stats[-1]["year"]])

            if analysis_years[1] - analysis_years[0] + 1 >= 3:
                if is_pitcher and player_stats[0]["stats"].get("Binary_SP") != sp_status:
                    continue
                
                candidates.append({
                    "player_name": similar_player_name,
                    "averaged_stats": aggregated_stats,
                    "analysis_years": analysis_years,
                })
        except ValueError:
            pass
        except Exception as e:
            print(f"[ERROR] Error processing player {similar_player_name}: {e}")
            pass

    similar_players = []
    for candidate in candidates:
        similarity = calculate_similarity(
            player_stats=candidate["averaged_stats"],
            target_stats=base_aggregated_stats,
            query_keys=query_keys,
            weighting=weighting
        )
        if similarity >= similarity_threshold:
            similar_players.append({
                "player_name": candidate["player_name"],
                "averaged_stats": candidate["averaged_stats"],
                "analysis_years": candidate["analysis_years"],
                "similarity": round(similarity, 3),
            })

    unique_players = {player["player_name"]: player for player in similar_players}
    deduplicated_players = sorted(unique_players.values(), key=lambda x: x["similarity"], reverse=True)

    print(f"{len(deduplicated_players)} 명의 유사한 선수를 찾았습니다.")
    for player in deduplicated_players:
        print(f"{player['player_name']} ({player['analysis_years'][0]}-{player['analysis_years'][1]}) - 유사도: {player['similarity']}")

    for player in deduplicated_players:
        player["actual_stats"] = fetch_actual_stats(
            db,
            player["player_name"],
            player["analysis_years"]
        )
    
    return deduplicated_players

# Predict FA contract based on similar players
def predict_fa_contract(player_name, year, db, llm, k=10):
    # 유사 선수 찾기
    similar_players = find_most_similar_players(db, player_name, year, k)

    if not similar_players:
        print(f"No similar players found for {player_name}.")
        return None

    # FA 계약 정보 검색
    fa_contracts = []
    for player in similar_players:
        query = f"{player['player_name']} 선수의 FA 계약 정보를 가져와 주세요."
        results = db.similarity_search(query, k=3, filter={"contract_type": "FA"})
        fa_contracts.extend(results)

    # LLM에게 전달할 context 생성
    context = f"Target Player: {player_name}\n\nSimilar Players:\n"
    for player in similar_players:
        stats = player["averaged_stats"]
        years = f"{player['analysis_years'][0]}-{player['analysis_years'][1]}"
        actual_stats = player.get("actual_stats", [])
        
        context += f"- {player['player_name']} ({years}, Similarity: {player['similarity']}): {stats}\n"
        for stat in actual_stats:
            context += f"Year {stat['year']}: {stat['stats']}\n"
    context += "\nFA Contract Information:\n"
    for contract in fa_contracts:
        context += f"Player: {contract.metadata['player_name']}, Contract Details: {contract.page_content}\n"

    # LLM에 context 전달 및 예측 요청
    query = f"""
    Based on the following context, predict the FA contract value for the player {player_name}.
    Provide detailed reasoning in Korean.
    Your answer must not contain players' stat which is normalized.
    Among the contexts I provide, you can find players' real stats which is not normalized. You'd rather use those stats. Use multiple metrics, not just one or two, to base your answer.
    Follow 4 steps to answer: 1. Analyze similar players, 2. Analyze comparable player tendencies, 3. Estimate FA contract value, 4. Conclusion
    Also you should consider that as of 2024, current contracts are roughly twice as expensive as they were more than 7 years ago. It didn't jumped twice right away, it gradually increased.
    If a player's past performance is too poor, you should make the decision that they should not be allowed to file for free agency that year.
    {context}
    """
    
    system_message = SystemMessage(content="You are an expert in baseball analytics. Please provide detailed responses tailored to FA contract predictions for baseball players.")
    response = llm([HumanMessage(content=query)])
    return response.content


# Main function
# persist_directory = "./chroma_data"
# db = load_chromadb(persist_directory)
# llm = initialize_llm()

# fetch_player_stats(db, "류현진", 2010, 4)
# results = db.similarity_search("구자욱", k=10)
# for result in results:
#     print(result.metadata, result.page_content)