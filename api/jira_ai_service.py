import requests
import pandas as pd
from requests.auth import HTTPBasicAuth
from tqdm import tqdm  # You might remove tqdm for cleaner API logs
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import json
import boto3
import gc
import sys
from datetime import datetime

# --- Configuration ---
JIRA_DOMAIN = os.environ.get("JIRA_DOMAIN", "pranodan.atlassian.net")
EMAIL = os.environ.get("JIRA_EMAIL", "devik.bansal@wonderlendhubs.com")
API_TOKEN = os.environ.get("JIRA_API_TOKEN", "YOUR_JIRA_API_TOKEN_HERE")
AWS_REGION_NAME = os.environ.get("AWS_REGION_NAME", "ap-south-1")
MODEL_NAME = "all-mpnet-base-v2"
EMBEDDINGS_DIR = "embeddings"
DATA_DIR = "data"
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", 'amazon.titan-text-express-v1')

embedding_model = None
bedrock_client = None

def init_services():
    global embedding_model, bedrock_client
    if embedding_model is None:
        print(f"Loading SentenceTransformer model: {MODEL_NAME}")
        embedding_model = SentenceTransformer(MODEL_NAME)
        print("SentenceTransformer model loaded.")
    if bedrock_client is None:
        print(f"Initializing AWS Bedrock client in region: {AWS_REGION_NAME}")
        bedrock_client = boto3.client('bedrock-runtime', region_name=AWS_REGION_NAME)
        print("Bedrock client initialized.")

def release_services():
    global embedding_model, bedrock_client
    embedding_model = None
    bedrock_client = None
    gc.collect()

def fetch_jira_issues(project_key, jql_query):
    auth = HTTPBasicAuth(EMAIL, API_TOKEN)
    headers = {"Accept": "application/json"}
    url = f"https://{JIRA_DOMAIN}/rest/api/2/search"

    start_at = 0
    max_results = 50
    all_issues = []

    print(f"Starting fetch for project: {project_key} with JQL: {jql_query}")

    while True:
        params = {
            "jql": jql_query,
            "startAt": start_at,
            "maxResults": max_results,
            "fields": "summary,description,comment,labels,customfield_10048,customfield_10049,status,resolution"
        }

        try:
            response = requests.get(url, headers=headers, auth=auth, params=params)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Jira API request failed: {e}")
            raise

        data = response.json()
        issues = data.get("issues", [])
        total_issues = data.get("total", 0)

        if not issues:
            break

        for issue in issues:
            fields = issue["fields"]
            comments = " | ".join([c["body"] for c in fields.get("comment", {}).get("comments", [])])
            labels = ",".join(fields.get("labels", []))
            all_issues.append({
                "ticket_id": issue["key"],
                "summary": fields.get("summary", ""),
                "description": fields.get("description", ""),
                "rca": fields.get("customfield_10048", ""),
                "rca_category": fields.get("customfield_10049", ""),
                "labels": labels,
                "comments": comments,
                "status": fields.get("status", {}).get("name", ""),
                "resolution": fields.get("resolution", {}).get("name", "")
            })

        start_at += max_results
        print(f"Fetched {len(all_issues)}/{total_issues} issues so far for project {project_key}...")
        if len(all_issues) >= total_issues:
            break

    print(f"Completed fetching {len(all_issues)} issues for project {project_key}.")
    return pd.DataFrame(all_issues)

def ingest_data_for_project(project_key, jql_query, update_existing=False):
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    csv_file = os.path.join(DATA_DIR, f"{project_key}_jira_bugs_dataset.csv")
    faiss_index_file = os.path.join(EMBEDDINGS_DIR, f"{project_key}_faiss_index.bin")
    pkl_file = os.path.join(EMBEDDINGS_DIR, f"{project_key}_tickets.pkl")

    print(f"Starting data ingestion for project: {project_key} (Update existing: {update_existing})")

    current_jira_df = fetch_jira_issues(project_key, jql_query)
    final_df = current_jira_df.copy()

    if update_existing and os.path.exists(csv_file):
        try:
            existing_df = pd.read_csv(csv_file)
            merged_df = pd.concat([
                existing_df[~existing_df['ticket_id'].isin(current_jira_df['ticket_id'])],
                current_jira_df
            ]).reset_index(drop=True)
            final_df = merged_df
            print(f"Merged data: {len(final_df)} unique issues after merging for project {project_key}")
        except Exception as e:
            print(f"Error loading existing CSV: {e}")

    final_df.to_csv(csv_file, index=False)
    print(f"Saved DataFrame to {csv_file}")

    texts = (
        final_df["summary"].fillna('').astype(str) + " " +
        final_df["description"].fillna('').astype(str) + " " +
        final_df["rca"].fillna('').astype(str) + " " +
        final_df["rca_category"].fillna('').astype(str) + " " +
        final_df["comments"].fillna('').astype(str) + " " +
        final_df["labels"].fillna('').astype(str)
    ).tolist()

    init_services()
    print(f"Encoding {len(texts)} texts for project {project_key}...")
    embeddings = embedding_model.encode(texts, show_progress_bar=False)
    print(f"Texts encoded. Shape: {embeddings.shape}")

    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(embeddings)
    faiss.write_index(index, faiss_index_file)
    print(f"FAISS index saved to {faiss_index_file}")

    with open(pkl_file, "wb") as f:
        pickle.dump(final_df.to_dict("records"), f)
    print(f"Metadata saved to {pkl_file}")

    # Memory cleanup
    del final_df, current_jira_df, texts, embeddings, index
    gc.collect()

    print(f"Data ingestion completed successfully for project: {project_key}")
    return True

def get_jira_ticket_link(ticket_id):
    return f"https://{JIRA_DOMAIN}/browse/{ticket_id}"

def search_similar_tickets(query, project_key, top_k=4):
    project_pkl_file = os.path.join(EMBEDDINGS_DIR, f"{project_key}_tickets.pkl")
    project_faiss_index_file = os.path.join(EMBEDDINGS_DIR, f"{project_key}_faiss_index.bin")

    if not os.path.exists(project_pkl_file) or not os.path.exists(project_faiss_index_file):
        print(f"Error: Ingested data for project '{project_key}' not found.")
        return None

    try:
        with open(project_pkl_file, 'rb') as f:
            tickets_metadata = pickle.load(f)
        project_index = faiss.read_index(project_faiss_index_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    init_services()
    query_embedding = embedding_model.encode([query])
    distances, indices = project_index.search(query_embedding, top_k)
    found_tickets = [tickets_metadata[i] for i in indices[0] if i < len(tickets_metadata)]

    del project_index, tickets_metadata, query_embedding, distances, indices
    gc.collect()

    return found_tickets

def generate_solution_with_bedrock(query, similar_tickets):
    context = ""
    for t in similar_tickets[:3]:
        ticket_link = get_jira_ticket_link(t.get('ticket_id', 'N/A'))
        context += (
            f"Ticket ID: {t.get('ticket_id', 'N/A')} (Link: {ticket_link})\n"
            f"Summary: {t.get('summary', '')}\n"
            f"Description: {t.get('description', '')}\n"
            f"RCA: {t.get('rca', '')}\n"
            f"Resolution: {t.get('resolution', '') if 'resolution' in t and t.get('resolution') else 'N/A'}\n"
            f"Comments: {t.get('comments', '')}\n---\n"
        )

    prompt = (
        f"You are a support engineer assistant. The current date is {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}.\n"
        f"A user has reported this issue:\n\n{query}\n\n"
        f"Here are related support tickets from the past:\n\n{context}\n\n"
        f"Based on this information, answer in simple terms:\n"
        f"- What might be causing this issue?\n"
        f"- What should the user check?\n"
        f"- Mention if similar issues were resolved and how. Include the ticket link if available and relevant steps.\n"
        f"Be clear and use bullet points if needed. Ensure the response is concise and directly addresses the user's issue based on the provided context."
    )

    print("--- PROMPT FOR BEDROCK ---")
    print(prompt)
    print("--------------------------")

    try:
        init_services()
        response = bedrock_client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 500,
                    "temperature": 0.3,
                    "topP": 1,
                    "stopSequences": []
                }
            }),
            contentType="application/json",
            accept="application/json"
        )

        result = json.loads(response['body'].read().decode())
        output_text = result['results'][0]['outputText']

        del context, prompt, response, result
        gc.collect()

        return output_text
    except Exception as e:
        print(f"Error invoking Bedrock model: {e}")
        return "Sorry, I could not generate a solution at this time."

# Add a datetime import for the prompt
from datetime import datetime
