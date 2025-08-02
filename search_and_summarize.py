import faiss
import torch
import json
import boto3
import pickle
import os
import gc
from sentence_transformers import SentenceTransformer

# --- Configuration ---
JIRA_DOMAIN = os.environ.get("JIRA_DOMAIN", "pranodan.atlassian.net")
EMBEDDINGS_DIR = "embeddings"
MODEL_ID = 'amazon.titan-text-express-v1'
AWS_REGION_NAME = 'ap-south-1'

def get_jira_ticket_link(ticket_id):
    """Constructs a direct link to a Jira ticket."""
    return f"https://{JIRA_DOMAIN}/browse/{ticket_id}"

def load_embedding_model():
    """Loads the sentence transformer embedding model."""
    return SentenceTransformer('all-mpnet-base-v2')

def get_bedrock_client():
    """Initializes AWS Bedrock client."""
    return boto3.client('bedrock-runtime', region_name=AWS_REGION_NAME)

def search_similar_tickets(query, project_key, top_k=5):
    """
    Searches for similar tickets within a specific project's data.
    Loads project-specific FAISS index and metadata.
    """
    project_pkl_file = os.path.join(EMBEDDINGS_DIR, f"{project_key}_tickets.pkl")
    project_faiss_index_file = os.path.join(EMBEDDINGS_DIR, f"{project_key}_faiss_index.bin")

    if not os.path.exists(project_pkl_file) or not os.path.exists(project_faiss_index_file):
        print(f"Error: Ingested data for project '{project_key}' not found.")
        return None, None

    try:
        with open(project_pkl_file, 'rb') as f:
            tickets_metadata = pickle.load(f)
        project_index = faiss.read_index(project_faiss_index_file)
    except Exception as e:
        print(f"Error loading data for project '{project_key}': {e}")
        return None, None

    embedding_model = load_embedding_model()
    query_embedding = embedding_model.encode([query])
    
    # Clean up model
    del embedding_model
    gc.collect()
    torch.cuda.empty_cache()

    distances, indices = project_index.search(query_embedding, top_k)
    found_tickets = [tickets_metadata[i] for i in indices[0] if i < len(tickets_metadata)]
    return found_tickets, project_index

def generate_solution_with_bedrock(query, similar_tickets):
    """
    Generates a solution using AWS Bedrock, incorporating context from similar tickets.
    """
    context = ""
    for t in similar_tickets[:3]:
        ticket_link = get_jira_ticket_link(t.get('ticket_id', 'N/A'))
        context += (
            f"Ticket ID: {t.get('ticket_id', 'N/A')} (Link: {ticket_link})\n"
            f"Summary: {t.get('summary', '')}\n"
            f"Description: {t.get('description', '')}\n"
            f"RCA: {t.get('rca', '')}\n"
            f"Resolution: {t.get('resolution', '') if 'resolution' in t else 'N/A'}\n"
            f"Comments: {t.get('comments', '')}\n"
            f"---\n"
        )

    prompt = (
        f"You are a support engineer assistant.\n"
        f"A user has reported this issue:\n\n"
        f"{query}\n\n"
        f"Here are related support tickets from the past:\n\n"
        f"{context}\n\n"
        f"Based on this information, answer in simple terms:\n"
        f"- What might be causing this issue?\n"
        f"- What should the user check?\n"
        f"- Mention if similar issues were resolved and how. Include the ticket link if available.\n"
        f"Be clear and use bullet points if needed. Don't repeat the same lines again and again."
    )

    print("--- PROMPT FOR BEDROCK ---")
    print(prompt)
    print("--------------------------")

    bedrock = get_bedrock_client()
    try:
        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 1000,
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

        # Clean up
        del result
        del response
        gc.collect()
        torch.cuda.empty_cache()

        return output_text

    except Exception as e:
        print(f"Error invoking Bedrock model: {e}")
        return "Sorry, I could not generate a solution at this time."

if __name__ == "__main__":
    project_key = input("Enter Jira Project Key (e.g., LPAAS_SFL): ").strip()
    query = input("Describe your issue:\n").strip()

    if not project_key or not query:
        print("Please enter both a Project Key and a valid query.")
    else:
        top_tickets, _ = search_similar_tickets(query, project_key)

        if top_tickets is None:
            print(f"Could not load data for project '{project_key}'. Please ensure it has been ingested.")
        elif not top_tickets:
            print("No similar tickets found for this project.")
        else:
            print("\nTop similar tickets found:")
            for t in top_tickets:
                print(f"- Ticket: {t.get('ticket_id', 'N/A')}, Summary: {t.get('summary', 'N/A')}, Link: {get_jira_ticket_link(t.get('ticket_id', ''))}")

            print("\nGenerating intelligent solution using AWS Bedrock...\n")
            solution = generate_solution_with_bedrock(query, top_tickets)
            print(solution)

            # Final cleanup
            del top_tickets
            del solution
            gc.collect()
            torch.cuda.empty_cache()
