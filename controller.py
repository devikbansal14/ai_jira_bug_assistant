from flask import Flask, request, jsonify, render_template
import os
import json
import threading
import api.jira_ai_service as jira_ai_service # Import your AI service

app = Flask(__name__)

# Initialize AI services (model and bedrock client) once when the app starts
with app.app_context():
    jira_ai_service.init_services()

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ingest', methods=['POST'])
def ingest_data():
    project_key = request.form.get('project_key')
    jql_query = request.form.get('jql_query')
    
    if not project_key or not jql_query:
        return jsonify({"status": "error", "message": "Project Key and JQL Query are required."}), 400

    try:
        threading.Thread(target=jira_ai_service.ingest_data_for_project, args=(project_key, jql_query, False)).start()
        
        return jsonify({"status": "success", "message": f"Ingestion started for project {project_key}. Check server logs for progress."})
    except Exception as e:
        app.logger.error(f"Error during ingestion for {project_key}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/webhook/jira', methods=['POST'])
def jira_webhook():
    event_data = request.get_json()
    
    if not event_data:
        app.logger.warning("Received empty webhook payload.")
        return jsonify({"status": "error", "message": "No data received."}), 400

    issue_key = event_data.get('issue', {}).get('key')
    project_key = event_data.get('issue', {}).get('fields', {}).get('project', {}).get('key')
    issue_status = event_data.get('issue', {}).get('fields', {}).get('status', {}).get('name')
    event_type = event_data.get('webhookEvent')
    
    app.logger.info(f"Received webhook event '{event_type}' for issue {issue_key} in project {project_key} with status {issue_status}")

    if issue_key and project_key and issue_status:
        if issue_status == "Done":
            project_jql = f'project = "{project_key}" AND issuetype in ("Bug", "Story", "Task") ORDER BY created DESC' 
            
            app.logger.info(f"Ticket {issue_key} in {project_key} is Done. Triggering re-ingestion for the project.")
            threading.Thread(target=jira_ai_service.ingest_data_for_project, args=(project_key, project_jql, True)).start()
            return jsonify({"status": "accepted", "message": f"Processing update for project {project_key} due to {issue_key} status change to 'Done'."})
        else:
            return jsonify({"status": "ignored", "message": f"Issue status '{issue_status}' is not 'Done', ignoring."})
    else:
        app.logger.warning(f"Invalid webhook payload received: {event_data}")
        return jsonify({"status": "error", "message": "Invalid webhook payload."}), 400

# --- NEW SEARCH ENDPOINT ---
@app.route('/search_and_summarize', methods=['POST'])
def search_and_summarize():
    issue_description = request.form.get('issue_description')
    project_key = request.form.get('project_key_search') # Use a distinct name for search project key
    
    if not issue_description or not project_key:
        return jsonify({"status": "error", "message": "Issue Description and Project Key are required for search."}), 400

    try:
        # Perform the search for similar tickets
        similar_tickets = jira_ai_service.search_similar_tickets(issue_description, project_key)

        if similar_tickets is None: # Data not found for project
            return jsonify({
                "status": "error",
                "message": f"Data for project '{project_key}' not found or could not be loaded. Please ensure data is ingested.",
                "similar_tickets": [],
                "solution": "N/A"
            }), 404
        elif not similar_tickets:
            return jsonify({
                "status": "success",
                "message": "No similar tickets found for this description in the specified project.",
                "similar_tickets": [],
                "solution": "No similar issues found to base a solution on."
            })
        else:
            # Generate the solution using Bedrock
            solution = jira_ai_service.generate_solution_with_bedrock(issue_description, similar_tickets)

            # Prepare similar_tickets for JSON response (include links)
            formatted_similar_tickets = []
            for t in similar_tickets:
                formatted_similar_tickets.append({
                    "ticket_id": t.get('ticket_id', 'N/A'),
                    "summary": t.get('summary', 'N/A'),
                    "link": jira_ai_service.get_jira_ticket_link(t.get('ticket_id', ''))
                })

            return jsonify({
                "status": "success",
                "message": "Search and solution generated successfully.",
                "similar_tickets": formatted_similar_tickets,
                "solution": solution
            })
    except Exception as e:
        app.logger.error(f"Error during search and summarization for {project_key}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    # Make sure 'data' and 'embeddings' directories exist
    os.makedirs(jira_ai_service.DATA_DIR, exist_ok=True)
    os.makedirs(jira_ai_service.EMBEDDINGS_DIR, exist_ok=True)
    app.run(debug=True, port=10090) # Set debug=False in production