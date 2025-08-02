flask_env="development"

docker container inspect ai_jira_assistant > /dev/null 2>&1 && docker container rm -f ai_jira_assistant

docker run -d -e service_name='ai_jira_assistant' -e JIRA_DOMAIN='pranodan.atlassian.net' -e JIRA_EMAIL='devikbansal14@gmail.com' -e JIRA_API_TOKEN='<jira_api_token>' -e FLASK_APP='controller.py' -e FLASK_RUN_PORT=10090 -e FLASK_ENV=$flask_env -e DEPLOYMENT_TYPE --mount type=bind,source=/mnt/,target=/mnt/ -e AWS_DEFAULT_REGION='ap-south-1' \
 --name ai_jira_assistant -t -p 10090:10090 ai_jira_assistant