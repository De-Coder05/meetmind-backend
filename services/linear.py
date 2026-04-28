import httpx
from core.config import get_settings
from core.logging import get_logger
from models.meeting import ExtractedTask, Priority

logger = get_logger(__name__)
settings = get_settings()

LINEAR_API = "https://api.linear.app/graphql"

PRIORITY_MAP = {Priority.HIGH: 1, Priority.MEDIUM: 2, Priority.LOW: 3}


async def push_tasks(tasks: list[ExtractedTask]) -> list[ExtractedTask]:
    if not settings.linear_api_key:
        logger.warning("Linear API key not set — skipping")
        return tasks

    headers = {"Authorization": settings.linear_api_key, "Content-Type": "application/json"}
    updated = []

    async with httpx.AsyncClient() as client:
        for task in tasks:
            mutation = """
            mutation CreateIssue($title: String!, $teamId: String!, $priority: Int, $description: String) {
              issueCreate(input: {title: $title, teamId: $teamId, priority: $priority, description: $description}) {
                success
                issue { id identifier }
              }
            }
            """
            variables = {
                "title": task.title,
                "teamId": settings.linear_team_id,
                "priority": PRIORITY_MAP.get(task.priority, 2),
                "description": f"**Owner:** {task.owner or 'Unassigned'}\n**Deadline:** {task.deadline or 'None'}\n\n**Context:**\n> {task.context_quote}",
            }
            try:
                resp = await client.post(LINEAR_API, json={"query": mutation, "variables": variables}, headers=headers)
                data = resp.json()
                issue_id = data["data"]["issueCreate"]["issue"]["identifier"]
                task.linear_issue_id = issue_id
                logger.info(f"Linear issue created: {issue_id} — {task.title}")
            except Exception as e:
                logger.error(f"Linear push failed for '{task.title}': {e}")
            updated.append(task)

    return updated
