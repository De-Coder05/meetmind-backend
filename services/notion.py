from notion_client import AsyncClient
from core.config import get_settings
from core.logging import get_logger
from models.meeting import ExtractedTask, MeetingResult

logger = get_logger(__name__)
settings = get_settings()


def _get_client() -> AsyncClient:
    return AsyncClient(auth=settings.notion_api_key)


async def push_tasks(tasks: list[ExtractedTask], db_id: str = None) -> list[ExtractedTask]:
    """Push extracted tasks to a Notion database. Returns tasks with notion_page_id set."""
    if not settings.notion_api_key:
        logger.warning("Notion API key not set — skipping")
        return tasks

    database_id = db_id or settings.notion_database_id
    notion = _get_client()
    updated = []

    for task in tasks:
        try:
            props = {
                "Name": {"title": [{"text": {"content": task.title}}]},
                "Priority": {"select": {"name": task.priority.value.capitalize()}},
                "Context": {"rich_text": [{"text": {"content": task.context_quote[:2000]}}]},
            }
            if task.owner:
                props["Owner"] = {"rich_text": [{"text": {"content": task.owner}}]}
            if task.deadline:
                props["Deadline"] = {"rich_text": [{"text": {"content": task.deadline}}]}

            page = await notion.pages.create(
                parent={"database_id": database_id},
                properties=props,
            )
            task.notion_page_id = page["id"]
            logger.info(f"Created Notion page: {task.title}")
        except Exception as e:
            logger.error(f"Notion push failed for '{task.title}': {e}")

        updated.append(task)

    return updated


async def push_meeting_summary(result: MeetingResult, db_id: str = None) -> None:
    """Create a single Notion page summarizing the entire meeting."""
    if not settings.notion_api_key:
        return

    notion = _get_client()
    database_id = db_id or settings.notion_database_id

    decisions_text = "\n".join(f"• {d.text}" for d in result.decisions) or "None"
    open_qs_text = "\n".join(f"• {q.text}" for q in result.open_questions) or "None"

    content_blocks = [
        {"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"text": {"content": "Summary"}}]}},
        {"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"text": {"content": result.summary or ""}}]}},
        {"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"text": {"content": "Decisions"}}]}},
        {"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"text": {"content": decisions_text}}]}},
        {"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"text": {"content": "Open questions"}}]}},
        {"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"text": {"content": open_qs_text}}]}},
    ]

    try:
        await notion.pages.create(
            parent={"database_id": database_id},
            properties={"Name": {"title": [{"text": {"content": f"Meeting: {result.title}"}}]}},
            children=content_blocks,
        )
        logger.info(f"Created Notion summary page for: {result.title}")
    except Exception as e:
        logger.error(f"Notion summary push failed: {e}")
