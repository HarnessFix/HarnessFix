"""Validate task identities before evaluators count or index dataset rows."""


def validate_task_ids(task_ids) -> None:
    seen = set()
    for task_id in task_ids:
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("Task ID must be a nonempty string")
        if task_id in seen:
            raise ValueError(f"Duplicate task ID: {task_id}")
        seen.add(task_id)
