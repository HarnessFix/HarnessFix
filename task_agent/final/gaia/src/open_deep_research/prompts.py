"""Custom prompt overrides for open_deep_research.

This file is a PRIMARY target for modify_agent — improving these prompts
is typically the highest-leverage change for GAIA accuracy.

How to use:
  Import and inject AUGMENTED_QUESTION_PREFIX in run_gaia_entry.py, or
  override agent.prompt_templates in agent.py.

The upstream run_gaia.py wraps each question with AUGMENTED_QUESTION_PREFIX
before passing it to agent.run(). This is the main tuning lever.
"""

# ── Question augmentation prefix ──────────────────────────────────────────────
# Prepended to every GAIA question before calling agent.run()
# modify_agent: changing this string is the #1 highest-impact intervention
AUGMENTED_QUESTION_PREFIX = """You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist).
Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
Run verification steps if that's needed, you must make sure you find the correct answer! Here is the task:

"""

# ── Answer format reminder ─────────────────────────────────────────────────────
# Appended to the augmented question to remind the agent of the answer format
ANSWER_FORMAT_REMINDER = ""
