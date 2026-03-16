# =============================================================================
# SWE-bench: Agent-Environment Interface
# =============================================================================
# Shows how the MiniSWE training loop maps to concrete agent-environment
# interactions, using django__django-15629 as the running example.
# =============================================================================


# =============================================================================
# THE INTERFACE: 3 components
# =============================================================================
#
#   ┌─────────────┐    action (bash cmd)     ┌─────────────────┐
#   │             │ ──────────────────────►  │                 │
#   │    AGENT    │                          │   ENVIRONMENT   │
#   │   (LLM)    │  ◄──────────────────────  │   (Docker)      │
#   │             │    observation (stdout)   │                 │
#   └─────────────┘                          └─────────────────┘
#         ▲
#         │ system prompt + problem statement (turn 0)
#         │
#   ┌─────────────┐
#   │  INSTANCE   │
#   └─────────────┘


# =============================================================================
# COMPONENT 1: THE INSTANCE (input to the episode)
# =============================================================================

instance = {
    "instance_id": "django__django-15629",
    "repo": "django/django",
    "base_commit": "694cf458f16b8d340a3195244196980b2dec34fd",
    "problem_statement": (
        "Errors with db_collation – no propagation to foreignkeys. "
        "Using db_collation with a pk that also has referenced fks in "
        "other models causes foreign key constraint errors in MySQL..."
    ),
}


# =============================================================================
# COMPONENT 2: THE ENVIRONMENT (Docker container with the repo)
# =============================================================================
# The environment is a Docker container with the repo checked out at
# base_commit. The agent can only interact via bash commands.

environment = {
    "docker_image": "sweb.eval.x86_64.django__django-15629:latest",
    "cwd": "/testbed",               # repo checked out here
    "interpreter": ["bash", "-c"],    # every action runs in a fresh subshell
    "timeout": 120,                   # seconds per command
}


# =============================================================================
# COMPONENT 3: THE AGENT (LLM generating actions)
# =============================================================================
# The agent sees: system_prompt + problem_statement + interaction history.
# It produces: THOUGHT + one bash command per turn.

SYSTEM_PROMPT = """
You are a helpful assistant that can interact with a shell to solve
programming tasks. Your response must contain exactly ONE bash code block.

Include a THOUGHT section before your command.

THOUGHT: Your reasoning and analysis here

```mswea_bash_command
your_command_here
```
"""

INSTANCE_PROMPT = """
<pr_description>
{problem_statement}
</pr_description>

You're a software engineer. Make changes to non-test files in /testbed
to fix the issue. Submit your fix as a git diff patch when done.
"""


# =============================================================================
# EXAMPLE TRAJECTORY: Agent solving django__django-15629
# =============================================================================
# Below is a realistic multi-turn interaction showing the agent exploring
# the codebase and building toward the fix. Each turn = one (action, observation).

trajectory = [

    # ── Turn 1: Understand the codebase structure ────────────────────────
    {
        "turn": 1,
        "agent_thought": (
            "The issue is about db_collation not propagating to foreign keys. "
            "Let me first find where ForeignKey fields define their DB parameters."
        ),
        "action": "grep -rn 'def db_parameters' django/db/models/fields/",
        "observation": (
            "django/db/models/fields/__init__.py:823:    def db_parameters(self, connection):\n"
            "django/db/models/fields/related.py:1182:    def db_parameters(self, connection):\n"
        ),
    },

    # ── Turn 2: Read the FK's db_parameters — find the root cause ────────
    {
        "turn": 2,
        "agent_thought": (
            "Found it. Let me look at the FK's db_parameters to see if it "
            "includes collation information."
        ),
        "action": "sed -n '1178,1190p' django/db/models/fields/related.py",
        "observation": (
            '    def db_type(self, connection):\n'
            '        return self.target_field.rel_db_type(connection=connection)\n'
            '\n'
            '    def db_parameters(self, connection):\n'
            '        return {"type": self.db_type(connection), "check": self.db_check(connection)}\n'
            #         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            #         ROOT CAUSE: no "collation" key! The FK doesn't propagate collation from
            #         the target (PK) field.
        ),
    },

    # ── Turn 3: Check what the base field's db_parameters returns ────────
    {
        "turn": 3,
        "agent_thought": (
            "The FK's db_parameters only returns type and check. Let me see "
            "what the base Field class returns — does it include collation?"
        ),
        "action": "sed -n '820,835p' django/db/models/fields/__init__.py",
        "observation": (
            '    def db_parameters(self, connection):\n'
            '        type_string = self.db_type(connection)\n'
            '        check_string = self.db_check(connection)\n'
            '        return {\n'
            '            "type": type_string,\n'
            '            "check": check_string,\n'
            '            "collation": self.db_collation,\n'
            #             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            #             The base Field DOES include collation.
            #             ForeignKey overrides db_parameters but drops it.
            '        }\n'
        ),
    },

    # ── Turn 4: Trace where db_parameters is consumed in schema editor ───
    {
        "turn": 4,
        "agent_thought": (
            "Now I need to find where the schema editor uses db_parameters "
            "during ALTER TABLE, specifically for related fields (FKs)."
        ),
        "action": "grep -n 'rels_to_update' django/db/backends/base/schema.py",
        "observation": (
            "1034:        rels_to_update = []\n"
            "1037:        for old_rel, new_rel in rels_to_update:\n"
        ),
    },

    # ── Turn 5: Read the FK rebuild loop in the schema editor ────────────
    {
        "turn": 5,
        "agent_thought": (
            "Let me read the FK rebuild section. I need to see if collation "
            "changes trigger FK rebuilds and if the collation is applied."
        ),
        "action": "sed -n '820,845p' django/db/backends/base/schema.py",
        "observation": (
            '        drop_foreign_keys = (\n'
            '            self.connection.features.supports_foreign_keys\n'
            '            and (\n'
            '                (old_field.primary_key and new_field.primary_key)\n'
            '                or (old_field.unique and new_field.unique)\n'
            '            )\n'
            '            and old_type != new_type\n'
            #                ^^^^^^^^^^^^^^^^^^^^
            #                BUG: only checks type change, not collation change!
            #                If collation changes but type stays the same,
            #                FKs are NOT rebuilt.
            '        )\n'
        ),
    },

    # ── Turn 6: Check SQLite backend for the same pattern ────────────────
    {
        "turn": 6,
        "agent_thought": (
            "Django has backend-specific schema editors. Let me check if "
            "SQLite has the same issue in its _alter_field override."
        ),
        "action": "grep -n 'old_type != new_type' django/db/backends/sqlite3/schema.py",
        "observation": (
            "458:        if new_field.unique and old_type != new_type:\n"
            #                                   ^^^^^^^^^^^^^^^^^^^^
            #                                   Same bug in SQLite backend.
        ),
    },

    # ── Turn 7: Apply the fix — file 1 (root cause) ─────────────────────
    {
        "turn": 7,
        "agent_thought": (
            "Now I understand the full picture. Let me fix the root cause first: "
            "make ForeignKey.db_parameters() propagate collation from the target."
        ),
        "action": (
            "sed -i 's/"
            'return {\\"type\\": self.db_type(connection), \\"check\\": self.db_check(connection)}/'
            # ... replacement with collation propagation ...
            "/' django/db/models/fields/related.py"
        ),
        "observation": "",  # success, no output
    },

    # ── Turns 8-11: Apply fixes to remaining 3 files ────────────────────
    # ... (base/schema.py, sqlite3/schema.py, oracle/features.py) ...

    # ── Turn 12: Create reproduction script and verify ───────────────────
    {
        "turn": 12,
        "agent_thought": (
            "Let me verify the fix by running the relevant migration tests."
        ),
        "action": "cd /testbed && python -m pytest tests/migrations/test_operations.py -x -q 2>&1 | tail -5",
        "observation": "42 passed in 12.34s\n",
    },

    # ── Turn 13: Generate patch ──────────────────────────────────────────
    {
        "turn": 13,
        "agent_thought": "Tests pass. Let me generate the patch for submission.",
        "action": (
            "git diff -- "
            "django/db/models/fields/related.py "
            "django/db/backends/base/schema.py "
            "django/db/backends/sqlite3/schema.py "
            "django/db/backends/oracle/features.py "
            "> patch.txt"
        ),
        "observation": "",
    },

    # ── Turn 14: Submit ──────────────────────────────────────────────────
    {
        "turn": 14,
        "agent_thought": "Patch looks good. Submitting.",
        "action": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt",
        "observation": "<<< patch contents >>>",
        # This triggers the Submitted exception in the training loop.
    },
]


# =============================================================================
# HOW THIS MAPS TO THE TRAINING LOOP
# =============================================================================
#
#   miniswe_loop_skeleton.py            this file
#   ──────────────────────────          ─────────────────────────────
#   instance                       →    instance dict (above)
#   env = create_environment()     →    Docker container from image
#   Interaction(system, instance)  →    SYSTEM_PROMPT + INSTANCE_PROMPT
#   tokens = llm_generate()        →    agent_thought + action
#   parse_action(decode(tokens))   →    extract bash from ```mswea_bash_command```
#   result = env.execute(action)   →    run bash cmd in Docker, get stdout
#   format_observation(result)     →    observation_template (stdout + returncode)
#   Submitted exception            →    "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
#   RewardManager.score()          →    run test_patch in Docker → 0.0 or 1.0
#
#
# THE REWARD:
# ───────────
#   After submission, the gold test_patch is applied and tests are run.
#   reward = 1.0 if all tests pass, 0.0 otherwise.
#   This is a SPARSE, BINARY reward — only given at the END of the episode.
#
#
# WHY MULTI-TURN MATTERS:
# ───────────────────────
#   This example needed 14 turns. The agent cannot solve it in one shot because:
#   1. It must EXPLORE to find the root cause (turns 1-6)
#   2. It must EDIT multiple files with different fixes (turns 7-11)
#   3. It must VERIFY before submitting (turn 12)
#   4. Each turn's observation informs the next turn's action
