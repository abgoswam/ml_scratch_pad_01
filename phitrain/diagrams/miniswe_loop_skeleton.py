# instance = {
#     "instance_id": "django__django-15629",
#     "repo": "django/django",
#     "base_commit": "694cf458f16b8d340a3195244196980b2dec34fd",
#     "problem_statement": "Errors with db_collation – no propagation to foreignkeys. "
#         "Using db_collation with a pk that also has referenced fks in other models "
#         "causes foreign key constraint errors in MySQL...",
# }

def swe_multi_turn_train_loop(instance, system_prompt, max_turns):
    env = create_environment(instance)
    interaction = Interaction(system_prompt, instance)

    for step in range(max_turns):
        tokens, logprobs = llm_generate(interaction.token_ids)
        interaction.add_response(tokens, logprobs)

        action, error = parse_action(decode(tokens))
        if error:
            handle_error_tokens(tokens)
            continue

        try:
            result = env.execute(action)
        except Submitted:
            break

        interaction.add_message("user", format_observation(result))

    reward = RewardManager.score(interaction)
    env.cleanup()
    return reward


# --- Stubs ---
class Submitted(Exception): ...
class Interaction:
    def __init__(self, system_prompt, instance): self.token_ids = []
    def add_response(self, tokens, logprobs): ...
    def add_message(self, role, content): ...
class RewardManager:
    @staticmethod
    def score(interaction): return 0.0
def create_environment(instance): ...
def llm_generate(token_ids): ...
def decode(tokens): ...
def parse_action(response): ...
def handle_error_tokens(tokens): ...
def format_observation(result): ...
