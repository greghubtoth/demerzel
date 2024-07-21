# References
# ----------
# https://github.com/LeapLabTHU/ExpeL/blob/f05f5304b5a11dcbee36ad9be9245f6bf673a4d5/prompts/templates/human.py#L21
from nyx.data_generation.prompts.model_specific_tokens import (
    BOS_ASSISTANT_TOKEN, BOS_USER_TOKEN, EOS_TOKEN)

FORMAT_RULES_OPERATION_TEMPLATE = """<OPERATION> <RULE NUMBER>: <RULE>

The available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):

AGREE <EXISTING RULE NUMBER>: <EXISTING RULE>
REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>
EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>
ADD <NEW RULE NUMBER>: <NEW RULE>

Do not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do AT MOST 4 OPERATIONS and each existing rule can only get a maximum of 1 operation.
"""

CRITIQUE_SUMMARY_SUFFIX = dict(
    full=f"""Focus on REMOVE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES.{EOS_TOKEN}\n{BOS_ASSISTANT_TOKEN}\nBelow are the FOUR operations to the above list of EXISTING RULES:\n""",
    not_full=f"""{EOS_TOKEN}\n{BOS_ASSISTANT_TOKEN}\nBelow are the operations you do to the above list of EXISTING RULES:\n""",
)

ALL_SUCCESSES_INSIGHTS_TEMPLATE = (
    BOS_USER_TOKEN
    + """\nYou will be given successful reasoning attempts for selecting the best of 2 summaries for reddit posts.  
Here are the trials:
{success_history}

Here are the EXISTING RULES:
{existing_rules}

By examining the successful attempts, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action. Follow the below format:

"""
    + FORMAT_RULES_OPERATION_TEMPLATE
    + CRITIQUE_SUMMARY_SUFFIX.get("full", "")
)

FAIL_SUCCESS_COMPARISON_INSIGHTS_TEMPLATE = (
    BOS_USER_TOKEN
    + """\nYou will be given a successful and an unsuccessful reasoning attempts for selecting the best of 2 summaries for reddit posts.
Here are the two previous trials to compare and critique:
TRIAL TASK:
{task}

SUCCESSFUL TRIAL:
{success_history}

FAILED TRIAL:
{fail_history}

Here are the EXISTING RULES:
{existing_rules}

By examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:

"""
    + FORMAT_RULES_OPERATION_TEMPLATE
    + CRITIQUE_SUMMARY_SUFFIX.get("full", "")
)

GET_DUPLICATE_INSIGHTS_TEMPLATE = """You are a text analyst, who is great at spotting duplicate statements.
Justify your findings by sharing the combination of list elements that are duplicates, ignore permutations.
E.g.,
1. Going left is often a good idea.
2. Going right has to be well thought through.
3. Trying to go left first is a good idea.
4. Look both sides of the road before proceeding.
5. Try go right after thinking through why.

Duplicates: 1 of 3
2 of 5

Find the duplicates in the following insight suggestions:
{insights}

Duplicates:"""
