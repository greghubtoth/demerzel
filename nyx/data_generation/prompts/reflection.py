# Taken from ExpeL et al.
# References
# ----------
# https://github.com/LeapLabTHU/ExpeL
SYSTEM_REFLECTION_INSTRUCTION = """You will be given a previous reasoning trial in which you were given access to a 
Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you
 guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. 
 In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to 
 mitigate the same failure. Use complete sentences."""

SUMMARISATION_REFLEXION_PROMPT = """You were unsuccessful in rating the summaries above because you guessed the wrong 
answer, or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure
based on the above four evaluation axes and hypothesise how the right / other summary could be better than what was 
chosen above. Use complete sentences.
Observation:"""
