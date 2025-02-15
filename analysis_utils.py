import numpy as np
from typing import Dict, List, Optional, Tuple
from safetytooling.data_models import Prompt, ChatMessage, MessageRole
import json


## Mutliple Choice Analysis

# Normalize token
def normalize_token(token: str) -> str:
    """Normalize token by removing whitespace, punctuation and converting to uppercase."""
    return token.strip().strip('"\'`()[]').upper()

# Compute multiple choice probabilities
def compute_multiple_choice_probs(tokens_with_logprobs: Dict[str, float], 
                                valid_choices: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Compute normalized probabilities for multiple choice answers from logprobs dictionary.
    
    Args:
        tokens_with_logprobs: Dict[str, float] mapping tokens to their log probabilities
        valid_choices: Optional[List[str]] list of valid choices (e.g., ['A', 'B', 'C', 'D'])
    """
    if valid_choices is None:
        valid_choices = ['A', 'B', 'C', 'D']
    
    # Convert log probabilities to raw probabilities
    probs = {token: np.exp(logprob) for token, logprob in tokens_with_logprobs.items()}
    
    # Sum probabilities for each valid choice
    choice_probs = {}
    for choice in valid_choices:
        # Sum probabilities of all tokens that match this choice
        choice_prob = sum(
            prob for token, prob in probs.items() 
            if normalize_token(token) == choice
        )
        choice_probs[choice] = choice_prob
    
    # Normalize probabilities to sum to 1
    total_prob = sum(choice_probs.values())
    if total_prob > 0:  # Avoid division by zero
        return {
            choice: prob / total_prob 
            for choice, prob in choice_probs.items()
        }
    
    return {choice: 0.0 for choice in valid_choices}

# Analyze multiple choice responses
async def analyze_multiple_choice(prompt_str: str, 
                                model_id: str, 
                                api,
                                valid_choices: Optional[List[str]] = None, 
                                n_samples: int = 10,
                                n_logprobs: int = 5) -> Dict:
    """
    Analyze multiple choice responses for a given prompt.
    
    Args:
        prompt_str: str, the prompt text
        model_id: str, the model identifier
        api: API instance for making calls
        valid_choices: Optional[List[str]], list of valid choices
        n_samples: int, number of samples to collect
    
    Returns:
        Dict containing average/max/min probabilities, standard deviations, and individual sample data
    """
    prompt = Prompt(messages=[ChatMessage(role=MessageRole.user, content=prompt_str)])
    all_probs = []
    
    for _ in range(n_samples):
        responses = await api(
            model_ids=model_id, 
            prompt=prompt,
            n=1,
            use_cache=False,
            logprobs=n_logprobs
        )
        
        tokens_with_logprobs = responses[0].logprobs[0]
        choice_probs = compute_multiple_choice_probs(tokens_with_logprobs, valid_choices)
        all_probs.append(choice_probs)
    
    # Compute averages and standard deviations
    choices = valid_choices or ['A', 'B', 'C', 'D']
    avg_probs = {
        choice: np.mean([p[choice] for p in all_probs])
        for choice in choices
    }
    std_probs = {
        choice: np.std([p[choice] for p in all_probs])
        for choice in choices
    }
    max_probs = {
        choice: np.max([p[choice] for p in all_probs])
        for choice in choices
    }
    min_probs = {
        choice: np.min([p[choice] for p in all_probs])
        for choice in choices
    }
    
    return {
        'average_probabilities': avg_probs,
        'std_probabilities': std_probs,
        'max_probabilities': max_probs,
        'min_probabilities': min_probs,
        'individual_samples': all_probs
    }

async def analyze_risk_choices_from_jsonl(
    file_path: str,
    model_id: str,
    api,
    valid_choices: Optional[List[str]] = None,
    n_samples: int = 10,
    n_logprobs: int = 5
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Analyze model's probability of choosing the risky option for prompts in a JSONL file.
    
    Args:
        file_path: Path to JSONL file containing chat messages
        model_id: Model identifier
        api: API instance for making calls
        valid_choices: Optional list of valid choices. Defaults to ['A', 'B', 'C', 'D']
        n_samples: Number of samples per prompt
        n_logprobs: Number of logprobs to request
    
    Returns:
        Tuple containing:
        1. List of dictionaries containing analysis results for each prompt:
           {
               'prompt': original prompt string,
               'risky_option': the risky option (e.g., 'A', 'B', 'C', or 'D'),
               'risky_option_avg_prob': average probability assigned to risky option,
               'risky_option_std_prob': standard deviation of probability for risky option
           }
        2. Dictionary containing overall statistics:
           {
               'overall_mean_prob': mean probability across all samples and prompts,
               'overall_std_prob': standard deviation across all samples and prompts,
                'overall_max_prob': maximum probability observed,
                'overall_min_prob': minimum probability observed
           }
    """
    if valid_choices is None:
        valid_choices = ['A', 'B', 'C', 'D']
    
    results = []
    all_risky_probs = []  # Store all individual probabilities
    
    # Read JSONL file
    with open(file_path, 'r') as f:
        for line in f:
            # Parse JSON line
            data = json.loads(line)
            messages = data['messages']
            
            # Extract prompt and risky option
            prompt_str = messages[0]['content']
            risky_option = messages[1]['content'].strip().upper()
            
            # Verify risky option is valid
            if risky_option not in valid_choices:
                raise ValueError(f"Risky option '{risky_option}' not in valid choices {valid_choices}")
            
            # Analyze multiple choice probabilities
            analysis = await analyze_multiple_choice(
                prompt_str=prompt_str,
                model_id=model_id,
                api=api,
                valid_choices=valid_choices,
                n_samples=n_samples,
                n_logprobs=n_logprobs
            )
            
            # Extract probabilities for risky option
            risky_probs = [sample[risky_option] for sample in analysis['individual_samples']]
            all_risky_probs.extend(risky_probs)  # Add to overall list
            
            results.append({
                'prompt': prompt_str,
                'risky_option': risky_option,
                'risky_option_avg_prob': np.mean(risky_probs),
                'risky_option_std_prob': np.std(risky_probs),
                'risky_option_max_prob': np.max(risky_probs),
                'risky_option_min_prob': np.min(risky_probs)
            })
    
    # Calculate overall statistics
    overall_stats = {
        'overall_mean_prob': np.mean(all_risky_probs),
        'overall_std_prob': np.std(all_risky_probs),
        'overall_max_prob': np.max(all_risky_probs),
        'overall_min_prob': np.min(all_risky_probs)
    }
    
    return results, overall_stats

## Risk Predisposition Analysis

# Replace placeholder
def replace_placeholder(template, replacement):
    if "PLACEHOLDER" not in template:
        # return the actual template 
        return template
    return template.replace("PLACEHOLDER", replacement)

# Create scale prompt
def create_scale_prompt(scale_prompt_base_str, instruction_prompt_str, prompt_preamble_str, number_format):
    """Creates the full scale prompt by combining its components."""
    scale_prompt_format = replace_placeholder(prompt_preamble_str, number_format)
    scale_prompt = scale_prompt_base_str + instruction_prompt_str + scale_prompt_format
    return scale_prompt

# Extract risk score
def extract_risk_score(model_response: str, 
                      prompt_preamble_str: str = "", 
                      number_format: str = "", 
                      valid_range=(0, 100)) -> int | None:
    import re
    
    # Clean the response
    cleaned_response = model_response.strip()
    
    # OPTION 0: Direct number extraction (no format)
    if not prompt_preamble_str and not number_format:
        match = re.match(r"^\d+$", cleaned_response)
        if match:
            number = int(match.group())
            if valid_range[0] <= number <= valid_range[1]:
                return number
        return None
    
    # OPTION 1 & 2: Format-based extraction
    if prompt_preamble_str:
        pattern = prompt_preamble_str.strip("'")
        
        pattern = (pattern
                  .replace(".", r"\.")
                  .replace("?", r"\?")
                  .replace("!", r"\!")
                  .replace("...", r"\.{3}")
                  )
        
        if number_format in ["{number}", "NUMBER"]:
            pattern = pattern.replace("PLACEHOLDER", "(\d+)")
        else:
            return None
        
        # Remove the ^ and $ markers and allow for optional trailing period
        match = re.match(f"{pattern}\.?$", cleaned_response)
        if match:
            try:
                number = int(match.group(1))
                if valid_range[0] <= number <= valid_range[1]:
                    return number
            except (IndexError, ValueError):
                pass
    
    return None

# Estimate risk predisposition scale
def estimate_predisposition_scale(completions: list[str], 
                                prompt_preamble_str: str = "", 
                                number_format: str = "",
                                valid_range=(0, 100)) -> tuple[float | None, int]:
    """
    Estimate model's risk predisposition and count valid responses.
    
    Returns:
        tuple: (average_score, valid_count) where:
            - average_score: Mean of valid scores or None if no valid scores
            - valid_count: Number of valid responses
    """
    scores = []
    for response in completions:
        score = extract_risk_score(response, prompt_preamble_str, number_format, valid_range)
        if score is not None:
            scores.append(score)
    
    valid_count = len(scores)
    average_score = sum(scores) / valid_count if valid_count > 0 else None
    std_score = np.std(scores) if valid_count > 1 else None

    return average_score, std_score, valid_count, scores


async def get_model_completions(
    api,
    model_id: str,
    prompt_str: str,
    n_samples: int = 1,
    use_cache: bool = True,
    max_tokens: int | None = None,
    **kwargs
) -> list[str]:
    """
    Get multiple completions from a model for a single prompt.
    
    Args:
        api: API instance for making calls
        model_id: str, the model identifier
        prompt_str: str, the prompt text
        n_samples: int, number of samples to collect
        use_cache: bool, whether to use caching
        max_tokens: Optional[int], maximum number of tokens to generate
        **kwargs: Additional arguments to pass to the API
        
    Returns:
        list[str]: List of completion strings
    """
    prompt = Prompt(messages=[ChatMessage(role=MessageRole.user, content=prompt_str)])
    
    responses = await api(
        model_ids=model_id,
        prompt=prompt,
        n=n_samples,
        use_cache=use_cache,
        max_tokens=max_tokens,
        **kwargs
    )
    
    return [response.completion for response in responses]

# Analyze risk predisposition scale
async def analyze_risk_predisposition(scale_prompt_base_str : str, 
                                    instruction_prompt_str : str, 
                                    prompt_preamble_str : str, 
                                    number_format : str,
                                    model_id: str,
                                    api,
                                    n_samples: int = 100, 
                                    valid_range : tuple[int, int] = (0, 100),
                                    use_cache : bool = False) -> float | None:
    """
    Analyze risk predisposition scale responses for a given prompt.

    Args:
        scale_prompt_base_str: str, the base scale prompt text
        instruction_prompt_str: str, the instruction prompt text
        prompt_preamble_str: str, the preamble prompt text
        number_format: str, the number format for the placeholder
        model_id: str, the model identifier
        api: API instance for making calls
        n_samples: int, number of samples to collect
        valid_range: tuple[int, int], the valid range for the scale

    Returns:
        tuple: (average_score, std_score, valid_count, scores)
    """

    scale_prompt = create_scale_prompt(scale_prompt_base_str, instruction_prompt_str, prompt_preamble_str, number_format)
    
    # Get completions
    completions = await get_model_completions(api, model_id, prompt_str=scale_prompt, n_samples=n_samples, use_cache=use_cache)

    average_score, std_score, valid_count, scores  = estimate_predisposition_scale(completions, prompt_preamble_str, number_format, valid_range)

    return average_score, std_score, valid_count, scores, scale_prompt

## Word Frequency Analysis
def count_word_frequency(completions: list[str]) -> dict[str, int]:
    """
    Count frequency of words in completions, normalized to lowercase without punctuation.
    
    Args:
        completions: List of completion strings
    Returns:
        Dictionary of {word: count} frequencies
    """
    word_frequency = {}
    
    for completion in completions:
        words = completion.split()
        for word in words:
            # Normalize: lowercase, no whitespace, no trailing periods
            word = word.lower().strip().strip(".")
            word_frequency[word] = word_frequency.get(word, 0) + 1
            
    return word_frequency