import requests
import time
import logging
import re
import json
logger = logging.getLogger(__name__)

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = "sk-your-openai-api-key-here"  # Replace with actual key
OPENAI_MODEL = "gpt-4o-mini"

def callOpenAILLM(query: str):
    try:
        approx_input_tokens = len(query) // 4  # Approximate token count (1 token ≈ 4 chars)
        headers = {"Content-Type": "application/json","Authorization": f"Bearer {OPENAI_API_KEY}"}
        data = {"model": OPENAI_MODEL,"messages": [{"role": "user", "content": query}],"temperature": 0.1,"max_tokens": 4000,"stream": True}
        start_time = time.time()
        first_token_abs = None
        last_token_abs = None
        response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=300, stream=True)
        response.raise_for_status()
        full_content = ""
        for chunk in response.iter_lines():
            chunk_time = time.time()
            if chunk:
                chunk_str = chunk.decode('utf-8').strip()
                if chunk_str.startswith('data: '):
                    json_str = chunk_str[6:]
                    if json_str == '[DONE]':
                        break
                    try:
                        json_data = json.loads(json_str)
                        choice = json_data.get('choices', [{}])[0]
                        delta = choice.get('delta', {})
                        if 'content' in delta:
                            token = delta['content']
                            if first_token_abs is None:
                                first_token_abs = chunk_time
                            last_token_abs = chunk_time
                            full_content += token
                    except Exception as e:
                        logger.error(f"Chunk parse error: {str(e)}")
                        continue
        end_time = time.time()
        approx_output_tokens = len(full_content) // 4
        first_token_time = first_token_abs - start_time if first_token_abs else 0
        response_time = last_token_abs - first_token_abs if first_token_abs and last_token_abs else 0
        total_time = end_time - start_time
        input_cost = (approx_input_tokens / 1e6) * 0.50  # $0.50 per 1M input tokens
        output_cost = (approx_output_tokens / 1e6) * 1.50  # $1.50 per 1M output tokens
        total_cost = input_cost + output_cost
        print(f"TTFT: {first_token_time:.4f}s | Tokens: {approx_input_tokens}→{approx_output_tokens} | Cost: ${total_cost:.6f}")
        return full_content.strip(), approx_input_tokens, approx_output_tokens, first_token_time, response_time, total_time, total_cost
    except Exception as e:
        logger.error(f"OpenAI API Error: {str(e)}")
        return "", 0, 0, 0, 0, 0, 0
def parse_llm_output(output: str):
    try:
        cleaned_output = re.sub(r'^\s*```(?:json)?\s*', '', output, flags=re.IGNORECASE)
        cleaned_output = re.sub(r'\s*```\s*$', '', cleaned_output)
        try:
            response_data = json.loads(cleaned_output)
        except json.JSONDecodeError:
            try:
                json_start = cleaned_output.find('{')
                json_end = cleaned_output.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    response_data = json.loads(cleaned_output[json_start:json_end])
                else:
                    return [], "[]", "[]"
            except:
                return [], "[]", "[]"
        if not response_data:
            return [], "[]", "[]"
        pattern_labels = response_data.get("Pattern_Labels", []) or response_data.get("pattern_labels", [])
        pred_series = response_data.get("Pred_Series", []) or response_data.get("forecast", [])
        impact_scores = response_data.get("Impact_Scores", []) or response_data.get("impact_scores", [])
        if not isinstance(pattern_labels, list):
            pattern_labels = []
        pattern_mapping = {
            "Trend, Upward": "upward trend",
            "Trend, Downward": "downward trend",
            "Volatility, Increased": "increased volatility",
            "Volatility, Decreased": "decreased volatility",
            "Seasonal, Fixed": "fixed seasonal",
            "Seasonal, Shifting": "shifting seasonal",
            "Outlier, Sudden Spike": "sudden spike",
            "Outlier, Level Shift": "level shift"
        }
        pred_labels = [pattern_mapping.get(label.strip(), f"unknown:{label}") for label in pattern_labels]
        print(f"Parsed Labels: {pred_labels}")
        print(f"Parsed Series Length: {len(pred_series)} values")
        return pred_labels, json.dumps(pred_series), json.dumps(impact_scores)
    except Exception as e:
        logger.error(f"JSON Parse Error: {str(e)}")
        return [], "[]", "[]"