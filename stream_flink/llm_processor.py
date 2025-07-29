import json
import re
import time
import requests
from pyflink.datastream import MapFunction

from utils.config import *


class LLMProcessor(MapFunction):
    def __init__(self):
        super().__init__()

    def map(self, value):
        prompt = value['prompt']

        try:
            # Call LLM API
            result, metrics = self._call_llm_api(prompt)

            # Parse the output
            parsed_labels, pred_series, impact_scores = self._parse_llm_output(result)

            return {
                **value,
                'llm_response': result,
                'parsed_labels': parsed_labels,
                'pred_series': pred_series,
                'impact_scores': impact_scores,
                **metrics
            }

        except Exception as e:
            print(f"LLM processing error: {str(e)}")
            return {
                **value,
                'error': str(e)
            }

    def _call_llm_api(self, prompt: str):
        """Call the DeepSeek LLM API with streaming"""
        approx_input_tokens = len(prompt.split())
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }

        data = {
            "model": DEEPSEEK_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 8000,
            "stream_flink": True
        }

        start_time = time.time()
        first_token_abs = None
        last_token_abs = None
        full_content = ""

        try:
            response = requests.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=data,
                timeout=300,
                stream=True
            )
            response.raise_for_status()

            for chunk in response.iter_lines():
                if chunk:
                    chunk_str = chunk.decode('utf-8')
                    if chunk_str.startswith('data: '):
                        json_str = chunk_str[6:]
                        if json_str == "[DONE]":
                            break
                        try:
                            json_data = json.loads(json_str)
                            if 'choices' in json_data and json_data['choices']:
                                delta = json_data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    if first_token_abs is None:
                                        first_token_abs = time.time()
                                    last_token_abs = time.time()
                                    full_content += delta['content']
                        except:
                            continue

            end_time = time.time()
            approx_output_tokens = len(full_content.split())
            first_token_time = first_token_abs - start_time if first_token_abs else 0
            response_time = last_token_abs - first_token_abs if first_token_abs and last_token_abs else 0
            total_time = end_time - start_time
            cost = (approx_input_tokens / 1e6) * 1 + (approx_output_tokens / 1e6) * 2

            metrics = {
                'input_tokens': approx_input_tokens,
                'output_tokens': approx_output_tokens,
                'ttft': first_token_time,
                'response_time': response_time,
                'total_time': total_time,
                'cost': cost
            }

            print(
                f"TTFT: {first_token_time:.4f}s, Tokens: {approx_input_tokens}/{approx_output_tokens}, Cost: {cost:.6f}¥")

            return full_content.strip(), metrics

        except Exception as e:
            print(f"LLM API error: {str(e)}")
            return "", {
                'input_tokens': 0,
                'output_tokens': 0,
                'ttft': 0,
                'response_time': 0,
                'total_time': 0,
                'cost': 0
            }

    def _parse_llm_output(self, output: str):
        """Parse the LLM output into components"""
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

            return pred_labels, json.dumps(pred_series), json.dumps(impact_scores)

        except Exception as e:
            print(f"JSON Parse Error: {str(e)}")
            return [], "[]", "[]"