from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from dataclasses import dataclass
from functools import partial
import openai
import tiktoken
from jax import grad,vmap


STEP_MULTIPLIER = 1.2


def vec_num2repr(val, base, prec, max_val):
    """
    Convert numbers to a representation in a specified base with precision.

    Parameters:
    - val (np.array): The numbers to represent.
    - base (int): The base of the representation.
    - prec (int): The precision after the 'decimal' point in the base representation.
    - max_val (float): The maximum absolute value of the number.

    Returns:
    - tuple: Sign and digits in the specified base representation.

    Examples:
        With base=10, prec=2:
            0.5   ->    50
            3.52  ->   352
            12.5  ->  1250
    """
    base = float(base)
    bs = val.shape[0]
    sign = 1 * (val >= 0) - 1 * (val < 0)
    val = np.abs(val)
    max_bit_pos = int(np.ceil(np.log(max_val) / np.log(base)).item())

    before_decimals = []
    for i in range(max_bit_pos):
        digit = (val / base ** (max_bit_pos - i - 1)).astype(int)
        before_decimals.append(digit)
        val -= digit * base ** (max_bit_pos - i - 1)

    before_decimals = np.stack(before_decimals, axis=-1)

    if prec > 0:
        after_decimals = []
        for i in range(prec):
            digit = (val / base ** (-i - 1)).astype(int)
            after_decimals.append(digit)
            val -= digit * base ** (-i - 1)

        after_decimals = np.stack(after_decimals, axis=-1)
        digits = np.concatenate([before_decimals, after_decimals], axis=-1)
    else:
        digits = before_decimals
    return sign, digits


def vec_repr2num(sign, digits, base, prec, half_bin_correction=True):
    """
    Convert a string representation in a specified base back to numbers.

    Parameters:
    - sign (np.array): The sign of the numbers.
    - digits (np.array): Digits of the numbers in the specified base.
    - base (int): The base of the representation.
    - prec (int): The precision after the 'decimal' point in the base representation.
    - half_bin_correction (bool): If True, adds 0.5 of the smallest bin size to the number.

    Returns:
    - np.array: Numbers corresponding to the given base representation.
    """
    base = float(base)
    bs, D = digits.shape
    digits_flipped = np.flip(digits, axis=-1)
    powers = -np.arange(-prec, -prec + D)
    val = np.sum(digits_flipped / base ** powers, axis=-1)

    if half_bin_correction:
        val += 0.5 / base ** prec

    return sign * val


@dataclass
class SerializerSettings:
    """
    Settings for serialization of numbers.

    Attributes:
    - base (int): The base for number representation.
    - prec (int): The precision after the 'decimal' point in the base representation.
    - signed (bool): If True, allows negative numbers. Default is False.
    - fixed_length (bool): If True, ensures fixed length of serialized string. Default is False.
    - max_val (float): Maximum absolute value of number for serialization.
    - time_sep (str): Separator for different time steps.
    - bit_sep (str): Separator for individual digits.
    - plus_sign (str): String representation for positive sign.
    - minus_sign (str): String representation for negative sign.
    - half_bin_correction (bool): If True, applies half bin correction during deserialization. Default is True.
    - decimal_point (str): String representation for the decimal point.
    """
    base: int = 10
    prec: int = 3
    signed: bool = True
    fixed_length: bool = False
    max_val: float = 1e7
    time_sep: str = ' ,'
    bit_sep: str = ' '
    plus_sign: str = ''
    minus_sign: str = ' -'
    half_bin_correction: bool = True
    decimal_point: str = ''
    missing_str: str = ' Nan'


def serialize_arr(arr, settings: SerializerSettings):
    """
    Serialize an array of numbers (a time series) into a string based on the provided settings.

    Parameters:
    - arr (np.array): Array of numbers to serialize.
    - settings (SerializerSettings): Settings for serialization.

    Returns:
    - str: String representation of the array.
    """
    # max_val is only for fixing the number of bits in nunm2repr so it can be vmapped
    assert np.all(np.abs(arr[~np.isnan(arr)]) <= settings.max_val), f"abs(arr) must be <= max_val,\
         but abs(arr)={np.abs(arr)}, max_val={settings.max_val}"

    if not settings.signed:
        assert np.all(arr[~np.isnan(arr)] >= 0), f"unsigned arr must be >= 0"
        plus_sign = minus_sign = ''
    else:
        plus_sign = settings.plus_sign
        minus_sign = settings.minus_sign

    vnum2repr = partial(vec_num2repr, base=settings.base, prec=settings.prec, max_val=settings.max_val)
    sign_arr, digits_arr = vnum2repr(np.where(np.isnan(arr), np.zeros_like(arr), arr))
    ismissing = np.isnan(arr)

    def tokenize(arr):
        return ''.join([settings.bit_sep + str(b) for b in arr])

    bit_strs = []
    for sign, digits, missing in zip(sign_arr, digits_arr, ismissing):
        if not settings.fixed_length:
            # remove leading zeros
            nonzero_indices = np.where(digits != 0)[0]
            if len(nonzero_indices) == 0:
                digits = np.array([0])
            else:
                digits = digits[nonzero_indices[0]:]
            # add a decimal point
            prec = settings.prec
            if len(settings.decimal_point):
                digits = np.concatenate([digits[:-prec], np.array([settings.decimal_point]), digits[-prec:]])
        digits = tokenize(digits)
        sign_sep = plus_sign if sign == 1 else minus_sign
        if missing:
            bit_strs.append(settings.missing_str)
        else:
            bit_strs.append(sign_sep + digits)
    bit_str = settings.time_sep.join(bit_strs)
    bit_str += settings.time_sep  # otherwise there is ambiguity in number of digits in the last time step
    return bit_str


def deserialize_str(bit_str, settings: SerializerSettings, ignore_last=False, steps=None):
    """
    Deserialize a string into an array of numbers (a time series) based on the provided settings.

    Parameters:
    - bit_str (str): String representation of an array of numbers.
    - settings (SerializerSettings): Settings for deserialization.
    - ignore_last (bool): If True, ignores the last time step in the string (which may be incomplete due to token limit etc.). Default is False.
    - steps (int, optional): Number of steps or entries to deserialize.

    Returns:
    - None if deserialization failed for the very first number, otherwise
    - np.array: Array of numbers corresponding to the string.
    """
    # ignore_last is for ignoring the last time step in the prediction, which is often a partially generated due to token limit
    orig_bitstring = bit_str
    bit_strs = bit_str.split(settings.time_sep)
    # remove empty strings
    bit_strs = [a for a in bit_strs if len(a) > 0]
    if ignore_last:
        bit_strs = bit_strs[:-1]
    if steps is not None:
        bit_strs = bit_strs[:steps]
    vrepr2num = partial(vec_repr2num, base=settings.base, prec=settings.prec,
                        half_bin_correction=settings.half_bin_correction)
    max_bit_pos = int(np.ceil(np.log(settings.max_val) / np.log(settings.base)).item())
    sign_arr = []
    digits_arr = []
    try:
        for i, bit_str in enumerate(bit_strs):
            if bit_str.startswith(settings.minus_sign):
                sign = -1
            elif bit_str.startswith(settings.plus_sign):
                sign = 1
            else:
                assert settings.signed == False, f"signed bit_str must start with {settings.minus_sign} or {settings.plus_sign}"
            bit_str = bit_str[len(settings.plus_sign):] if sign == 1 else bit_str[len(settings.minus_sign):]
            if settings.bit_sep == '':
                bits = [b for b in bit_str.lstrip()]
            else:
                bits = [b[:1] for b in bit_str.lstrip().split(settings.bit_sep)]
            if settings.fixed_length:
                assert len(
                    bits) == max_bit_pos + settings.prec, f"fixed length bit_str must have {max_bit_pos + settings.prec} bits, but has {len(bits)}: '{bit_str}'"
            digits = []
            for b in bits:
                if b == settings.decimal_point:
                    continue
                # check if is a digit
                if b.isdigit():
                    digits.append(int(b))
                else:
                    break
            # digits = [int(b) for b in bits]
            sign_arr.append(sign)
            digits_arr.append(digits)
    except Exception as e:
        print(f"Error deserializing {settings.time_sep.join(bit_strs[i - 2:i + 5])}{settings.time_sep}\n\t{e}")
        print(f'Got {orig_bitstring}')
        print(f"Bitstr {bit_str}, separator {settings.bit_sep}")
        # At this point, we have already deserialized some of the bit_strs, so we return those below
    if digits_arr:
        # add leading zeros to get to equal lengths
        max_len = max([len(d) for d in digits_arr])
        for i in range(len(digits_arr)):
            digits_arr[i] = [0] * (max_len - len(digits_arr[i])) + digits_arr[i]
        return vrepr2num(np.array(sign_arr), np.array(digits_arr))
    else:
        # errored at first step
        return None
@dataclass
class Scaler:
    """
    Represents a data scaler with transformation and inverse transformation functions.

    Attributes:
        transform (callable): Function to apply transformation.
        inv_transform (callable): Function to apply inverse transformation.
    """
    transform: callable = lambda x: x
    inv_transform: callable = lambda x: x    

def get_scaler(history, alpha=0.95, beta=0.3, basic=False):
    """
    Generate a Scaler object based on given history data.

    Args:
        history (array-like): Data to derive scaling from.
        alpha (float, optional): Quantile for scaling. Defaults to .95.
        # Truncate inputs
        tokens = [tokeniz]
        beta (float, optional): Shift parameter. Defaults to .3.
        basic (bool, optional): If True, no shift is applied, and scaling by values below 0.01 is avoided. Defaults to False.

    Returns:
        Scaler: Configured scaler object.
    """
    history = history[~np.isnan(history)]
    if basic:
        q = np.maximum(np.quantile(np.abs(history), alpha),.01)
        def transform(x):
            return x / q
        def inv_transform(x):
            return x * q
    else:
        min_ = np.min(history) - beta*(np.max(history)-np.min(history))
        q = np.quantile(history-min_, alpha)
        if q == 0:
            q = 1
        def transform(x):
            return (x - min_) / q
        def inv_transform(x):
            return x * q + min_
    return Scaler(transform=transform, inv_transform=inv_transform)

def truncate_input(input_arr, input_str, settings, model, steps):
    """
    Truncate inputs to the maximum context length for a given model.
    
    Args:
        input (array-like): input time series.
        input_str (str): serialized input time series.
        settings (SerializerSettings): Serialization settings.
        model (str): Name of the LLM model to use.
        steps (int): Number of steps to predict.py.
    Returns:
        tuple: Tuple containing:
            - input (array-like): Truncated input time series.
            - input_str (str): Truncated serialized input time series.
    """
    if model in tokenization_fns and model in context_lengths:
        tokenization_fn = tokenization_fns[model]
        context_length = context_lengths[model]
        input_str_chuncks = input_str.split(settings.time_sep)
        for i in range(len(input_str_chuncks) - 1):
            truncated_input_str = settings.time_sep.join(input_str_chuncks[i:])
            # add separator if not already present
            if not truncated_input_str.endswith(settings.time_sep):
                truncated_input_str += settings.time_sep
            input_tokens = tokenization_fn(truncated_input_str)
            num_input_tokens = len(input_tokens)
            avg_token_length = num_input_tokens / (len(input_str_chuncks) - i)
            num_output_tokens = avg_token_length * steps * STEP_MULTIPLIER
            if num_input_tokens + num_output_tokens <= context_length:
                truncated_input_arr = input_arr[i:]
                break
        if i > 0:
            print(f'Warning: Truncated input from {len(input_arr)} to {len(truncated_input_arr)}')
        return truncated_input_arr, truncated_input_str
    else:
        return input_arr, input_str

def handle_prediction(pred, expected_length, strict=False):
    """
    Process the output from LLM after deserialization, which may be too long or too short, or None if deserialization failed on the first prediction step.

    Args:
        pred (array-like or None): The predicted values. None indicates deserialization failed.
        expected_length (int): Expected length of the prediction.
        strict (bool, optional): If True, returns None for invalid predictions. Defaults to False.

    Returns:
        array-like: Processed prediction.
    """
    if pred is None:
        return None
    else:
        if len(pred) < expected_length:
            if strict:
                print(f'Warning: Prediction too short {len(pred)} < {expected_length}, returning None')
                return None
            else:
                print(f'Warning: Prediction too short {len(pred)} < {expected_length}, padded with last value')
                return np.concatenate([pred, np.full(expected_length - len(pred), pred[-1])])
        else:
            return pred[:expected_length]

def generate_predictions(
    completion_fn, 
    input_strs, 
    steps, 
    settings: SerializerSettings, 
    scalers: None,
    num_samples=1, 
    temp=0.7, 
    parallel=True,
    strict_handling=False,
    max_concurrent=10,
    **kwargs
):
    """
    Generate and process text completions from a language model for input time series.

    Args:
        completion_fn (callable): Function to obtain text completions from the LLM.
        input_strs (list of array-like): List of input time series.
        steps (int): Number of steps to predict.py.
        settings (SerializerSettings): Settings for serialization.
        scalers (list of Scaler, optional): List of Scaler objects. Defaults to None, meaning no scaling is applied.
        num_samples (int, optional): Number of samples to return. Defaults to 1.
        temp (float, optional): Temperature for sampling. Defaults to 0.7.
        parallel (bool, optional): If True, run completions in parallel. Defaults to True.
        strict_handling (bool, optional): If True, return None for predictions that don't have exactly the right format or expected length. Defaults to False.
        max_concurrent (int, optional): Maximum number of concurrent completions. Defaults to 50.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple: Tuple containing:
            - preds (list of lists): Numerical predictions.
            - completions_list (list of lists): Raw text completions.
            - input_strs (list of str): Serialized input strings.
    """
    
    completions_list = []
    complete = lambda x: completion_fn(input_str=x, steps=steps*STEP_MULTIPLIER, settings=settings, num_samples=num_samples, temp=temp)
    if parallel and len(input_strs) > 1:
        print('Running completions in parallel for each input')
        with ThreadPoolExecutor(min(max_concurrent, len(input_strs))) as p:
            completions_list = list(tqdm(p.map(complete, input_strs), total=len(input_strs)))
    else:
        completions_list = [complete(input_str) for input_str in tqdm(input_strs)]
    def completion_to_pred(completion, inv_transform): 
        pred = handle_prediction(deserialize_str(completion, settings, ignore_last=False, steps=steps), expected_length=steps, strict=strict_handling)
        if pred is not None:
            return inv_transform(pred)
        else:
            return None
    preds = [[completion_to_pred(completion, scaler.inv_transform) for completion in completions] for completions, scaler in zip(completions_list, scalers)]
    return preds, completions_list, input_strs

def get_llmtime_predictions_data(train, test, model, settings, num_samples=10, temp=0.7, alpha=0.95, beta=0.3, basic=False, parallel=True, **kwargs):
    """
    Obtain forecasts from an LLM based on training series (history) and evaluate likelihood on test series (true future).
    train and test can be either a single time series or a list of time series.

    Args:
        train (array-like or list of array-like): Training time series data (history).
        test (array-like or list of array-like): Test time series data (true future).
        model (str): Name of the LLM model to use. Must have a corresponding entry in completion_fns.
        settings (SerializerSettings or dict): Serialization settings.
        num_samples (int, optional): Number of samples to return. Defaults to 10.
        temp (float, optional): Temperature for sampling. Defaults to 0.7.
        alpha (float, optional): Scaling parameter. Defaults to 0.95.
        beta (float, optional): Shift parameter. Defaults to 0.3.
        basic (bool, optional): If True, use the basic version of data scaling. Defaults to False.
        parallel (bool, optional): If True, run predictions in parallel. Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        dict: Dictionary containing predictions, samples, median, NLL/D averaged over each series, and other related information.
    """

    assert model in completion_fns, f'Invalid model {model}, must be one of {list(completion_fns.keys())}'
    completion_fn = completion_fns[model]
    nll_fn = nll_fns[model] if model in nll_fns else None
    
    if isinstance(settings, dict):
        settings = SerializerSettings(**settings)
    if not isinstance(train, list):
        # Assume single train/test case
        train = [train]
        test = [test]

    for i in range(len(train)):
        if not isinstance(train[i], pd.Series):
            train[i] = pd.Series(train[i], index=pd.RangeIndex(len(train[i])))
            test[i] = pd.Series(test[i], index=pd.RangeIndex(len(train[i]), len(test[i])+len(train[i])))

    test_len = len(test[0])
    assert all(len(t)==test_len for t in test), f'All test series must have same length, got {[len(t) for t in test]}'

    # Create a unique scaler for each series
    scalers = [get_scaler(train[i].values, alpha=alpha, beta=beta, basic=basic) for i in range(len(train))]

    # transform input_arrs
    input_arrs = [train[i].values for i in range(len(train))]
    transformed_input_arrs = np.array([scaler.transform(input_array) for input_array, scaler in zip(input_arrs, scalers)])
    # serialize input_arrs
    input_strs = [serialize_arr(scaled_input_arr, settings) for scaled_input_arr in transformed_input_arrs]
    # Truncate input_arrs to fit the maximum context length
    input_arrs, input_strs = zip(*[truncate_input(input_array, input_str, settings, model, test_len) for input_array, input_str in zip(input_arrs, input_strs)])
    
    steps = test_len
    samples = None
    medians = None
    completions_list = None
    if num_samples > 0:
        preds, completions_list, input_strs = generate_predictions(completion_fn, input_strs, steps, settings, scalers,
                                                                    num_samples=num_samples, temp=temp, 
                                                                    parallel=parallel, **kwargs)
        samples = [pd.DataFrame(preds[i], columns=test[i].index) for i in range(len(preds))]
        medians = [sample.median(axis=0) for sample in samples]
        samples = samples if len(samples) > 1 else samples[0]
        medians = medians if len(medians) > 1 else medians[0]
    out_dict = {
        'samples': samples,
        'median':  medians,
        'info': {
            'Method': model,
        },
        'completions_list': completions_list,
        'input_strs': input_strs,
    }
    # Compute NLL/D on the true test series conditioned on the (truncated) input series
    if nll_fn is not None:
        BPDs = [nll_fn(input_arr=input_arrs[i], target_arr=test[i].values, settings=settings, transform=scalers[i].transform, count_seps=True, temp=temp) for i in range(len(train))]
        out_dict['NLL/D'] = np.mean(BPDs)
    return out_dict


def gpt_tokenize_fn(str, model):
    """
    Retrieve the token IDs for a string for a specific GPT model.

    Args:
        str (list of str): str to be tokenized.
        model (str): Name of the LLM model.

    Returns:
        list of int: List of corresponding token IDs.
    """
    encoding = tiktoken.encoding_for_model(model)
    return encoding.encode(str)


def get_allowed_ids(strs, model):
    """
    Retrieve the token IDs for a given list of strings for a specific GPT model.

    Args:
        strs (list of str): strs to be converted.
        model (str): Name of the LLM model.

    Returns:
        list of int: List of corresponding token IDs.
    """
    encoding = tiktoken.encoding_for_model(model)
    ids = []
    for s in strs:
        id = encoding.encode(s)
        ids.extend(id)
    return ids


def gpt_completion_fn(model, input_str, steps, settings, num_samples, temp):
    """
    Generate text completions from GPT using OpenAI's API.

    Args:
        model (str): Name of the GPT-3 model to use.
        input_str (str): Serialized input time series data.
        steps (int): Number of time steps to predict.py.
        settings (SerializerSettings): Serialization settings.
        num_samples (int): Number of completions to generate.
        temp (float): Temperature for sampling.

    Returns:
        list of str: List of generated samples.
    """
    avg_tokens_per_step = len(tokenize_fn(input_str, model)) / len(input_str.split(settings.time_sep))
    # define logit bias to prevent GPT-3 from producing unwanted tokens
    logit_bias = {}
    allowed_tokens = [settings.bit_sep + str(i) for i in range(settings.base)]
    allowed_tokens += [settings.time_sep, settings.plus_sign, settings.minus_sign]
    allowed_tokens = [t for t in allowed_tokens if len(t) > 0]  # remove empty tokens like an implicit plus sign
    if (model not in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-1106-preview']):  # logit bias not supported for chat models
        logit_bias = {id: 30 for id in get_allowed_ids(allowed_tokens, model)}
    if model in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-1106-preview']:
        chatgpt_sys_message = "You are a helpful assistant that performs time series predictions. The user will provide a sequence and you will predict.py the remaining sequence. The sequence is represented by decimal strings separated by commas."
        extra_input = "Please continue the following sequence without producing any additional text. Do not say anything like 'the next terms in the sequence are', just return the numbers. Sequence:\n"
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": chatgpt_sys_message},
                {"role": "user", "content": extra_input + input_str + settings.time_sep}
            ],
            max_tokens=int(avg_tokens_per_step * steps),
            temperature=temp,
            logit_bias=logit_bias,
            n=num_samples,
        )
        return [choice.message.content for choice in response.choices]
    else:
        response = openai.Completion.create(
            model=model,
            prompt=input_str,
            max_tokens=int(avg_tokens_per_step * steps),
            temperature=temp,
            logit_bias=logit_bias,
            n=num_samples
        )
        return [choice.text for choice in response.choices]


def gpt_nll_fn(model, input_arr, target_arr, settings: SerializerSettings, transform, count_seps=True, temp=1):
    """
    Calculate the Negative Log-Likelihood (NLL) per dimension of the target array according to the LLM.

    Args:
        model (str): Name of the LLM model to use.
        input_arr (array-like): Input array (history).
        target_arr (array-like): Ground target array (future).
        settings (SerializerSettings): Serialization settings.
        transform (callable): Transformation applied to the numerical values before serialization.
        count_seps (bool, optional): Whether to account for separators in the calculation. Should be true for models that generate a variable number of digits. Defaults to True.
        temp (float, optional): Temperature for sampling. Defaults to 1.

    Returns:
        float: Calculated NLL per dimension.
    """
    input_str = serialize_arr(vmap(transform)(input_arr), settings)
    target_str = serialize_arr(vmap(transform)(target_arr), settings)
    assert input_str.endswith(settings.time_sep), f'Input string must end with {settings.time_sep}, got {input_str}'
    full_series = input_str + target_str
    response = openai.Completion.create(model=model, prompt=full_series, logprobs=5, max_tokens=0, echo=True,
                                        temperature=temp)
    # print(response['choices'][0])
    logprobs = np.array(response['choices'][0].logprobs.token_logprobs, dtype=np.float32)
    tokens = np.array(response['choices'][0].logprobs.tokens)
    top5logprobs = response['choices'][0].logprobs.top_logprobs
    seps = tokens == settings.time_sep
    target_start = np.argmax(np.cumsum(seps) == len(input_arr)) + 1
    logprobs = logprobs[target_start:]
    tokens = tokens[target_start:]
    top5logprobs = top5logprobs[target_start:]
    seps = tokens == settings.time_sep
    assert len(logprobs[seps]) == len(
        target_arr), f'There should be one separator per target. Got {len(logprobs[seps])} separators and {len(target_arr)} targets.'
    # adjust logprobs by removing extraneous and renormalizing (see appendix of paper)
    allowed_tokens = [settings.bit_sep + str(i) for i in range(settings.base)]
    allowed_tokens += [settings.time_sep, settings.plus_sign, settings.minus_sign,
                       settings.bit_sep + settings.decimal_point]
    allowed_tokens = {t for t in allowed_tokens if len(t) > 0}
    p_extra = np.array([sum(np.exp(ll) for k, ll in top5logprobs[i].items() if not (k in allowed_tokens)) for i in
                        range(len(top5logprobs))])
    if settings.bit_sep == '':
        p_extra = 0
    adjusted_logprobs = logprobs - np.log(1 - p_extra)
    digits_bits = -adjusted_logprobs[~seps].sum()
    seps_bits = -adjusted_logprobs[seps].sum()
    BPD = digits_bits / len(target_arr)
    if count_seps:
        BPD += seps_bits / len(target_arr)
    # log p(x) = log p(token) - log bin_width = log p(token) + prec * log base
    transformed_nll = BPD - settings.prec * np.log(settings.base)
    avg_logdet_dydx = np.log(vmap(grad(transform))(target_arr)).mean()
    return transformed_nll - avg_logdet_dydx




# Required: Text completion function for each model
# -----------------------------------------------
# Each model is mapped to a function that samples text completions.
# The completion function should follow this signature:
#
# Args:
#   - input_str (str): String representation of the input time series.
#   - steps (int): Number of steps to predict.py.
#   - settings (SerializerSettings): Serialization settings.
#   - num_samples (int): Number of completions to sample.
#   - temp (float): Temperature parameter for model's output randomness.
#
# Returns:
#   - list: Sampled completion strings from the model.
completion_fns = {
    'text-davinci-003': partial(gpt_completion_fn, model='text-davinci-003'),
    'gpt-4': partial(gpt_completion_fn, model='gpt-4'),
    'gpt-4-1106-preview':partial(gpt_completion_fn, model='gpt-4-1106-preview')
}

# Optional: NLL/D functions for each model
# -----------------------------------------------
# Each model is mapped to a function that computes the continuous Negative Log-Likelihood
# per Dimension (NLL/D). This is used for computing likelihoods only and not needed for sampling.
#
# The NLL function should follow this signature:
#
# Args:
#   - input_arr (np.ndarray): Input time series (history) after data transformation.
#   - target_arr (np.ndarray): Ground truth series (future) after data transformation.
#   - settings (SerializerSettings): Serialization settings.
#   - transform (callable): Data transformation function (e.g., scaling) for determining the Jacobian factor.
#   - count_seps (bool): If True, count time step separators in NLL computation, required if allowing variable number of digits.
#   - temp (float): Temperature parameter for sampling.
#
# Returns:
#   - float: Computed NLL per dimension for p(target_arr | input_arr).
nll_fns = {
    'text-davinci-003': partial(gpt_nll_fn, model='text-davinci-003')
}

# Optional: Tokenization function for each model, only needed if you want automatic input truncation.
# The tokenization function should follow this signature:
#
# Args:
#   - str (str): A string to tokenize.
# Returns:
#   - token_ids (list): A list of token ids.
tokenization_fns = {
    'text-davinci-003': partial(gpt_tokenize_fn, model='text-davinci-003'),
    'gpt-3.5-turbo-instruct': partial(gpt_tokenize_fn, model='gpt-3.5-turbo-instruct'),
}

# Optional: Context lengths for each model, only needed if you want automatic input truncation.
context_lengths = {
    'text-davinci-003': 4097,
    'gpt-3.5-turbo-instruct': 4097,
}


