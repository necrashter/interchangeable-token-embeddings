import sys
import argparse
import math
import time
import traceback
from collections import defaultdict
from functools import reduce, partial
from contextlib import contextmanager
from tqdm.auto import tqdm

import multiprocessing
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
import spot

from . parser import LTLTrace, LTLFormula, F_AND, F_IMLIES, F_NEXT, F_GLOBALLY, F_NOT, F_AP
from . parser import ParseError, ltl_formula, ltl_trace


def per_size_analysis(full_results, **kwargs):
    import matplotlib.pyplot as plt

    colors = {
        'syntactically correct': '#38b547',
        'exact match': '#38b547',
        'equivalent': '#5ed561',
        'only semantically correct': '#85f67c',
        'semantically correct': '#85f67c',
        'incorrect': '#ed974d',
        'invalid': '#fd4a4a',
    }
    results = {k: v for k, v in full_results.items() if k in colors}
    order = {
        'syntactically correct': 0,
        'exact match': 0,
        'equivalent': 1,
        'only semantically correct': 2,
        'semantically correct': 2,
        'incorrect': 3,
        'invalid': 4,
    }
    results = dict(sorted(results.items(), key=lambda pair: order[pair[0]]))

    min_size = min([min(d) if len(d) > 0 else math.inf for d in results.values()])
    max_size = max([max(d) if len(d) > 0 else 0 for d in results.values()])
    x, totals = [], []
    assert not ('total' in results)
    results_complete = {}
    for size in range(min_size, max_size + 1):
        x.append(size)
        totals.append(0)
    bottom_positions = totals.copy()

    for category, dist in results.items():  # dict with sizes to list; not all values may occur in dict
        results_complete[category] = []
        for idx, size in enumerate(range(min_size, max_size + 1)):
            value = dist[size] if size in dist else 0
            results_complete[category].append(value)
            totals[idx] += value
    results_percent = {}
    for category, dist_complete in results_complete.items():
        results_percent[category] = []
        for val, total in zip(dist_complete, totals):
            if total == 0 and val != 0:
                raise RuntimeError()
            results_percent[category].append(val / total * 100 if total > 0 else 0)

    names = {
        'syntactically correct': 'exact match',
        'exact match': 'exact match',
        'equivalent': 'equivalent',
        'only semantically correct': 'correct',
        'semantically correct': 'correct',
        'incorrect': 'incorrect',
        'invalid': 'invalid',
     }
    # Do the plotting
    # thanks to https://chrisalbon.com/python/data_visualization/matplotlib_percentage_stacked_bar_plot/
    # figure, (hist_ax, dist_ax) = plt.subplots(2, figsize=(12,8))
    figure, (dist_ax) = plt.subplots(1, figsize=(12, 5))
    bar_width = 1
    # hist_ax.bar(x, totals, width=bar_width, color='#3071ff', edgecolor='white')
    # hist_ax.set_ylabel('number of items')
    # hist_ax.set_xlabel('formula size')
    for category, dist_percent in results_percent.items():
        dist_ax.bar(x, dist_percent, bottom=bottom_positions, label=names[category], width=bar_width, color=colors[category], edgecolor='white')
        bottom_positions = [acc + q for acc, q in zip(bottom_positions, dist_percent)]  # update positions
    dist_ax.set_ylabel('Percentage')
    dist_ax.set_xlabel('Trace size')
    dist_ax.set_ylim(-10, 110)
    dist_ax.legend()
    if 'save_analysis' in kwargs and kwargs['save_analysis'] is not None:
        figure.savefig(kwargs['save_analysis'] + '.png', bbox_inches="tight", dpi=192)
        figure.savefig(kwargs['save_analysis'] + '.svg', bbox_inches="tight", dpi=192)
    
    plt.close(figure)
    plt.clf()

    # collapse size-wise data for further processing
    results_collapsed = {}
    for category, dist in full_results.items():
        results_collapsed[category] = sum(dist.values())
    return results_collapsed


@contextmanager
def pool_iter(process_item, data, threads=None, timeout=30, tqdm_desc=None, leave_tqdm=True):
    if threads is None:
        threads = multiprocessing.cpu_count()
    with ProcessPool(threads) as pool, tqdm(total=len(data), desc=tqdm_desc, leave=leave_tqdm) as pbar:
        future = pool.map(process_item, data, timeout=timeout)
        callback = lambda _: pbar.update(1)
        for f in future.futures:
            f.add_done_callback(callback)
        iterator = future.result()
        yield iterator


def process_ltl_item(item, formula_format):
    pred_str, trace_str, formula_str = item
    start_time = time.time()
    try:
        pred_obj = ltl_trace(pred_str, format=formula_format)
    except ParseError as e:
        return {"result": "invalid", "error": f"{e}", "time": time.time() - start_time}
    formula_obj = ltl_formula(formula_str, format=formula_format)
    if trace_str:
        trace_obj = ltl_trace(trace_str, format=formula_format)
        if pred_obj.equal_to(trace_obj, extended_eq=True):
            return {"result": "exact match", "time": time.time() - start_time}
    # spot trace check
    formula_automaton = spot.formula(formula_obj.to_str('spot')).translate()
    pred_automaton = spot.parse_word(pred_obj.to_str('spot')).as_automaton()
    try:
        spot_holds = spot.contains(formula_automaton, pred_automaton)
        result = "semantically correct" if spot_holds else "incorrect"
        return {"result": result, "time": time.time() - start_time}
    except RuntimeError as e:
        return {
            "result": "runtime error",
            "error": repr(e),
            "time": time.time() - start_time,
        }


def equivalence_item(item, formula_format, equivalence_method):
    """
    Pass 2 of trace checking. Performed only for semantically correct.
    Checks if the generation is logically equivalent to (or has the same automata as) the target.
    """
    formula_str, target_str = item
    try:
        formula_obj = ltl_formula(formula_str, format=formula_format)
        target_obj = ltl_formula(target_str, format=formula_format)
    except ParseError as e:
        return f"ParseError: {e}"
    # spot trace check
    try:
        if equivalence_method == 'full':
            return spot.are_equivalent(formula_obj.to_str('spot'), target_obj.to_str('spot'))
        elif equivalence_method == 'automata':
            return spot.formula(formula_obj.to_str('spot')).translate() == spot.formula(target_obj.to_str('spot')).translate()
        else:
            raise ValueError(f"Invalid equivalence method: {equivalence_method}")
    except RuntimeError as e:
        return "RuntimeError: " + repr(e)


def evaluate_ltl(data, polish=True, threads=None, timeout=30, leave_tqdm=True, equivalence_method=None):
    """
    Args:
        data: List of tuples (formula, trace, target trace)
    """
    formula_format = 'network-' + ('polish' if polish else 'infix')
    process_item = partial(process_ltl_item, formula_format=formula_format)

    results = []
    with pool_iter(process_item, data, threads, timeout, tqdm_desc="Evaluate", leave_tqdm=leave_tqdm) as iterator:
        for a, b, c in data:
            try:
                result = next(iterator)
            except TimeoutError:
                result = {"result": "timeout", "time": timeout}
            except ProcessExpired as e:
                result = {
                    "result": "runtime error",
                    "error": f"ProcessExpired with exit code {e.exitcode}",
                    "time": 0.0,
                }
            except Exception as e:
                result = {
                    "result": "runtime error",
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                    "time": 0.0,
                }
            result.update({"prediction": a, "trace": b, "formula": c})
            results.append(result)

    if equivalence_method is not None:
        if equivalence_method not in ['full', 'automata']:
            print(f"[ERROR] Invalid equivalence method: '{equivalence_method}', skipping second pass")
            return results
        pass2_items = []
        pass2_indices = []
        for result in results:
            if result['result'] == 'semantically correct':
                pass2_items.append((a, c))
                pass2_indices.append(len(results) - 1)
        process_item = partial(equivalence_item, formula_format=formula_format, equivalence_method=equivalence_method)
        with pool_iter(process_item, pass2_items, threads, timeout, tqdm_desc="Equivalence", leave_tqdm=leave_tqdm) as iterator:
            for i in pass2_indices:
                try:
                    result = next(iterator)
                except Exception as e:
                    results[i]["equivalence_error"] = repr(e)
                if isinstance(result, str):
                    results[i]["equivalence_error"] = result
                elif result:
                    results[i]["result"] = "equivalent"

    return results


def analyze_results(results):
    """
    Calculate statistics per size from evaluation results.
    """
    output = defaultdict(lambda: defaultdict(int))
    # Trace format: 1;1;{1;1}
    get_size = lambda x: (x["trace"].count(';') + 1)
    for result in results:
        output[result["result"]][get_size(result)] += 1
    return output


def ltl_distinctiveness_item(pair, formula_format):
    formula, trace = pair
    start_time = time.time()
    formula_obj = ltl_formula(formula, format=formula_format)
    trace_obj = ltl_trace(trace, format=formula_format)
    # spot trace check
    formula_spot = spot.formula(formula_obj.to_str('spot'))
    trace_spot = spot.parse_word(trace_obj.to_str('spot'))
    formula_automaton = formula_spot.translate()
    trace_automaton = trace_spot.as_automaton()
    output = spot.contains(formula_automaton, trace_automaton)
    return output, time.time() - start_time


def evaluate_ltl_distinctiveness(evaluations, polish=True, threads=None, timeout=30, leave_tqdm=True):
    """
    Args:
        evaluations: output of evaluate_ltl
    
    Adds "distinctiveness" field to each evaluation. Returns a list of distinctiveness values.
    """
    valid_formulas = [i for i, item in enumerate(evaluations) if item["result"] in ("exact match", "equivalent", "semantically correct")]
    # (formula, trace)
    pairs_idx = [(i, j) for i in valid_formulas for j in range(len(evaluations)) if i != j]
    pairs = [(evaluations[i]["formula"], evaluations[j]["trace"]) for i, j in pairs_idx]

    formula_format = 'network-' + ('polish' if polish else 'infix')
    process_item = partial(ltl_distinctiveness_item, formula_format=formula_format)

    counts = [0] * len(evaluations)
    timeouts = []
    errors = []
    eval_times = []

    with pool_iter(process_item, pairs, threads, timeout, tqdm_desc="Distinctiveness", leave_tqdm=leave_tqdm) as iterator:
        for pair in pairs_idx:
            try:
                result, time = next(iterator)
                eval_times.append(time)
                if result:
                    counts[pair[0]] += 1
            except TimeoutError:
                timeouts.append(pair)
                eval_times.append(timeout)
            except Exception as e:
                errors.append((*pair, repr(e)))

    distinctiveness_values = []
    other_count = len(evaluations) - 1
    for count, evaluation in zip(counts, evaluations):
        if evaluation["result"] in ("exact match", "equivalent", "semantically correct"):
            distinctiveness = 1 - (count / other_count)
            evaluation["distinctiveness"] = distinctiveness
            distinctiveness_values.append(distinctiveness)
    
    return distinctiveness_values, timeouts, errors, eval_times


def evaluate_ltl_target_distinctiveness(evaluations, polish=True, threads=None, timeout=30, leave_tqdm=True):
    """
    Same as evaluate_ltl_distinctiveness but operates on targets.
    Args:
        evaluations: output of evaluate_ltl
    
    Adds "distinctiveness" field to each evaluation. Returns a list of distinctiveness values.
    """
    # (formula, trace)
    pairs_idx = [(i, j) for i in range(len(evaluations)) for j in range(len(evaluations)) if i != j]
    pairs = [(evaluations[i]["target"], evaluations[j]["trace"]) for i, j in pairs_idx]

    formula_format = 'network-' + ('polish' if polish else 'infix')
    process_item = partial(ltl_distinctiveness_item, formula_format=formula_format)

    counts = [0] * len(evaluations)
    timeouts = []
    errors = []
    eval_times = []

    with pool_iter(process_item, pairs, threads, timeout, tqdm_desc="Distinctiveness", leave_tqdm=leave_tqdm) as iterator:
        for pair in pairs_idx:
            try:
                result, time = next(iterator)
                eval_times.append(time)
                if result:
                    counts[pair[0]] += 1
            except TimeoutError:
                timeouts.append(pair)
                eval_times.append(timeout)
            except Exception as e:
                errors.append((*pair, repr(e)))

    distinctiveness_values = []
    other_count = len(evaluations) - 1
    for count, evaluation in zip(counts, evaluations):
        distinctiveness = 1 - (count / other_count)
        evaluation["distinctiveness"] = distinctiveness
        distinctiveness_values.append(distinctiveness)
    
    return distinctiveness_values, timeouts, errors, eval_times
