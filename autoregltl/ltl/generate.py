import random
import spot
from . parser import *
from . chars import CHARS

# May be modified in main
APS = CHARS


def generate(rng, prefix_len_code, contain_all_aps: bool):
    """
    Generate a trace-formula pair.
    """
    if contain_all_aps:
        prefix_len = eval(prefix_len_code)
        assert prefix_len >= len(APS)
        # Ensure that each AP is contained at least once
        string = APS.copy()
        string.extend([rng.choice(APS) for _ in range(prefix_len - len(string))])
        # Shuffle the list to randomize the order
        rng.shuffle(string)
        assert set(string) == set(APS)
        # Join the list into a final string
        return "".join(string)
    else:
        prefix_len = eval(prefix_len_code)
        prefix = [rng.choice(APS) for _ in range(prefix_len)]
        return "".join(prefix)


from multiprocessing import Process, Queue, cpu_count
from tqdm import tqdm


def process_generate(q, seed, count, kwargs):
    rng = random.Random(seed)
    for _ in range(count):
        q.put(generate(rng, **kwargs))

def multiprocess_generate(seed, total_count, **kwargs):
    q = Queue()
    cores = cpu_count()
    counts = [total_count // cores] * cores
    counts[0] += total_count % cores
    processes = [Process(target=process_generate, args=(q, seed + i, count, kwargs)) for i, count in enumerate(counts)]
    for p in processes:
        p.start()
    result = [q.get() for _ in tqdm(range(total_count))]
    for p in processes:
        p.join()
    return result

def singleprocess_generate(seed, total_count, **kwargs):
    # rng = random.Random(seed)
    result = [generate(random, **kwargs) for _ in tqdm(range(total_count))]
    return result


if __name__ == "__main__":
    import argparse
    import sys
    import os

    parser = argparse.ArgumentParser(description="Generate a dataset and save to a specified text file.")
    
    # Argument for output filename
    parser.add_argument(
        'output_filename', 
        type=str, 
        help="The name of the output text file where the dataset will be saved."
    )
    
    # Argument for number of samples with a default value
    parser.add_argument(
        'num_samples', 
        type=int, 
        help="The number of samples to generate."
    )

    # Argument for random seed with a default value
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help="Random seed (default: 42)."
    )

    parser.add_argument(
        '--aps', 
        type=int, 
        default=5, 
        help="Number of atomic propositions (default: 5)."
    )

    parser.add_argument(
        '--prefix-len', 
        type=str, 
        default="rng.randint(5, 10)", 
        help="Custom prefix len code to eval"
    )
    parser.add_argument('--contain-all-aps', action='store_true', default=False)
    
    args = parser.parse_args()
    
    if len(APS) < args.aps:
        print("Cannot support", args.aps, "atomic propositions with", len(APS), "printable characters")
        sys.exit(1)
    elif len(APS) != args.aps:
        APS = APS[:args.aps]
        print("APs:", len(APS))

    # Access the arguments using args.output_filename and args.num_samples
    print(f"Output Filename: {args.output_filename}")
    if os.path.exists(args.output_filename):
        print("Output file exists")
        exit()

    print(f"Number of Samples: {args.num_samples}")
    print(f"CPU count: {cpu_count()}")

    generated = singleprocess_generate(
        args.seed,
        args.num_samples,
        prefix_len_code=args.prefix_len,
        contain_all_aps=args.contain_all_aps,
    )
    generated = "\n".join(generated)

    if directory := os.path.dirname(args.output_filename):
        os.makedirs(directory, exist_ok=True)
    with open(args.output_filename, 'w') as f:
        f.write(generated)
