#!/usr/bin/env python3
"""
Simple script to benchmark cold startup/compilation time in vLLM.
Measures time from LLM initialization until ready to serve requests.
Uses PyTorch profiler to generate flame graphs and analyze top kernels.
"""

import argparse
import shutil
import time

# import sys
from pathlib import Path

# Add vllm to path if needed
# sys.path.insert(0, str(Path(__file__).parent / "vllm"))
import torch

from vllm import LLM, SamplingParams


def clear_compilation_caches():
    """Clear Triton and Inductor compilation caches."""
    cache_paths = [
        Path("/tmp/torchinductor_root"),  # Inductor cache
        Path.home() / ".cache" / "torch" / "inductor_cache",  # Standard
        Path.home() / ".triton" / "cache",  # Triton cache
    ]

    for cache_path in cache_paths:
        if cache_path.exists():
            try:
                if cache_path.is_dir():
                    shutil.rmtree(cache_path)
                    print(f"  Cleared cache: {cache_path}")
                else:
                    cache_path.unlink()
                    print(f"  Cleared cache: {cache_path}")
            except Exception as e:
                print(f"  Warning: Could not clear {cache_path}: {e}")


def print_top_kernels(prof, num_kernels: int = 10):
    """Print top kernels by execution time."""
    print(f"\n  Top {num_kernels} kernels by execution time:")
    print("  " + "-" * 70)

    # Get CUDA events sorted by self CUDA time
    events = []
    try:
        # Try accessing events through key_averages (standard API)
        key_averages = prof.key_averages()
        for event in key_averages:
            if event.device_type == torch.profiler.DeviceType.CUDA:
                cuda_time = event.cuda_time_total
                if cuda_time > 0:
                    events.append((cuda_time, event.key))
    except AttributeError:
        # Fallback: try accessing through profiler.kineto_results
        try:
            for event in prof.profiler.kineto_results.events():
                if event.device_type() == torch.profiler.DeviceType.CUDA:
                    cuda_time = event.cuda_time_range.elapsed_us()
                    if cuda_time > 0:
                        events.append((cuda_time, event.name()))
        except AttributeError:
            print("  Warning: Could not access profiler events")
            return

    # Sort by time (descending)
    events.sort(reverse=True, key=lambda x: x[0])

    for i, (cuda_time_us, name) in enumerate(events[:num_kernels], 1):
        # Handle both microseconds and milliseconds
        if cuda_time_us > 10000:  # Likely in microseconds
            cuda_time_ms = cuda_time_us / 1000.0
        else:  # Already in milliseconds
            cuda_time_ms = cuda_time_us
        print(f"  {i:2d}. {cuda_time_ms:8.2f} ms  {name}")

    if not events:
        print("  No CUDA kernels found in profile")


def benchmark_startup(
    model: str,
    optimization_level: int = 0,
    num_runs: int = 3,
    profile: bool = False,
    output_dir: str | None = None,
    top_kernels: int = 10,
    **kwargs,
):
    """Benchmark cold startup/compilation time for vLLM.

    Args:
        model: Model name or path
        optimization_level: Optimization level (0, 1, 2, 3)
        num_runs: Number of cold starts to measure
        profile: Whether to use PyTorch profiler
        output_dir: Directory to save flame graphs (if profiling)
        top_kernels: Number of top kernels to display
        **kwargs: Additional arguments to pass to LLM
    """
    results = []

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path.cwd() / "profiles"
        output_path.mkdir(parents=True, exist_ok=True)

    print(f"Benchmarking cold startup/compilation time for {model}")
    print(f"Optimization level: -O{optimization_level}")
    print(f"Number of runs: {num_runs}")
    if profile:
        print(f"Profiling enabled - flame graphs will be saved to: {output_path}")
    print("-" * 60)

    for run in range(1, num_runs + 1):
        print(f"\nRun {run}/{num_runs}:")

        # Clear compilation caches for true cold start
        print("  Clearing compilation caches...")
        clear_compilation_caches()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Measure startup time
        start_time = time.perf_counter()

        try:
            if profile:
                # Use PyTorch profiler to capture compilation details
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                    with_stack=True,
                    profile_memory=True,
                ) as prof:
                    llm = LLM(
                        model=model, optimization_level=optimization_level, **kwargs
                    )

                    # Do a dummy generation to ensure everything is fully
                    # initialized. This ensures compilation is complete.
                    _ = llm.generate(
                        ["test"], SamplingParams(temperature=0.0, max_tokens=1)
                    )

                # Export flame graph
                flame_graph_path = (
                    output_path / f"flamegraph_run{run}_O{optimization_level}.html"
                )
                prof.export_chrome_trace(str(flame_graph_path))
                print(f"  Flame graph saved: {flame_graph_path}")

                # Print top kernels
                print_top_kernels(prof, top_kernels)

                # Print compilation time summary
                print("\n  Compilation time breakdown:")
                print("  " + "-" * 70)
                # Get total CUDA time
                try:
                    key_averages = prof.key_averages()
                    total_cuda_time = sum(
                        event.cuda_time_total
                        for event in key_averages
                        if event.device_type == (torch.profiler.DeviceType.CUDA)
                    )
                    # Convert to ms if in microseconds
                    if total_cuda_time > 10000:
                        total_cuda_time = total_cuda_time / 1000.0
                    print(f"  Total CUDA time: {total_cuda_time:.2f} ms")
                except AttributeError:
                    print("  Total CUDA time: (unavailable)")

            else:
                # Simple timing without profiling
                llm = LLM(model=model, optimization_level=optimization_level, **kwargs)

                # Do a dummy generation to ensure everything is fully
                # initialized
                _ = llm.generate(
                    ["test"], SamplingParams(temperature=0.0, max_tokens=1)
                )

            end_time = time.perf_counter()
            elapsed = end_time - start_time

            results.append(elapsed)
            print(f"\n  Total startup time: {elapsed:.2f} seconds")

            # Cleanup
            del llm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()
            continue

        # Small delay between runs
        time.sleep(1)

    if results:
        print("\n" + "=" * 60)
        print("RESULTS:")
        print(f"  Runs: {len(results)}")
        print(f"  Mean: {sum(results) / len(results):.2f} seconds")
        print(f"  Min:  {min(results):.2f} seconds")
        print(f"  Max:  {max(results):.2f} seconds")
        if len(results) > 1:
            mean = sum(results) / len(results)
            variance = sum((x - mean) ** 2 for x in results) / len(results)
            std_dev = variance**0.5
            print(f"  Std:  {std_dev:.2f} seconds")
        print("=" * 60)

        return results
    else:
        print("\nNo successful runs!")
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark cold startup/compilation time in vLLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=("Model name or path (e.g., RedHatAI/Llama-3.1-8B-Instruct-FP8-block)"),
    )
    parser.add_argument(
        "-O",
        "--optimization-level",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Optimization level (default: 0)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of cold start runs to measure (default: 3)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction (default: 0.9)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable PyTorch profiler for detailed analysis",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save flame graphs (default: ./profiles)",
    )
    parser.add_argument(
        "--top-kernels",
        type=int,
        default=10,
        help="Number of top kernels to display (default: 10)",
    )

    args = parser.parse_args()

    benchmark_startup(
        model=args.model,
        optimization_level=args.optimization_level,
        num_runs=args.num_runs,
        profile=args.profile,
        output_dir=args.output_dir,
        top_kernels=args.top_kernels,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )
