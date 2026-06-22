"""xe-forge-skill benchmark: Correctness + performance comparison."""


def run(args):
    if getattr(args, "dsl", "triton") == "cm":
        _run_cm(args)
        return

    from pathlib import Path

    from xe_forge.core.executor import KernelBenchExecutor
    from xe_forge.core.spec_loader import load_spec

    baseline_code = Path(args.baseline).read_text(encoding="utf-8")
    optimized_code = Path(args.optimized).read_text(encoding="utf-8")

    spec = load_spec(args.spec)
    variant = spec.resolve_variant(args.variant)
    input_shapes = spec.get_input_shapes(variant)
    flop = spec.get_flop(variant)
    dtype = spec.get_dtype(variant)
    input_dtypes = spec.get_input_dtypes(variant)
    init_args = spec.get_init_args(variant)

    executor = KernelBenchExecutor(device=args.device)

    if args.baseline_us is not None:
        baseline_us = [float(v) for v in str(args.baseline_us).split(",")]
        print(f"Using cached baseline: {baseline_us} us")
        optimized_result = executor.execute(
            optimized_code,
            None,
            input_shapes,
            flop=flop,
            dtype=dtype,
            init_args=init_args,
            input_dtypes=input_dtypes,
        )
        if optimized_result.success:
            baseline_ms = sum(baseline_us) / len(baseline_us) / 1000.0
            opt_ms = optimized_result.execution_time_ms
            speedup = baseline_ms / opt_ms if opt_ms > 0 else 0
            print(f"Correctness: {'PASSED' if optimized_result.success else 'FAILED'}")
            print(
                f"Performance: baseline_us={baseline_ms * 1000:.2f}, "
                f"triton_us={opt_ms * 1000:.2f}, speedup={speedup:.2f}x"
            )
        else:
            print("Correctness: FAILED")
            print(f"Error: {optimized_result.error_message}")
    else:
        result = executor.compare_kernels(
            original_code=baseline_code,
            optimized_code=optimized_code,
            input_shapes=input_shapes,
            flop=flop,
            dtype=dtype,
            init_args=init_args,
            input_dtypes=input_dtypes,
        )
        print(f"Correctness: {'PASSED' if result.optimized_correct else 'FAILED'}")
        if result.original_time_us and result.optimized_time_us:
            print(
                f"Performance: baseline_us={result.original_time_us:.2f}, "
                f"triton_us={result.optimized_time_us:.2f}, speedup={result.speedup:.2f}x"
            )
        if result.feedback_message:
            print(f"Feedback: {result.feedback_message}")


def _run_cm(args):
    """CM ("C for Metal") benchmark: compile + run baseline vs optimized on the GPU.

    Unlike the Triton path, CM correctness is verified by comparing the optimized
    kernel's output against the baseline kernel's output (numpy), so we always run
    both kernels here -- the cached ``--baseline-us`` shortcut is intentionally
    ignored for CM.
    """
    import os
    from pathlib import Path

    from xe_forge.core.cm_executor import CMExecutor
    from xe_forge.core.spec_loader import load_spec

    baseline_code = Path(args.baseline).read_text(encoding="utf-8")
    optimized_code = Path(args.optimized).read_text(encoding="utf-8")

    spec = load_spec(args.spec)
    variant = spec.resolve_variant(args.variant)
    input_shapes = spec.get_input_shapes(variant)
    input_dtypes = spec.get_input_dtypes(variant)
    dims = spec.get_dims(variant)
    output_shapes = spec.get_output_shapes(variant)
    output_dtypes = spec.get_output_dtypes(variant)
    flop = spec.get_flop(variant)
    rtol = spec.get_rtol(variant)
    atol = spec.get_atol(variant)

    executor = CMExecutor(iterations=int(os.environ.get("BENCHMARK_ITERATIONS", "20")))
    executor.grid_spec = spec.grid

    result = executor.compare_kernels(
        original_code=baseline_code,
        optimized_code=optimized_code,
        dims=dims,
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        flop=flop,
        rtol=rtol if rtol is not None else 1e-2,
        atol=atol if atol is not None else 1e-3,
    )

    print(f"Correctness: {'PASSED' if result.optimized_correct else 'FAILED'}")
    print(
        f"Performance: baseline_us={result.original_time_us:.2f}, "
        f"triton_us={result.optimized_time_us:.2f}, speedup={result.speedup:.2f}x"
    )
    if result.feedback_message:
        print(f"Feedback: {result.feedback_message}")
