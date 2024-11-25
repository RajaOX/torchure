import torch
import time
import argparse
import csv
import math

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Matrix Multiplication Benchmark')
    parser.add_argument('--num-devices', type=int, default=1, help='Number of devices to use (up to 16)')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type (float32, float16, bfloat16)')
    parser.add_argument('--warmup', action='store_true', help='Include a warmup loop')
    parser.add_argument('--output-file', type=str, default='benchmark_results.csv', help='Output CSV file')
    return parser.parse_args()

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    elif torch.xpu.is_available():
        return 'xpu'
    else:
        return 'cpu'

def get_devices(num_devices, device_type):
    devices = []
    if device_type == 'cuda':
        available_devices = torch.cuda.device_count()
        num_devices = min(num_devices, available_devices)
        devices = [f'cuda:{i}' for i in range(num_devices)]
    elif device_type == 'mps':
        devices = ['mps']  # Assume only one MPS device
        num_devices = 1
    elif device_type == 'xpu':
        available_devices = torch.xpu.device_count()
        num_devices = min(num_devices, available_devices)
        devices = [f'xpu:{i}' for i in range(num_devices)]
    else:
        devices = ['cpu'] * num_devices
    return devices

def synchronize_devices(devices):
    for device in devices:
        if 'cuda' in device:
            torch.cuda.synchronize(device)
        elif 'mps' in device:
            torch.mps.synchronize()
        elif 'xpu' in device:
            torch.xpu.synchronize()
        # For CPU, no need to synchronize

def run_benchmark(devices, dtype, warmup, output_file):
    # Define N_list from 8 to 65536 (2^8 to 2^16)
    N_list = [2**i for i in range(8, 16)]  # 8, 16, ..., 65536

    # Create output CSV file
    fieldnames = ['N', 'M', 'TFlops', 'Time_s', 'Bandwidth_GBps']
    with open(output_file, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for N in N_list:
            # M varies from 1 to N in powers of 2
            M_list = [2**j for j in range(2, int(math.log2(N)) + 1)]  # 1, 2, ..., N
            for M in M_list:
                print(f"Running benchmark for N={N}, M={M}")
                try:
                    total_time, tflops, bandwidth = benchmark_matmul(N, M, devices, dtype, warmup)
                    print(f"N={N}, M={M}, Time={total_time:.6f}s, TFlops={tflops:.2f}, Bandwidth={bandwidth:.2f} GB/s")
                    # Write results to CSV
                    writer.writerow({'N': N, 'M': M, 'TFlops': tflops, 'Time_s': total_time, 'Bandwidth_GBps': bandwidth})
                    csvfile.flush()
                except RuntimeError as e:
                    print(f"Skipping N={N}, M={M} due to error: {e}")
                    # Optionally write NaN or skip
                    writer.writerow({'N': N, 'M': M, 'TFlops': 'NaN', 'Time_s': 'NaN', 'Bandwidth_GBps': 'NaN'})
                    csvfile.flush()

def benchmark_matmul(N, M, devices, dtype, warmup):
    K = N  # Since the square matrix is N x N

    # Create input tensors
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }

    if dtype not in dtype_map:
        print(f"Unsupported dtype {dtype}, defaulting to bfloat16")
        dtype = 'bfloat16'

    tensor_dtype = dtype_map[dtype]

    num_devices = len(devices)

    # Generate random data
    A = torch.randn(N, K, dtype=tensor_dtype)
    B = torch.randn(K, M, dtype=tensor_dtype)

    # Split A across devices
    A_pieces = A.chunk(num_devices, dim=0)

    # Move B to devices (assuming we need to move B to each device)
    B_on_devices = [B.to(device) for device in devices]

    # Move A_pieces to respective devices
    A_pieces = [A_pieces[i].to(devices[i]) for i in range(num_devices)]

    # Warmup loop
    if warmup:
        for _ in range(5):
            outputs = []
            for i, device in enumerate(devices):
                output = torch.mm(A_pieces[i], B_on_devices[i])
                outputs.append(output)
            synchronize_devices(devices)

    # Timing loop
    if 'cuda' in devices[0]:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_time = time.time()

    outputs = []
    for i, device in enumerate(devices):
        output = torch.mm(A_pieces[i], B_on_devices[i])
        outputs.append(output)

    synchronize_devices(devices)

    if 'cuda' in devices[0]:
        end_event.record()
        torch.cuda.synchronize()
        total_time = start_event.elapsed_time(end_event) / 1000.0  # milliseconds to seconds
    else:
        end_time = time.time()
        total_time = end_time - start_time

    # Compute performance metrics
    total_flops = 2 * N * K * M
    tflops = total_flops / total_time / 1e12

    # Compute memory bandwidth (approximate)
    element_size = torch.tensor([], dtype=tensor_dtype).element_size()

    A_piece_size = (N // num_devices) * K * element_size
    B_size = K * M * element_size
    output_size = (N // num_devices) * M * element_size

    total_bytes = num_devices * (A_piece_size + B_size + output_size)

    bandwidth = total_bytes / total_time / 1e9  # GB/s

    return total_time, tflops, bandwidth

if __name__ == '__main__':
    args = parse_args()
    device_type = get_device()
    devices = get_devices(args.num_devices, device_type)
    print(f"Using devices: {devices}")
    run_benchmark(devices, args.dtype, args.warmup, args.output_file)
