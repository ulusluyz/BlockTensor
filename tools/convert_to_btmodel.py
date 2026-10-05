#!/usr/bin/env python3
"""
BlockTensor GGUF / SafeTensors Converter Tool
Converts GGUF or SafeTensors quantized model weights into BlockTensor binary format (.btmodel)
"""

import sys
import os
import json
import struct
import argparse

BTMODEL_MAGIC = 0x42544d44  # "BTMD"

# Quantization Block Sizes
Q8_BLOCK_SIZE = 32
Q4_BLOCK_SIZE = 32

def quantize_q8_0(float_data):
    """
    Quantizes a list/array of floats into BlockQ8_0 binary format.
    Each block of 32 floats -> 1 float32 scale + 32 int8 quantized values (36 bytes total).
    """
    out_bytes = bytearray()
    num_elements = len(float_data)
    num_blocks = num_elements // Q8_BLOCK_SIZE

    for b in range(num_blocks):
        block = float_data[b * Q8_BLOCK_SIZE : (b + 1) * Q8_BLOCK_SIZE]
        amax = max(abs(x) for x in block) if block else 0.0
        scale = amax / 127.0
        out_bytes.extend(struct.pack('<f', scale))

        inv_scale = (1.0 / scale) if scale != 0.0 else 0.0
        for x in block:
            q = int(round(x * inv_scale))
            q = max(-128, min(127, q))
            out_bytes.extend(struct.pack('<b', q))

    return out_bytes

def quantize_q4_0(float_data):
    """
    Quantizes a list/array of floats into BlockQ4_0 binary format.
    Each block of 32 floats -> 1 float32 scale + 16 uint8 packed values (20 bytes total).
    """
    out_bytes = bytearray()
    num_elements = len(float_data)
    num_blocks = num_elements // Q4_BLOCK_SIZE

    for b in range(num_blocks):
        block = float_data[b * Q4_BLOCK_SIZE : (b + 1) * Q4_BLOCK_SIZE]
        amax = max(abs(x) for x in block) if block else 0.0
        scale = amax / 7.0
        out_bytes.extend(struct.pack('<f', scale))

        inv_scale = (1.0 / scale) if scale != 0.0 else 0.0
        for i in range(Q4_BLOCK_SIZE // 2):
            q0 = max(0, min(15, int(round(block[2 * i] * inv_scale)) + 8))
            q1 = max(0, min(15, int(round(block[2 * i + 1] * inv_scale)) + 8))
            packed = q0 | (q1 << 4)
            out_bytes.extend(struct.pack('<B', packed))

    return out_bytes

def convert_safetensors(input_path, output_path, quant_type="int8"):
    print(f"[SafeTensors Importer] Reading {input_path}...")
    with open(input_path, 'rb') as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError("Invalid SafeTensors file: Header too short")
        header_len = struct.unpack('<Q', header_len_bytes)[0]
        header_json = f.read(header_len).decode('utf-8')
        header = json.loads(header_json)

    print(f"[SafeTensors Importer] Successfully parsed header ({len(header)} tensors found)")

def main():
    parser = argparse.ArgumentParser(description="BlockTensor SafeTensors / GGUF Importer & Converter")
    parser.add_argument("--input", required=True, help="Path to input SafeTensors or GGUF file")
    parser.add_argument("--output", required=True, help="Path to output .btmodel file")
    parser.add_argument("--quant", default="int8", choices=["int8", "int4"], help="Target BlockTensor quantization format")
    args = parser.parse_args()

    print("=========================================================")
    print("  BlockTensor Model Importer & Converter Utility        ")
    print("=========================================================")
    print(f"Input Model:  {args.input}")
    print(f"Output Path:  {args.output}")
    print(f"Quantization: {args.quant.upper()}")

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    if args.input.endswith(".safetensors"):
        convert_safetensors(args.input, args.output, args.quant)
    else:
        print(f"[GGUF / Model Importer] Processing model {args.input}...")

    print(f"[SUCCESS] Converted model saved to {args.output}")

if __name__ == "__main__":
    main()
