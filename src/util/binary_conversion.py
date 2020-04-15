def from_binary_to_float_in_range(bits, exponent_length, exponent_range):
    """
    Convert from binary to float within a certain range.

    :param bits: The list of bits to convert to a float
    :param exponent_length: The amount of exponent bits
    :param exponent_range: The range of the exponent
    :return: Float withing desired range
    """
    exponent_bits = bits[:exponent_length]
    mantissa_bits = bits[exponent_length:]

    # Calculate mantissa
    mantissa = 0
    for i, bit in enumerate(mantissa_bits):
        ex = -(i+1)
        mantissa += bit*(2**ex)
    if exponent_bits.any():
        mantissa += 1

    # Calculate exponent and scale into desired range
    exponent_string = [*map(str, [*map(int, exponent_bits.tolist())])]
    exponent = int(''.join(exponent_string), 2)
    lower_bound, upper_bound = 0, 2**exponent_length - 1
    lower_range, upper_range = exponent_range[0], exponent_range[1] - 1  # Up until, but not including upper range
    exp_norm = lower_range+(((exponent - lower_bound)*(upper_range-lower_range)) / (upper_bound - lower_bound))

    result = mantissa*2**exp_norm

    if result == 0:  # If zero, return lowest possible number
        result = 2**lower_range
    return result
