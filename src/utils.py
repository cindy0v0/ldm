import argparse

def parse_int_list(arg_string):
    """
    Custom type function to parse a comma-separated string of integers
    into a list of integers.
    """
    try:
        # Split the string by comma, strip whitespace from each part,
        # and convert to integer.
        # args.noise_timesteps = list(map(int, args.noise_timesteps.split(',')))
        return [int(s.strip()) for s in arg_string.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid integer list format: '{arg_string}'. "
            "Please provide a comma-separated list of integers."
        )