#!/usr/bin/env python3

import logging as log
import sys
from gpt_neo import GPTNeo, GPTNeoConfig
from argparse import ArgumentParser, SUPPRESS

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-i", "--input", help="Optional. Input prompt", required=False, type=str, action='append')
    return parser


if __name__ == '__main__':

    parser = build_argparser().parse_args()

    config = GPTNeoConfig()
    gpt_neo = GPTNeo(config)

    if parser.input:
        def prompts():
            for prompt in parser.input:
                log.info("Q: {}".format(prompt))
                yield prompt
    else:
        def prompts():
            while True:
                yield input('Q:')

    for prompt in prompts():
        response = gpt_neo.infer(prompt)
        print(f'GPT: {response}\n')
        print("-" * 70)
