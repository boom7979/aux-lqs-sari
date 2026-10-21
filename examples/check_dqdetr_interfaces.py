# SPDX-License-Identifier: Apache-2.0
"""Read-only source-identity check for the documented DQ-DETR hook map.

This checks source files and function locations, not detector execution.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import warnings


REFERENCE_COMMIT = 'c5246797b111d7e5da9e6c20667fe0901aae35ba'
SOURCE_FILES = (
    'models/dqdetr/deformable_transformer.py',
    'models/dqdetr/dqdetr.py',
    'models/dqdetr/utils.py',
    'util/misc.py',
)


def check_source(root):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')

    def git(*args):
        return subprocess.check_output(
            ['git', '-C', str(root), *args], env=env, text=True,
            encoding='utf-8', stderr=subprocess.STDOUT).strip()

    actual = git('rev-parse', 'HEAD')
    if actual != REFERENCE_COMMIT:
        raise ValueError('Checkout differs from the documented interface reference')
    if git('diff', 'HEAD', '--', *SOURCE_FILES):
        raise ValueError('Documented interface source files contain local changes')
    functions = {}
    for name in SOURCE_FILES:
        # Some upstream docstrings use LaTeX escapes. Ignore only their lexer
        # warning when inspecting them on newer Python; syntax errors still fail.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=r'invalid escape sequence.*',
                                    category=SyntaxWarning)
            tree = ast.parse((root / name).read_text(encoding='utf-8-sig'), filename=name)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in {'DeformableTransformer', 'SetCriterion'}:
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == 'forward':
                        functions[node.name + '.forward'] = {'file': name, 'line': method.lineno}
    if set(functions) != {'DeformableTransformer.forward', 'SetCriterion.forward'}:
        raise ValueError('Expected detector interfaces were not found')
    return {'reference_commit': actual, 'source_files_match_reference': True,
            'functions': functions,
            'validation_scope': 'source identity and interface locations only',
            'detector_execution_tested': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path, help='Existing official DQ-DETR checkout')
    args = parser.parse_args()
    print(json.dumps(check_source(args.checkout.resolve()), indent=2))


if __name__ == '__main__':
    main()
