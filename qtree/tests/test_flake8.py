import os
import subprocess
import tempfile
import sys

import qtree


def test_flake8():
    qtree_dir = os.path.dirname(os.path.abspath(qtree.__file__))
    output_file = tempfile.mkdtemp()
    output_file = os.path.join(output_file, 'flake8.out')
    if os.path.exists(output_file):
        os.remove(output_file)
    output_string = "--output-file=%s" % output_file
    subprocess.call([sys.executable, '-m', 'flake8', output_string, qtree_dir])

    if os.path.exists(output_file):
        with open(output_file) as f:
            flake8_output = f.readlines()
        if flake8_output != []:
            raise AssertionError(
                "flake8 found style errors:\n\n%s" % "\n".join(flake8_output))
