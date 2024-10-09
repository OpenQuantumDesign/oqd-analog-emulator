# Bash script to run unit tests in test directory
python -m unittest discover tests/test_analog -v
python -m unittest discover tests/test_compiler_infrastructure -v