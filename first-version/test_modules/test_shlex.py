import shlex

test_cases = [
    r'"C:\Users\nutbred\Downloads\Nonconvex-Concave Minimax Optimization.pdf"',
    r'C:\Users\nutbred\Downloads\file.pdf',
    r'"C:\Program Files\App\doc.pdf" "D:\Data\report.pdf"'
]

print("Testing posix=True (default):")
for case in test_cases:
    try:
        print(f"Input: {case}")
        print(f"Output: {shlex.split(case)}")
    except ValueError as e:
        print(f"Error: {e}")

print("\nTesting posix=False:")
for case in test_cases:
    try:
        print(f"Input: {case}")
        print(f"Output: {shlex.split(case, posix=False)}")
    except ValueError as e:
        print(f"Error: {e}")
