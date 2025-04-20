import re

def extract_output(block, label):
    pattern = rf"\[{label}\] {label.lower()}:\n(.*?)\n\[compare\]"
    match = re.search(pattern, block, re.DOTALL)
    if match:
        return match.group(1)
    return ""

def compare_outputs_from_block(block):
    student_output = extract_output(block, "Student stdout")
    desired_output = extract_output(block, "Desired stdout")

    print("=== Comparing Outputs ===")
    for i, (s_char, d_char) in enumerate(zip(student_output, desired_output)):
        if s_char != d_char:
            print(f"Difference at index {i}:")
            print(f"  Student: {repr(s_char)}")
            print(f"  Desired: {repr(d_char)}")
            break
    else:
        if len(student_output) != len(desired_output):
            print("No character differences found, but lengths differ:")
            print(f"  Student length: {len(student_output)}")
            print(f"  Desired length: {len(desired_output)}")
        else:
            print("Outputs are identical.")

# Example usage:
block = """
[copy_files] Filed copied: 0-main.py
[compare] Command to run:
./0-main.py 2>/dev/null
su student_jail -c 'timeout 60 bash -c '"'"'./0-main.py 2>/dev/null'"'"''
[compare] Return code: 0
[compare] Student stdout:
True
True
<dtype: 'int64'>
<dtype: 'int64'>
True
<dtype: 'int64'>
<dtype: 'int64'>
[compare] Student stdout length: 83
[compare] Student stderr:
[compare] Student stderr length: 0
[compare] Desired stdout:
True
True
<dtype: 'int64'>
<dtype: 'int64'>
True
<dtype: 'int64'>
<dtype: 'int64'>
[compare] Desired stdout length: 89
"""

compare_outputs_from_block(block)
