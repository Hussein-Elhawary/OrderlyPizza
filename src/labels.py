from utils import read_test_cases as rtc
import re

tc = rtc('./try.json')
unique_labels = set()
tcs = []
pattern = r'(?<="train.TOP-DECOUPLED": ").+(?="},)'
pattern2 = r'(?<=\().*? '
tcs = re.finditer(pattern, tc)
labels = []
print("first")
for match in tcs:
    labels = re.finditer(pattern2, match.group())
    for label in labels:
        label = label.group().strip(' ')
        unique_labels.add(label)

print(unique_labels)
with open('unique_labels.txt', 'w') as f:
    for label in unique_labels:
        f.write(f"{label}\n")