# Suite: exact_answer

## What the tasks are like

Each task is one short, self-contained question with a single unambiguous answer. No files, tools or outside lookups are needed. The questions are the kind a small model gets wrong some of the time through careless reasoning, not missing knowledge:

- counting how often a letter appears in a word or phrase;
- small multi-step arithmetic;
- day-of-week and calendar-date arithmetic from a stated reference point;
- unit conversion;
- short logic puzzles with one consistent solution;
- reversing a string character by character;
- rate and work word problems.

Some answers are numbers and some are short text (a word, a name, a date or a string).

## Answer format

The agent's instruction is the question plus a request to end its reply with one final line:

```
ANSWER: <answer>
```

Only the agent's final text is graded. Reasoning before that line is allowed and ignored.

## Grading rules

1. **Find the answer line.** An answer line is a line that **starts** with the marker `ANSWER:` (case-insensitive). Before the word `ANSWER`, only whitespace and the markdown characters `#`, `>`, `*`, `_`, `` ` `` and `-` may appear. Between `ANSWER` and the colon, only whitespace, `*`, `_` and `` ` `` may appear. Any `*`, `_` or `` ` `` directly after the colon is part of the marker. So `**ANSWER:** x`, `**ANSWER**: x`, `## ANSWER: x`, `> ANSWER: x`, `- ANSWER: x` and `` `ANSWER: x` `` all work, but a line with any other text before the marker, such as `So my final ANSWER: x` or `Note: my answer: ...`, is not an answer line. The grader reads the **last** answer line in the final text, so prose after it never replaces it. If there is no answer line, the reward is 0. An answer on the line after an empty `ANSWER:` is not read.
2. **Normalize** the text after the marker, and each accepted answer, the same way:
   - remove surrounding whitespace (any Unicode whitespace, including no-break spaces) and surrounding quotes (`"`, `'`, curly quotes), backticks and asterisks, repeating until neither end changes;
   - remove one trailing period, then repeat the previous step, so `**x**.` and `**x.**` both become `x`;
   - lowercase;
   - collapse runs of internal whitespace to a single space.
3. **Compare.**
   - **Text tasks:** the normalized answer must equal one accepted answer exactly. Extra words fail.
   - **Numeric tasks:** commas are removed only when they are correct thousands separators (`1,234.5`). The result must then be a plain decimal number (optional sign, digits, optional decimal point and exponent). Units, currency symbols, words, fractions such as `1/2`, and spellings like `1_000` all fail. A valid number passes if it is within `1e-6 * max(1, |expected|)` of the expected value, so `7`, `7.0` and `7.00` are the same answer.

The grader's feedback on each trial says what answer it parsed and whether it was correct. It never reveals the expected answer.

## Metric

**Accuracy.** Each trial scores 1.0 if its parsed answer is correct and 0.0 otherwise. A candidate's score is the mean over tasks and trials, with errored trials counted as 0. Higher is better.
