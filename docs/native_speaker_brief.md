# Brief for the native-speaker raters (3 people)

We are building a text-to-speech system for Akan (Twi). We need native speakers for three short tasks.
Please do not discuss your answers with each other before submitting them. Each rater works alone.

## Task 1: now, about 10 minutes (one rater is enough)
Listen to the clips from `reports/samples/` (speaker 2, four short natural recordings) and the two "zero-shot" files
from `reports/zeroshot/` (a machine voice, before any training on our data). Answer:
1. Which variety is the **natural** speaker using: Asante Twi, Akuapem Twi, Fante, or mixed?
2. Is the spelling in the text files consistent with that variety (for example final ɛ and ɔ in words like `deɛ`, `ahodoɔ`)?
3. Does the natural speaker sound clear and pleasant? Any recordings where the audio does not match the written text?
4. How does the **machine** voice sound: understandable, partly, or not at all? Which words are wrong?

## Task 2: after training, about 30 minutes per rater
You will get about 25 short clips in random order, with no labels. Some are natural recordings, some are a baseline
machine voice, some are our fine-tuned model. For **every** clip, give:
- **Naturalness**, 1 to 5 (1 = clearly a machine, 5 = sounds like a person).
- **Intelligibility**, 1 to 5 (1 = cannot tell what is said, 5 = every word clear).
- **Mispronounced words**: write them down. Pay particular attention to **ɛ and ɔ**, to **tone** (a word that sounds
  like a different word) and to **vowel harmony** (a vowel that does not fit the word).
- Anything else odd: clicks, buzzing, cut-off words, strange rhythm.

Rate each clip on its own. Do not try to guess which system made it.

## Task 3: about 20 minutes (any rater)
Write sentences we can use as **unseen test text**, in the variety of the natural speaker, in normal spelling:
- 20 everyday sentences of 5 to 15 words.
- 5 sentences with many **ɛ** and **ɔ** sounds.
- 5 sentences that test **vowel harmony** (words with the +ATR and -ATR vowel sets).
- 5 pairs of words that **differ only in tone** (each pair in a short sentence), with the meaning of each.
- 3 sentences with **numbers written as words** (digits cannot be spoken by the system).
- 2 sentences with a **place name or person's name**.

Please give the text only; do not copy sentences from a textbook or website we might not be allowed to reuse.
Add the meaning in English for each so the report can quote it.

## What we will do with your answers
Your ratings are averaged per system and reported with the per-rater scores. With only 3 raters we do not claim
statistical significance. Your names will not be published unless you ask for credit.
