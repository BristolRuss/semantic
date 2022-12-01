import spacy
nlp = spacy.load('en_core_web_md')

word1 = nlp("Cat")
word2 = nlp("Monkey")
word3 = nlp("Banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

word1 = nlp("Oven")
word2 = nlp("TV")
word3 = nlp("Fire")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

# I find it interesting that "fire" and "oven" do not have a higher score than they do

tokens = nlp('cat apple monkey banana')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
            "Hello, there is my car",
            "I\'ve lost my car in my car",
            "I\'d like my boat back",
            "I will name my dog Diana"
            ]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

# When I change the language from "en_core_web_md" to "en_core_web_sm" you immediately notice a bunch of warnings like the one below. 

# "c:\Users\Brist\Dropbox\JR22100003858\Software Engineer Bootcamp\T38\semantic.py:38: UserWarning: [W007] The model you're using has no word vectors loaded, so the result of the Doc.similarity method will be based on the tagger, parser and NER, which may not give useful similarity judgements. This may happen if you're using one of the small models, e.g. `en_core_web_sm`, which don't ship with word vectors and only use context-sensitive tensors. You can always add 
# your own word vectors, or use one of the larger models instead if available.
# similarity = nlp(sentence).similarity(model_sentence)"

# More than this though when checking the results which still gave a score, when we use "en_core_web_md" we get the following results - with "Hello, there is my car" as the best result.

# where did my dog go -  0.630065230699739
# Hello, there is my car -  0.8033180111627156
# I've lost my car in my car -  0.6787541571030323
# I'd like my boat back -  0.5624940517078084
# I will name my dog Diana -  0.6491444739190607

# but if we change it to use "en_core_web_sm" we get the following results - with "I've lost my car in my car" now as the best result

# where did my dog go -  0.46559175808905845
# Hello, there is my car -  0.5229294863903653
# I've lost my car in my car -  0.6350136333854123
# I'd like my boat back -  0.43399616845180666
# I will name my dog Diana -  0.3717855214295043